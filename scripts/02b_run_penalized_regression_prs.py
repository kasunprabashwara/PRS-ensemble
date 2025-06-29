import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib
import os

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


import config
from utils import run_plink_command

def clump_snps(gwas_sumstats_file, ld_ref_bfile, clump_out_prefix, p1, r2, kb):
    """Clump SNPs using PLINK based on GWAS p-values."""
    print(f"Clumping SNPs from {gwas_sumstats_file}")
    args = [
        "--bfile", ld_ref_bfile,
        "--clump", gwas_sumstats_file,
        "--clump-p1", str(p1),
        "--clump-r2", str(r2),
        "--clump-kb", str(kb),
        "--clump-snp-field", "SNP",
        "--clump-field", "P"
    ]
    run_plink_command(args, "Clumping", log_prefix=clump_out_prefix)

    clumped_file = f"{clump_out_prefix}.clumped"
    if not os.path.exists(clumped_file) or os.path.getsize(clumped_file) == 0:
        print(f"Warning: Clumping output file {clumped_file} is empty. No SNPs selected.")
        return []

    clumped_df = pd.read_csv(clumped_file, delim_whitespace=True)
    return clumped_df['SNP'].unique().tolist() if 'SNP' in clumped_df else []

def extract_genotypes(plink_bfile_prefix, snp_list, out_prefix, keep_ids_file=None):
    """Extract specified SNPs for specified individuals and recode to additive format (0,1,2)."""
    if not snp_list:
        print("SNP list is empty, cannot extract genotypes.")
        return pd.DataFrame()

    print(f"Extracting {len(snp_list)} SNPs...")
    snp_list_file = f"{out_prefix}_snp_list.txt"
    with open(snp_list_file, 'w') as f:
        for snp in snp_list:
            f.write(f"{snp}\n")

    args = [
        "--bfile", plink_bfile_prefix,
        "--extract", snp_list_file,
        "--recode", "A", # Additive format (0, 1, 2)
        "--out", out_prefix
    ]
    if keep_ids_file and os.path.exists(keep_ids_file):
        args.extend(["--keep", keep_ids_file])

    run_plink_command(args, "Genotype Extraction") # PLINK1 does not need log_prefix with --out

    raw_file = f"{out_prefix}.raw"
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"PLINK .raw file not found: {raw_file}. Genotype extraction failed.")

    geno_df = pd.read_csv(raw_file, delim_whitespace=True)
    geno_df.rename(columns={'#IID': 'IID'}, inplace=True)
    geno_df.set_index('IID', inplace=True)
    # Drop metadata columns that are not genotypes
    meta_cols = ['FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    geno_df = geno_df.drop(columns=[col for col in meta_cols if col in geno_df.columns])
    # Rename SNP columns from 'SNP_A' to 'SNP'
    geno_df.columns = [col.rsplit('_', 1)[0] for col in geno_df.columns]
    return geno_df

def train_penalized_model(X_train, y_train):
    """Train a penalized logistic regression model with cross-validation."""
    print("Training Penalized Regression model (Logistic ElasticNet)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    cv = StratifiedKFold(n_splits=config.PENREG_CV_FOLDS, shuffle=True, random_state=42)
    model = LogisticRegressionCV(
        Cs=10,
        cv=cv,
        penalty='elasticnet',
        solver='saga',
        l1_ratios=config.PENREG_L1_RATIOS,
        max_iter=5000,
        n_jobs=-1,
        random_state=42,
        scoring='roc_auc'
    )
    model.fit(X_train_scaled, y_train)

    print(f"Best parameters: l1_ratio={model.l1_ratio_[0]:.2f}, C={model.C_[0]:.4f}")
    return model, scaler

def main():
    """Main function to run the penalized regression PRS pipeline."""
    print("--- Starting Penalized Regression PRS Calculation ---")

    # 1. Clump SNPs
    clump_output_prefix = os.path.join(config.RESULTS_DIR, "penreg/clumped_snps")
    clumped_snps = clump_snps(
        gwas_sumstats_file=config.GWAS_PROCESSED_FILE,
        ld_ref_bfile=config.LD_REF_PLINK_PREFIX,
        clump_out_prefix=clump_output_prefix,
        p1=config.PENREG_CLUMP_P1,
        r2=config.PENREG_CLUMP_R2,
        kb=config.PENREG_CLUMP_KB
    )
    if not clumped_snps:
        print("No SNPs left after clumping. Exiting Penalized Regression step.")
        # Create empty PRS files to not break the pipeline
        for dset in ['train', 'test']:
            pd.DataFrame(columns=['IID', 'PRS_penreg']).to_csv(os.path.join(config.BASE_PRS_DIR, f"{dset}_penreg.prs"), sep='\t', index=False)
        return
    joblib.dump(clumped_snps, os.path.join(config.MODEL_DIR, "penreg_snps.joblib"))

    # 2. Extract genotypes and train model on the training set
    print("\nProcessing training data...")
    train_geno_prefix = os.path.join(config.RESULTS_DIR, "penreg/train_genotypes")
    train_genos = extract_genotypes(
        plink_bfile_prefix=config.TARGET_GENO_PREFIX,
        snp_list=clumped_snps,
        out_prefix=train_geno_prefix,
        keep_ids_file=config.TRAIN_IDS_FILE
    )
    pheno_df = pd.read_csv(config.PHENO_FILE)
    pheno_df['IID'] = pheno_df['IID'].astype(str)
    
    train_data = train_genos.join(pheno_df.set_index('IID'), how='inner')
    X_train = train_data[clumped_snps]
    y_train = train_data[config.TARGET_DISEASE_COLUMN]

    model, scaler = train_penalized_model(X_train, y_train)
    joblib.dump(model, os.path.join(config.MODEL_DIR, "penreg_model.joblib"))
    joblib.dump(scaler, os.path.join(config.MODEL_DIR, "penreg_scaler.joblib"))

    # 3. Generate PRS for both training and testing sets
    for dset, ids_file in [('train', config.TRAIN_IDS_FILE), ('test', config.TEST_IDS_FILE)]:
        print(f"\nGenerating Penalized Regression PRS for {dset} set...")
        geno_prefix = os.path.join(config.RESULTS_DIR, f"penreg/{dset}_genotypes_full")
        
        all_genos = extract_genotypes(
            plink_bfile_prefix=config.TARGET_GENO_PREFIX,
            snp_list=clumped_snps,
            out_prefix=geno_prefix,
            keep_ids_file=ids_file
        )

        if all_genos.empty:
            print(f"No genotypes extracted for {dset} set. Skipping PRS calculation.")
            continue
        
        # Ensure columns are in the same order as training
        all_genos = all_genos[clumped_snps]
        X_all_scaled = scaler.transform(all_genos)
        
        # Use decision_function for a continuous score
        prs = model.decision_function(X_all_scaled)

        prs_df = pd.DataFrame({'IID': all_genos.index, 'PRS_penreg': prs})
        output_file = os.path.join(config.BASE_PRS_DIR, f"{dset}_penreg.prs")
        prs_df.to_csv(output_file, sep='\t', index=False)
        print(f"Penalized regression PRS for {dset} set saved to {output_file}")

    print("--- Penalized Regression PRS Calculation Finished ---")

if __name__ == '__main__':
    main()