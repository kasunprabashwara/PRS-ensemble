import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
import joblib
import os
import config
from scripts.utils import run_plink_command, load_phenotypes, load_ids_from_file

def clump_snps_plink(gwas_sumstats_file, plink_bfile_for_ld, clump_out_prefix,
                     p1_thresh, p2_thresh, r2_thresh, kb_thresh,
                     clump_snp_field='SNP', clump_p_field='P'):
    """Clump SNPs using PLINK based on GWAS p-values and LD from a reference bfile."""
    os.makedirs(os.path.dirname(clump_out_prefix), exist_ok=True)
    args = [
        "--bfile", plink_bfile_for_ld,
        "--clump", gwas_sumstats_file,
        "--clump-p1", str(p1_thresh),
        "--clump-p2", str(p2_thresh),
        "--clump-r2", str(r2_thresh),
        "--clump-kb", str(kb_thresh),
        "--clump-snp-field", clump_snp_field, # Specifies SNP ID column in sumstats
        "--clump-field", clump_p_field,    # Specifies P-value column in sumstats
        "--out", clump_out_prefix
    ]
    run_plink_command(args)
    
    clumped_file = f"{clump_out_prefix}.clumped"
    if not os.path.exists(clumped_file) or os.path.getsize(clumped_file) == 0:
        print(f"Warning: Clumping output file {clumped_file} is empty or not found.")
        return []
    
    clumped_df = pd.read_csv(clumped_file, delim_whitespace=True)
    return clumped_df['SNP'].unique().tolist() if 'SNP' in clumped_df else []


def extract_genotypes_for_snps(plink_bfile_prefix, snp_list, out_prefix, keep_ids_file=None):
    """Extract specified SNPs for specified individuals and recode to additive format (0,1,2)."""
    snp_list_file = f"{out_prefix}_snp_list.txt"
    with open(snp_list_file, 'w') as f:
        for snp in snp_list:
            f.write(f"{snp}\n")
    
    args = [
        "--bfile", plink_bfile_prefix,
        "--extract", snp_list_file,
        "--recode", "A", # Additive genetic model (0,1,2 dosages for A1)
        "--out", out_prefix
    ]
    if keep_ids_file and os.path.exists(keep_ids_file):
        args.extend(["--keep", keep_ids_file])
    
    run_plink_command(args, cmd_name="PLINK") # Using PLINK 1.9 for --recode A
    
    raw_file = f"{out_prefix}.raw"
    if not os.path.exists(raw_file):
         raise FileNotFoundError(f"PLINK .raw file not found: {raw_file}. SNP extraction might have failed.")
    
    # Read .raw file. SNP columns are named SNPID_EFFECTALLELE
    # The effect allele for --recode A is A1 from the .bim file.
    # We need to be careful about matching these SNPID_EFFECTALLELE columns with our snp_list
    geno_df = pd.read_csv(raw_file, delim_whitespace=True)
    
    # Rename IID column if necessary
    if 'IID' not in geno_df.columns and '#IID' in geno_df.columns:
        geno_df.rename(columns={'#IID': 'IID'}, inplace=True)
    elif 'IID' not in geno_df.columns and 'ID' in geno_df.columns: # some plink versions
         geno_df.rename(columns={'ID': 'IID'}, inplace=True)


    # Identify SNP dosage columns. These usually end with _A, _C, _G, or _T
    # We need to map them back to the original SNP IDs from snp_list
    # This part can be tricky due to allele encoding (_A might not mean A is the counted allele)
    # PLINK's --recode A counts the A1 allele from the BIM file.
    # We need to read the BIM file for the extracted SNPs to confirm the A1 alleles.
    bim_df = pd.read_csv(f"{out_prefix}.bim", delim_whitespace=True, header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])
    bim_map = pd.Series(bim_df.A1.values, index=bim_df.SNP).to_dict()

    snp_dosage_cols = []
    final_snp_names_in_order = [] # SNPs that were successfully extracted and in order
    
    for snp_id in snp_list: # Iterate in the order of our original snp_list
        if snp_id in bim_map: # If the SNP was present in the plink output
            effect_allele = bim_map[snp_id]
            col_name_in_raw = f"{snp_id}_{effect_allele}"
            if col_name_in_raw in geno_df.columns:
                snp_dosage_cols.append(col_name_in_raw)
                final_snp_names_in_order.append(snp_id)
            # else: # SNP might have been monomorphic or removed
            # print(f"Warning: SNP {snp_id} (expected col {col_name_in_raw}) not found in .raw columns after extraction.")
        # else:
            # print(f"Warning: SNP {snp_id} not found in .bim file after extraction ({out_prefix}.bim).")


    if not snp_dosage_cols:
        print("Warning: No SNP dosage columns could be identified or matched from .raw file.")
        return pd.DataFrame(columns=['IID']), []

    # Select IID and only the identified SNP dosage columns, in the order of final_snp_names_in_order
    geno_df_selected = geno_df[['IID'] + snp_dosage_cols].set_index('IID')
    # Rename columns to just SNP IDs for easier use
    geno_df_selected.columns = final_snp_names_in_order
    
    # Clean up temporary files
    # for ext in ['.log', '.nosex', '.raw', '_snp_list.txt', '.bim', '.bed', '.fam']: # Keep .bim for allele check
    for ext in ['.log', '.nosex', '.raw', '_snp_list.txt']:
         if os.path.exists(f"{out_prefix}{ext}"): os.remove(f"{out_prefix}{ext}")
    if os.path.exists(f"{out_prefix}.bim"): os.remove(f"{out_prefix}.bim") # remove intermediate bim
    if os.path.exists(f"{out_prefix}.bed"): os.remove(f"{out_prefix}.bed") 
    if os.path.exists(f"{out_prefix}.fam"): os.remove(f"{out_prefix}.fam")


    return geno_df_selected, final_snp_names_in_order


def train_penalized_regression(X_snp, X_cov, y, cv_folds, l1_ratios, is_binary_trait=True):
    """Trains an ElasticNet (or Logistic) regression model."""
    X_full = np.concatenate((X_snp, X_cov), axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    if is_binary_trait:
        model = LogisticRegressionCV(
            Cs=10, # Inverse of regularization strength; smaller values specify stronger regularization
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            penalty='elasticnet',
            solver='saga', # saga solver supports elasticnet
            l1_ratios=l1_ratios,
            max_iter=2000, # Increased max_iter
            random_state=42,
            scoring='roc_auc' # Use AUC for model selection if binary
        )
    else:
        model = ElasticNetCV(
            l1_ratio=l1_ratios,
            cv=cv_folds,
            random_state=42,
            max_iter=2000, # Increased max_iter
            n_alphas=100 # Number of alphas along the regularization path
        )
    
    model.fit(X_scaled, y)
    return model, scaler


def run_penalized_regression_prs_pipeline(
    formatted_gwas_file, # Standardized GWAS sumstats
    train_bfile_prefix,  # PLINK prefix for training genotype data (for LD clumping & model training)
    train_pheno_file,    # Phenotype file for training individuals
    train_ids_file,      # File containing IIDs for the training set
    output_dir,          # Directory to save model, coefficients, and PRS
    model_name="PenReg",
    is_binary_trait=True
    ):
    """Full pipeline for training a penalized regression PRS model."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Clump SNPs using training data LD
    print("Step 1: Clumping SNPs...")
    clump_out_prefix = os.path.join(output_dir, f"{model_name}_clumped")
    # Use the training genotype data for LD calculation in clumping
    # The GWAS sumstats provide the p-values.
    selected_snps = clump_snps_plink(
        gwas_sumstats_file=formatted_gwas_file,
        plink_bfile_for_ld=train_bfile_prefix, # LD from training data
        clump_out_prefix=clump_out_prefix,
        p1_thresh=config.PENREG_CLUMP_P1, p2_thresh=config.PENREG_CLUMP_P2,
        r2_thresh=config.PENREG_CLUMP_R2, kb_thresh=config.PENREG_CLUMP_KB,
        clump_p_field='P' # Assuming 'P' is the p-value col in formatted_gwas_file
    )
    if not selected_snps:
        print("No SNPs selected after clumping. Penalized regression cannot proceed.")
        # Create empty outputs
        pd.DataFrame(columns=['IID', f'PRS_{model_name}']).to_csv(os.path.join(output_dir, f"{model_name}.prs"), sep='\t', index=False)
        return os.path.join(output_dir, f"{model_name}.prs"), None, None, []

    print(f"Selected {len(selected_snps)} SNPs after clumping.")
    pd.DataFrame(selected_snps, columns=['SNP']).to_csv(os.path.join(output_dir, f"{model_name}_selected_snps.txt"), index=False)

    # 2. Extract genotypes for selected SNPs for training individuals
    print("Step 2: Extracting training genotypes for selected SNPs...")
    train_geno_extract_prefix = os.path.join(output_dir, f"{model_name}_train_genotypes")
    X_train_snp_df, snp_features_ordered = extract_genotypes_for_snps(
        train_bfile_prefix, selected_snps, train_geno_extract_prefix, keep_ids_file=train_ids_file
    )
    if X_train_snp_df.empty:
        print("No genotype data extracted for training. Penalized regression cannot proceed.")
        pd.DataFrame(columns=['IID', f'PRS_{model_name}']).to_csv(os.path.join(output_dir, f"{model_name}.prs"), sep='\t', index=False)
        return os.path.join(output_dir, f"{model_name}.prs"), None, None, []

    # 3. Load phenotypes and covariates for training individuals
    print("Step 3: Loading training phenotypes and covariates...")
    pheno_train_df = load_phenotypes(train_pheno_file)
    train_ids = load_ids_from_file(train_ids_file)
    pheno_train_df = pheno_train_df[pheno_train_df['IID'].isin(train_ids)]

    # Merge genotypes with phenotypes
    train_data = pheno_train_df.merge(X_train_snp_df.reset_index(), on="IID", how="inner")
    train_data = train_data.dropna(subset=[config.TARGET_DISEASE_COLUMN] + snp_features_ordered + config.COVARIATE_COLUMNS)

    if train_data.empty:
        print("No overlapping samples after merging genotypes and phenotypes for training. Cannot proceed.")
        pd.DataFrame(columns=['IID', f'PRS_{model_name}']).to_csv(os.path.join(output_dir, f"{model_name}.prs"), sep='\t', index=False)
        return os.path.join(output_dir, f"{model_name}.prs"), None, None, []

    X_snp_values = train_data[snp_features_ordered].values
    X_cov_values = train_data[config.COVARIATE_COLUMNS].values
    y_values = train_data[config.TARGET_DISEASE_COLUMN].values

    # 4. Train penalized regression model
    print("Step 4: Training penalized regression model...")
    model, scaler = train_penalized_regression(
        X_snp_values, X_cov_values, y_values,
        cv_folds=config.PENREG_CV_FOLDS,
        l1_ratios=config.ELASTICNET_L1_RATIOS, # Re-use ensemble's L1 ratios for this base model too
        is_binary_trait=is_binary_trait
    )
    
    # Save model and scaler
    joblib.dump(model, os.path.join(output_dir, f"{model_name}_model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, f"{model_name}_scaler.joblib"))
    print(f"Trained model and scaler saved to {output_dir}")

    # Extract SNP coefficients (weights for PRS calculation)
    # For LogisticRegressionCV, coef_ is (1, n_features). For ElasticNetCV, it's (n_features,)
    coefficients = model.coef_.flatten()
    snp_coeffs = coefficients[:len(snp_features_ordered)]
    # cov_coeffs = coefficients[len(snp_features_ordered):]

    snp_effects_df = pd.DataFrame({'SNP': snp_features_ordered, 'EffectWeight': snp_coeffs})
    snp_effects_df.to_csv(os.path.join(output_dir, f"{model_name}_snp_effects.txt"), sep='\t', index=False)

    # 5. Calculate PRS for the training set itself (optional, but useful for ensemble input)
    print("Step 5: Calculating PRS for training set...")
    X_full_scaled = scaler.transform(np.concatenate((X_snp_values, X_cov_values), axis=1))
    
    if hasattr(model, "decision_function"): # LogisticRegression or similar
        # The decision_function gives the raw score (log-odds ratio for logistic)
        # This is typically what's used as PRS from logistic models
        prs_scores_train = model.decision_function(X_full_scaled)
    else: # ElasticNetRegressor
        prs_scores_train = model.predict(X_full_scaled)
        
    prs_df_train = pd.DataFrame({'IID': train_data['IID'], f'PRS_{model_name}': prs_scores_train})
    output_prs_file = os.path.join(config.BASE_PRS_DIR, f"{os.path.basename(train_bfile_prefix)}_{model_name}.prs")
    prs_df_train.to_csv(output_prs_file, sep='\t', index=False)
    print(f"Penalized Regression PRS for training data saved to {output_prs_file}")
    
    return output_prs_file, os.path.join(output_dir, f"{model_name}_model.joblib"), \
           os.path.join(output_dir, f"{model_name}_scaler.joblib"), \
           snp_features_ordered


def calculate_penreg_prs_on_new_data(
    target_bfile_prefix, target_ids_file, # Genotypes for new data, and IDs to process
    trained_model_path, trained_scaler_path, # From training
    snp_features_ordered, # List of SNPs (in order) used in the trained model
    pheno_file_new_data, # Phenotypes for new data (for covariates)
    output_prs_file,
    model_name="PenReg"
    ):
    """Calculates PRS on new data using a pre-trained penalized regression model."""
    print(f"Calculating {model_name} PRS for new data: {target_bfile_prefix}")
    
    # 1. Extract genotypes for the specific SNPs for the new data individuals
    new_data_geno_extract_prefix = os.path.join(os.path.dirname(output_prs_file), f"{model_name}_newdata_genotypes")
    X_new_snp_df, extracted_snp_names = extract_genotypes_for_snps(
        target_bfile_prefix, snp_features_ordered, new_data_geno_extract_prefix, keep_ids_file=target_ids_file
    )
    if X_new_snp_df.empty:
        print(f"No genotype data extracted for new data {target_bfile_prefix}. Cannot calculate PRS.")
        pd.DataFrame(columns=['IID', f'PRS_{model_name}']).to_csv(output_prs_file, sep='\t', index=False)
        return output_prs_file

    # Ensure columns in X_new_snp_df match snp_features_ordered
    # Reindex to ensure correct order and handle missing SNPs (fill with 0 - mean imputation after scaling)
    X_new_snp_df = X_new_snp_df.reindex(columns=snp_features_ordered).fillna(0)


    # 2. Load phenotypes/covariates for new data
    pheno_new_df = load_phenotypes(pheno_file_new_data)
    new_ids = load_ids_from_file(target_ids_file)
    pheno_new_df = pheno_new_df[pheno_new_df['IID'].isin(new_ids)]

    # Merge genotypes with phenotypes
    new_data_merged = pheno_new_df.merge(X_new_snp_df.reset_index(), on="IID", how="inner")
    # Drop rows if critical covariates are missing, but not if only phenotype is missing (for PRS calculation)
    new_data_merged = new_data_merged.dropna(subset=snp_features_ordered + config.COVARIATE_COLUMNS)


    if new_data_merged.empty:
        print(f"No overlapping samples in new data {target_bfile_prefix} after merging. Cannot calculate PRS.")
        pd.DataFrame(columns=['IID', f'PRS_{model_name}']).to_csv(output_prs_file, sep='\t', index=False)
        return output_prs_file

    X_snp_new_values = new_data_merged[snp_features_ordered].values
    X_cov_new_values = new_data_merged[config.COVARIATE_COLUMNS].values
    
    # 3. Load trained model and scaler
    model = joblib.load(trained_model_path)
    scaler = joblib.load(trained_scaler_path)
    
    # 4. Scale features using the *trained* scaler
    X_full_new = np.concatenate((X_snp_new_values, X_cov_new_values), axis=1)
    X_full_new_scaled = scaler.transform(X_full_new) # Use transform, not fit_transform
    
    # 5. Predict PRS
    if hasattr(model, "decision_function"):
        prs_scores_new = model.decision_function(X_full_new_scaled)
    else:
        prs_scores_new = model.predict(X_full_new_scaled)
        
    prs_df_new = pd.DataFrame({'IID': new_data_merged['IID'], f'PRS_{model_name}': prs_scores_new})
    prs_df_new.to_csv(output_prs_file, sep='\t', index=False)
    print(f"{model_name} PRS for new data saved to {output_prs_file}")
    return output_prs_file


if __name__ == "__main__":
    # Example usage for training
    formatted_gwas = os.path.join(config.GWAS_SUMSTATS_DIR, "disease_X_gwas_formatted.txt")
    # Ensure formatted GWAS and PLINK files for training exist (see 01_data_preprocessing and 02a_run_ldpred2 for dummy creation)
    # For clumping and training, use QCed training set PLINK files
    train_bfile = config.UKB_GENO_PREFIX + "_qc" 
    train_pheno = config.UKB_PHENO_FILE
    train_ids = config.UKB_TRAIN_IDS_FILE
    
    penreg_output_dir = os.path.join(config.BASE_PRS_DIR, "penreg_model_training")

    # Create dummy files if they don't exist for script flow
    if not os.path.exists(formatted_gwas):
        pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(1,501)], 'CHR': ['1']*500, 'BP': range(10000, 10500),
            'A1': ['A']*500, 'A2': ['G']*500, 'BETA': pd.np.random.randn(500)*0.05,
            'SE': [0.01]*500, 'P': pd.np.random.rand(500), 'N': [95000]*500
        }).to_csv(formatted_gwas, sep='\t', index=False)
        print(f"Created dummy formatted GWAS: {formatted_gwas}")

    # Check for plink files used for LD in clumping
    if not os.path.exists(train_bfile + ".bim"):
        print(f"Warning: Training bfile {train_bfile} for LD clumping not found. Skipping PenReg training.")
    elif not os.path.exists(train_pheno):
        print(f"Warning: Training phenotype file {train_pheno} not found. Skipping PenReg training.")
    elif not os.path.exists(train_ids):
        print(f"Warning: Training IDs file {train_ids} not found. Skipping PenReg training.")
    else:
        is_binary = pd.read_csv(train_pheno)[config.TARGET_DISEASE_COLUMN].nunique() == 2
        print(f"Target trait is binary: {is_binary}")

        prs_file_train, model_path, scaler_path, snp_list_trained = run_penalized_regression_prs_pipeline(
            formatted_gwas_file=formatted_gwas,
            train_bfile_prefix=train_bfile,
            train_pheno_file=train_pheno,
            train_ids_file=train_ids,
            output_dir=penreg_output_dir,
            model_name="PenRegElasticNet",
            is_binary_trait=is_binary
        )
        print(f"Penalized regression training complete. PRS for training set: {prs_file_train}")

        # Example usage for calculating PRS on new data (e.g., UKB validation set)
        # valid_bfile = config.UKB_GENO_PREFIX + "_qc" # Assuming QCed files cover all individuals
        # valid_ids = config.UKB_VALID_IDS_FILE
        # valid_pheno = config.UKB_PHENO_FILE # Pheno file covers all
        # output_prs_valid_file = os.path.join(config.BASE_PRS_DIR, f"ukb_valid_PenRegElasticNet.prs")
        #
        # if os.path.exists(model_path) and os.path.exists(valid_ids):
        #     calculate_penreg_prs_on_new_data(
        #         target_bfile_prefix=valid_bfile,
        #         target_ids_file=valid_ids,
        #         trained_model_path=model_path,
        #         trained_scaler_path=scaler_path,
        #         snp_features_ordered=snp_list_trained, # from training output
        #         pheno_file_new_data=valid_pheno,
        #         output_prs_file=output_prs_valid_file,
        #         model_name="PenRegElasticNet"
        #     )
        #     print(f"PRS calculation for validation set complete: {output_prs_valid_file}")
        # else:
        #      print("Skipping PRS calculation on validation set due to missing model, scaler, or ID file.")