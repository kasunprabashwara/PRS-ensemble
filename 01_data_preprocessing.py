import os
import pandas as pd
import numpy as np
import config
from utils import run_plink_command

def format_gwas_sumstats(raw_gwas_file, processed_gwas_file, col_map):
    """
    Standardizes GWAS summary statistics to a common format.
    Expected output columns: SNP, CHR, BP, A1, A2, BETA, SE, P
    """
    print(f"Processing GWAS summary statistics: {raw_gwas_file}")
    df = pd.read_csv(raw_gwas_file, sep='\t', low_memory=False)

    # Create a variant ID if a dedicated SNP column is not specified or present
    if 'snp_col' not in col_map or col_map['snp_col'] not in df.columns:
        print("SNP column not in data, creating variant ID from CHR:BP:A1:A2.")
        df['variant_id'] = (df[col_map['chr_col']].astype(str) + ':' +
                            df[col_map['bp_col']].astype(str) + ':' +
                            df[col_map['ea_col']].astype(str) + ':' +
                            df[col_map['oa_col']].astype(str))
        col_map['snp_col'] = 'variant_id'

    # Rename columns based on the provided map
    rename_map = {v: k.replace('_col', '') for k, v in col_map.items()}
    df.rename(columns=rename_map, inplace=True)

    # --- Column Standardization and Type Conversion ---
    final_cols = {}
    final_cols['SNP'] = df['snp']
    final_cols['CHR'] = pd.to_numeric(df['chr'].astype(str).str.replace('chr', ''), errors='coerce')
    final_cols['BP'] = pd.to_numeric(df['bp'], errors='coerce')
    final_cols['A1'] = df['ea'].str.upper() # Effect Allele
    final_cols['A2'] = df['oa'].str.upper() # Other Allele
    final_cols['P'] = pd.to_numeric(df['pval'], errors='coerce')
    final_cols['SE'] = pd.to_numeric(df['se'], errors='coerce')
    final_cols['N'] = pd.to_numeric(df['n'], errors='coerce')

    # Calculate BETA. If 'beta' is z_score, it's not a true BETA.
    # A true BETA can be derived if not present: BETA = Z / sqrt(2*EAF*(1-EAF)*(N+Z^2))
    # However, many tools can work with Z-scores directly or Z/SE. We will provide a simple BETA.
    # The LDpred2 R script will recalculate it from Z and SE anyway.
    if 'beta_col' in col_map:
        final_cols['BETA'] = pd.to_numeric(df['beta'], errors='coerce')
    else: # If z_score was mapped to beta_col
        print("Calculating pseudo-BETA from Z-score and SE.")
        final_cols['BETA'] = df['beta'] * df['se']

    processed_df = pd.DataFrame(final_cols)

    # --- QC on summary statistics ---
    print("Performing QC on summary statistics...")
    processed_df.dropna(subset=['SNP', 'CHR', 'BP', 'A1', 'A2', 'BETA', 'SE', 'P'], inplace=True)
    processed_df = processed_df[processed_df['SE'] > 0]
    processed_df = processed_df[(processed_df['P'] > 0) & (processed_df['P'] <= 1)]
    processed_df = processed_df[processed_df['A1'].str.match('^[ACGT]$') & processed_df['A2'].str.match('^[ACGT]$')]
    processed_df = processed_df[processed_df['CHR'].isin(range(1, 23))]
    processed_df['CHR'] = processed_df['CHR'].astype(int)
    processed_df['BP'] = processed_df['BP'].astype(int)
    processed_df.drop_duplicates(subset=['SNP'], keep='first', inplace=True)

    processed_df.to_csv(processed_gwas_file, sep='\t', index=False)
    print(f"Processed GWAS summary stats saved to: {processed_gwas_file} ({len(processed_df)} variants)")
    return processed_gwas_file

def create_dummy_data_if_needed():
    """Creates dummy genotype, phenotype, and ID files if they don't exist."""
    print("\n--- Checking for target data and creating dummies if needed ---")
    
    # 1. Dummy Target Genotypes
    if not os.path.exists(config.TARGET_GENO_PREFIX + ".bed"):
        print(f"Target genotype file not found. Creating dummy data at: {config.TARGET_GENO_PREFIX}")
        # Create a dummy .map and .ped file, then convert with PLINK
        map_file = config.TARGET_GENO_PREFIX + ".map"
        ped_file = config.TARGET_GENO_PREFIX + ".ped"
        with open(map_file, 'w') as f:
            f.write("1 rs123 0 1000\n")
            f.write("1 rs456 0 2000\n")
        
        with open(ped_file, 'w') as f:
            for i in range(1, 201): # 200 individuals
                case_status = i % 2 # 100 cases, 100 controls
                sex = 1 if i % 3 == 0 else 2
                f.write(f"FAM{i} IID{i} 0 0 {sex} {case_status+1} A A G C\n")
        
        run_plink_command([
            "--ped", ped_file,
            "--map", map_file,
            "--make-bed"
        ], log_prefix=config.TARGET_GENO_PREFIX)
        os.remove(map_file)
        os.remove(ped_file)

    # 2. Dummy Phenotype File
    if not os.path.exists(config.PHENO_FILE):
        print(f"Phenotype file not found. Creating dummy data at: {config.PHENO_FILE}")
        fam_df = pd.read_csv(config.TARGET_GENO_PREFIX + ".fam", delim_whitespace=True, header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'])
        fam_df['AGE'] = np.random.randint(40, 80, size=len(fam_df))
        for i in range(1, 11):
            fam_df[f'PC{i}'] = np.random.randn(len(fam_df))
        fam_df.rename(columns={'PHENO': config.TARGET_DISEASE_COLUMN}, inplace=True)
        fam_df[config.TARGET_DISEASE_COLUMN] = fam_df[config.TARGET_DISEASE_COLUMN] - 1 # Convert 1/2 to 0/1
        fam_df[['IID', config.TARGET_DISEASE_COLUMN, 'AGE', 'SEX'] + [f'PC{i}' for i in range(1,11)]].to_csv(config.PHENO_FILE, index=False)

    # 3. Dummy ID Split Files
    if not os.path.exists(config.TRAIN_IDS_FILE) or not os.path.exists(config.TEST_IDS_FILE):
        print("ID split files not found. Creating train/test split.")
        fam_df = pd.read_csv(config.TARGET_GENO_PREFIX + ".fam", delim_whitespace=True, header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'])
        train_ids = fam_df.sample(frac=0.7, random_state=42)
        test_ids = fam_df.drop(train_ids.index)
        train_ids[['FID', 'IID']].to_csv(config.TRAIN_IDS_FILE, sep='\t', index=False, header=False)
        test_ids[['FID', 'IID']].to_csv(config.TEST_IDS_FILE, sep='\t', index=False, header=False)

    # 4. Dummy LD Reference
    if not os.path.exists(config.LD_REF_PLINK_PREFIX + ".bed"):
        print(f"LD Reference not found. Creating dummy data at: {config.LD_REF_PLINK_PREFIX}")
        run_plink_command([
            "--bfile", config.TARGET_GENO_PREFIX,
            "--make-bed"
        ], log_prefix=config.LD_REF_PLINK_PREFIX)

    print("--- Dummy data check complete ---")

def main():
    print("--- Starting Data Preprocessing ---")
    # 1. Format GWAS summary statistics
    format_gwas_sumstats(
        raw_gwas_file=config.GWAS_RAW_FILE,
        processed_gwas_file=config.GWAS_PROCESSED_FILE,
        col_map=config.GWAS_COLS
    )

    # 2. Create dummy data for subsequent steps if real data is not present
    create_dummy_data_if_needed()
    
    print("--- Data Preprocessing Finished ---")

if __name__ == '__main__':
    main()