import pandas as pd
import subprocess
import os
import config
from scripts.utils import run_plink_command

def format_gwas_sumstats(raw_gwas_file, processed_gwas_file,
                         snp_col, chr_col, bp_col, a1_col, a2_col,
                         beta_col=None, or_col=None, se_col=None, pval_col=None, freq_col=None, n_col=None,
                         output_beta_col='BETA', output_se_col='SE', output_p_col='P',
                         output_freq_col='FREQ', output_n_col='N'):
    """
    Standardizes GWAS summary statistics to a common format.
    Expected output columns: SNP, CHR, BP, A1, A2, FREQ (of A1), BETA, SE, P, N_eff
    """
    print(f"Processing GWAS summary statistics: {raw_gwas_file}")
    df = pd.read_csv(raw_gwas_file, delim_whitespace=True) # Adjust delimiter if needed

    col_map = {
        snp_col: 'SNP',
        chr_col: 'CHR',
        bp_col: 'BP',
        a1_col: 'A1', # Effect allele
        a2_col: 'A2'  # Other allele
    }
    if freq_col: col_map[freq_col] = output_freq_col
    if pval_col: col_map[pval_col] = output_p_col
    if n_col: col_map[n_col] = output_n_col
    
    df.rename(columns=col_map, inplace=True)

    if beta_col:
        df[output_beta_col] = df[beta_col]
    elif or_col:
        df[output_beta_col] = pd.np.log(df[or_col].astype(float)) # Convert OR to log(OR) for BETA
    else:
        raise ValueError("Either beta_col or or_col must be provided.")

    if se_col:
        df[output_se_col] = df[se_col]
    elif beta_col and pval_col: # Estimate SE from BETA and P if SE not available
        from scipy.stats import norm
        df[output_p_col] = df[output_p_col].astype(float)
        df[output_beta_col] = df[output_beta_col].astype(float)
        # Filter out p-values of 0 or 1 to avoid issues with norm.ppf
        valid_p = (df[output_p_col] > 0) & (df[output_p_col] < 1)
        df.loc[valid_p, output_se_col] = abs(df.loc[valid_p, output_beta_col] / norm.ppf(df.loc[valid_p, output_p_col] / 2))
        df[output_se_col].fillna(df[output_se_col].median(), inplace=True) # Basic imputation for missing SE
        print(f"Estimated SE for {valid_p.sum()} SNPs. Imputed {pd.isna(df[output_se_col]).sum()} missing SEs with median.")
    else:
        raise ValueError("SE column (se_col) or BETA and P-value columns (beta_col, pval_col) must be provided to get SE.")

    required_out_cols = ['SNP', 'CHR', 'BP', 'A1', 'A2', output_beta_col, output_se_col, output_p_col]
    if output_freq_col in df.columns: required_out_cols.append(output_freq_col)
    if output_n_col in df.columns: required_out_cols.append(output_n_col)
    
    df = df[required_out_cols]
    df = df.dropna(subset=['SNP', 'CHR', 'BP', 'A1', 'A2', output_beta_col, output_se_col, output_p_col])
    df['CHR'] = df['CHR'].astype(str).str.replace('chr', '', case=False) # Remove "chr" prefix if present

    # Further QC (example: remove non-biallelic, non-standard chromosomes, MAF if FREQ available)
    df = df[df['A1'].str.match('^[ACGT]$') & df['A2'].str.match('^[ACGT]$')] # Keep only A,C,G,T
    df = df[df['CHR'].isin([str(i) for i in range(1, 23)] + ['X', 'Y', 'MT'])] # Standard chromosomes
    
    # Ensure N column is present for LDpred2, can be N_eff or N
    if output_n_col not in df.columns and config.GWAS_N_EFFECTIVE_COL in df.columns: # if N_eff is separate
        df[output_n_col] = df[config.GWAS_N_EFFECTIVE_COL]
    elif output_n_col not in df.columns:
        # Try to infer N from a global value if not per-SNP (less ideal)
        # This is a placeholder, you might need to get N from GWAS paper
        global_n = 100000 # EXAMPLE: Replace with actual average N if per-SNP N is missing
        print(f"Warning: N column ('{output_n_col}') not found. Using a global N of {global_n}. This is not ideal for LDpred2.")
        df[output_n_col] = global_n


    df.to_csv(processed_gwas_file, sep='\t', index=False)
    print(f"Processed GWAS summary stats saved to: {processed_gwas_file}")
    return processed_gwas_file


def qc_genotypes_plink(bfile_prefix_in, qc_bfile_prefix_out,
                       mind=0.02, geno=0.02, maf=0.01, hwe=1e-6,
                       keep_ids_file=None, extract_snps_file=None):
    """Performs QC on genotype data using PLINK."""
    args = [
        "--bfile", bfile_prefix_in,
        "--mind", str(mind),      # Missingness per individual
        "--geno", str(geno),      # Missingness per SNP
        "--maf", str(maf),        # Minor allele frequency
        "--hwe", str(hwe),        # Hardy-Weinberg equilibrium
        "--autosome",             # Keep only autosomal SNPs, consider if X chr is needed
        "--make-bed",
        "--out", qc_bfile_prefix_out
    ]
    if keep_ids_file and os.path.exists(keep_ids_file):
        args.extend(["--keep", keep_ids_file])
    if extract_snps_file and os.path.exists(extract_snps_file):
        args.extend(["--extract", extract_snps_file])

    print(f"Performing QC on {bfile_prefix_in}, output to {qc_bfile_prefix_out}")
    run_plink_command(args)
    print(f"QCed genotype data saved to {qc_bfile_prefix_out}.{{bed,bim,fam}}")


def split_ids_train_val_test(ids_list, train_frac=0.7, val_frac=0.15, output_dir="data/uk_biobank/"):
    """Splits IDs into train, validation, and test sets and saves them."""
    import numpy as np
    np.random.shuffle(ids_list)
    n = len(ids_list)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    
    train_ids = ids_list[:n_train]
    val_ids = ids_list[n_train : n_train + n_val]
    test_ids = ids_list[n_train + n_val :]

    def save_ids(ids, filepath):
        fam_df = pd.read_csv(config.UKB_GENO_PREFIX + ".fam", delim_whitespace=True, header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'])
        ids_to_save_df = fam_df[fam_df['IID'].astype(str).isin(ids)][['FID', 'IID']]
        ids_to_save_df.to_csv(filepath, sep='\t', header=False, index=False)
        print(f"Saved {len(ids)} IDs to {filepath}")

    os.makedirs(output_dir, exist_ok=True)
    save_ids(train_ids, os.path.join(output_dir, "ukb_train_ids.txt"))
    save_ids(val_ids, os.path.join(output_dir, "ukb_valid_ids.txt"))
    save_ids(test_ids, os.path.join(output_dir, "ukb_test_ids.txt"))
    print("Train/Val/Test ID files created.")


if __name__ == "__main__":
    # --- 1. Format Example GWAS Summary Statistics ---
    # This is a placeholder. You need to provide your raw GWAS file and map its columns.
    # Example: If your raw GWAS file is "my_gwas.txt"
    # raw_gwas_path = os.path.join(config.GWAS_SUMSTATS_DIR, "my_raw_gwas.txt")
    # Create a dummy raw GWAS file for demonstration if it doesn't exist
    # if not os.path.exists(raw_gwas_path):
    #     dummy_data = {
    #         'rsid': [f'rs{i}' for i in range(1,1001)], 'chrom': ['1']*1000, 'pos': range(10000, 11000),
    #         'allele1': ['A']*1000, 'allele0': ['G']*1000, 'beta_val': pd.np.random.randn(1000)*0.05,
    #         'se_val': [0.01]*1000, 'pval_info': pd.np.random.rand(1000), 'n_samples': [95000]*1000,
    #         'freq_allele1': pd.np.random.uniform(0.01, 0.99, 1000)
    #     }
    #     pd.DataFrame(dummy_data).to_csv(raw_gwas_path, sep='\t', index=False)
    #     print(f"Created dummy raw GWAS file: {raw_gwas_path}")

    # processed_gwas_path = os.path.join(config.GWAS_SUMSTATS_DIR, "disease_X_gwas_formatted.txt")
    # format_gwas_sumstats(
    #     raw_gwas_file=raw_gwas_path,
    #     processed_gwas_file=processed_gwas_path,
    #     snp_col='rsid', chr_col='chrom', bp_col='pos',
    #     a1_col='allele1', a2_col='allele0', # Ensure a1 is effect allele
    #     beta_col='beta_val', se_col='se_val', pval_col='pval_info',
    #     freq_col = 'freq_allele1', n_col = 'n_samples'
    # )
    print("GWAS formatting step needs to be configured with your actual file and column names.")

    # --- 2. QC Genotype Data (Example for UK Biobank) ---
    # This assumes you have config.UKB_GENO_PREFIX pointing to raw PLINK files.
    # Create dummy PLINK files for demonstration if they don't exist
    # dummy_bim = pd.DataFrame({0: [1]*5, 1: [f'rs{i}' for i in range(5)], 2:[0]*5, 3:range(100,105), 4:['A']*5, 5:['G']*5})
    # dummy_fam = pd.DataFrame({0: [f'FAM{i}' for i in range(10)], 1: [f'ID{i}' for i in range(10)], 2:[0]*10, 3:[0]*10, 4:[1]*10, 5:[-9]*10})
    # if not os.path.exists(config.UKB_GENO_PREFIX + ".bed"):
    #     print(f"Creating dummy PLINK files for {config.UKB_GENO_PREFIX}")
    #     dummy_bim.to_csv(config.UKB_GENO_PREFIX + ".bim", sep='\t', header=False, index=False)
    #     dummy_fam.to_csv(config.UKB_GENO_PREFIX + ".fam", sep='\t', header=False, index=False)
    #     with open(config.UKB_GENO_PREFIX + ".bed", "wb") as f: # Create a tiny valid BED file header
    #        f.write(b'\x6c\x1b\x01') # Magic number for PLINK bed file
    #        # This BED file won't be usable for analysis, just for file existence
    #        num_individuals = len(dummy_fam)
    #        num_snps = len(dummy_bim)
    #        # Each SNP takes ceil(num_individuals / 4) bytes
    #        for _ in range(num_snps):
    #            f.write(b'\x55' * ((num_individuals + 3) // 4) ) # Fill with some data (01010101)

    # ukb_qc_prefix = config.UKB_GENO_PREFIX + "_qc"
    # qc_genotypes_plink(config.UKB_GENO_PREFIX, ukb_qc_prefix)
    print("Genotype QC step needs actual PLINK files and config.UKB_GENO_PREFIX to be set.")

    # --- 3. Split UK Biobank IDs for Training, Validation, Test ---
    # (Assuming UKB_GENO_PREFIX.fam exists and is populated)
    # if os.path.exists(config.UKB_GENO_PREFIX + ".fam"):
    #     fam_df = pd.read_csv(config.UKB_GENO_PREFIX + ".fam", delim_whitespace=True, header=None, usecols=[1])
    #     all_ukb_ids = fam_df[1].astype(str).tolist()
    #     if all_ukb_ids:
    #          split_ids_train_val_test(all_ukb_ids, output_dir=os.path.dirname(config.UKB_TRAIN_IDS_FILE))
    #     else:
    #          print("No IDs found in .fam file to split.")
    # else:
    #     print(f"{config.UKB_GENO_PREFIX}.fam not found. Skipping ID splitting.")
    print("ID splitting step needs a .fam file at config.UKB_GENO_PREFIX.")
    print("Preprocessing script outline complete. Manual configuration and execution needed.")