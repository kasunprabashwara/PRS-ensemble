import subprocess
import os
import pandas as pd
import config
from scripts.utils import run_plink_command # If needed for LD calculation separately

def prepare_ldpred2_sumstats_input(formatted_gwas_file, ldpred2_sumstats_file,
                                   ld_ref_bim_file):
    """
    Prepares GWAS summary statistics for LDpred2.
    LDpred2 expects specific columns: chr, pos, a0, a1, beta, beta_se, p, n_eff (or n)
    It also requires matching SNPs with an LD reference. bigsnpr::snp_match is often used.
    This Python function is a simplified version; using bigsnpr's R functions is more robust.
    """
    print(f"Preparing sumstats for LDpred2: {formatted_gwas_file} -> {ldpred2_sumstats_file}")
    sumstats = pd.read_csv(formatted_gwas_file, sep='\t')
    
    # Rename columns to LDpred2 defaults if necessary
    # Assuming formatted_gwas_file has: SNP, CHR, BP, A1 (effect), A2, BETA, SE, P, N_eff/N
    col_map = {
        'CHR': 'chr', 'BP': 'pos',
        'A1': 'a1', 'A2': 'a0', # LDpred2: a0=ref, a1=eff. Your A1 should be effect allele.
        'BETA': 'beta', 'SE': 'beta_se',
        'P': 'p'
    }
    # Determine N column
    if config.GWAS_N_EFFECTIVE_COL in sumstats.columns:
        col_map[config.GWAS_N_EFFECTIVE_COL] = 'n_eff'
    elif 'N' in sumstats.columns:
        col_map['N'] = 'n_eff' # Use N as n_eff if N_eff not present
    else:
        raise ValueError("N or N_eff column required in summary statistics for LDpred2.")

    sumstats.rename(columns=col_map, inplace=True)
    
    # Ensure required columns exist
    required_ldpred2_cols = ['chr', 'pos', 'a0', 'a1', 'beta', 'beta_se', 'p', 'n_eff']
    if 'SNP' in sumstats.columns: required_ldpred2_cols.append('SNP') # Keep SNP ID if present

    missing_cols = [col for col in required_ldpred2_cols if col not in sumstats.columns and col != 'SNP']
    if missing_cols:
        raise ValueError(f"Missing required columns for LDpred2 input: {missing_cols} in {formatted_gwas_file} after mapping.")

    sumstats = sumstats[required_ldpred2_cols]

    # Basic filtering for sanity (LDpred2/bigsnpr does more robust matching)
    sumstats = sumstats.dropna()
    sumstats = sumstats[sumstats['beta_se'] > 0] # SE must be positive

    # If you have an LD reference BIM file, you can do SNP matching here or rely on LDpred2.R script
    if ld_ref_bim_file and os.path.exists(ld_ref_bim_file):
        print(f"Matching sumstats SNPs with LD reference: {ld_ref_bim_file}")
        ld_bim = pd.read_csv(ld_ref_bim_file, sep='\t', header=None, names=['chr_ld', 'SNP', 'cm', 'pos_ld', 'a1_ld', 'a0_ld'])
        # This is a simplified match; bigsnpr::snp_match handles allele flipping, strand issues etc.
        sumstats_matched = pd.merge(sumstats, ld_bim[['SNP']], on='SNP', how='inner')
        print(f"Original sumstats SNPs: {len(sumstats)}, Matched with LD ref: {len(sumstats_matched)}")
        sumstats_matched.to_csv(ldpred2_sumstats_file, sep='\t', index=False)
    else:
        print(f"Warning: LD reference BIM file not provided or not found: {ld_ref_bim_file}. SNP matching might be incomplete.")
        sumstats.to_csv(ldpred2_sumstats_file, sep='\t', index=False)
        
    print(f"LDpred2-ready sumstats saved to: {ldpred2_sumstats_file}")
    return ldpred2_sumstats_file


def run_ldpred2_prs(
    rscript_executable,
    ldpred2_r_script, # Path to LDpred2.R or your wrapper
    sumstats_file,    # Prepared sumstats for LDpred2
    ld_ref_plink_prefix, # PLINK prefix for LD reference panel (bed/bim/fam)
    target_geno_plink_prefix, # PLINK prefix for target cohort to calculate PRS on
    output_prs_file_prefix, # Prefix for output PRS files
    h2_init=config.LDPRED2_H2_INIT,
    p_causal_seq=None, # For grid model, e.g. "0.001,0.01,0.1"
    sparse=config.LDPRED2_SPARSE_FLAG,
    n_cores=4,
    keep_target_ids_file=None # File with IIDs of target individuals
    ):
    """
    Runs LDpred2 using a provided R script.
    This function assumes an R script that takes command-line arguments for these files.
    You might need to adjust the R script or this call.
    """
    os.makedirs(os.path.dirname(output_prs_file_prefix), exist_ok=True)
    
    # --- Method 1: Using LDpred2.R from privefl/bigsnpr (requires bigsnpr installation in R) ---
    # This script often requires an LD matrix or a .rds file of LD reference.
    # Creating the LD reference .rds file is usually a preliminary step done in R using bigsnpr.
    # For this example, let's assume LDpred2.R can directly use PLINK files for LD ref if specified.
    # This is a HYPOTHETICAL command structure. You MUST adapt it to your LDpred2.R script.
    
    # If your LDpred2.R script expects a pre-computed LD matrix (.rds file):
    # 1. Create LD matrix .rds file (usually once per LD reference panel)
    #    This is typically done in R:
    #    R> library(bigsnpr)
    #    R> snp_readBed("ld_ref.bed") # Creates .rds and .bk files
    #    R> obj.bigSNP <- snp_attach("ld_ref.rds")
    #    R> G <- obj.bigSNP$genotypes
    #    R> CHR <- obj.bigSNP$map$chromosome
    #    R> POS <- obj.bigSNP$map$physical.pos
    #    R> # Optional: compute LD for specific regions / store correlation matrix
    #    R> # This part is complex and depends on LDpred2's exact needs (full corr vs. sparse)

    # Let's try to call a generic LDpred2.R script assuming it handles PLINK LD ref
    cmd_ldpred2_auto = [
        rscript_executable,
        ldpred2_r_script,
        "--sumstats", sumstats_file,
        "--ld_ref_plink", ld_ref_plink_prefix, # Pass PLINK prefix for LD ref
        "--target_plink", target_geno_plink_prefix, # Pass PLINK prefix for target geno
        "--out", f"{output_prs_file_prefix}_auto.prs",
        "--model", "auto",
        "--n_cores", str(n_cores)
    ]
    if h2_init is not None:
        cmd_ldpred2_auto.extend(["--h2_init", str(h2_init)])
    if keep_target_ids_file and os.path.exists(keep_target_ids_file):
        cmd_ldpred2_auto.extend(["--keep_target_ids", keep_target_ids_file])

    print(f"Running LDpred2 (auto mode): {' '.join(cmd_ldpred2_auto)}")
    # subprocess.run(cmd_ldpred2_auto, check=True, capture_output=True, text=True)
    print(f"Placeholder: LDpred2 auto execution. Command: {' '.join(cmd_ldpred2_auto)}")
    # Simulate output file for now
    dummy_iids = pd.read_csv(f"{target_geno_plink_prefix}.fam", sep='\s+', header=None)[1]
    pd.DataFrame({'IID': dummy_iids, 'PRS_LDpred2_auto': pd.np.random.randn(len(dummy_iids))}).to_csv(f"{output_prs_file_prefix}_auto.prs", sep='\t', index=False)


    # Example for LDpred2-grid (if p_causal_seq is provided)
    if p_causal_seq:
        p_str = ",".join(map(str, p_causal_seq))
        cmd_ldpred2_grid = [
            rscript_executable,
            ldpred2_r_script,
            "--sumstats", sumstats_file,
            "--ld_ref_plink", ld_ref_plink_prefix,
            "--target_plink", target_geno_plink_prefix,
            "--out", f"{output_prs_file_prefix}_grid.prs", # R script might produce multiple files per p
            "--model", "grid",
            "--p_causal", p_str,
            "--n_cores", str(n_cores)
        ]
        if h2_init is not None: cmd_ldpred2_grid.extend(["--h2_init", str(h2_init)])
        if keep_target_ids_file: cmd_ldpred2_grid.extend(["--keep_target_ids", keep_target_ids_file])

        print(f"Running LDpred2 (grid mode): {' '.join(cmd_ldpred2_grid)}")
        # subprocess.run(cmd_ldpred2_grid, check=True, capture_output=True, text=True)
        print(f"Placeholder: LDpred2 grid execution. Command: {' '.join(cmd_ldpred2_grid)}")
        # Simulate one output file for grid; real script might make many
        pd.DataFrame({'IID': dummy_iids, 'PRS_LDpred2_grid_p_best': pd.np.random.randn(len(dummy_iids))}).to_csv(f"{output_prs_file_prefix}_grid_best.prs", sep='\t', index=False)


    if sparse:
        cmd_ldpred2_sparse = [
             rscript_executable, ldpred2_r_script,
             "--sumstats", sumstats_file, "--ld_ref_plink", ld_ref_plink_prefix,
             "--target_plink", target_geno_plink_prefix,
             "--out", f"{output_prs_file_prefix}_sparse.prs",
             "--model", "sparse", "--n_cores", str(n_cores)
        ]
        # ... add other params ...
        print(f"Running LDpred2 (sparse mode): {' '.join(cmd_ldpred2_sparse)}")
        # subprocess.run(cmd_ldpred2_sparse, check=True, capture_output=True, text=True)
        print(f"Placeholder: LDpred2 sparse execution. Command: {' '.join(cmd_ldpred2_sparse)}")
        pd.DataFrame({'IID': dummy_iids, 'PRS_LDpred2_sparse': pd.np.random.randn(len(dummy_iids))}).to_csv(f"{output_prs_file_prefix}_sparse.prs", sep='\t', index=False)

    print(f"LDpred2 PRS calculation(s) placeholder complete. Output prefix: {output_prs_file_prefix}")
    # Return paths to generated PRS files (actual paths depend on R script output naming)
    # This is a guess; your R script will determine the actual output file names.
    generated_files = []
    if os.path.exists(f"{output_prs_file_prefix}_auto.prs"): generated_files.append(f"{output_prs_file_prefix}_auto.prs")
    if os.path.exists(f"{output_prs_file_prefix}_grid_best.prs"): generated_files.append(f"{output_prs_file_prefix}_grid_best.prs")
    if os.path.exists(f"{output_prs_file_prefix}_sparse.prs"): generated_files.append(f"{output_prs_file_prefix}_sparse.prs")
    return generated_files


if __name__ == "__main__":
    # Ensure config.LDPRED2_RSCRIPT_PATH is correctly set!
    if not os.path.exists(config.LDPRED2_RSCRIPT_PATH):
        print(f"Error: LDpred2 R script not found at {config.LDPRED2_RSCRIPT_PATH}. Please check config.py.")
    else:
        # Example: Processed GWAS sumstats from step 01
        formatted_gwas_file = os.path.join(config.GWAS_SUMSTATS_DIR, "disease_X_gwas_formatted.txt")
        ldpred2_sumstats_input = os.path.join(config.GWAS_SUMSTATS_DIR, "disease_X_gwas_for_ldpred2.txt")

        # Create dummy formatted GWAS if not exists
        if not os.path.exists(formatted_gwas_file):
            dummy_gwas_data = {
                'SNP': [f'rs{i}' for i in range(1,501)], 'CHR': ['1']*500, 'BP': range(10000, 10500),
                'A1': ['A']*500, 'A2': ['G']*500, 'BETA': pd.np.random.randn(500)*0.05,
                'SE': [0.01]*500, 'P': pd.np.random.rand(500), 'N': [95000]*500
            }
            pd.DataFrame(dummy_gwas_data).to_csv(formatted_gwas_file, sep='\t', index=False)
            print(f"Created dummy formatted GWAS: {formatted_gwas_file}")

        # Create dummy LD ref bim file for SNP matching example
        dummy_ld_ref_bim_path = config.LD_REF_PREFIX + ".bim"
        if not os.path.exists(dummy_ld_ref_bim_path):
            os.makedirs(os.path.dirname(dummy_ld_ref_bim_path), exist_ok=True)
            # Create bim with some overlapping and some non-overlapping SNPs
            bim_snps = [f'rs{i}' for i in range(1, 251)] + [f'rs{i+500}' for i in range(1,251)] # some overlap, some not
            dummy_bim_ld = pd.DataFrame({
                0: ['1']*500, 1: bim_snps, 2:[0]*500, 3:range(20000, 20500), 4:['A']*500, 5:['G']*500
            })
            dummy_bim_ld.to_csv(dummy_ld_ref_bim_path, sep='\t', header=False, index=False)
            print(f"Created dummy LD reference BIM: {dummy_ld_ref_bim_path}")


        prepare_ldpred2_sumstats_input(
            formatted_gwas_file,
            ldpred2_sumstats_input,
            ld_ref_bim_file=config.LD_REF_PREFIX + ".bim" # Assumes .bim file exists for LD ref
        )

        # Target data for PRS calculation (e.g., UKB QCed data, training split)
        target_prefix = config.UKB_GENO_PREFIX + "_qc" # Use QCed data
        ids_to_keep = config.UKB_TRAIN_IDS_FILE # Calculate on training set

        # Create dummy target plink files and ID file if they don't exist
        if not os.path.exists(target_prefix + ".fam"):
             print(f"Creating dummy target PLINK files for {target_prefix}")
             fam_df_ids = [f'ID_train_{i}' for i in range(50)] + [f'ID_other_{i}' for i in range(50)]
             dummy_fam_target = pd.DataFrame({0: [f'FAM{i}' for i in range(100)], 1: fam_df_ids, 2:[0]*100, 3:[0]*100, 4:[1]*100, 5:[-9]*100})
             dummy_bim_target = pd.DataFrame({0: ['1']*5, 1: [f'rs{i}' for i in range(5)], 2:[0]*5, 3:range(100,105), 4:['A']*5, 5:['G']*5})
             dummy_fam_target.to_csv(target_prefix + ".fam", sep='\t', header=False, index=False)
             dummy_bim_target.to_csv(target_prefix + ".bim", sep='\t', header=False, index=False)
             with open(target_prefix + ".bed", "wb") as f: f.write(b'\x6c\x1b\x01' + (b'\x55' * (((100+3)//4)*5)) )
        
        if not os.path.exists(ids_to_keep):
            train_sample_ids = [f'ID_train_{i}' for i in range(50)] # subset of fam IDs
            # Need FID and IID for PLINK's --keep
            fam_data_for_ids = pd.read_csv(target_prefix + ".fam", delim_whitespace=True, header=None, names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'])
            ids_to_save_df = fam_data_for_ids[fam_data_for_ids['IID'].isin(train_sample_ids)][['FID', 'IID']]
            ids_to_save_df.to_csv(ids_to_keep, sep='\t', header=False, index=False)
            print(f"Created dummy ID file for keep: {ids_to_keep}")


        if os.path.exists(ldpred2_sumstats_input):
            run_ldpred2_prs(
                rscript_executable=config.RSCRIPT_EXECUTABLE,
                ldpred2_r_script=config.LDPRED2_RSCRIPT_PATH,
                sumstats_file=ldpred2_sumstats_input,
                ld_ref_plink_prefix=config.LD_REF_PREFIX, # Make sure .bed, .bim, .fam exist for this
                target_geno_plink_prefix=target_prefix,
                output_prs_file_prefix=os.path.join(config.BASE_PRS_DIR, f"ukb_train_ldpred2"),
                p_causal_seq=config.LDPRED2_P_SEQ, # Example for grid model
                sparse=config.LDPRED2_SPARSE_FLAG,
                keep_target_ids_file=ids_to_keep
            )
        else:
            print(f"LDpred2 input sumstats not found: {ldpred2_sumstats_input}")