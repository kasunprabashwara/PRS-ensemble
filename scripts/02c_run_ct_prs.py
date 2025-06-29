import pandas as pd
import os
import sys

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import config
from utils import run_plink_command

def run_ct_prs(gwas_file, target_bfile, p_thresholds, clump_r2, clump_kb, keep_ids_file, output_dir):
    """
    Performs a highly robust Clumping and Thresholding (C+T) PRS calculation
    using the os.chdir() strategy for reliable PLINK output.
    """
    dataset_name = "train" if "train" in os.path.basename(keep_ids_file) else "test"
    print(f"--- Starting C+T PRS Calculation for {dataset_name} set ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Store the original working directory
    original_wd = os.getcwd()
    
    try:
        # Change to the output directory to ensure all PLINK files land here
        os.chdir(output_dir)

        # 1. Clump the GWAS summary statistics once
        print("Step 1: Clumping GWAS summary statistics...")
        clump_out_prefix = "clumped_snps"
        # We need to provide absolute paths to input files now that we've changed directory
        abs_gwas_file = os.path.join(project_root, os.path.relpath(gwas_file, project_root))
        abs_ld_ref_bfile = os.path.join(project_root, os.path.relpath(config.LD_REF_PLINK_PREFIX, project_root))
        
        args_clump = [
            "--bfile", abs_ld_ref_bfile,
            "--clump", abs_gwas_file,
            "--clump-p1", str(config.CT_CLUMP_P1),
            "--clump-r2", str(clump_r2),
            "--clump-kb", str(clump_kb),
            "--clump-snp-field", "SNP",
            "--clump-field", "P",
            "--out", clump_out_prefix
        ]
        run_plink_command(args_clump, "C+T Clumping")
        
        clumped_file = f"{clump_out_prefix}.clumped"
        if not os.path.exists(clumped_file) or os.path.getsize(clumped_file) == 0:
            print("Clumping resulted in an empty SNP set. C+T cannot proceed.")
            # Change back to original directory before returning
            os.chdir(original_wd)
            return

        clumped_snps_df = pd.read_csv(clumped_file, sep=r'\s+')
        
        # 2. Load the full GWAS to get p-values for clumped SNPs
        gwas_df = pd.read_csv(abs_gwas_file, sep='\t')
        clumped_gwas = pd.merge(clumped_snps_df[['SNP']], gwas_df, on='SNP', how='inner')

        # 3. Iterate through p-value thresholds
        print("Step 2: Calculating PRS for each p-value threshold...")
        # Get absolute paths for other input files
        abs_target_bfile = os.path.join(project_root, os.path.relpath(target_bfile, project_root))
        abs_keep_ids_file = os.path.join(project_root, os.path.relpath(keep_ids_file, project_root))

        for p_thresh in p_thresholds:
            p_thresh_str = f"{p_thresh:g}"
            print(f"  - Processing p-value threshold: {p_thresh_str}")
            
            threshold_snps_df = clumped_gwas[clumped_gwas['P'] <= p_thresh]

            # Define the final PRS file path in the central directory
            final_prs_path = os.path.join(config.BASE_PRS_DIR, f"{dataset_name}_ct_p{p_thresh_str}.prs")

            if threshold_snps_df.empty:
                print(f"    -> No SNPs pass p-value threshold {p_thresh_str}. Creating empty PRS file.")
                ids_df = pd.read_csv(abs_keep_ids_file, sep=r'\s+', header=None)
                empty_prs_df = pd.DataFrame({'IID': ids_df[1], f'PRS_ct_p{p_thresh_str}': 0.0})
                # Save directly to the final destination
                empty_prs_df.to_csv(final_prs_path, sep='\t', index=False)
                continue

            # Prepare the scoring file for this threshold
            score_df = threshold_snps_df[['SNP', 'A1', 'BETA']]
            score_file = f"p{p_thresh_str}_score.txt"
            score_df.to_csv(score_file, sep='\t', index=False, header=True)
            
            # Call PLINK's --score command
            prs_out_prefix = f"prs_p{p_thresh_str}"
            args_score = [
                "--bfile", abs_target_bfile,
                "--score", os.path.abspath(score_file), "1", "2", "3", "header",
                "--keep", abs_keep_ids_file,
                "--out", prs_out_prefix
            ]
            run_plink_command(args_score, f"C+T Scoring (p={p_thresh_str})")
            
            plink_profile_file = f"{prs_out_prefix}.profile"
            if os.path.exists(plink_profile_file):
                profile_df = pd.read_csv(plink_profile_file, sep=r'\s+')
                profile_df = profile_df[['IID', 'SCORE']]
                profile_df.rename(columns={'SCORE': f'PRS_ct_p{p_thresh_str}'}, inplace=True)
                
                # Save the final PRS file to the central base_prs directory
                profile_df.to_csv(final_prs_path, sep='\t', index=False)
                print(f"    -> Saved C+T PRS to: {final_prs_path}")
            else:
                print(f"    -> WARNING: Expected PLINK output file not found: {plink_profile_file}")

    finally:
        # Crucially, change back to the original directory, even if errors occur
        os.chdir(original_wd)
        print(f"--- C+T PRS Calculation for {dataset_name} set Finished ---")


def main():
    """Main function to run C+T for both train and test sets."""
    # Run for training set
    run_ct_prs(
        gwas_file=config.GWAS_PROCESSED_FILE,
        target_bfile=config.TARGET_GENO_PREFIX,
        p_thresholds=config.CT_P_THRESHOLDS,
        clump_r2=config.CT_CLUMP_R2,
        clump_kb=config.CT_CLUMP_KB,
        keep_ids_file=config.TRAIN_IDS_FILE,
        output_dir=os.path.join(config.RESULTS_DIR, "ct_prs/train_workspace")
    )

    # Run for testing set
    run_ct_prs(
        gwas_file=config.GWAS_PROCESSED_FILE,
        target_bfile=config.TARGET_GENO_PREFIX,
        p_thresholds=config.CT_P_THRESHOLDS,
        clump_r2=config.CT_CLUMP_R2,
        clump_kb=config.CT_CLUMP_KB,
        keep_ids_file=config.TEST_IDS_FILE,
        output_dir=os.path.join(config.RESULTS_DIR, "ct_prs/test_workspace")
    )

if __name__ == "__main__":
    main()