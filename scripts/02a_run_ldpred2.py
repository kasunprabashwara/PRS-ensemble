import subprocess
import os

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import config

def run_ldpred2_auto(
    rscript_path,
    gwas_file,
    ld_ref_prefix,
    target_geno_prefix,
    output_prefix,
    keep_ids_file=None,
    n_cores=4):
    """
    Runs the LDpred2-auto R script using subprocess.
    """
    print("--- Starting LDpred2-auto PRS Calculation ---")
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    cmd = [
        config.RSCRIPT_EXECUTABLE,
        rscript_path,
        "--sumstats", gwas_file,
        "--ld_ref", ld_ref_prefix,
        "--target_geno", target_geno_prefix,
        "--out", output_prefix,
        "--n_cores", str(n_cores)
    ]

    if keep_ids_file and os.path.exists(keep_ids_file):
        cmd.extend(["--keep", keep_ids_file])

    if config.LDPRED2_H2_EST is not None:
        cmd.extend(["--h2_est", str(config.LDPRED2_H2_EST)])

    print("Running LDpred2 R command:")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True, text=True, capture_output=False) # Set capture_output to False to see live output
        print("LDpred2 script executed successfully.")
        print(f"PRS scores and betas saved with prefix: {output_prefix}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("!!!!!! ERROR running LDpred2 R script !!!!!!")
        if isinstance(e, subprocess.CalledProcessError):
            print(e.stdout)
            print(e.stderr)
        raise

def main():
    """
    Main function to run LDpred2 for both training and testing sets.
    """
    # Check for necessary R script
    if not os.path.exists(config.LDPRED2_R_SCRIPT):
        print(f"FATAL: LDpred2 R script not found at {config.LDPRED2_R_SCRIPT}.")
        print("Please ensure the `ldpred2_wrapper.R` file is in the `scripts/` directory.")
        return

    # 1. Generate PRS for the Training set
    print("\nGenerating LDpred2 PRS for Training Set...")
    run_ldpred2_auto(
        rscript_path=config.LDPRED2_R_SCRIPT,
        gwas_file=config.GWAS_PROCESSED_FILE,
        ld_ref_prefix=config.LD_REF_PLINK_PREFIX,
        target_geno_prefix=config.TARGET_GENO_PREFIX,
        output_prefix=os.path.join(config.BASE_PRS_DIR, "train_ldpred2"),
        keep_ids_file=config.TRAIN_IDS_FILE
    )

    # 2. Generate PRS for the Test set
    print("\nGenerating LDpred2 PRS for Test Set...")
    run_ldpred2_auto(
        rscript_path=config.LDPRED2_R_SCRIPT,
        gwas_file=config.GWAS_PROCESSED_FILE,
        ld_ref_prefix=config.LD_REF_PLINK_PREFIX,
        target_geno_prefix=config.TARGET_GENO_PREFIX,
        output_prefix=os.path.join(config.BASE_PRS_DIR, "test_ldpred2"),
        keep_ids_file=config.TEST_IDS_FILE
    )
    print("--- LDpred2 PRS Calculation Finished for all sets ---")

if __name__ == "__main__":
    main()