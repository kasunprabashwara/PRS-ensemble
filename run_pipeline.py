import os
import subprocess
import sys

# Temporarily add project root to sys.path to find config
# This ensures the runner script itself can find config.py
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
import config


def run_script(script_path):
    """Helper function to run a python script from the project root."""
    print(f"\n{'='*25} Running Script: {os.path.basename(script_path)} {'='*25}")
    # Construct the full path to the script inside the scripts directory
    script_abs_path = os.path.join(config.SCRIPTS_DIR, os.path.basename(script_path))
    
    try:
        # **THE FIX IS HERE**: We tell subprocess to run the command
        # as if we were standing in the project's root directory.
        process = subprocess.run(
            ["python", script_abs_path],
            check=True,
            capture_output=True,
            text=True,
            cwd=config.PROJECT_ROOT 
        )
        print(f"--- STDOUT: {os.path.basename(script_path)} ---")
        print(process.stdout)
        if process.stderr:
            print(f"--- STDERR: {os.path.basename(script_path)} ---")
            print(process.stderr)
        print(f"{'='*25} Finished Script: {os.path.basename(script_path)} {'='*25}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!!!!! ERROR running {script_path} !!!!!!")
        print("Return Code:", e.returncode)
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"!!!!!! SCRIPT NOT FOUND: {script_abs_path} !!!!!!")
        print("This can happen if the script name is wrong or it's not in the scripts/ folder.")
        return False


def main():
    """
    Main function to run the entire PRS ensemble pipeline.
    """
    print("Starting the PRS Ensemble Pipeline...")

    # Define the sequence of script basenames to run
    scripts_to_run = [
        "01_data_preprocessing.py",
        "02a_run_ldpred2.py",
        "02b_run_penalized_regression_prs.py",
        "02c_run_ct_prs.py",
        "03_ensemble_model.py",
        "04_calculate_final_prs_and_evaluate.py",
        "05_interpretability.py"
    ]

    # --- Pre-run Checks ---
    print("\n--- Performing pre-run checks ---")
    all_checks_ok = True
    # Check for GWAS file
    if not os.path.exists(config.GWAS_RAW_FILE):
        print(f"FATAL ERROR: GWAS file not found at {config.GWAS_RAW_FILE}.")
        print("Please place your GWAS summary statistics file there.")
        all_checks_ok = False

    # Check for executables in PATH
    for exe in [config.PLINK_EXECUTABLE, config.RSCRIPT_EXECUTABLE]:
        try:
            # Use a common flag that works for most tools
            subprocess.run([exe, "--version"], capture_output=True, check=True, text=True)
            print(f"Found executable: {exe}")
        except (subprocess.CalledProcessError, FileNotFoundError):
             try: # PLINK1 has a different help flag
                subprocess.run([exe, "--help"], capture_output=True, text=True)
                print(f"Found executable: {exe}")
             except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"FATAL ERROR: Executable '{exe}' not found in your system PATH.")
                print("Please install it or provide its full path in config.py.")
                all_checks_ok = False


    if not all_checks_ok:
        print("\nPre-run checks failed. Please fix the issues and try again.")
        return
    else:
        print("--- Pre-run checks passed ---\n")

    # Run each script in order
    for script in scripts_to_run:
        if not run_script(script):
            print("\nPipeline execution failed at one of the steps. Please check the error messages above.")
            return

    print("\n🎉🎉🎉 PRS Ensemble Pipeline Completed Successfully! 🎉🎉🎉")
    print(f"Final evaluation results (metrics, plots) are in: {config.EVALUATION_DIR}")
    print(f"Interpretability plots are in: {config.INTERPRET_DIR}")
    print(f"Final ensemble scores are in: {config.ENSEMBLE_PRS_DIR}")


if __name__ == "__main__":
    main()