# In scripts/utils.py

import sys
import os

# Get the absolute path of the project's root directory by going one level up from the script's directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now, the rest of the imports will work correctly
import subprocess
import pandas as pd

import config


# --- THIS IS THE FUNCTION TO FIX ---
def run_plink_command(args, cmd_name="PLINK", log_prefix=None):
    """Helper function to run PLINK commands and log output."""
    print(f"Running {cmd_name}: {' '.join(args)}")
    try:
        # Add log file argument using --out, which PLINK understands
        if log_prefix:
            log_dir = os.path.dirname(log_prefix)
            os.makedirs(log_dir, exist_ok=True)
            # The '--out' argument tells PLINK where to write its .log, .bed, .bim, .fam etc.
            # This replaces the need for separate stdout/stderr redirection for logging.
            args.extend(["--out", log_prefix])

        process = subprocess.run(
            [config.PLINK_EXECUTABLE] + args,
            check=True,
            capture_output=True,
            text=True
        )
        # PLINK often prints useful info to stderr, so we print both
        print(f"--- STDOUT: {cmd_name} ---")
        print(process.stdout)
        if process.stderr:
            print(f"--- STDERR: {cmd_name} ---")
            print(process.stderr)
        return process.stdout, process.stderr
    except FileNotFoundError:
        print(f"Error: {config.PLINK_EXECUTABLE} not found. Please check config.py and your PATH.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"!!!!!! ERROR during {cmd_name} execution !!!!!!")
        print("Return Code:", e.returncode)
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        raise
# --- END OF THE FUNCTION TO FIX ---


def load_phenotypes(pheno_file, iid_col='IID'):
    """Loads and prepares phenotype data."""
    if not os.path.exists(pheno_file):
        raise FileNotFoundError(f"Phenotype file not found: {pheno_file}")

    pheno_df = pd.read_csv(pheno_file)
    pheno_df[iid_col] = pheno_df[iid_col].astype(str)

    required_cols = [iid_col, config.TARGET_DISEASE_COLUMN] + config.COVARIATE_COLUMNS
    missing_cols = [col for col in required_cols if col not in pheno_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in phenotype file {pheno_file}: {missing_cols}")

    return pheno_df[required_cols]

def load_ids_from_file(ids_file_path):
    """Loads a list of IIDs from a file (FID IID format)."""
    if not os.path.exists(ids_file_path):
        print(f"Warning: ID file {ids_file_path} not found. Returning empty list.")
        return []
    ids_df = pd.read_csv(ids_file_path, header=None, delim_whitespace=True)
    if ids_df.shape[1] >= 2: # FID and IID
        return ids_df[1].astype(str).tolist()
    return []

def read_prs_files(prs_file_paths):
    """Reads multiple PRS files and merges them by IID."""
    all_prs_dfs = []
    base_prs_score_cols = []

    for f_path in prs_file_paths:
        try:
            prs_df = pd.read_csv(f_path, delim_whitespace=True)
            if '#IID' in prs_df.columns:
                prs_df.rename(columns={'#IID': 'IID'}, inplace=True)

            if 'IID' not in prs_df.columns:
                print(f"Warning: 'IID' column not found in {f_path}. Skipping.")
                continue
            prs_df['IID'] = prs_df['IID'].astype(str)

            # Find the PRS score column
            potential_prs_cols = [col for col in prs_df.columns if col.upper().startswith(('PRS', 'SCORE'))]
            if not potential_prs_cols:
                 potential_prs_cols = [col for col in prs_df.columns if col not in ['FID', 'IID', 'PHENO'] and pd.api.types.is_numeric_dtype(prs_df[col])]

            if not potential_prs_cols:
                print(f"Warning: No numeric PRS column found in {f_path}. Skipping.")
                continue
            prs_col_name = potential_prs_cols[0]

            # Create a unique name from the file name, e.g., 'ukb_train_ldpred2.prs' -> 'PRS_ldpred2'
            basename = os.path.basename(f_path).split('.')[0]
            # Remove common prefixes/suffixes
            method_name = basename.replace('train_', '').replace('test_', '').replace('_prs', '')
            unique_prs_col_name = f"PRS_{method_name}"

            prs_df.rename(columns={prs_col_name: unique_prs_col_name}, inplace=True)
            base_prs_score_cols.append(unique_prs_col_name)
            all_prs_dfs.append(prs_df[['IID', unique_prs_col_name]].set_index('IID'))

        except Exception as e:
            print(f"Could not load or process PRS file {f_path}: {e}")
            continue

    if not all_prs_dfs:
        return pd.DataFrame(columns=['IID']), []

    # Use outer join to keep all individuals, then reset index to bring IID back as a column
    merged_prs = pd.concat(all_prs_dfs, axis=1, join='outer').reset_index()
    return merged_prs, sorted(list(set(base_prs_score_cols)))