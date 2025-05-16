import subprocess
import pandas as pd
import os
import config

def run_plink_command(args, cmd_name="PLINK"):
    """Helper function to run PLINK commands."""
    executable = config.PLINK_EXECUTABLE if cmd_name == "PLINK" else config.PLINK2_EXECUTABLE
    cmd = [executable] + args
    print(f"Running {cmd_name}: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"{cmd_name} Error stdout:\n{stdout}")
            print(f"{cmd_name} Error stderr:\n{stderr}")
            raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)
        print(f"{cmd_name} stdout:\n{stdout}")
        if stderr: # PLINK often prints info to stderr
             print(f"{cmd_name} stderr:\n{stderr}")
        return stdout, stderr
    except FileNotFoundError:
        print(f"Error: {executable} not found. Please check config.py and your PATH.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error during {cmd_name} execution: {e}")
        raise

def load_phenotypes(pheno_file, iid_col='IID', disease_col=config.TARGET_DISEASE_COLUMN, covariate_cols=None):
    """Loads and prepares phenotype data."""
    if covariate_cols is None:
        covariate_cols = config.COVARIATE_COLUMNS
    pheno_df = pd.read_csv(pheno_file)
    pheno_df[iid_col] = pheno_df[iid_col].astype(str)
    
    required_cols = [iid_col, disease_col] + covariate_cols
    missing_cols = [col for col in required_cols if col not in pheno_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in phenotype file {pheno_file}: {missing_cols}")
        
    return pheno_df[[iid_col] + [disease_col] + covariate_cols].dropna()


def load_ids_from_file(ids_file_path):
    """Loads a list of IIDs from a file (one IID per line, or FID IID)."""
    if not os.path.exists(ids_file_path):
        print(f"Warning: ID file {ids_file_path} not found. Returning empty list.")
        return []
    ids_df = pd.read_csv(ids_file_path, header=None, delim_whitespace=True)
    if ids_df.shape[1] == 1: # Only IIDs
        return ids_df[0].astype(str).tolist()
    elif ids_df.shape[1] >= 2: # FID and IID
        return ids_df[1].astype(str).tolist()
    return []

def read_prs_files(prs_file_paths):
    """Reads multiple PRS files and merges them by IID."""
    all_prs_dfs = []
    base_prs_score_cols = []

    for f_path in prs_file_paths:
        try:
            prs_df = pd.read_csv(f_path, delim_whitespace=True) # Adjust sep if needed
            # Standardize IID column name if it varies (e.g. 'IID', '#IID', 'id')
            if '#IID' in prs_df.columns: prs_df.rename(columns={'#IID': 'IID'}, inplace=True)
            elif 'id' in prs_df.columns: prs_df.rename(columns={'id': 'IID'}, inplace=True)
            
            if 'IID' not in prs_df.columns:
                print(f"Warning: 'IID' column not found in {f_path}. Skipping.")
                continue

            prs_df['IID'] = prs_df['IID'].astype(str)
            
            # Identify PRS column (often 'PRS', 'SCORE', or the only other numeric column)
            potential_prs_cols = [col for col in prs_df.columns if col not in ['IID', 'FID', 'PHENO'] and pd.api.types.is_numeric_dtype(prs_df[col])]
            if not potential_prs_cols:
                print(f"Warning: No numeric PRS column found in {f_path}. Skipping.")
                continue
            
            prs_col_name = potential_prs_cols[0] # Take the first one
            if len(potential_prs_cols) > 1:
                print(f"Warning: Multiple potential PRS columns in {f_path} ({potential_prs_cols}). Using '{prs_col_name}'.")

            # Rename PRS column to be unique if file name implies method
            method_name = os.path.basename(f_path).split('.')[0].replace('_prs', '').replace('_scores', '')
            unique_prs_col_name = f"PRS_{method_name}"
            prs_df.rename(columns={prs_col_name: unique_prs_col_name}, inplace=True)
            
            base_prs_score_cols.append(unique_prs_col_name)
            all_prs_dfs.append(prs_df[['IID', unique_prs_col_name]].set_index('IID'))
        except Exception as e:
            print(f"Could not load or process PRS file {f_path}: {e}")
            continue
    
    if not all_prs_dfs:
        return pd.DataFrame(columns=['IID']), []

    merged_prs = pd.concat(all_prs_dfs, axis=1).reset_index()
    return merged_prs, base_prs_score_cols