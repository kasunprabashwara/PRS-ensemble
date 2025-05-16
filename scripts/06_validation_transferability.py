import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
import os
import config
from scripts.utils import load_phenotypes, load_ids_from_file
import matplotlib.pyplot as plt

def evaluate_prs_performance(
    prs_file, # File containing IID and PRS score (e.g., 'PRS_Ensemble')
    pheno_file,
    ids_file, # File with IIDs for the specific dataset to evaluate on
    prs_col_name=None, # Automatically detected if None, or specify e.g. 'PRS_Ensemble'
    dataset_name="Dataset",
    output_metrics_file=None,
    output_roc_plot_file=None
    ):
    """Calculates and prints evaluation metrics for a given PRS."""
    print(f"\n--- Evaluating PRS Performance for {dataset_name} ---")
    print(f"PRS file: {prs_file}")

    try:
        prs_df = pd.read_csv(prs_file, delim_whitespace=True) # robust to tab or space
        if '#IID' in prs_df.columns: prs_df.rename(columns={'#IID': 'IID'}, inplace=True)
        prs_df['IID'] = prs_df['IID'].astype(str)
    except FileNotFoundError:
        print(f"Error: PRS file not found: {prs_file}")
        return None
    except Exception as e:
        print(f"Error reading PRS file {prs_file}: {e}")
        return None


    if prs_col_name is None: # Try to auto-detect PRS column
        potential_prs_cols = [col for col in prs_df.columns if col not in ['IID', 'FID', 'PHENO'] and pd.api.types.is_numeric_dtype(prs_df[col])]
        if not potential_prs_cols:
            print(f"Error: No numeric PRS column found in {prs_file}.")
            return None
        prs_col_name = potential_prs_cols[0]
        print(f"Auto-detected PRS column: {prs_col_name}")
    elif prs_col_name not in prs_df.columns:
        print(f"Error: Specified PRS column '{prs_col_name}' not found in {prs_file}.")
        return None

    pheno_all = load_phenotypes(pheno_file)
    target_ids = load_ids_from_file(ids_file)
    pheno_df = pheno_all[pheno_all['IID'].isin(target_ids)]

    eval_data = pd.merge(pheno_df, prs_df[['IID', prs_col_name]], on="IID", how="inner")
    eval_data = eval_data.dropna(subset=[config.TARGET_DISEASE_COLUMN, prs_col_name])

    if eval_data.empty:
        print(f"No overlapping samples with phenotype and PRS for {dataset_name}.")
        return None

    y_true = eval_data[config.TARGET_DISEASE_COLUMN].values
    prs_scores = eval_data[prs_col_name].values
    
    metrics = {'Dataset': dataset_name, 'PRS_Method': prs_col_name, 'N_Samples': len(y_true)}
    
    if len(np.unique(y_true)) == 0: # Should not happen if data is present
        print(f"Warning: Only one class present in y_true for {dataset_name}. Cannot evaluate.")
        return metrics
    if len(np.unique(y_true)) == 1 and len(y_true) > 0 : # Only one class, but some data
        print(f"Warning: Only one class ({np.unique(y_true)[0]}) present in y_true for {dataset_name}. Cannot calculate AUC/AP.")
        return metrics


    is_binary = len(np.unique(y_true)) == 2
    if is_binary:
        try:
            auc = roc_auc_score(y_true, prs_scores)
            ap = average_precision_score(y_true, prs_scores)
            metrics.update({'AUC': auc, 'AP': ap})
            print(f"AUC: {auc:.4f}")
            print(f"Average Precision (PR-AUC): {ap:.4f}")

            if output_roc_plot_file:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_true, prs_scores)
                plt.figure()
                plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {dataset_name} - {prs_col_name}')
                plt.legend(loc='lower right')
                plt.savefig(output_roc_plot_file)
                plt.close()
                print(f"ROC curve plot saved to {output_roc_plot_file}")

        except ValueError as e: # if only one class in y_true after filtering NAs
            print(f"ValueError during metric calculation (likely only one class present): {e}")
            metrics.update({'AUC': np.nan, 'AP': np.nan})


        # Odds Ratio per SD (requires logistic regression)
        # prs_scaled = StandardScaler().fit_transform(prs_scores.reshape(-1, 1))
        # try:
        #     import statsmodels.api as sm
        #     # Add covariates if you want adjusted OR
        #     # X_ logistic = sm.add_constant(np.column_stack((prs_scaled, eval_data[config.COVARIATE_COLUMNS].values)))
        #     X_logistic = sm.add_constant(prs_scaled)
        #     logit_model = sm.Logit(y_true, X_logistic).fit(disp=0)
        #     or_per_sd = np.exp(logit_model.params[1])
        #     or_conf_int = np.exp(logit_model.conf_int().iloc[1].values)
        #     metrics.update({'OR_per_SD': or_per_sd, 'OR_CI_low': or_conf_int[0], 'OR_CI_high': or_conf_int[1]})
        #     print(f"OR per SD: {or_per_sd:.3f} (95% CI: {or_conf_int[0]:.3f}-{or_conf_int[1]:.3f})")
        # except Exception as e:
        #     print(f"Could not calculate OR per SD: {e}")
        #     metrics.update({'OR_per_SD': np.nan})


    else: # Quantitative trait
        r_sq = r2_score(y_true, prs_scores)
        corr = np.corrcoef(y_true, prs_scores)[0, 1]
        metrics.update({'R2': r_sq, 'Correlation': corr})
        print(f"R-squared: {r_sq:.4f}")
        print(f"Correlation: {corr:.4f}")

    if output_metrics_file:
        # Append to metrics file
        mode = 'a' if os.path.exists(output_metrics_file) else 'w'
        header = mode == 'w'
        pd.DataFrame([metrics]).to_csv(output_metrics_file, mode=mode, header=header, index=False)
        print(f"Metrics saved to {output_metrics_file}")
        
    return metrics


if __name__ == "__main__":
    all_results = []
    overall_metrics_summary_file = os.path.join(config.VALIDATION_DIR, "all_evaluation_metrics.csv")
    if os.path.exists(overall_metrics_summary_file): os.remove(overall_metrics_summary_file) # Start fresh

    # --- Evaluate on UK Biobank Test Set ---
    ukb_test_ensemble_prs_file = os.path.join(config.ENSEMBLE_PRS_DIR, "ukb_test_EnsembleElasticNet.prs") # Output from 03
    
    # Create dummy ensemble PRS file for UKB test if it doesn't exist
    if not os.path.exists(ukb_test_ensemble_prs_file):
        print(f"Creating dummy ensemble PRS file for UKB Test: {ukb_test_ensemble_prs_file}")
        ukb_test_ids = load_ids_from_file(config.UKB_TEST_IDS_FILE)
        if not ukb_test_ids : ukb_test_ids = [f"testID{i}" for i in range(20)] # if id file missing
        pd.DataFrame({'IID': ukb_test_ids, 'PRS_EnsembleElasticNet': pd.np.random.randn(len(ukb_test_ids))}).to_csv(ukb_test_ensemble_prs_file, sep='\t', index=False)

    if not os.path.exists(config.UKB_PHENO_FILE): print(f"UKB Pheno file missing: {config.UKB_PHENO_FILE}")
    if not os.path.exists(config.UKB_TEST_IDS_FILE): print(f"UKB Test IDs file missing: {config.UKB_TEST_IDS_FILE}")

    if os.path.exists(ukb_test_ensemble_prs_file) and os.path.exists(config.UKB_PHENO_FILE) and os.path.exists(config.UKB_TEST_IDS_FILE):
        metrics_ukb_test = evaluate_prs_performance(
            prs_file=ukb_test_ensemble_prs_file,
            pheno_file=config.UKB_PHENO_FILE,
            ids_file=config.UKB_TEST_IDS_FILE,
            prs_col_name="PRS_EnsembleElasticNet", # Adjust if your ensemble output has a different PRS col name
            dataset_name="UKB_TestSet",
            output_metrics_file=overall_metrics_summary_file,
            output_roc_plot_file=os.path.join(config.VALIDATION_DIR, "roc_ukb_test_ensemble.png")
        )
        if metrics_ukb_test: all_results.append(metrics_ukb_test)
    else:
        print("Skipping UKB Test Set evaluation due to missing PRS, phenotype, or ID files.")


    # --- Evaluate on FinnGen (Transferability) ---
    finngen_ensemble_prs_file = os.path.join(config.ENSEMBLE_PRS_DIR, "finngen_EnsembleElasticNet.prs") # Output from 03
    
    # Create dummy ensemble PRS file for FinnGen if it doesn't exist
    if not os.path.exists(finngen_ensemble_prs_file):
        print(f"Creating dummy ensemble PRS file for FinnGen: {finngen_ensemble_prs_file}")
        # Try to get FinnGen IDs for dummy file, assumes a FINNGEN_IDS_FILE might exist from config
        finngen_ids = []
        if hasattr(config, 'FINNGEN_IDS_FILE') and config.FINNGEN_IDS_FILE and os.path.exists(config.FINNGEN_IDS_FILE):
            finngen_ids = load_ids_from_file(config.FINNGEN_IDS_FILE)
        if not finngen_ids: finngen_ids = [f"fgID{i}" for i in range(20)]
        pd.DataFrame({'IID': finngen_ids, 'PRS_EnsembleElasticNet': pd.np.random.randn(len(finngen_ids))}).to_csv(finngen_ensemble_prs_file, sep='\t', index=False)

    if not os.path.exists(config.FINNGEN_PHENO_FILE): print(f"FinnGen Pheno file missing: {config.FINNGEN_PHENO_FILE}")
    # Assuming FinnGen IDs for evaluation would be all IDs present in the FINNGEN_PHENO_FILE that match the PRS file,
    # or you'd create a specific FINNGEN_EVAL_IDS_FILE. For simplicity, using all from pheno file.
    # Or if config.FINNGEN_IDS_FILE exists, use that.
    finngen_eval_ids_path = config.FINNGEN_PHENO_FILE # Use pheno file to get all possible IDs and then inner join with PRS.
    # A specific ID file is better:
    # finngen_eval_ids_path = getattr(config, 'FINNGEN_IDS_FILE', None)
    # if not finngen_eval_ids_path or not os.path.exists(finngen_eval_ids_path):
    #     print("Warning: FINNGEN_IDS_FILE not specified or found. Using all IDs from pheno for FinnGen eval.")
    #     # Create a dummy one based on pheno if pheno exists
    #     if os.path.exists(config.FINNGEN_PHENO_FILE):
    #         temp_fg_ids = pd.read_csv(config.FINNGEN_PHENO_FILE)['IID'].unique()
    #         # Need FID, IID format for load_ids_from_file if using that strictly
    #         # For now, this example will rely on inner merge logic within evaluate_prs_performance
    # else :
    #    print(f"Using FinnGen IDs from: {finngen_eval_ids_path}")


    if os.path.exists(finngen_ensemble_prs_file) and os.path.exists(config.FINNGEN_PHENO_FILE) :
        # If using all IDs from pheno file that are in PRS:
        # Create a temporary ID file from the pheno file's IIDs for the function
        temp_fg_ids_df = pd.read_csv(config.FINNGEN_PHENO_FILE)[['IID']] # Assumes IID col
        temp_fg_ids_df.insert(0, 'FID', temp_fg_ids_df['IID']) # Dummy FID
        temp_finngen_ids_file = os.path.join(config.VALIDATION_DIR, "temp_finngen_eval_ids.txt")
        temp_fg_ids_df.to_csv(temp_finngen_ids_file, sep='\t', header=False, index=False)


        metrics_finngen = evaluate_prs_performance(
            prs_file=finngen_ensemble_prs_file,
            pheno_file=config.FINNGEN_PHENO_FILE,
            ids_file=temp_finngen_ids_file, # Use all IDs from FinnGen pheno file that are in PRS
            prs_col_name="PRS_EnsembleElasticNet",
            dataset_name="FinnGen_Transfer",
            output_metrics_file=overall_metrics_summary_file,
            output_roc_plot_file=os.path.join(config.VALIDATION_DIR, "roc_finngen_ensemble.png")
        )
        if metrics_finngen: all_results.append(metrics_finngen)
        if os.path.exists(temp_finngen_ids_file): os.remove(temp_finngen_ids_file)
    else:
        print("Skipping FinnGen evaluation due to missing PRS or phenotype files.")

    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n\n--- Overall Evaluation Summary ---")
        print(summary_df)
        # summary_df.to_csv(overall_metrics_summary_file, index=False) # Already appended
    else:
        print("\nNo evaluation results to summarize.")