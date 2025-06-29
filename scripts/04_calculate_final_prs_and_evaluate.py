import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import joblib
import os
import glob
import matplotlib.pyplot as plt
import config

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from utils import load_phenotypes, load_ids_from_file, read_prs_files

def calculate_ensemble_prs(
    base_prs_files_pattern,
    pheno_file,
    ids_file,
    model_dir,
    output_prs_file):
    """Calculates final ensemble PRS on new data using a trained model."""
    print(f"\n--- Calculating Final Ensemble PRS for {os.path.basename(ids_file)} ---")
    model_name = config.ENSEMBLE_MODEL_NAME

    # 1. Load trained model, scalers, and feature list
    try:
        ensemble_model = joblib.load(os.path.join(model_dir, f"{model_name}_model.joblib"))
        scaler_prs = joblib.load(os.path.join(model_dir, f"{model_name}_scaler_prs.joblib"))
        scaler_cov = joblib.load(os.path.join(model_dir, f"{model_name}_scaler_cov.joblib"))
        with open(os.path.join(model_dir, f"{model_name}_features.txt"), 'r') as f:
            feature_names_from_training = [line.strip() for line in f]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not load trained ensemble model files from {model_dir}. Please run 03_ensemble_model.py first. Error: {e}")

    # 2. Load and merge base PRS scores for the new data
    prs_files_new = glob.glob(base_prs_files_pattern)
    if not prs_files_new:
        raise FileNotFoundError(f"No base PRS files found for new data pattern: {base_prs_files_pattern}")
    
    merged_base_prs_new_df, _ = read_prs_files(prs_files_new)
    pheno_new_all = load_phenotypes(pheno_file)
    target_ids_new = load_ids_from_file(ids_file)
    pheno_new_df = pheno_new_all[pheno_new_all['IID'].isin(target_ids_new)]

    # 3. Merge and prepare data, ensuring all required columns are present
    ensemble_new_data = pd.merge(merged_base_prs_new_df, pheno_new_df, on="IID", how="inner")
    
    base_prs_features_trained = [f for f in feature_names_from_training if f.startswith("PRS_")]
    covariate_features_trained = [f for f in feature_names_from_training if f in config.COVARIATE_COLUMNS]
    
    missing_cols = [c for c in base_prs_features_trained + covariate_features_trained if c not in ensemble_new_data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in new data required for prediction: {missing_cols}")
    
    ensemble_new_data.dropna(subset=base_prs_features_trained + covariate_features_trained, inplace=True)
    if ensemble_new_data.empty:
        print("Warning: No overlapping samples found for PRS calculation. Output will be empty.")
        pd.DataFrame(columns=['IID', f'PRS_{model_name}']).to_csv(output_prs_file, sep='\t', index=False)
        return

    # 4. Apply scalers and predict
    X_prs_new = ensemble_new_data[base_prs_features_trained].values
    X_cov_new = ensemble_new_data[covariate_features_trained].values
    X_prs_new_scaled = scaler_prs.transform(X_prs_new)
    X_cov_new_scaled = scaler_cov.transform(X_cov_new)
    X_ensemble_features_new = np.concatenate((X_prs_new_scaled, X_cov_new_scaled), axis=1)

    final_prs_scores = ensemble_model.decision_function(X_ensemble_features_new)
    
    prs_df_final = pd.DataFrame({'IID': ensemble_new_data['IID'], f'PRS_{model_name}': final_prs_scores})
    prs_df_final.to_csv(output_prs_file, sep='\t', index=False)
    print(f"Final ensemble PRS saved to {output_prs_file}")

def evaluate_prs(prs_file, pheno_file, ids_file, dataset_name, output_dir):
    """Calculates and saves evaluation metrics (AUC, AP) and an ROC plot."""
    print(f"\n--- Evaluating PRS Performance for {dataset_name} ---")
    
    prs_df = pd.read_csv(prs_file, sep='\t')
    pheno_all = load_phenotypes(pheno_file)
    target_ids = load_ids_from_file(ids_file)
    pheno_df = pheno_all[pheno_all['IID'].isin(target_ids)]
    
    eval_data = pd.merge(pheno_df, prs_df, on="IID", how="inner")
    prs_col_name = [col for col in prs_df.columns if col.startswith('PRS_')][0]
    eval_data.dropna(subset=[config.TARGET_DISEASE_COLUMN, prs_col_name], inplace=True)

    if eval_data.empty or eval_data[config.TARGET_DISEASE_COLUMN].nunique() < 2:
        print("Not enough data or classes to evaluate performance.")
        return

    y_true = eval_data[config.TARGET_DISEASE_COLUMN]
    y_scores = eval_data[prs_col_name]

    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    metrics = {
        'Dataset': dataset_name,
        'PRS_Method': 'Ensemble',
        'N_Samples': len(y_true),
        'N_Cases': int(y_true.sum()),
        'AUC': auc,
        'Average_Precision': ap
    }
    print(pd.DataFrame([metrics]))
    
    # Save metrics
    metrics_file = os.path.join(output_dir, f"metrics_{dataset_name}.csv")
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    
    # Create and save ROC plot
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend(loc="lower right")
    plot_file = os.path.join(output_dir, f"roc_curve_{dataset_name}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Metrics and ROC plot saved to {output_dir}")

def main():
    # 1. Calculate final ensemble PRS for the test set
    test_prs_output_file = os.path.join(config.ENSEMBLE_PRS_DIR, "test_ensemble_prs.prs")
    calculate_ensemble_prs(
        base_prs_files_pattern=os.path.join(config.BASE_PRS_DIR, "test_*.prs"),
        pheno_file=config.PHENO_FILE,
        ids_file=config.TEST_IDS_FILE,
        model_dir=config.ENSEMBLE_MODEL_DIR,
        output_prs_file=test_prs_output_file
    )

    # 2. Evaluate the performance on the test set
    evaluate_prs(
        prs_file=test_prs_output_file,
        pheno_file=config.PHENO_FILE,
        ids_file=config.TEST_IDS_FILE,
        dataset_name="Internal_Test_Set",
        output_dir=config.EVALUATION_DIR
    )
    print("--- Final PRS Calculation and Evaluation Finished ---")

if __name__ == "__main__":
    main()