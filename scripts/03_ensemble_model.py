import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


import config
from utils import load_phenotypes, load_ids_from_file, read_prs_files

def train_ensemble(
    base_prs_files_pattern,
    pheno_file,
    ids_file,
    model_output_dir):
    """Trains the ensemble model using LogisticElasticNetCV."""
    print("--- Starting Ensemble Model Training ---")
    os.makedirs(model_output_dir, exist_ok=True)

    # 1. Load and merge all base PRS scores for the training individuals
    print(f"Loading base PRS files for training: {base_prs_files_pattern}")
    prs_files = glob.glob(base_prs_files_pattern)
    if not prs_files:
        raise FileNotFoundError(f"No base PRS files found for pattern {base_prs_files_pattern}. Cannot train ensemble.")
    
    merged_base_prs_df, base_prs_score_cols = read_prs_files(prs_files)
    if merged_base_prs_df.empty or not base_prs_score_cols:
        raise ValueError("No data after reading base PRS files. Cannot train ensemble.")

    # 2. Load phenotypes and covariates for the training individuals
    pheno_df_all = load_phenotypes(pheno_file)
    target_ids = load_ids_from_file(ids_file)
    pheno_df = pheno_df_all[pheno_df_all['IID'].isin(target_ids)]

    # 3. Merge and prepare data for scikit-learn
    ensemble_train_data = pd.merge(merged_base_prs_df, pheno_df, on="IID", how="inner")
    ensemble_train_data = ensemble_train_data.dropna(
        subset=[config.TARGET_DISEASE_COLUMN] + base_prs_score_cols + config.COVARIATE_COLUMNS
    )
    if ensemble_train_data.empty:
        raise ValueError("No overlapping samples after merging base PRS and phenotypes for training.")

    X_prs = ensemble_train_data[base_prs_score_cols].values
    X_cov = ensemble_train_data[config.COVARIATE_COLUMNS].values
    y = ensemble_train_data[config.TARGET_DISEASE_COLUMN].values

    scaler_prs = StandardScaler().fit(X_prs)
    scaler_cov = StandardScaler().fit(X_cov)
    X_prs_scaled = scaler_prs.transform(X_prs)
    X_cov_scaled = scaler_cov.transform(X_cov)

    X_ensemble_features = np.concatenate((X_prs_scaled, X_cov_scaled), axis=1)
    all_feature_names = base_prs_score_cols + config.COVARIATE_COLUMNS

    # 4. Train LogisticElasticNetCV model
    cv_splitter = StratifiedKFold(n_splits=config.CV_FOLDS_ENSEMBLE, shuffle=True, random_state=123)
    ensemble_model = LogisticRegressionCV(
        Cs=10, cv=cv_splitter, penalty='elasticnet', solver='saga',
        l1_ratios=config.ELASTICNET_L1_RATIOS, max_iter=5000, random_state=123, scoring='roc_auc', n_jobs=-1
    )
    print("Fitting ensemble model...")
    ensemble_model.fit(X_ensemble_features, y)

    # 5. Save model, scalers, and feature names
    model_name = config.ENSEMBLE_MODEL_NAME
    joblib.dump(ensemble_model, os.path.join(model_output_dir, f"{model_name}_model.joblib"))
    joblib.dump(scaler_prs, os.path.join(model_output_dir, f"{model_name}_scaler_prs.joblib"))
    joblib.dump(scaler_cov, os.path.join(model_output_dir, f"{model_name}_scaler_cov.joblib"))
    with open(os.path.join(model_output_dir, f"{model_name}_features.txt"), 'w') as f:
        for feature in all_feature_names: f.write(f"{feature}\n")
    
    coeffs = ensemble_model.coef_.flatten()
    model_coeffs_df = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': coeffs})
    model_coeffs_df.to_csv(os.path.join(model_output_dir, f"{model_name}_coefficients.csv"), index=False)
    
    print(f"Ensemble model and artifacts saved to {model_output_dir}")
    print("--- Ensemble Model Training Finished ---")

def main():
    train_ensemble(
        base_prs_files_pattern=os.path.join(config.BASE_PRS_DIR, "train_*.prs"),
        pheno_file=config.PHENO_FILE,
        ids_file=config.TRAIN_IDS_FILE,
        model_output_dir=config.ENSEMBLE_MODEL_DIR
    )

if __name__ == "__main__":
    main()