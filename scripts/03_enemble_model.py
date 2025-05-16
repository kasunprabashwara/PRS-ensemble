import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob
import config
from scripts.utils import load_phenotypes, load_ids_from_file, read_prs_files

def train_ensemble(
    base_prs_files_pattern, # e.g., "results/base_prs_scores/ukb_train_*.prs"
    pheno_file,
    ids_file, # File with IIDs for this dataset (e.g., training IDs)
    model_output_dir,
    model_name="EnsembleElasticNet",
    is_binary_trait=True
    ):
    """Trains the ensemble model using (Logistic)ElasticNetCV."""
    os.makedirs(model_output_dir, exist_ok=True)

    # 1. Load and merge all base PRS scores for the specified individuals
    print(f"Loading base PRS files matching: {base_prs_files_pattern}")
    prs_files = glob.glob(base_prs_files_pattern)
    if not prs_files:
        print(f"No base PRS files found for pattern {base_prs_files_pattern}. Cannot train ensemble.")
        return None, None, None, []
    
    print(f"Found base PRS files: {prs_files}")
    merged_base_prs_df, base_prs_score_cols = read_prs_files(prs_files)

    if merged_base_prs_df.empty or not base_prs_score_cols:
        print("No data after reading base PRS files. Cannot train ensemble.")
        return None, None, None, []

    # 2. Load phenotypes and covariates for the specified individuals
    pheno_df_all = load_phenotypes(pheno_file)
    target_ids = load_ids_from_file(ids_file)
    pheno_df = pheno_df_all[pheno_df_all['IID'].isin(target_ids)]

    # Merge PRS with phenotypes
    ensemble_train_data = pd.merge(merged_base_prs_df, pheno_df, on="IID", how="inner")
    ensemble_train_data = ensemble_train_data.dropna(
        subset=[config.TARGET_DISEASE_COLUMN] + base_prs_score_cols + config.COVARIATE_COLUMNS
    )

    if ensemble_train_data.empty:
        print("No overlapping samples after merging base PRS and phenotypes for ensemble training.")
        return None, None, None, []

    # 3. Prepare data for scikit-learn
    X_prs = ensemble_train_data[base_prs_score_cols].values
    X_cov = ensemble_train_data[config.COVARIATE_COLUMNS].values
    y = ensemble_train_data[config.TARGET_DISEASE_COLUMN].values

    # Scale PRS scores and covariates separately (as they might have different distributions)
    scaler_prs = StandardScaler()
    X_prs_scaled = scaler_prs.fit_transform(X_prs)
    
    scaler_cov = StandardScaler()
    X_cov_scaled = scaler_cov.fit_transform(X_cov)

    X_ensemble_features = np.concatenate((X_prs_scaled, X_cov_scaled), axis=1)
    all_feature_names = base_prs_score_cols + config.COVARIATE_COLUMNS

    # 4. Train (Logistic)ElasticNetCV model
    if is_binary_trait:
        print("Binary phenotype detected, using LogisticRegressionCV for ensemble.")
        cv_splitter = StratifiedKFold(n_splits=config.CV_FOLDS_ENSEMBLE, shuffle=True, random_state=123)
        ensemble_model = LogisticRegressionCV(
            Cs=10, cv=cv_splitter, penalty='elasticnet', solver='saga',
            l1_ratios=config.ELASTICNET_L1_RATIOS, max_iter=2000, random_state=123, scoring='roc_auc'
        )
    else:
        print("Quantitative phenotype detected, using ElasticNetCV for ensemble.")
        cv_splitter = KFold(n_splits=config.CV_FOLDS_ENSEMBLE, shuffle=True, random_state=123)
        ensemble_model = ElasticNetCV(
            l1_ratio=config.ELASTICNET_L1_RATIOS, cv=cv_splitter, random_state=123, max_iter=2000
        )
        
    ensemble_model.fit(X_ensemble_features, y)

    # Save model, scalers, and feature names
    model_path = os.path.join(model_output_dir, f"{model_name}_model.joblib")
    scaler_prs_path = os.path.join(model_output_dir, f"{model_name}_scaler_prs.joblib")
    scaler_cov_path = os.path.join(model_output_dir, f"{model_name}_scaler_cov.joblib")
    features_path = os.path.join(model_output_dir, f"{model_name}_features.txt")

    joblib.dump(ensemble_model, model_path)
    joblib.dump(scaler_prs, scaler_prs_path)
    joblib.dump(scaler_cov, scaler_cov_path)
    with open(features_path, 'w') as f:
        for feature in all_feature_names:
            f.write(f"{feature}\n")
    
    print(f"Ensemble model, scalers, and feature list saved to {model_output_dir}")

    # Store model coefficients for inspection
    coeffs = ensemble_model.coef_.flatten()
    model_coeffs_df = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': coeffs})
    model_coeffs_df.to_csv(os.path.join(model_output_dir, f"{model_name}_coefficients.csv"), index=False)
    print(f"Ensemble model coefficients saved.")

    return model_path, scaler_prs_path, scaler_cov_path, all_feature_names


def calculate_ensemble_prs_on_new_data(
    base_prs_files_pattern_new_data, # Pattern for base PRS files for the new dataset
    pheno_file_new_data,
    ids_file_new_data, # IDs for the new dataset
    trained_ensemble_model_path,
    trained_scaler_prs_path,
    trained_scaler_cov_path,
    feature_names_from_training, # Order of features model was trained on
    output_prs_file,
    model_name="EnsembleElasticNet"
    ):
    """Calculates ensemble PRS on new data using a trained ensemble model."""

    print(f"Calculating {model_name} PRS for new data using base PRS from: {base_prs_files_pattern_new_data}")
    prs_files_new = glob.glob(base_prs_files_pattern_new_data)
    if not prs_files_new:
        print(f"No base PRS files found for new data pattern {base_prs_files_pattern_new_data}.")
        pd.DataFrame(columns=['IID', f'PRS_{model_name}']).to_csv(output_prs_file, sep='\t', index=False)
        return output_prs_file
        
    merged_base_prs_new_df, base_prs_score_cols_new = read_prs_files(prs_files_new)

    if merged_base_prs_new_df.empty:
        print("No data after reading base PRS files for new data.")
        pd.DataFrame(columns=['IID', f'PRS_{model_name}']).to_csv(output_prs_file, sep='\t', index=False)
        return output_prs_file

    pheno_new_all = load_phenotypes(pheno_file_new_data)
    target_ids_new = load_ids_from_file(ids_file_new_data)
    pheno_new_df = pheno_new_all[pheno_new_all['IID'].isin(target_ids_new)]

    ensemble_new_data = pd.merge(merged_base_prs_new_df, pheno_new_df, on="IID", how="inner")
    # Drop if covariates are missing; disease column not strictly needed for PRS calculation itself
    ensemble_new_data = ensemble_new_data.dropna(subset=base_prs_score_cols_new + config.COVARIATE_COLUMNS)


    if ensemble_new_data.empty:
        print("No overlapping samples after merging base PRS and phenotypes for new data.")
        pd.DataFrame(columns=['IID', f'PRS_{model_name}']).to_csv(output_prs_file, sep='\t', index=False)
        return output_prs_file

    # Ensure features are in the same order as during training
    # And that all necessary base PRS and covariate columns are present
    base_prs_features_trained = [f for f in feature_names_from_training if f.startswith("PRS_")]
    covariate_features_trained = [f for f in feature_names_from_training if f in config.COVARIATE_COLUMNS]
    
    # Check for missing features in the new data that were used in training
    missing_base_prs_cols = [col for col in base_prs_features_trained if col not in ensemble_new_data.columns]
    if missing_base_prs_cols:
        raise ValueError(f"Missing base PRS columns in new data that were used for training: {missing_base_prs_cols}")
    
    X_prs_new = ensemble_new_data[base_prs_features_trained].values
    X_cov_new = ensemble_new_data[covariate_features_trained].values

    # Load trained model and scalers
    ensemble_model = joblib.load(trained_ensemble_model_path)
    scaler_prs = joblib.load(trained_scaler_prs_path)
    scaler_cov = joblib.load(trained_scaler_cov_path)

    X_prs_new_scaled = scaler_prs.transform(X_prs_new) # Use transform
    X_cov_new_scaled = scaler_cov.transform(X_cov_new) # Use transform
    X_ensemble_features_new = np.concatenate((X_prs_new_scaled, X_cov_new_scaled), axis=1)

    if hasattr(ensemble_model, "decision_function"):
        final_prs_scores = ensemble_model.decision_function(X_ensemble_features_new)
    else:
        final_prs_scores = ensemble_model.predict(X_ensemble_features_new)
        
    prs_df_final = pd.DataFrame({'IID': ensemble_new_data['IID'], f'PRS_{model_name}': final_prs_scores})
    prs_df_final.to_csv(output_prs_file, sep='\t', index=False)
    print(f"Ensemble PRS for new data saved to {output_prs_file}")
    return output_prs_file


if __name__ == "__main__":
    # Train ensemble model (e.g., on UKB training set's base PRS)
    # Assumes base PRS files like ukb_train_ldpred2_auto.prs, ukb_train_PenRegElasticNet.prs etc. exist in BASE_PRS_DIR
    base_prs_pattern_train = os.path.join(config.BASE_PRS_DIR, f"ukb_train_*.prs") # Adjust basename if needed
    
    # Create dummy base PRS files for training if they don't exist
    train_ids_list = load_ids_from_file(config.UKB_TRAIN_IDS_FILE)
    if not train_ids_list: # if file is empty or not found, create some dummy ids
        print(f"Warning: {config.UKB_TRAIN_IDS_FILE} is empty or not found. Creating dummy IDs for ensemble training flow.")
        train_ids_list = [f'ID_train_{i}' for i in range(50)] # Placeholder
        # Save these dummy IDs if the file was indeed missing
        if not os.path.exists(config.UKB_TRAIN_IDS_FILE):
             os.makedirs(os.path.dirname(config.UKB_TRAIN_IDS_FILE), exist_ok=True)
             pd.DataFrame({'FID':train_ids_list, 'IID':train_ids_list}).to_csv(config.UKB_TRAIN_IDS_FILE, sep='\t', header=False, index=False)


    if not glob.glob(base_prs_pattern_train) and train_ids_list:
        print(f"Creating dummy base PRS files for pattern: {base_prs_pattern_train}")
        pd.DataFrame({'IID': train_ids_list, 'PRS_ukb_train_dummyLDpred2': pd.np.random.randn(len(train_ids_list))}).to_csv(os.path.join(config.BASE_PRS_DIR, "ukb_train_dummyLDpred2.prs"), sep='\t', index=False)
        pd.DataFrame({'IID': train_ids_list, 'PRS_ukb_train_dummyPenReg': pd.np.random.randn(len(train_ids_list))}).to_csv(os.path.join(config.BASE_PRS_DIR, "ukb_train_dummyPenReg.prs"), sep='\t', index=False)

    # Check if pheno file exists
    if not os.path.exists(config.UKB_PHENO_FILE):
        print(f"Warning: Phenotype file {config.UKB_PHENO_FILE} not found. Cannot train ensemble.")
    elif not os.path.exists(config.UKB_TRAIN_IDS_FILE):
        print(f"Warning: Training IDs file {config.UKB_TRAIN_IDS_FILE} not found. Cannot train ensemble.")
    else:
        is_binary = pd.read_csv(config.UKB_PHENO_FILE)[config.TARGET_DISEASE_COLUMN].nunique() == 2
        model_p, scaler_p, scaler_c, features_p = train_ensemble(
            base_prs_files_pattern=base_prs_pattern_train,
            pheno_file=config.UKB_PHENO_FILE,
            ids_file=config.UKB_TRAIN_IDS_FILE,
            model_output_dir=config.ENSEMBLE_MODEL_DIR,
            is_binary_trait=is_binary
        )

        # Calculate ensemble PRS for validation set
        # base_prs_pattern_valid = os.path.join(config.BASE_PRS_DIR, f"ukb_valid_*.prs")
        # output_ensemble_prs_valid = os.path.join(config.ENSEMBLE_PRS_DIR, f"ukb_valid_ensemble.prs")
        #
        # valid_ids_list = load_ids_from_file(config.UKB_VALID_IDS_FILE)
        # if not glob.glob(base_prs_pattern_valid) and valid_ids_list:
        #    print(f"Creating dummy base PRS files for validation: {base_prs_pattern_valid}")
        #    pd.DataFrame({'IID': valid_ids_list, 'PRS_ukb_valid_dummyLDpred2': pd.np.random.randn(len(valid_ids_list))}).to_csv(os.path.join(config.BASE_PRS_DIR, "ukb_valid_dummyLDpred2.prs"), sep='\t', index=False)
        #    pd.DataFrame({'IID': valid_ids_list, 'PRS_ukb_valid_dummyPenReg': pd.np.random.randn(len(valid_ids_list))}).to_csv(os.path.join(config.BASE_PRS_DIR, "ukb_valid_dummyPenReg.prs"), sep='\t', index=False)

        # if model_p and os.path.exists(config.UKB_VALID_IDS_FILE): # Check if model training was successful
        #     calculate_ensemble_prs_on_new_data(
        #         base_prs_files_pattern_new_data=base_prs_pattern_valid,
        #         pheno_file_new_data=config.UKB_PHENO_FILE,
        #         ids_file_new_data=config.UKB_VALID_IDS_FILE,
        #         trained_ensemble_model_path=model_p,
        #         trained_scaler_prs_path=scaler_p,
        #         trained_scaler_cov_path=scaler_c,
        #         feature_names_from_training=features_p,
        #         output_prs_file=output_ensemble_prs_valid
        #     )
        #     print(f"Ensemble PRS for validation set calculated: {output_ensemble_prs_valid}")
        # else:
        #     print("Skipping ensemble PRS calculation for validation set (model not trained or IDs missing).")