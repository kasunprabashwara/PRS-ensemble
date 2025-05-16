import pandas as pd
import numpy as np
import shap # Make sure this is installed: pip install shap
import joblib
import os
import matplotlib.pyplot as plt
import config

def explain_ensemble_coefficients(ensemble_model_dir, model_name="EnsembleElasticNet"):
    """Loads and displays ensemble model coefficients."""
    coeffs_file = os.path.join(ensemble_model_dir, f"{model_name}_coefficients.csv")
    features_file = os.path.join(ensemble_model_dir, f"{model_name}_features.txt")
    
    if not os.path.exists(coeffs_file) or not os.path.exists(features_file):
        print(f"Coefficient or feature file not found in {ensemble_model_dir}. Run ensemble training first.")
        return None

    coeffs_df = pd.read_csv(coeffs_file)
    print("\n--- Ensemble Model Coefficients (Importance of Base PRS and Covariates) ---")
    print(coeffs_df)

    # Plotting coefficients
    coeffs_df_sorted = coeffs_df.sort_values(by='Coefficient', ascending=False)
    plt.figure(figsize=(10, max(6, len(coeffs_df_sorted) * 0.3))) # Adjust height based on num features
    plt.barh(coeffs_df_sorted['Feature'], coeffs_df_sorted['Coefficient'])
    plt.xlabel("Coefficient Value (Weight in Ensemble)")
    plt.title("Ensemble Model Feature Importances (Coefficients)")
    plt.tight_layout()
    plot_path = os.path.join(config.INTERPRET_DIR, f"{model_name}_coefficient_plot.png")
    plt.savefig(plot_path)
    print(f"Coefficient plot saved to {plot_path}")
    plt.close()
    return coeffs_df

def run_shap_on_ensemble(
    base_prs_files_pattern, # e.g., for UKB Test set: "results/base_prs_scores/ukb_test_*.prs"
    pheno_file,
    ids_file, # e.g., config.UKB_TEST_IDS_FILE
    trained_ensemble_model_path,
    trained_scaler_prs_path,
    trained_scaler_cov_path,
    feature_names_from_training, # From ensemble training
    output_shap_plot_prefix,
    max_display_shap=15
    ):
    """Runs SHAP analysis on the trained ensemble model to explain contributions of base PRS and covariates."""
    os.makedirs(os.path.dirname(output_shap_plot_prefix), exist_ok=True)
    
    print(f"\n--- Running SHAP Analysis on Ensemble Model ---")
    print(f"Using base PRS from: {base_prs_files_pattern}")

    # 1. Prepare data for SHAP (similar to calculate_ensemble_prs_on_new_data)
    prs_files = glob.glob(base_prs_files_pattern)
    if not prs_files:
        print(f"No base PRS files found for SHAP data pattern {base_prs_files_pattern}.")
        return
        
    merged_base_prs_df, base_prs_score_cols_shap = read_prs_files(prs_files)
    if merged_base_prs_df.empty:
        print("No data after reading base PRS files for SHAP.")
        return

    pheno_all = load_phenotypes(pheno_file)
    target_ids = load_ids_from_file(ids_file)
    pheno_df = pheno_all[pheno_all['IID'].isin(target_ids)]

    shap_data_merged = pd.merge(merged_base_prs_df, pheno_df, on="IID", how="inner")
    # Keep only necessary columns and drop NAs for rows that would be used in prediction
    base_prs_features_trained = [f for f in feature_names_from_training if f.startswith("PRS_")]
    covariate_features_trained = [f for f in feature_names_from_training if f in config.COVARIATE_COLUMNS]
    
    required_shap_cols = base_prs_features_trained + covariate_features_trained
    shap_data_merged = shap_data_merged.dropna(subset=required_shap_cols)


    if shap_data_merged.empty:
        print("No overlapping samples for SHAP analysis.")
        return

    X_prs_shap = shap_data_merged[base_prs_features_trained].values
    X_cov_shap = shap_data_merged[covariate_features_trained].values

    # Load trained model and scalers
    ensemble_model = joblib.load(trained_ensemble_model_path)
    scaler_prs = joblib.load(trained_scaler_prs_path)
    scaler_cov = joblib.load(trained_scaler_cov_path)

    X_prs_shap_scaled = scaler_prs.transform(X_prs_shap)
    X_cov_shap_scaled = scaler_cov.transform(X_cov_shap)
    X_ensemble_features_shap = np.concatenate((X_prs_shap_scaled, X_cov_shap_scaled), axis=1)
    
    # Convert to DataFrame for SHAP explainer if needed, with correct feature names
    X_df_shap = pd.DataFrame(X_ensemble_features_shap, columns=feature_names_from_training)

    # 2. Create SHAP Explainer
    # For linear models (ElasticNet, LogisticRegression), LinearExplainer is efficient.
    # If the model is a tree-based model (not the case here), TreeExplainer.
    # KernelExplainer is model-agnostic but slower.
    if isinstance(ensemble_model, (ElasticNetCV, LogisticRegressionCV)) or \
       hasattr(ensemble_model, 'coef_'): # Check if it's likely a linear model
        print("Using shap.LinearExplainer for the ensemble model.")
        # The masker is used to handle feature perturbation for non-linear models,
        # for LinearExplainer, just passing the data itself is common.
        # Using X_df_shap as the background data. For linear models, this choice is less critical.
        explainer = shap.LinearExplainer(ensemble_model, X_df_shap)
    else: # Fallback to KernelExplainer (slower)
        print("Warning: Ensemble model is not recognized as linear. Using shap.KernelExplainer (can be slow).")
        # KernelExplainer needs a background dataset for permutation. A subset of training data is common.
        # For simplicity, using the provided X_df_shap. For large datasets, subsample X_df_shap.
        background_data = shap.sample(X_df_shap, min(100, X_df_shap.shape[0])) # Use at most 100 samples for background
        explainer = shap.KernelExplainer(ensemble_model.predict_proba if hasattr(ensemble_model, "predict_proba") else ensemble_model.predict,
                                         background_data)
    
    shap_values = explainer.shap_values(X_df_shap) # For binary classification, this might be for class 1

    # 3. Generate SHAP plots
    # Summary plot (shows feature importance and impact)
    plt.figure()
    shap.summary_plot(shap_values, X_df_shap, plot_type="bar", max_display=max_display_shap, show=False)
    plt.title(f"SHAP Summary Plot (Bar) for Ensemble Model\n(Base PRS & Covariate Contributions)")
    plt.tight_layout()
    plt.savefig(f"{output_shap_plot_prefix}_summary_bar.png")
    plt.close()
    
    plt.figure()
    # For binary classification from LogisticRegressionCV, shap_values might have two sets if predict_proba was used
    # or if explainer gives for both classes. If decision_function was used, it's one set.
    # If shap_values is a list (e.g. for multi-class or for each class in binary), pick one (e.g. class 1)
    shap_values_for_plot = shap_values
    if isinstance(shap_values, list) and len(shap_values) == 2: # Common for binary classification shap_values output
        shap_values_for_plot = shap_values[1] # Assuming positive class is index 1

    shap.summary_plot(shap_values_for_plot, X_df_shap, max_display=max_display_shap, show=False)
    plt.title(f"SHAP Summary Plot (Dot) for Ensemble Model\n(Base PRS & Covariate Contributions)")
    plt.tight_layout()
    plt.savefig(f"{output_shap_plot_prefix}_summary_dot.png")
    plt.close()
    
    print(f"SHAP summary plots saved with prefix: {output_shap_plot_prefix}")

    # SNP-level interpretability from base models (e.g., PenReg model)
    # If one of your base models was the penalized regression (02b), its coefficients
    # directly indicate SNP importance *for that base model*.
    # You could load those SNP effects and display the top ones.
    # E.g., penreg_snp_effects_file = os.path.join(config.BASE_PRS_DIR, "penreg_model_training/PenRegElasticNet_snp_effects.txt")
    # if os.path.exists(penreg_snp_effects_file):
    #    penreg_effects = pd.read_csv(penreg_snp_effects_file, sep='\t')
    #    penreg_effects['AbsEffect'] = penreg_effects['EffectWeight'].abs()
    #    top_snps_penreg = penreg_effects.sort_values(by='AbsEffect', ascending=False).head(15)
    #    print("\n--- Top SNP contributors to Penalized Regression Base Model ---")
    #    print(top_snps_penreg[['SNP', 'EffectWeight']])


if __name__ == "__main__":
    ensemble_dir = config.ENSEMBLE_MODEL_DIR
    coeffs = explain_ensemble_coefficients(ensemble_dir)

    # Run SHAP on the ensemble model using, for example, UKB test set data
    # Ensure that base PRS scores for the UKB test set have been generated first
    # And the ensemble model has been trained.
    
    trained_model = os.path.join(ensemble_dir, "EnsembleElasticNet_model.joblib")
    scaler_prs_p = os.path.join(ensemble_dir, "EnsembleElasticNet_scaler_prs.joblib")
    scaler_cov_p = os.path.join(ensemble_dir, "EnsembleElasticNet_scaler_cov.joblib")
    features_p_file = os.path.join(ensemble_dir, "EnsembleElasticNet_features.txt")

    if all(os.path.exists(p) for p in [trained_model, scaler_prs_p, scaler_cov_p, features_p_file]):
        with open(features_p_file, 'r') as f:
            trained_features = [line.strip() for line in f if line.strip()]

        # Use UKB test set for SHAP example
        base_prs_pattern_test = os.path.join(config.BASE_PRS_DIR, f"ukb_test_*.prs") # Adjust if needed
        
        # Create dummy base PRS files for test set if they don't exist (for SHAP flow)
        test_ids_list = load_ids_from_file(config.UKB_TEST_IDS_FILE)
        if not test_ids_list:
            print(f"Warning: {config.UKB_TEST_IDS_FILE} is empty or not found. Creating dummy IDs for SHAP flow.")
            test_ids_list = [f'ID_test_{i}' for i in range(30)] # Placeholder
            if not os.path.exists(config.UKB_TEST_IDS_FILE):
                 os.makedirs(os.path.dirname(config.UKB_TEST_IDS_FILE), exist_ok=True)
                 pd.DataFrame({'FID':test_ids_list, 'IID':test_ids_list}).to_csv(config.UKB_TEST_IDS_FILE, sep='\t', header=False, index=False)

        if not glob.glob(base_prs_pattern_test) and test_ids_list:
            print(f"Creating dummy base PRS files for SHAP test data: {base_prs_pattern_test}")
            # Ensure dummy PRS files align with features expected by the loaded ensemble model
            for feat in trained_features:
                if feat.startswith("PRS_"): # This is a base PRS feature
                    # Use a simplified name for the dummy file based on the feature name
                    dummy_file_name_part = feat.replace("PRS_ukb_train_", "").replace("PRS_","") # try to make it generic
                    dummy_prs_file_path = os.path.join(config.BASE_PRS_DIR, f"ukb_test_{dummy_file_name_part}.prs")
                    if not os.path.exists(dummy_prs_file_path) and "dummy" not in dummy_prs_file_path: # Avoid re-creating general dummies
                         # Check if this dummy file corresponds to a trained feature
                         pd.DataFrame({'IID': test_ids_list, feat: pd.np.random.randn(len(test_ids_list))}).to_csv(dummy_prs_file_path, sep='\t', index=False)
                         print(f"Created dummy file: {dummy_prs_file_path} for SHAP feature {feat}")


        if not os.path.exists(config.UKB_PHENO_FILE):
             print(f"Warning: Phenotype file {config.UKB_PHENO_FILE} not found. Cannot run SHAP.")
        elif not os.path.exists(config.UKB_TEST_IDS_FILE):
             print(f"Warning: Test IDs file {config.UKB_TEST_IDS_FILE} not found. Cannot run SHAP.")
        else:
            run_shap_on_ensemble(
                base_prs_files_pattern=base_prs_pattern_test,
                pheno_file=config.UKB_PHENO_FILE,
                ids_file=config.UKB_TEST_IDS_FILE,
                trained_ensemble_model_path=trained_model,
                trained_scaler_prs_path=scaler_prs_p,
                trained_scaler_cov_path=scaler_cov_p,
                feature_names_from_training=trained_features,
                output_shap_plot_prefix=os.path.join(config.INTERPRET_DIR, "EnsembleSHAP_UKBtest")
            )
    else:
        print("Ensemble model/scalers/features not found. Skipping SHAP analysis.")

    print("\nNote on SNP-level interpretability:")
    print("To get SNP-level contributions to the *final ensemble PRS*, you would typically:")
    print("1. For each base model that is SNP-based (e.g., Penalized Regression, LDpred2):")
    print("   Obtain its per-SNP effect sizes (betas).")
    print("2. Get the weight of that base model in the ensemble (from ensemble coefficients).")
    print("3. The effective contribution of a SNP through a particular base model is (SNP_beta_base * weight_base_in_ensemble).")
    print("4. Sum these effective contributions across all base models a SNP participates in.")
    print("This provides an 'effective meta-SNP beta' for the ensemble. SHAP is not directly applied to this whole multi-stage process in one go easily.")
    print("XPRS paper's approach might involve applying SHAP to a specific model structure that allows such decomposition.")