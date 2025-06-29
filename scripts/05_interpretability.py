import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


import config

def plot_ensemble_coefficients(model_dir, output_dir):
    """Loads and plots the coefficients of the trained ensemble model."""
    print("--- Generating Ensemble Interpretability Plot ---")
    model_name = config.ENSEMBLE_MODEL_NAME
    coeffs_file = os.path.join(model_dir, f"{model_name}_coefficients.csv")

    if not os.path.exists(coeffs_file):
        print(f"Coefficient file not found: {coeffs_file}. Run 03_ensemble_model.py first.")
        return

    coeffs_df = pd.read_csv(coeffs_file)
    coeffs_df = coeffs_df.sort_values(by='Coefficient', ascending=False)
    
    # Separate PRS from covariates for clearer plotting
    prs_coeffs = coeffs_df[coeffs_df['Feature'].str.startswith('PRS_')].copy()
    cov_coeffs = coeffs_df[~coeffs_df['Feature'].str.startswith('PRS_')].copy()

    # Clean up feature names for plotting
    prs_coeffs['Feature'] = prs_coeffs['Feature'].str.replace('PRS_', '')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [3, 1]})

    # Plot for base PRS methods
    sns.barplot(x='Coefficient', y='Feature', data=prs_coeffs, ax=axes[0], palette='viridis')
    axes[0].set_title('Ensemble Weights for Base PRS Methods', fontsize=16)
    axes[0].set_xlabel('Coefficient (Weight in Ensemble)', fontsize=12)
    axes[0].set_ylabel('Base PRS Method', fontsize=12)

    # Plot for covariates
    sns.barplot(x='Coefficient', y='Feature', data=cov_coeffs, ax=axes[1], palette='plasma')
    axes[1].set_title('Covariate Weights', fontsize=16)
    axes[1].set_xlabel('Coefficient', fontsize=12)
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_coefficient_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    print("\n--- Ensemble Model Coefficients (Higher is more important) ---")
    print(coeffs_df.to_string())
    print(f"\nCoefficient plot saved to {plot_path}")
    print("--- Interpretability Step Finished ---")

def main():
    plot_ensemble_coefficients(
        model_dir=config.ENSEMBLE_MODEL_DIR,
        output_dir=config.INTERPRET_DIR
    )

if __name__ == "__main__":
    main()