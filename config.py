"""
Configuration file for the PRS-ensemble pipeline.
"""
import os

# --- Project Directories ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# --- Executables ---
# IMPORTANT: Ensure these are in your system's PATH or provide full paths.
PLINK_EXECUTABLE = "plink"
PLINK2_EXECUTABLE = "plink2"
RSCRIPT_EXECUTABLE = "Rscript"

# --- Input Data: GWAS Summaary Statistics ---
GWAS_RAW_FILE = os.path.join(DATA_DIR, "GCST013197.tsv")
GWAS_PROCESSED_FILE = os.path.join(DATA_DIR, "AD_gwas_formatted.txt")

# Column mapping for your specific GWAS file
GWAS_COLS = {
    "chr_col": "chromosome",
    "bp_col": "base_pair_location",
    "ea_col": "effect_allele",
    "oa_col": "other_allele",
    "beta_col": "z_score", # Note: Using Z-score as beta; will be scaled by SE in formatting
    "se_col": "standard_error",
    "eaf_col": "effect_allele_frequency",
    "pval_col": "p_value",
    "n_col": "n",
    # A unique variant ID will be created during preprocessing if not present
    "snp_col": "variant_id"
}

# --- Input Data: Genotypes and Phenotypes ---
# IMPORTANT: Replace these with paths to your actual data.
# The pipeline will generate DUMMY data for these paths if they don't exist.

# LD reference panel (e.g., 1000 Genomes EUR subset)
LD_REF_PLINK_PREFIX = os.path.join(DATA_DIR, "ld_ref/1000G.EUR")
# Target genotype data for PRS calculation
TARGET_GENO_PREFIX = os.path.join(DATA_DIR, "target_genotypes/target_data")
PHENO_FILE = os.path.join(DATA_DIR, "phenotypes.csv")

# --- Input Data: Subject IDs for splitting ---
IDS_DIR = os.path.join(DATA_DIR, "subject_ids")
TRAIN_IDS_FILE = os.path.join(IDS_DIR, "train_ids.txt")
TEST_IDS_FILE = os.path.join(IDS_DIR, "test_ids.txt")

# --- Phenotype and Covariates ---
TARGET_DISEASE_COLUMN = "AD_CASE" # Name of the phenotype column in PHENO_FILE
COVARIATE_COLUMNS = ['AGE', 'SEX'] + [f'PC{i}' for i in range(1, 11)]

# --- PRS Method Parameters ---

# 1. LDpred2 Parameters
LDPRED2_R_SCRIPT = os.path.join(SCRIPTS_DIR, "ldpred2_wrapper.R")
LDPRED2_H2_EST = None # Let LDpred2-auto estimate heritability

# 2. Penalized Regression (PenReg) Parameters
PENREG_CLUMP_P1 = 0.05
PENREG_CLUMP_R2 = 0.1
PENREG_CLUMP_KB = 250
PENREG_CV_FOLDS = 5
PENREG_L1_RATIOS = [0.1, 0.5, 0.8, 0.95, 1]

# 3. C+T (Clumping and Thresholding) Parameters
CT_CLUMP_P1 = 1
CT_CLUMP_R2 = 0.1
CT_CLUMP_KB = 250
CT_P_THRESHOLDS = [5e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0]

# --- Ensemble Model Parameters ---
ENSEMBLE_MODEL_NAME = "EnsembleAD"
CV_FOLDS_ENSEMBLE = 5
ELASTICNET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1]

# --- Output Directories ---
BASE_PRS_DIR = os.path.join(RESULTS_DIR, "base_prs")
ENSEMBLE_MODEL_DIR = os.path.join(MODEL_DIR, "ensemble")
ENSEMBLE_PRS_DIR = os.path.join(RESULTS_DIR, "ensemble_prs")
EVALUATION_DIR = os.path.join(RESULTS_DIR, "evaluation")
INTERPRET_DIR = os.path.join(RESULTS_DIR, "interpretability")

# --- Create directories if they don't exist ---
for d in [DATA_DIR, RESULTS_DIR, MODEL_DIR, IDS_DIR, BASE_PRS_DIR, ENSEMBLE_MODEL_DIR, ENSEMBLE_PRS_DIR, EVALUATION_DIR, INTERPRET_DIR, os.path.join(DATA_DIR, 'ld_ref'), os.path.join(DATA_DIR, 'target_genotypes')]:
    os.makedirs(d, exist_ok=True)