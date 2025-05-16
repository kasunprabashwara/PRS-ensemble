# File Paths
GWAS_SUMSTATS_DIR = "data/gwas_summary_stats/" # Ensure files here are well-formatted
UKB_GENO_PREFIX = "data/uk_biobank/genotypes/ukb_subset" # PLINK prefix (no .bed/.bim/.fam)
UKB_PHENO_FILE = "data/uk_biobank/phenotypes/ukb_pheno.csv"
UKB_TRAIN_IDS_FILE = "data/uk_biobank/ukb_train_ids.txt" # File with IIDs for training set
UKB_VALID_IDS_FILE = "data/uk_biobank/ukb_valid_ids.txt" # File with IIDs for validation set
UKB_TEST_IDS_FILE = "data/uk_biobank/ukb_test_ids.txt"   # File with IIDs for test set

FINNGEN_GENO_PREFIX = "data/finngen/genotypes/finngen_subset"
FINNGEN_PHENO_FILE = "data/finngen/phenotypes/finngen_pheno.csv"
# FINNGEN_IDS_FILE = "data/finngen/finngen_all_ids.txt" # If you need to subset FinnGen for testing

LD_REF_DIR = "data/ld_reference/" # Directory containing LD reference panel files
# Example LD reference prefix (e.g., 1000 Genomes EUR subset in PLINK format)
# Ensure this reference panel matches the ancestry of your GWAS sumstats primarily.
LD_REF_PREFIX = "data/ld_reference/1kg_eur_phase3_common_snps" # PLINK prefix for LD reference

# Tools
PLINK_EXECUTABLE = "plink" # Or full path: "/path/to/plink"
PLINK2_EXECUTABLE = "plink2" # Or full path: "/path/to/plink2" (some tasks are faster)
RSCRIPT_EXECUTABLE = "Rscript" # Or full path: "/path/to/Rscript"

# Path to the LDpred2.R script (download from LDpred2 GitHub repository)
# e.g., https://github.com/privefl/bigsnpr/blob/master/tmp-save/LDpred2.R
LDPRED2_RSCRIPT_PATH = "/path/to/your/LDpred2.R"
# Path to the helper script for LDpred2 input prep if needed (from bigsnpr examples)
# BIGSNPR_HELPER_RSCRIPT_PATH = "/path/to/your/bigsnpr_ldpred2_helper.R"

# Parameters
TARGET_DISEASE_COLUMN = "disease_status" # Name of phenotype column (0/1 for binary)
COVARIATE_COLUMNS = ["age", "sex", "PC1", "PC2", "PC3", "PC4", "PC5"] # Example covariates
GWAS_N_EFFECTIVE_COL = "N_eff" # Effective sample size in sumstats, or 'N' if total N.

# Cross-validation & Hyperparameters for Ensemble
CV_FOLDS_ENSEMBLE = 5
ELASTICNET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0] # For ensemble ElasticNet

# LDpred2 parameters (can be tuned)
LDPRED2_H2_INIT = None # Let LDpred2 estimate, or provide a value e.g., 0.1
LDPRED2_P_SEQ = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2] # Sequence of p (causal prop.) for grid model
LDPRED2_SPARSE_FLAG = False # Whether to run sparse model as well/instead

# Penalized Regression (PROSPER-like base model)
PENREG_CLUMP_P1 = 1         # P-value threshold for index SNPs in PLINK clumping
PENREG_CLUMP_P2 = 1         # P-value threshold for secondary SNPs
PENREG_CLUMP_R2 = 0.1       # LD r2 threshold for clumping
PENREG_CLUMP_KB = 500       # KB window for clumping
PENREG_CV_FOLDS = 5         # CV folds for inner ElasticNetCV for this base model

# Output directories
BASE_PRS_DIR = "results/base_prs_scores/"
ENSEMBLE_PRS_DIR = "results/ensemble_prs_scores/"
ENSEMBLE_MODEL_DIR = "results/ensemble_model/"
VALIDATION_DIR = "results/validation_transferability/"
INTERPRET_DIR = "results/interpretability/"

# Ensure directories exist
import os
os.makedirs(BASE_PRS_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_PRS_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_MODEL_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(INTERPRET_DIR, exist_ok=True)
os.makedirs(GWAS_SUMSTATS_DIR, exist_ok=True)
os.makedirs(LD_REF_DIR, exist_ok=True)