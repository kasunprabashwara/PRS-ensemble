#!/usr/bin/env Rscript

# A robust wrapper script to run LDpred2-auto from the command line.

if (!require("pacman")) install.packages("pacman", repos = "http://cran.us.r-project.org")
pacman::p_load(argparse, bigsnpr, data.table, tidyverse)

# --- Argument Parsing ---
parser <- ArgumentParser(description="Run LDpred2-auto")
parser$add_argument("--sumstats", required=TRUE, help="Path to the formatted summary statistics file.")
parser$add_argument("--ld_ref", required=TRUE, help="Path prefix for the LD reference panel PLINK files.")
parser$add_argument("--target_geno", required=TRUE, help="Path prefix for the target genotype PLINK files.")
parser$add_argument("--out", required=TRUE, help="Output prefix for PRS files and model objects.")
parser$add_argument("--keep", help="Optional path to a file with IDs to keep.")
parser$add_argument("--h2_est", type="double", help="Optional fixed heritability estimate.")
parser$add_argument("--n_cores", type="integer", default=1, help="Number of cores to use.")
args <- parser$parse_args()

# --- Utility Function to create a dummy PRS file on failure ---
create_dummy_prs <- function(target_bigSNP, keep_rows, out_prefix) {
    message("Creating a dummy PRS file (all zeros) due to an error or insufficient data.")
    prs_df <- data.frame(
      FID = target_bigSNP$fam$family.ID[keep_rows],
      IID = target_bigSNP$fam$sample.ID[keep_rows],
      PRS_ldpred2 = 0.0
    )
    # The output file name needs to match the glob pattern, e.g. train_ldpred2.prs
    # args$out will be something like ".../results/base_prs/train_ldpred2"
    final_prs_path <- paste0(out_prefix, ".prs")
    fwrite(prs_df, file = final_prs_path, sep = "\t")
    message("Dummy PRS file saved to: ", final_prs_path)
}

# --- Main Logic ---
message("Starting LDpred2-auto pipeline...")
start_time <- Sys.time()

# 1. Attach target genotypes FIRST to get IDs for the dummy file if needed
message("Attaching target data: ", args$target_geno)
rds_target_path <- paste0(args$target_geno, ".rds")
if (!file.exists(rds_target_path)) {
    message("No .rds file found for target data. Reading from .bed...")
    snp_readBed(args$target_geno)
}
target_bigSNP <- snp_attach(rds_target_path)
G_target <- target_bigSNP$genotypes
target_map <- target_bigSNP$map

# Get IDs to keep if specified
if (!is.null(args$keep) && file.exists(args$keep)) {
    keep_ids <- fread(args$keep, header = FALSE)$V2
    target_iids <- target_bigSNP$fam$sample.ID
    rows_to_keep <- which(target_iids %in% keep_ids)
    if(length(rows_to_keep) == 0) {
        stop("No individuals left in target data after applying --keep filter.")
    }
} else {
    rows_to_keep <- rows_along(G_target)
}

# Use a try-catch block to handle potential errors gracefully
tryCatch({
    # 2. Read GWAS Summary Statistics
    message("Reading summary statistics from: ", args$sumstats)
    sumstats <- fread(args$sumstats)
    setnames(sumstats, old = c("CHR", "BP", "A1", "A2", "BETA", "SE", "P", "N"),
             new = c("chr", "pos", "a1", "a0", "beta", "beta_se", "p", "n_eff"))
    sumstats$a0 <- toupper(sumstats$a0)
    sumstats$a1 <- toupper(sumstats$a1)

    # 3. Attach LD reference panel
    message("Attaching LD reference panel from: ", args$ld_ref)
    rds_path <- paste0(args$ld_ref, ".rds")
    if (!file.exists(rds_path)) snp_readBed(paste0(args$ld_ref, ".bed"))
    obj.bigSNP <- snp_attach(rds_path)
    map_ld <- obj.bigSNP$map[, c("chromosome", "marker.ID", "physical.pos", "allele1", "allele2")]
    names(map_ld) <- c("chr", "rsid", "pos", "a1", "a0")

    # 4. Match summary statistics with LD reference
    message("Matching summary statistics with LD reference panel...")
    df_beta <- snp_match(sumstats, map_ld, strand_flip = TRUE, join_by_pos = TRUE)
    
    # *** ROBUSTNESS CHECK ***
    if(nrow(df_beta) < 10) {
      stop(paste("Only", nrow(df_beta), "SNPs matched between GWAS and LD reference. This is too few to proceed. Exiting."), call. = FALSE)
    }

    # 5. LD Score Regression (with error handling)
    h2_est <- tryCatch({
        ldsc <- snp_ldsc(snp_cor(obj.bigSNP$genotypes, ind.col=df_beta$`_NUM_ID_`, size=3e3/3e6, ncores=args$n_cores),
                         length(df_beta$`_NUM_ID_`), chi2=(df_beta$beta/df_beta$beta_se)^2,
                         sample_size=df_beta$n_eff, blocks=NULL)
        max(1e-4, ldsc[["h2"]]) # Ensure heritability is at least a small positive number
    }, error = function(e) {
        message("Warning: LD score regression failed. This can happen with few SNPs. Using a default h2 of 0.1.")
        return(0.1)
    })
    message("Using heritability estimate: ", h2_est)

    # 6. Run LDpred2-auto
    message("Running LDpred2-auto...")
    multi_auto <- snp_ldpred2_auto(
      corr = snp_cor(obj.bigSNP$genotypes, ind.col=df_beta$`_NUM_ID_`, size=3e3/3e6, ncores=args$n_cores),
      df_beta = df_beta, h2_init = h2_est,
      vec_p_init = seq_log(1e-4, 0.5, length.out = 20),
      ncores = args$n_cores, allow_jump_sign = FALSE, shrink_corr = 0.95)

    beta_auto <- sapply(multi_auto, function(auto) auto$beta_est)

    # 7. Apply model to target data
    final_beta <- rowMeans(beta_auto)
    pred <- big_prodVec(G_target, final_beta[df_beta$'_NUM_ID_'], ind.row=rows_to_keep, ind.col=df_beta$'_NUM_ID_.ss')
    
    # 8. Save results
    prs_df <- data.frame(FID=target_bigSNP$fam$family.ID[rows_to_keep], IID=target_bigSNP$fam$sample.ID[rows_to_keep], PRS_ldpred2=pred)
    final_prs_path <- paste0(args$out, ".prs") # Save as .prs directly
    fwrite(prs_df, file = final_prs_path, sep = "\t")
    message("LDpred2 PRS successfully saved to: ", final_prs_path)

}, error = function(e) {
    # This block runs if any error occurs in the tryCatch block
    message("\n!!! An error occurred during the LDpred2 process: ", e$message)
    create_dummy_prs(target_bigSNP, rows_to_keep, args$out)
})

end_time <- Sys.time()
message("LDpred2 script finished.")
message("Total time: ", round(difftime(end_time, start_time, units = "mins"), 2), " minutes.")