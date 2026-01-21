# =============================================================================
# R Script: scinference Vignette - Generate Data and Results
# =============================================================================
# This script replicates the R vignette examples and saves:
# 1. The generated data (Y0, Y1)
# 2. All results for comparison with Python
# =============================================================================

# Load the scinference package
library(scinference)

# Create output directory
output_dir <- "/Users/anzony.quisperojas/Documents/GitHub/python/scinference/notebooks/r_output"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# EXAMPLE 1: CONFORMAL INFERENCE
# =============================================================================
cat("=== CONFORMAL INFERENCE EXAMPLE ===\n")

set.seed(12345)

J   <- 50
T0  <- 50
T1  <- 5

w       <- rep(0, J)
w[1:3]  <- 1/3
Y0      <- matrix(rnorm((T0+T1)*J), (T0+T1), J)
Y1      <- Y0 %*% w + rnorm(T0+T1)

Y1[(T0+1):(T0+T1)] <- Y1[(T0+1):(T0+T1)] + 2

# Save conformal data
write.csv(Y0, file.path(output_dir, "Y0_conformal.csv"), row.names = FALSE)
write.csv(data.frame(Y1 = Y1), file.path(output_dir, "Y1_conformal.csv"), row.names = FALSE)

cat("Data dimensions:\n")
cat(sprintf("  Y0: %d x %d\n", nrow(Y0), ncol(Y0)))
cat(sprintf("  Y1: %d\n", length(Y1)))
cat(sprintf("  True effect: 2\n\n"))

# --- Test H0: theta = 4 with Moving Block ---
cat("P-values for H0: theta=4 (Moving Block Permutations)\n")
cat(strrep("=", 55), "\n")

result_sc_mb <- scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                             estimation_method="sc", permutation_method="mb")
cat(sprintf("Synthetic Control:     p-value = %.8f\n", result_sc_mb$p_val))

result_did_mb <- scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                              estimation_method="did", permutation_method="mb")
cat(sprintf("Difference-in-Diff:    p-value = %.8f\n", result_did_mb$p_val))

result_classo_mb <- scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                                 estimation_method="classo", permutation_method="mb")
cat(sprintf("Constrained Lasso:     p-value = %.8f\n\n", result_classo_mb$p_val))

# --- Test H0: theta = 4 with IID ---
cat("P-values for H0: theta=4 (IID Permutations, n_perm=5000)\n")
cat(strrep("=", 55), "\n")

set.seed(42)  # For reproducibility
result_did_iid <- scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                               estimation_method="did", permutation_method="iid")
cat(sprintf("Difference-in-Diff:    p-value = %.8f\n", result_did_iid$p_val))

result_sc_iid <- scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                              estimation_method="sc", permutation_method="iid")
cat(sprintf("Synthetic Control:     p-value = %.8f\n", result_sc_iid$p_val))

result_classo_iid <- scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                                  estimation_method="classo", permutation_method="iid")
cat(sprintf("Constrained Lasso:     p-value = %.8f\n\n", result_classo_iid$p_val))

# --- Confidence Intervals ---
cat("90% Pointwise Confidence Intervals (Synthetic Control)\n")
cat(strrep("=", 55), "\n")

obj_ci <- scinference(Y1, Y0, T1=T1, T0=T0, estimation_method="sc",
                       ci=TRUE, ci_grid=seq(-2, 8, 0.1))

cat(sprintf("%-10s %-15s %-15s\n", "Period", "Lower Bound", "Upper Bound"))
cat(strrep("-", 40), "\n")
for (t in 1:T1) {
  cat(sprintf("%-10d %-15.2f %-15.2f\n", t, obj_ci$lb[t], obj_ci$ub[t]))
}
cat("\n")

# --- Test True Null ---
result_true <- scinference(Y1, Y0, T1=T1, T0=T0, theta0=2,
                            estimation_method="sc", permutation_method="mb")
cat(sprintf("Testing true null H0: theta=2\n"))
cat(sprintf("P-value: %.8f\n\n", result_true$p_val))

# =============================================================================
# EXAMPLE 2: T-TEST INFERENCE
# =============================================================================
cat("=== T-TEST INFERENCE EXAMPLE ===\n")

set.seed(12345)

J_t   <- 30
T0_t  <- 30
T1_t  <- 30

w_t       <- rep(0, J_t)
w_t[1:3]  <- 1/3
Y0_t      <- matrix(rnorm((T0_t+T1_t)*J_t), (T0_t+T1_t), J_t)
Y1_t      <- Y0_t %*% w_t + rnorm(T0_t+T1_t)

Y1_t[(T0_t+1):(T0_t+T1_t)] <- Y1_t[(T0_t+1):(T0_t+T1_t)] + 2

# Save t-test data
write.csv(Y0_t, file.path(output_dir, "Y0_ttest.csv"), row.names = FALSE)
write.csv(data.frame(Y1 = Y1_t), file.path(output_dir, "Y1_ttest.csv"), row.names = FALSE)

cat("Data dimensions:\n")
cat(sprintf("  Y0: %d x %d\n", nrow(Y0_t), ncol(Y0_t)))
cat(sprintf("  Y1: %d\n", length(Y1_t)))
cat(sprintf("  True ATT: 2\n\n"))

# --- T-test K=2 ---
cat("T-test Results (K=2 cross-fits)\n")
cat(strrep("=", 40), "\n")

ttest_K2 <- scinference(Y1_t, Y0_t, T1=T1_t, T0=T0_t, inference_method="ttest", K=2)
cat(sprintf("ATT estimate:     %.8f\n", ttest_K2$att))
cat(sprintf("Standard Error:   %.8f\n", ttest_K2$se))
cat(sprintf("90%% CI:           [%.8f, %.8f]\n\n", ttest_K2$lb, ttest_K2$ub))

# --- T-test K=3 ---
cat("T-test Results (K=3 cross-fits)\n")
cat(strrep("=", 40), "\n")

ttest_K3 <- scinference(Y1_t, Y0_t, T1=T1_t, T0=T0_t, inference_method="ttest", K=3)
cat(sprintf("ATT estimate:     %.8f\n", ttest_K3$att))
cat(sprintf("Standard Error:   %.8f\n", ttest_K3$se))
cat(sprintf("90%% CI:           [%.8f, %.8f]\n\n", ttest_K3$lb, ttest_K3$ub))

# --- T-test DID ---
cat("T-test Results (DID method, K=2)\n")
cat(strrep("=", 40), "\n")

ttest_did <- scinference(Y1_t, Y0_t, T1=T1_t, T0=T0_t, inference_method="ttest",
                          estimation_method="did", K=2)
cat(sprintf("ATT estimate:     %.8f\n", ttest_did$att))
cat(sprintf("Standard Error:   %.8f\n", ttest_did$se))
cat(sprintf("90%% CI:           [%.8f, %.8f]\n\n", ttest_did$lb, ttest_did$ub))

# =============================================================================
# SAVE ALL RESULTS TO JSON-LIKE FORMAT
# =============================================================================
results <- list(
  conformal = list(
    sc_mb_pval = result_sc_mb$p_val,
    did_mb_pval = result_did_mb$p_val,
    classo_mb_pval = result_classo_mb$p_val,
    did_iid_pval = result_did_iid$p_val,
    sc_iid_pval = result_sc_iid$p_val,
    classo_iid_pval = result_classo_iid$p_val,
    ci_lb = obj_ci$lb,
    ci_ub = obj_ci$ub,
    true_null_pval = result_true$p_val
  ),
  ttest = list(
    K2_att = ttest_K2$att,
    K2_se = ttest_K2$se,
    K2_lb = ttest_K2$lb,
    K2_ub = ttest_K2$ub,
    K3_att = ttest_K3$att,
    K3_se = ttest_K3$se,
    K3_lb = ttest_K3$lb,
    K3_ub = ttest_K3$ub,
    did_att = ttest_did$att,
    did_se = ttest_did$se,
    did_lb = ttest_did$lb,
    did_ub = ttest_did$ub
  )
)

# Save results as CSV for easy reading
results_df <- data.frame(
  metric = c(
    "conformal_sc_mb_pval", "conformal_did_mb_pval", "conformal_classo_mb_pval",
    "conformal_did_iid_pval", "conformal_sc_iid_pval", "conformal_classo_iid_pval",
    "conformal_ci_lb_1", "conformal_ci_lb_2", "conformal_ci_lb_3", "conformal_ci_lb_4", "conformal_ci_lb_5",
    "conformal_ci_ub_1", "conformal_ci_ub_2", "conformal_ci_ub_3", "conformal_ci_ub_4", "conformal_ci_ub_5",
    "conformal_true_null_pval",
    "ttest_K2_att", "ttest_K2_se", "ttest_K2_lb", "ttest_K2_ub",
    "ttest_K3_att", "ttest_K3_se", "ttest_K3_lb", "ttest_K3_ub",
    "ttest_did_att", "ttest_did_se", "ttest_did_lb", "ttest_did_ub"
  ),
  value = c(
    result_sc_mb$p_val, result_did_mb$p_val, result_classo_mb$p_val,
    result_did_iid$p_val, result_sc_iid$p_val, result_classo_iid$p_val,
    obj_ci$lb[1], obj_ci$lb[2], obj_ci$lb[3], obj_ci$lb[4], obj_ci$lb[5],
    obj_ci$ub[1], obj_ci$ub[2], obj_ci$ub[3], obj_ci$ub[4], obj_ci$ub[5],
    result_true$p_val,
    ttest_K2$att, ttest_K2$se, ttest_K2$lb, ttest_K2$ub,
    ttest_K3$att, ttest_K3$se, ttest_K3$lb, ttest_K3$ub,
    ttest_did$att, ttest_did$se, ttest_did$lb, ttest_did$ub
  )
)

write.csv(results_df, file.path(output_dir, "r_results.csv"), row.names = FALSE)

cat("\n=== FILES SAVED ===\n")
cat(sprintf("Data: %s/Y0_conformal.csv, Y1_conformal.csv\n", output_dir))
cat(sprintf("Data: %s/Y0_ttest.csv, Y1_ttest.csv\n", output_dir))
cat(sprintf("Results: %s/r_results.csv\n", output_dir))
cat("\nDone!\n")
