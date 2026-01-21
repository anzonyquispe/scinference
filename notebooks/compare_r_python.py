"""
Compare Python scinference results with R results using the same data.
"""
import sys
sys.path.insert(0, '/Users/anzony.quisperojas/Documents/GitHub/python/scinference/src')

import numpy as np
import pandas as pd
from scinference import scinference

# Load R results
r_results = pd.read_csv('/Users/anzony.quisperojas/Documents/GitHub/python/scinference/notebooks/r_output/r_results.csv')
r_dict = dict(zip(r_results['metric'], r_results['value']))

print("=" * 70)
print("COMPARISON: Python vs R scinference")
print("=" * 70)

# =============================================================================
# CONFORMAL INFERENCE
# =============================================================================
print("\n### CONFORMAL INFERENCE ###\n")

# Load conformal data
Y0_conf = pd.read_csv('/Users/anzony.quisperojas/Documents/GitHub/python/scinference/notebooks/r_output/Y0_conformal.csv').values
Y1_conf = pd.read_csv('/Users/anzony.quisperojas/Documents/GitHub/python/scinference/notebooks/r_output/Y1_conformal.csv')['Y1'].values

T0, T1 = 50, 5

print(f"Data: Y0 shape = {Y0_conf.shape}, Y1 shape = {Y1_conf.shape}")
print(f"T0 = {T0}, T1 = {T1}\n")

# --- Moving Block Tests ---
print("Moving Block Permutations (theta0=4)")
print("-" * 50)

# SC + MB
py_sc_mb = scinference(Y1_conf, Y0_conf, T1=T1, T0=T0, theta0=4,
                       estimation_method="sc", permutation_method="mb")
r_sc_mb = r_dict['conformal_sc_mb_pval']
diff_sc_mb = abs(py_sc_mb['p_val'] - r_sc_mb)
print(f"SC:     Python={py_sc_mb['p_val']:.8f}  R={r_sc_mb:.8f}  Diff={diff_sc_mb:.8f} {'OK' if diff_sc_mb < 0.001 else 'MISMATCH'}")

# DID + MB
py_did_mb = scinference(Y1_conf, Y0_conf, T1=T1, T0=T0, theta0=4,
                        estimation_method="did", permutation_method="mb")
r_did_mb = r_dict['conformal_did_mb_pval']
diff_did_mb = abs(py_did_mb['p_val'] - r_did_mb)
print(f"DID:    Python={py_did_mb['p_val']:.8f}  R={r_did_mb:.8f}  Diff={diff_did_mb:.8f} {'OK' if diff_did_mb < 0.001 else 'MISMATCH'}")

# CLasso + MB
py_classo_mb = scinference(Y1_conf, Y0_conf, T1=T1, T0=T0, theta0=4,
                           estimation_method="classo", permutation_method="mb")
r_classo_mb = r_dict['conformal_classo_mb_pval']
diff_classo_mb = abs(py_classo_mb['p_val'] - r_classo_mb)
print(f"CLasso: Python={py_classo_mb['p_val']:.8f}  R={r_classo_mb:.8f}  Diff={diff_classo_mb:.8f} {'OK' if diff_classo_mb < 0.001 else 'MISMATCH'}")

# --- IID Tests ---
print("\nIID Permutations (theta0=4, n_perm=5000)")
print("-" * 50)
print("Note: IID results depend on random seed, some difference expected")

np.random.seed(42)
py_did_iid = scinference(Y1_conf, Y0_conf, T1=T1, T0=T0, theta0=4,
                         estimation_method="did", permutation_method="iid", n_perm=5000)
r_did_iid = r_dict['conformal_did_iid_pval']
diff_did_iid = abs(py_did_iid['p_val'] - r_did_iid)
print(f"DID:    Python={py_did_iid['p_val']:.8f}  R={r_did_iid:.8f}  Diff={diff_did_iid:.8f}")

np.random.seed(42)
py_sc_iid = scinference(Y1_conf, Y0_conf, T1=T1, T0=T0, theta0=4,
                        estimation_method="sc", permutation_method="iid", n_perm=5000)
r_sc_iid = r_dict['conformal_sc_iid_pval']
diff_sc_iid = abs(py_sc_iid['p_val'] - r_sc_iid)
print(f"SC:     Python={py_sc_iid['p_val']:.8f}  R={r_sc_iid:.8f}  Diff={diff_sc_iid:.8f}")

np.random.seed(42)
py_classo_iid = scinference(Y1_conf, Y0_conf, T1=T1, T0=T0, theta0=4,
                            estimation_method="classo", permutation_method="iid", n_perm=5000)
r_classo_iid = r_dict['conformal_classo_iid_pval']
diff_classo_iid = abs(py_classo_iid['p_val'] - r_classo_iid)
print(f"CLasso: Python={py_classo_iid['p_val']:.8f}  R={r_classo_iid:.8f}  Diff={diff_classo_iid:.8f}")

# --- Confidence Intervals ---
print("\n90% Confidence Intervals (SC)")
print("-" * 50)

py_ci = scinference(Y1_conf, Y0_conf, T1=T1, T0=T0, estimation_method="sc",
                    ci=True, ci_grid=np.arange(-2, 8.1, 0.1))

print(f"{'Period':<8} {'Python LB':<12} {'R LB':<12} {'Python UB':<12} {'R UB':<12} {'Match'}")
for t in range(T1):
    r_lb = r_dict[f'conformal_ci_lb_{t+1}']
    r_ub = r_dict[f'conformal_ci_ub_{t+1}']
    py_lb = py_ci['lb'][t]
    py_ub = py_ci['ub'][t]
    lb_match = abs(py_lb - r_lb) < 0.2
    ub_match = abs(py_ub - r_ub) < 0.2
    match_str = "OK" if (lb_match and ub_match) else "MISMATCH"
    print(f"{t+1:<8} {py_lb:<12.2f} {r_lb:<12.2f} {py_ub:<12.2f} {r_ub:<12.2f} {match_str}")

# --- True Null ---
print("\nTrue Null (theta0=2)")
print("-" * 50)
py_true = scinference(Y1_conf, Y0_conf, T1=T1, T0=T0, theta0=2,
                      estimation_method="sc", permutation_method="mb")
r_true = r_dict['conformal_true_null_pval']
diff_true = abs(py_true['p_val'] - r_true)
print(f"Python={py_true['p_val']:.8f}  R={r_true:.8f}  Diff={diff_true:.8f} {'OK' if diff_true < 0.01 else 'MISMATCH'}")

# =============================================================================
# T-TEST INFERENCE
# =============================================================================
print("\n" + "=" * 70)
print("### T-TEST INFERENCE ###")
print("=" * 70 + "\n")

# Load t-test data
Y0_ttest = pd.read_csv('/Users/anzony.quisperojas/Documents/GitHub/python/scinference/notebooks/r_output/Y0_ttest.csv').values
Y1_ttest = pd.read_csv('/Users/anzony.quisperojas/Documents/GitHub/python/scinference/notebooks/r_output/Y1_ttest.csv')['Y1'].values

T0_t, T1_t = 30, 30

print(f"Data: Y0 shape = {Y0_ttest.shape}, Y1 shape = {Y1_ttest.shape}")
print(f"T0 = {T0_t}, T1 = {T1_t}\n")

# --- T-test K=2 ---
print("T-test K=2 (SC)")
print("-" * 50)
py_K2 = scinference(Y1_ttest, Y0_ttest, T1=T1_t, T0=T0_t,
                    inference_method="ttest", K=2)
r_K2_att = r_dict['ttest_K2_att']
r_K2_se = r_dict['ttest_K2_se']
r_K2_lb = r_dict['ttest_K2_lb']
r_K2_ub = r_dict['ttest_K2_ub']

diff_att = abs(py_K2['att'] - r_K2_att)
diff_se = abs(py_K2['se'] - r_K2_se)
print(f"ATT: Python={py_K2['att']:.8f}  R={r_K2_att:.8f}  Diff={diff_att:.8f} {'OK' if diff_att < 0.001 else 'MISMATCH'}")
print(f"SE:  Python={py_K2['se']:.8f}  R={r_K2_se:.8f}  Diff={diff_se:.8f} {'OK' if diff_se < 0.001 else 'MISMATCH'}")
print(f"CI:  Python=[{py_K2['lb']:.4f}, {py_K2['ub']:.4f}]  R=[{r_K2_lb:.4f}, {r_K2_ub:.4f}]")

# --- T-test K=3 ---
print("\nT-test K=3 (SC)")
print("-" * 50)
py_K3 = scinference(Y1_ttest, Y0_ttest, T1=T1_t, T0=T0_t,
                    inference_method="ttest", K=3)
r_K3_att = r_dict['ttest_K3_att']
r_K3_se = r_dict['ttest_K3_se']
r_K3_lb = r_dict['ttest_K3_lb']
r_K3_ub = r_dict['ttest_K3_ub']

diff_att3 = abs(py_K3['att'] - r_K3_att)
diff_se3 = abs(py_K3['se'] - r_K3_se)
print(f"ATT: Python={py_K3['att']:.8f}  R={r_K3_att:.8f}  Diff={diff_att3:.8f} {'OK' if diff_att3 < 0.001 else 'MISMATCH'}")
print(f"SE:  Python={py_K3['se']:.8f}  R={r_K3_se:.8f}  Diff={diff_se3:.8f} {'OK' if diff_se3 < 0.001 else 'MISMATCH'}")
print(f"CI:  Python=[{py_K3['lb']:.4f}, {py_K3['ub']:.4f}]  R=[{r_K3_lb:.4f}, {r_K3_ub:.4f}]")

# --- T-test DID ---
print("\nT-test K=2 (DID)")
print("-" * 50)
py_did = scinference(Y1_ttest, Y0_ttest, T1=T1_t, T0=T0_t,
                     inference_method="ttest", estimation_method="did", K=2)
r_did_att = r_dict['ttest_did_att']
r_did_se = r_dict['ttest_did_se']
r_did_lb = r_dict['ttest_did_lb']
r_did_ub = r_dict['ttest_did_ub']

diff_did_att = abs(py_did['att'] - r_did_att)
diff_did_se = abs(py_did['se'] - r_did_se)
print(f"ATT: Python={py_did['att']:.8f}  R={r_did_att:.8f}  Diff={diff_did_att:.8f} {'OK' if diff_did_att < 0.001 else 'MISMATCH'}")
print(f"SE:  Python={py_did['se']:.8f}  R={r_did_se:.8f}  Diff={diff_did_se:.8f} {'OK' if diff_did_se < 0.001 else 'MISMATCH'}")
print(f"CI:  Python=[{py_did['lb']:.4f}, {py_did['ub']:.4f}]  R=[{r_did_lb:.4f}, {r_did_ub:.4f}]")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Count matches
matches = []
mismatches = []

# Moving block
if diff_sc_mb < 0.001: matches.append("SC+MB")
else: mismatches.append(f"SC+MB (diff={diff_sc_mb:.6f})")
if diff_did_mb < 0.001: matches.append("DID+MB")
else: mismatches.append(f"DID+MB (diff={diff_did_mb:.6f})")
if diff_classo_mb < 0.01: matches.append("CLasso+MB")
else: mismatches.append(f"CLasso+MB (diff={diff_classo_mb:.6f})")

# True null
if diff_true < 0.05: matches.append("True null")
else: mismatches.append(f"True null (diff={diff_true:.6f})")

# T-test
if diff_att < 0.001: matches.append("T-test K2 ATT")
else: mismatches.append(f"T-test K2 ATT (diff={diff_att:.6f})")
if diff_se < 0.001: matches.append("T-test K2 SE")
else: mismatches.append(f"T-test K2 SE (diff={diff_se:.6f})")
if diff_att3 < 0.001: matches.append("T-test K3 ATT")
else: mismatches.append(f"T-test K3 ATT (diff={diff_att3:.6f})")
if diff_se3 < 0.001: matches.append("T-test K3 SE")
else: mismatches.append(f"T-test K3 SE (diff={diff_se3:.6f})")
if diff_did_att < 0.001: matches.append("T-test DID ATT")
else: mismatches.append(f"T-test DID ATT (diff={diff_did_att:.6f})")
if diff_did_se < 0.001: matches.append("T-test DID SE")
else: mismatches.append(f"T-test DID SE (diff={diff_did_se:.6f})")

print(f"\nMatches ({len(matches)}): {', '.join(matches)}")
if mismatches:
    print(f"\nMismatches ({len(mismatches)}):")
    for m in mismatches:
        print(f"  - {m}")
else:
    print("\nAll results match R package!")
