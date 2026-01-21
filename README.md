# scinference

A Python package for inference methods for synthetic control and related methods.

This is a Python port of the [R scinference package](https://github.com/kwuthrich/scinference) by Victor Chernozhukov, Kaspar Wuthrich, and Yinchu Zhu.

## Installation

```bash
pip install scinference
```

## Quick Start

```python
import numpy as np
from scinference import scinference

# Generate example data
np.random.seed(12345)
J, T0, T1 = 50, 50, 5
Y0 = np.random.randn(T0 + T1, J)
w = np.zeros(J)
w[:3] = 1/3
Y1 = Y0 @ w + np.random.randn(T0 + T1)
Y1[T0:] += 2  # Add treatment effect of 2

# Test null hypothesis theta0=4 using synthetic control
result = scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                     estimation_method="sc", permutation_method="mb")
print(f"p-value: {result['p_val']:.4f}")
```

## Conformal Inference

The conformal inference method works with a small number of post-treatment periods. It supports three estimation methods and two permutation approaches.

### Testing a Null Hypothesis

```python
# Test H0: theta = 4 with different estimation methods

# Synthetic Control with Moving Block permutations
result_sc = scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                        estimation_method="sc", permutation_method="mb")
print(f"SC p-value: {result_sc['p_val']:.4f}")

# Difference-in-Differences
result_did = scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                         estimation_method="did", permutation_method="mb")
print(f"DID p-value: {result_did['p_val']:.4f}")

# Constrained Lasso
result_classo = scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                            estimation_method="classo", permutation_method="mb")
print(f"CLasso p-value: {result_classo['p_val']:.4f}")
```

### IID Permutations

For IID permutations (randomly drawn), use `permutation_method="iid"`:

```python
result_iid = scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                         estimation_method="sc", permutation_method="iid",
                         n_perm=5000)
print(f"p-value (IID): {result_iid['p_val']:.4f}")
```

### Pointwise Confidence Intervals

Compute pointwise confidence intervals by specifying a grid of values to test:

```python
# Compute 90% pointwise confidence intervals
result_ci = scinference(Y1, Y0, T1=T1, T0=T0,
                        estimation_method="sc",
                        ci=True,
                        ci_grid=np.arange(-2, 8.1, 0.1),
                        alpha=0.1)

print("90% Confidence Intervals:")
for t in range(T1):
    print(f"  Period {t+1}: [{result_ci['lb'][t]:.2f}, {result_ci['ub'][t]:.2f}]")
```

## T-test Based Inference

The t-test method requires a **larger number of post-treatment periods** (typically T1 >= 20). It provides estimates of the Average Treatment Effect on the Treated (ATT).

```python
# Data with more post-treatment periods
T0, T1 = 30, 30
Y0 = np.random.randn(T0 + T1, 30)
Y1 = Y0.mean(axis=1) + np.random.randn(T0 + T1)
Y1[T0:] += 2  # Treatment effect

# T-test with K=2 cross-fits
result = scinference(Y1, Y0, T1=T1, T0=T0,
                     inference_method="ttest", K=2, alpha=0.1)

print(f"ATT estimate: {result['att']:.4f}")
print(f"Standard Error: {result['se']:.4f}")
print(f"90% CI: [{result['lb']:.4f}, {result['ub']:.4f}]")
```

### T-test with Different Cross-fits

```python
# K=3 cross-fits for potentially better performance
result_K3 = scinference(Y1, Y0, T1=T1, T0=T0,
                        inference_method="ttest", K=3)

# Using DID estimation instead of SC
result_did = scinference(Y1, Y0, T1=T1, T0=T0,
                         inference_method="ttest",
                         estimation_method="did", K=2)
```

## API Reference

### Main Function

```python
scinference(Y1, Y0, T1, T0,
            inference_method="conformal",
            alpha=0.1,
            ci=False,
            theta0=0,
            estimation_method="sc",
            permutation_method="mb",
            ci_grid=None,
            n_perm=5000,
            K=2)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Y1` | array | Required | Outcome for treated unit (T x 1) |
| `Y0` | array | Required | Outcomes for control units (T x J) |
| `T1` | int | Required | Number of post-treatment periods |
| `T0` | int | Required | Number of pre-treatment periods |
| `inference_method` | str | "conformal" | "conformal" or "ttest" |
| `alpha` | float | 0.1 | Significance level |
| `ci` | bool | False | Compute confidence intervals |
| `theta0` | float/array | 0 | Null hypothesis value |
| `estimation_method` | str | "sc" | "did", "sc", or "classo" |
| `permutation_method` | str | "mb" | "mb" (moving block) or "iid" |
| `ci_grid` | array | None | Grid for CI computation |
| `n_perm` | int | 5000 | Permutations for IID method |
| `K` | int | 2 | Number of cross-fits (t-test) |

### Returns

**Conformal inference:**
- `p_val`: p-value for the null hypothesis
- `lb`: lower bounds of pointwise CIs (if `ci=True`)
- `ub`: upper bounds of pointwise CIs (if `ci=True`)

**T-test:**
- `att`: Average Treatment Effect on the Treated
- `se`: Standard error
- `lb`: Lower bound of confidence interval
- `ub`: Upper bound of confidence interval

## Estimation Methods

- **did**: Difference-in-differences (simple average of controls)
- **sc**: Synthetic control (Abadie et al.) - constrained weighted average
- **classo**: Constrained lasso - L1-penalized with sum-to-one constraint

## Development

```bash
# Clone the repository
git clone https://github.com/anzonyquispe/scinference.git
cd scinference

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=scinference
```

## References

- Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). "An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls." *Journal of the American Statistical Association*. [arXiv:1712.09089](https://arxiv.org/abs/1712.09089)

- Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2019). "Practical and robust t-test based inference for synthetic control and related methods." [arXiv:1812.10820](https://arxiv.org/abs/1812.10820)

## License

GPL-3.0
