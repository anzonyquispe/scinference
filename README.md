# scinference

A Python package for inference methods for synthetic control and related methods based on Chernozhukov et al. (2020).

This is a Python port of the [R scinference package](https://github.com/kwuthrich/scinference).

## Installation

```bash
# Install from source
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Conformal Inference

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
Y1[T0:] += 2  # Add treatment effect

# Test null hypothesis theta0=4
result = scinference(Y1, Y0, T1=T1, T0=T0, theta0=4,
                     estimation_method="sc", permutation_method="mb")
print(f"p-value: {result['p_val']:.4f}")

# Compute confidence intervals
result_ci = scinference(Y1, Y0, T1=T1, T0=T0, estimation_method="sc",
                        ci=True, ci_grid=np.arange(-2, 8, 0.1))
print(f"Lower bounds: {result_ci['lb']}")
print(f"Upper bounds: {result_ci['ub']}")
```

### T-test Inference

```python
# With larger T1 for t-test
T1 = 30
result = scinference(Y1, Y0, T1=T1, T0=T0, inference_method="ttest", K=2)
print(f"ATT: {result['att']:.4f}")
print(f"SE: {result['se']:.4f}")
print(f"95% CI: [{result['lb']:.4f}, {result['ub']:.4f}]")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Y1` | array | Required | Outcome data for treated unit (T x 1) |
| `Y0` | array | Required | Outcome data for control units (T x J) |
| `T1` | int | Required | Number of post-treatment periods |
| `T0` | int | Required | Number of pre-treatment periods |
| `inference_method` | str | "conformal" | "conformal" or "ttest" |
| `alpha` | float | 0.1 | Significance level |
| `ci` | bool | False | Whether to compute confidence intervals |
| `theta0` | float/array | 0 | Null hypothesis for treatment effect |
| `estimation_method` | str | "sc" | "did", "sc", or "classo" |
| `permutation_method` | str | "mb" | "mb" (moving block) or "iid" |
| `ci_grid` | array | None | Grid for confidence interval |
| `n_perm` | int | 5000 | Number of permutations (iid method) |
| `K` | int | 2 | Number of cross-fits (t-test) |

## Returns

**Conformal inference:**
- `p_val`: p-value for the null hypothesis
- `lb`: lower bounds of pointwise CIs (if ci=True)
- `ub`: upper bounds of pointwise CIs (if ci=True)

**T-test:**
- `att`: Average Treatment Effect on the Treated
- `se`: Standard error
- `lb`: Lower bound of confidence interval
- `ub`: Upper bound of confidence interval

## Estimation Methods

- **did**: Difference-in-differences
- **sc**: Synthetic control (Abadie et al.)
- **classo**: Constrained lasso (only for conformal)

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=scinference
```

## References

Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2020). An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls. arXiv preprint.

## License

GPL-3.0
