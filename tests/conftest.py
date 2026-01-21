"""
Pytest configuration and shared fixtures for scinference tests.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def conformal_test_data(data_dir):
    """Load test data for conformal inference tests."""
    Y0 = pd.read_csv(data_dir / "Y0_test.csv").values
    Y1 = pd.read_csv(data_dir / "Y1_test.csv")["Y1"].values
    return {"Y0": Y0, "Y1": Y1, "T0": 50, "T1": 5}


@pytest.fixture
def ttest_test_data(data_dir):
    """Load test data for t-test tests."""
    Y0 = pd.read_csv(data_dir / "Y0_ttest.csv").values
    Y1 = pd.read_csv(data_dir / "Y1_ttest.csv")["Y1"].values
    return {"Y0": Y0, "Y1": Y1, "T0": 30, "T1": 30}


@pytest.fixture
def simple_data():
    """Generate simple synthetic data for basic tests."""
    np.random.seed(12345)
    J = 50
    T0 = 50
    T1 = 5
    T = T0 + T1

    # Generate control outcomes
    Y0 = np.random.randn(T, J)

    # Generate treated outcomes as weighted average of controls + noise + treatment
    w = np.zeros(J)
    w[:3] = 1 / 3
    Y1 = Y0 @ w + np.random.randn(T)
    Y1[T0:] += 2  # Add treatment effect of 2

    return {"Y0": Y0, "Y1": Y1, "T0": T0, "T1": T1, "true_effect": 2}


@pytest.fixture
def large_t1_data():
    """Generate data with large T1 for t-test tests."""
    np.random.seed(42)
    J = 50
    T0 = 30
    T1 = 30
    T = T0 + T1

    Y0 = np.random.randn(T, J)
    w = np.zeros(J)
    w[:3] = 1 / 3
    Y1 = Y0 @ w + np.random.randn(T)
    Y1[T0:] += 1.5  # Add treatment effect

    return {"Y0": Y0, "Y1": Y1, "T0": T0, "T1": T1}
