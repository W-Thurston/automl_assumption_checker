import numpy as np
import pandas as pd


def generate_linear_data(
    n_samples: int = 100, noise_std: float = 1.0, seed: int | list = None
) -> pd.DataFrame:
    """Generate simple linear data: y = 3x + noise"""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=n_samples)
    noise = rng.normal(0, noise_std, size=n_samples)
    y = 3 * X + noise
    return pd.DataFrame({"x": X, "y": y})


def generate_heteroskedastic_data(
    n_samples: int = 100, seed: int | list = None
) -> pd.DataFrame:
    """Variance of error increases with X (classic heteroskedasticity)"""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=n_samples)
    noise = rng.normal(0, 0.5 + 0.5 * np.abs(X), size=n_samples)
    y = 3 * X + noise
    return pd.DataFrame({"x": X, "y": y})


def generate_multicollinear_data(
    n_samples: int = 100, seed: int | list = None
) -> pd.DataFrame:
    """Two highly correlated predictors (collinear)"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n_samples)
    x2 = x1 + rng.normal(0, 0.01, size=n_samples)  # x2 ≈ x1
    noise = rng.normal(0, 1, size=n_samples)
    y = 2 * x1 + 3 * x2 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def generate_nonlinear_data(
    n_samples: int = 100, seed: int | list = None
) -> pd.DataFrame:
    """y = sin(x) + noise -- violates linearity"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=n_samples)
    noise = rng.normal(0, 0.3, size=n_samples)
    y = np.sin(X) + noise
    return pd.DataFrame({"x": X, "y": y})


def list_simulations() -> dict:
    return {
        "linear": generate_linear_data,
        "heteroskedastic": generate_heteroskedastic_data,
        "multicollinear": generate_multicollinear_data,
        "nonlinear": generate_nonlinear_data,
    }
