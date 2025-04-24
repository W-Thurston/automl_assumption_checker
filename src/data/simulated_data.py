# app/data/simulated_data.py
import numpy as np
import pandas as pd

__all__ = [
    "generate_linear_data",
    "generate_heteroscedastic_data",
    "generate_multicollinear_data",
    "generate_nonlinear_data",
    "generate_skewed_data",
    "list_simulations",
]


def generate_linear_data(
    n_samples: int = 100, noise_std: float = 1.0, seed: int = None
) -> pd.DataFrame:
    """
    Generate simple linear data: y = 3x + noise

    Args:
        n_samples (int, optional): Number of observations to generate. Defaults to 100.
        noise_std (float, optional): Standard deviation of the noise. Defaults to 1.0.
        seed (int, optional): Randomness seed. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns 'x' and 'y'
            where y = 3x + noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=n_samples)
    noise = rng.normal(0, noise_std, size=n_samples)
    y = 3 * X + noise
    return pd.DataFrame({"x": X, "y": y})


def generate_heteroscedastic_data(
    n_samples: int = 100, seed: int = None
) -> pd.DataFrame:
    """
    Generate data where residual variance increases with X (heteroscedasticity).

    Args:
        n_samples (int, optional): Number of observations to generate. Defaults to 100.
        seed (int, optional): Randomness seed. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns 'x' and 'y'
            where y = 3x + noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=n_samples)
    noise = rng.normal(0, 0.5 + 0.5 * np.abs(X), size=n_samples)
    y = 3 * X + noise
    return pd.DataFrame({"x": X, "y": y})


def generate_multicollinear_data(
    n_samples: int = 100, seed: int = None
) -> pd.DataFrame:
    """
    Generate data with two highly correlated predictors (multicollinearity).

    Args:
        n_samples (int, optional): Number of observations to generate. Defaults to 100.
        seed (int, optional): Randomness seed. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns 'x1', 'x2', and 'y'
            where y = 2 * x1 + 3 * x2 + noise.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n_samples)
    x2 = x1 + rng.normal(0, 0.01, size=n_samples)  # x2 ≈ x1
    noise = rng.normal(0, 1, size=n_samples)
    y = 2 * x1 + 3 * x2 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def generate_nonlinear_data(n_samples: int = 100, seed: int = None) -> pd.DataFrame:
    """
    Generate nonlinear data using y = sin(x) + noise.

    Args:
        n_samples (int, optional): Number of observations to generate. Defaults to 100.
        seed (int, optional): Randomness seed. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns 'x' and 'y'
            where y = np.sin(X) + noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=n_samples)
    noise = rng.normal(0, 0.3, size=n_samples)
    y = np.sin(X) + noise
    return pd.DataFrame({"x": X, "y": y})


def generate_skewed_data(n_samples: int = 100, seed: int = None) -> pd.DataFrame:
    """
    Generate linear-looking data with clearly non-normal
     residuals using an exponential noise component.

    Args:
        n_samples (int, optional): Number of observations to generate. Defaults to 100.
        seed (int, optional): Randomness seed. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns 'x' and 'y'.
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.random.normal(loc=0, scale=1, size=n_samples)
    # Exponential errors are positively skewed → residuals will be non-normal
    error = np.random.exponential(scale=1.0, size=n_samples)
    y = 2 * x + error
    return pd.DataFrame({"x": x, "y": y})


def generate_autocorrelated_data(
    n_samples: int = 100, seed: int = None
) -> pd.DataFrame:
    """
    Generate linear data with autocorrelated residual structure.

    Simulates a simple trend with a sinusoidal component added to induce
    autocorrelation in the residuals — commonly used to test independence
    violations (e.g., time series drift).

    Args:
        n_samples (int, optional): Number of observations to generate. Defaults to 100.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns 'x' (sequential) and 'y' (autocorrelated)
    """
    np.random.seed(seed)
    x = np.arange(n_samples)
    y = x + np.sin(np.linspace(0, 10 * np.pi, n_samples))
    return pd.DataFrame({"x": x, "y": y})


def generate_passing_influence_data(
    n_samples: int = 150, seed: int = 42
) -> pd.DataFrame:
    """
    Generate a clean linear dataset unlikely to trigger influence check failures.
    Characteristics:
    - Moderately sized sample (to soften thresholds)
    - No extreme leverage or outliers
    - Centered, normally distributed predictors
    """
    np.random.seed(seed)
    x = np.random.normal(0, 0.6, n_samples)  # low variance
    y = 2.5 * x + np.random.normal(0, 0.25, n_samples)  # low residual noise

    return pd.DataFrame({"x": x, "y": y})


def list_simulations() -> dict:
    """
    Return a dictionary of available simulated data generators.

    Returns:
        dict: Keys are simulation names (e.g., 'linear', 'nonlinear') and
            values are callable generator functions.
    """
    return {
        "linear": generate_linear_data,
        "heteroscedastic": generate_heteroscedastic_data,
        "multicollinear": generate_multicollinear_data,
        "nonlinear": generate_nonlinear_data,
        "skewed": generate_skewed_data,
        "autocorrelated": generate_autocorrelated_data,
        "passing_influence": generate_passing_influence_data,
    }
