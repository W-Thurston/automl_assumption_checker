# app/core/dispatcher.py
from typing import Callable, Dict

import pandas as pd

from app.core.homoscedasticity import check_homoscedasticity
from app.core.linearity import check_linearity
from app.core.types import AssumptionResult

__all__ = ["check_assumption", "run_all_checks"]

# Define the registry mapping assumption names to functions
ASSUMPTION_CHECKS: Dict[
    str, Callable[[pd.Series, pd.Series, bool], AssumptionResult]
] = {
    "linearity": check_linearity,
    "homoscedasticity": check_homoscedasticity,
    # Add more as you implement them
}


def check_assumption(
    name: str, X: pd.Series, y: pd.Series, return_plot: bool = False
) -> AssumptionResult:
    """
    Run the specified assumption check by name.

    Args:
        name (str): assumption name
        X (pd.Series): Predictor (1D)
        y (pd.Series): Response (1D)
        return_plot (bool, optional): Whether to return base64-encoded
            PNG of the plot. Defaults to False.

    Returns:
        AssumptionResult: An object containing the outcome of the
            assumption check, including whether the check passed,
            a summary message, diagnostic details, optional residuals
            and fitted values, and an optional base64-encoded plot.
    """
    if name not in ASSUMPTION_CHECKS:
        raise ValueError(f"Unknown assumption: '{name}'")

    return ASSUMPTION_CHECKS[name](X, y, return_plot)


def run_all_checks(
    X: pd.Series, y: pd.Series, return_plot: bool = False
) -> Dict[str, AssumptionResult]:
    """
    Run all registered assumption checks and return a dictionary of results.

    Args:
        X (pd.Series): Predictor (1D)
        y (pd.Series): Response (1D)
        return_plot (bool, optional): Whether to return base64-encoded
            PNG of the plot. Defaults to False.

    Returns:
        Dict[str, AssumptionResult]: A dictionary of assumption names
            mapped to their result objects.
    """
    results = {}
    for name, func in ASSUMPTION_CHECKS.items():
        results[name] = func(X, y, return_plot)
    return results
