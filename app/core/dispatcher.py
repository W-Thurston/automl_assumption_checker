# app/core/dispatcher.py
from typing import Dict, Tuple

import pandas as pd

from app.core import homoscedasticity  # noqa: F401
from app.core import linearity  # noqa: F401
from app.core import multicollinearity  # noqa: F401
from app.core import normality  # noqa: F401
from app.core.registry import ASSUMPTION_CHECKS
from app.core.types import AssumptionResult
from app.models.base_model_wrapper import BaseModelWrapper
from app.models.utils import get_model_wrapper

__all__ = ["check_assumption", "run_all_checks"]


def check_assumption(
    name: str, X: pd.Series, y: pd.Series, return_plot: bool = False
) -> AssumptionResult:
    """
    Run the specified assumption check by name.

    Args:
        name (str): assumption name
        X (pd.Series or pd.DataFrame): Predictor values (1D or multivariate)
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

    if isinstance(X, pd.Series):
        X = X.to_frame()

    return ASSUMPTION_CHECKS[name](X, y, return_plot)


def run_all_checks(
    X: pd.Series, y: pd.Series, model_type=None, return_plot: bool = False
) -> Tuple[Dict[str, AssumptionResult], BaseModelWrapper]:
    """
    Run all registered assumption checks and return a dictionary of results.

    Args:
        X (pd.Series or pd.DataFrame): Predictor values (1D or multivariate)
        y (pd.Series): Response (1D)
        return_plot (bool, optional): Whether to return base64-encoded
            PNG of the plot. Defaults to False.

    Returns:
        Dict[str, AssumptionResult]: A dictionary of assumption names
            mapped to their result objects.
    """
    results = {}

    if isinstance(X, pd.Series):
        X = X.to_frame()

    model_wrapper = get_model_wrapper(model_type, X, y)

    for name, func in ASSUMPTION_CHECKS.items():
        if model_type not in getattr(func, "_model_types", ["linear"]):
            continue
        results[name] = func(X, y, model_wrapper=model_wrapper, return_plot=return_plot)
    return results, model_wrapper
