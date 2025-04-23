# src/core/linearity.py
"""
Check linearity assumption using:
    - Plots:
        - Residuals vs fitted plot
    - Statistical tests:
        - R²
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

from src.config import LINEARITY_R2_THRESHOLD, R2_SEVERITY_THRESHOLDS
from src.core.registry import register_assumption
from src.core.types import AssumptionResult
from src.utils import build_result, classify_severity, fig_to_base64

__all__ = ["check_linearity"]


@register_assumption("linearity", model_types=["linear"])
def check_linearity(
    X: pd.Series, y: pd.Series, return_plot: bool = False, model_wrapper=None
) -> AssumptionResult:
    """
    Check linearity assumption using:
    - Plots:
        - Residuals vs fitted plot
    - Statistical tests:
        - R²

    Args:
        X (pd.Series): Predictor (1D)
        y (pd.Series): Response (1D)
        return_plot (bool, optional): Whether to return base64-encoded
            PNG of the plot. Defaults to False.

    Returns:
        AssumptionResult: Structured diagnostic output.
    """
    if isinstance(X, pd.DataFrame):
        if X.shape[1] > 1:
            return build_result(
                name="linearity",
                passed=True,
                summary="Linearity check not run: only supports one predictor.",
                details={
                    "note": (
                        "Linearity check skipped — "
                        + "only valid for single predictor inputs."
                    )
                },
                plot_base64=None,
                severity="low",
                recommendation=None,
                flag="info",
            )
        X = X.iloc[:, 0]  # Convert to Series

    # Guard for if model_wrapper is None
    if model_wrapper is None:
        from src.models.utils import get_model_wrapper

        model_wrapper = get_model_wrapper("linear", X, y)

    # Fit simple linear model to input data
    residuals = model_wrapper.residuals()
    y_pred = model_wrapper.fitted()

    # Coefficient of determination (R²) measures goodness of fit
    r2 = r2_score(y, y_pred)

    # Use config threshold to determine pass/fail status
    passed = r2 > LINEARITY_R2_THRESHOLD

    # Use centralized classifier to determine diagnostic severity
    severity = classify_severity(r2, R2_SEVERITY_THRESHOLDS)

    # Suggest transformation or feature engineering if linearity is poor
    recommendation = (
        None
        if passed
        else "Consider transforming your features or engineering new ones."
    )

    # Used for visual/UI indication - can help prioritize failed assumptions
    flag = "info" if passed else "warning"

    # Generate residual vs fitted plot if requested
    encoded = None
    if return_plot:
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, alpha=0.7)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted (Linearity Check)")
        encoded = fig_to_base64(fig)

    # Package the diagnostic results using the shared builder
    return build_result(
        name="linearity",
        passed=passed,
        summary=f"R² = {r2:.2f} → {'Pass' if passed else 'Fail'}",
        details={"r_squared": r2, "r2_threshold": LINEARITY_R2_THRESHOLD},
        residuals=residuals,
        fitted=y_pred,
        plot_base64=encoded,
        severity=severity,
        recommendation=recommendation,
        flag=flag,
    )
