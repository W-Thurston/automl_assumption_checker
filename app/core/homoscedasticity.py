# app/core/homoscedasticity.py
"""
Check homoscedasticity assumption using Breusch-Pagan test
    and residuals vs fitted plot.
"""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from app.config import HOMOSCEDASTICITY_PVAL_THRESHOLD, PVAL_SEVERITY_THRESHOLDS
from app.core.registry import register_assumption
from app.core.types import AssumptionResult
from app.utils import build_result, classify_severity, fig_to_base64

__all__ = ["check_homoscedasticity"]


@register_assumption("homoscedasticity")
def check_homoscedasticity(
    X: pd.Series, y: pd.Series, return_plot: bool = False
) -> AssumptionResult:
    """
    Check for homoscedasticity using residual vs fitted plot and Breusch-Pagan test.

    Args:
        X (pd.Series): Predictor (1D)
        y (pd.Series): Response (1D)
        return_plot (bool, optional): Whether to return a plot. Defaults to False.

    Returns:
        AssumptionResult: Structured diagnostic output.
    """

    # Fit OLS model using statsmodels
    X_reshaped = X.values.reshape(-1, 1)
    model = sm.OLS(y, sm.add_constant(X_reshaped)).fit()
    residuals = model.resid
    fitted = model.fittedvalues

    # Breusch-Pagan test checks for non-constant residual variance
    _, pval, _, _ = het_breuschpagan(residuals, sm.add_constant(X_reshaped))
    passed = pval > HOMOSCEDASTICITY_PVAL_THRESHOLD

    # Classify severity of violation based on p-value
    severity = classify_severity(pval, PVAL_SEVERITY_THRESHOLDS)

    # Recommend next steps if residuals are heteroskedastic
    recommendation = (
        None
        if passed
        else (
            "Consider using weighted least squares or "
            "transforming your response variable."
        )
    )

    # Set flag for UI or prioritization
    flag = "info" if passed else "warning"

    # Plot residuals vs fitted values if requested
    encoded = None
    if return_plot:
        fig, ax = plt.subplots()
        ax.scatter(fitted, residuals, alpha=0.7)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted (Homoscedasticity Check)")
        encoded = fig_to_base64(fig)

    # Package the diagnostic results using the shared builder
    return build_result(
        name="homoscedasticity",
        passed=passed,
        summary=f"Breusch-Pagan p = {pval:.4f} → {'Pass' if passed else 'Fail'}",
        details={
            "breusch_pagan_pval": pval,
            "homoscedasticity_pval_threshold": HOMOSCEDASTICITY_PVAL_THRESHOLD,
        },
        residuals=residuals,
        fitted=fitted,
        plot_base64=encoded,
        severity=severity,
        recommendation=recommendation,
        flag=flag,
    )
