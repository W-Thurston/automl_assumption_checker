# src/core/influence.py
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.regressionplots import influence_plot
from statsmodels.stats.outliers_influence import OLSInfluence

from src.config import (
    COOKS_SEVERITY_THRESHOLDS,
    INFLUENCE_COOKS_THRESHOLD,
    INFLUENCE_DFBETA_THRESHOLD,
    INFLUENCE_LEVERAGE_THRESHOLD,
)
from src.core.registry import register_assumption
from src.core.types import AssumptionResult
from src.models.utils import get_model_wrapper
from src.utils import classify_severity, fig_to_base64


@register_assumption("influence", model_types=["linear"])
def check_influence(X, y, return_plot=False, model_wrapper=None) -> AssumptionResult:
    """
    Check for influential observations using Cook's Distance, Leverage, and DFBETAs.

    Args:
        X (pd.DataFrame): Predictor variables
        y (pd.Series): Response variable
        return_plot (bool, optional): Whether to return visual diagnostics.
            Defaults to False.
        model_wrapper (optional): A prefit model wrapper object.

    Returns:
        AssumptionResult: Structured output containing diagnostics and optional plots.
    """
    if model_wrapper is None:
        model_wrapper = get_model_wrapper("linear", X, y)

    influence: OLSInfluence = model_wrapper.get_influence()

    cooks = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    dfbetas = influence.dfbetas

    max_cook = float(np.max(cooks))
    mean_leverage = float(np.mean(leverage))
    max_leverage = float(np.max(leverage))
    max_dfbeta = float(np.max(np.abs(dfbetas)))

    # Dynamically establish thresholds
    n, p = X.shape
    cooks_thresh = INFLUENCE_COOKS_THRESHOLD(n)
    leverage_thresh = INFLUENCE_LEVERAGE_THRESHOLD(n, p)
    dfbeta_thresh = INFLUENCE_DFBETA_THRESHOLD(n)

    dfbeta_cutoff = dfbeta_thresh
    n_extreme_dfbeta = int(np.sum(np.abs(dfbetas) > dfbeta_cutoff))

    # Define pass/fail: fail if any one exceeds threshold
    passed = (
        max_cook <= cooks_thresh
        and max_leverage <= leverage_thresh
        and max_dfbeta <= dfbeta_cutoff
    )

    # Classify based on max Cook’s Distance
    severity = classify_severity(max_cook, COOKS_SEVERITY_THRESHOLDS)

    recommendation = (
        None
        if passed
        else (
            "Investigate high-leverage or influential observations. "
            "Points with high Cook's Distance, DFBETAs, or "
            "leverage may unduly influence the model. "
            "Consider robust regression methods or removing these points "
            "after further review."
        )
    )

    plots = []
    if return_plot:
        # Cook's Distance vs Observation
        fig1, ax1 = plt.subplots()
        ax1.stem(np.arange(len(cooks)), cooks, markerfmt=",", basefmt=" ")
        ax1.axhline(y=cooks_thresh, color="red", linestyle="--", label="Threshold")
        ax1.set_title("Cook's Distance by Observation")
        ax1.set_xlabel("Observation Index")
        ax1.set_ylabel("Cook's Distance")
        ax1.legend()
        plots.append(
            {
                "title": "Cook's Distance",
                "type": "cooks_distance",
                "image": fig_to_base64(fig1),
            }
        )

        # Influence Plot (Leverage vs Standardized Residuals)
        fig2 = influence_plot(model_wrapper.model, criterion="cooks")
        plots.append(
            {
                "title": "Influence Plot (Leverage vs Residuals)",
                "type": "influence_plot",
                "image": fig_to_base64(fig2),
            }
        )

    return AssumptionResult(
        name="influence",
        passed=passed,
        summary=(
            f"Max Cook's Distance = {max_cook:.4f}" f" → {'Pass' if passed else 'Fail'}"
        ),
        details={
            "max_cooks_distance": max_cook,
            "cooks_distance_threshold": cooks_thresh,
            "max_leverage": max_leverage,
            "leverage_threshold": leverage_thresh,
            "mean_leverage": mean_leverage,
            "max_dfbeta": max_dfbeta,
            "dfbeta_threshold": dfbeta_cutoff,
            "num_large_dfbetas": n_extreme_dfbeta,
        },
        plots=plots,
        flag="warning" if not passed else "info",
        severity=severity,
        recommendation=recommendation,
    )
