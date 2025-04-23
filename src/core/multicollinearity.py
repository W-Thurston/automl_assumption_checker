# src/core/multicollinearity.py
"""
Check multicollinearity assumption using:
    - Plots:
        - heatmap of correlation matrix
    - Statistical tests:
        - Variance Inflation Factor (VIF)
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import VIF_SEVERITY_THRESHOLDS, VIF_THRESHOLD
from src.core.registry import register_assumption
from src.core.types import AssumptionResult
from src.utils import build_result, classify_severity, fig_to_base64

__all__ = ["check_multicollinearity"]


@register_assumption("multicollinearity", model_types=["linear"])
def check_multicollinearity(
    X: pd.DataFrame, y: pd.Series, return_plot: bool = False, model_wrapper=None
) -> AssumptionResult:
    """
    Check multicollinearity assumption using:
    - Plots:
        - heatmap of correlation matrix
    - Statistical tests:
        - Variance Inflation Factor (VIF)

    Args:
        X (pd.DataFrame): Predictor or Feature values (n, p≥2)
        return_plot (bool, optional): Whether to return a plot. Defaults to False.

    Returns:
        AssumptionResult: Structured diagnostic output.
    """
    # Skip multicollinearity check if features is less than 2
    if X.shape[1] < 2:
        return build_result(
            name="multicollinearity",
            passed=True,
            summary="Only one predictor — multicollinearity not applicable.",
            details={
                "note": "Multicollinearity requires at least two predictor variables."
            },
            plot_base64=None,
            severity="low",
            recommendation=None,
            flag="info",
        )

    # Calculate VIF for each independent variable
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]

    # Use config threshold to determine pass/fail for each feature
    vif_data["passed"] = vif_data["VIF"].apply(lambda x: 1 if x <= VIF_THRESHOLD else 0)

    # Use centralized classifier to determine diagnostic severity
    vif_data["severity"] = vif_data["VIF"].apply(
        lambda x: classify_severity(x, VIF_SEVERITY_THRESHOLDS)
    )

    # Overall severity based on "worst" of the three
    severity = max(
        vif_data["severity"].values,
        key=lambda s: ["low", "moderate", "high"].index(s),
    )

    # Check if all VIF values are below 5
    passed = all(vif_data["VIF"] < 5)

    # Recommend next steps if features are correlated
    recommendation = (
        None
        if passed
        else (
            "Consider removing one of the correlated features"
            + " or combining them into a single feature"
        )
    )

    # Set flag for UI or prioritization
    flag = "info" if passed else "warning"

    # Flattened details per feature for aligned rendering in report.py
    feature_detail_rows = {
        f"{row['feature']} (VIF)": row["VIF"]
        for row in vif_data.to_dict(orient="records")
    }
    threshold_detail_rows = {
        f"{row['feature']} threshold": VIF_THRESHOLD
        for row in vif_data.to_dict(orient="records")
    }

    details = {
        **feature_detail_rows,
        **threshold_detail_rows,
        "max_variance_inflation_factor": vif_data["VIF"].max(),
        "multicollinearity_vif_threshold": VIF_THRESHOLD,
    }

    # Plot heatmap of correlation matrix
    encoded = None
    if return_plot:
        fig, ax = plt.subplots()
        corr = X.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation of feature values")
        encoded = fig_to_base64(fig)

    # Package the diagnostic results using the shared builder
    return build_result(
        name="multicollinearity",
        passed=passed,
        summary=(
            f"Max VIF among predictors = {vif_data.loc[:, 'VIF'].max():.2f} → "
            f"{'Pass' if passed else 'Fail'}"
        ),
        details=details,
        plot_base64=encoded,
        severity=severity,
        recommendation=recommendation,
        flag=flag,
    )
