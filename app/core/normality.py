# app/core/normality.py
"""
Check normality assumption using Q-Q plot, historgram, and
the statistical tests:
    - Shapiro-Wilk test
    - D'Agostino and Pearson's test
    - Anderson-Darling test
"""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm  # Q-Q plot
from scipy.stats import anderson, normaltest, shapiro

from app.config import NORMALITY_PVAL_THRESHOLD, PVAL_SEVERITY_THRESHOLDS
from app.core.registry import register_assumption
from app.core.types import AssumptionResult
from app.utils import build_result, classify_severity, fig_to_base64

__all__ = ["check_normality"]


@register_assumption("normality")
def check_normality(
    X: pd.Series, y: pd.Series, return_plot: bool = False
) -> AssumptionResult:
    """
    Check normality assumption using Q-Q plot, historgram, and
    the statistical tests: Shapiro-Wilk and Normaltest

    Args:
        X (pd.Series): Predictor (1D)
        y (pd.Series): Response (1D)
        return_plot (bool, optional): Whether to return a plot. Defaults to False.

    Returns:
        AssumptionResult: Structured diagnostic output.
    """

    # Assure correct formatting of X values
    X_reshaped = X.values.reshape(-1, 1)
    model = sm.OLS(y, sm.add_constant(X_reshaped)).fit()
    residuals = model.resid
    fitted = model.fittedvalues

    # Shapiro-Wilks test checks if data comes from a normally distributed population
    _, shapiro_pval = shapiro(residuals)
    shapiro_passed = shapiro_pval > NORMALITY_PVAL_THRESHOLD

    # Classify severity of violation based on Shapiro p-value
    shapiro_severity = classify_severity(shapiro_pval, PVAL_SEVERITY_THRESHOLDS)

    # Normal test checks whether a sample differs from a normal distribution
    _, dagostino_pval = normaltest(residuals)
    dagostino_passed = dagostino_pval > NORMALITY_PVAL_THRESHOLD

    # Classify severity of violation based on Normaltest p-value
    dagostino_severity = classify_severity(dagostino_pval, PVAL_SEVERITY_THRESHOLDS)

    # Anderson test checks whether a sample differs from a specified distribution
    anderson_result = anderson(residuals, dist="norm")
    anderson_stat = anderson_result.statistic
    anderson_critical = anderson_result.critical_values[2]  # 5% level
    anderson_passed = anderson_stat < anderson_critical

    # Manually assign severity based on how far we are from the critical value
    # You can define custom thresholds later if needed
    anderson_severity = "low" if anderson_passed else "high"

    # Overall severity based on "worst" of the three
    severity = max(
        [shapiro_severity, dagostino_severity, anderson_severity],
        key=lambda s: ["low", "moderate", "high"].index(s),
    )

    passed = sum([shapiro_passed, dagostino_passed, anderson_passed]) >= 2

    # Recommend next steps if residuals are not from a normal distribution
    recommendation = (
        None if passed else "Consider log-transforming Y or using robust regression."
    )

    # Set flag for UI or prioritization
    flag = "info" if passed else "warning"

    # Plot Q-Q plot and Histogram of residuals if requested
    plots = []
    if return_plot:
        # Q-Q Plot
        fig1 = sm.qqplot(residuals, line="45")
        fig1.suptitle("Q-Q Plot (Normality Check)")
        plots.append(
            {
                "title": "Q-Q Plot",
                "type": "qq",
                "image": fig_to_base64(fig1),
            }
        )

        # Histogram
        fig2, ax = plt.subplots()
        ax.hist(residuals, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_title("Histogram of Residuals")
        plots.append(
            {
                "title": "Histogram",
                "type": "histogram",
                "image": fig_to_base64(fig2),
            }
        )

    overall_str = "Pass (≥ 2 of 3 passed)" if passed else "Fail (≤ 2 of 3 passed)"

    # Package the diagnostic results using the shared builder
    return build_result(
        name="normality",
        passed=passed,
        summary=(
            f"Shapiro-Wilk p = {shapiro_pval:.4f} → "
            f"{'Pass' if shapiro_passed else 'Fail'}, "
            f"D'Agostino p = {dagostino_pval:.4f} → "
            f"{'Pass' if dagostino_passed else 'Fail'}, "
            f"Anderson stat = {anderson_stat:.4f} < (crit = {anderson_critical:.4f}) → "
            f"{'Pass' if anderson_passed else 'Fail'}"
            f" | Overall → {overall_str}"
        ),
        details={
            "shapiro_pval": shapiro_pval,
            "dagostino_pval": dagostino_pval,
            "anderson_stat": anderson_stat,
            "anderson_critical_5pct": anderson_critical,
            "normality_pval_threshold": NORMALITY_PVAL_THRESHOLD,
            "tests_used:": [
                "Shapiro-Wilk (tests overall shape)",
                "D'Agostino-Pearson (tests skew/kurtosis)",
                "Anderson-Darling (emphasizes tails)",
            ],
        },
        residuals=residuals,
        fitted=fitted,
        plots=plots,
        severity=severity,
        recommendation=recommendation,
        flag=flag,
    )
