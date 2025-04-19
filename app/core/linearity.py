# app/core/linearity.py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from app.core.types import AssumptionResult
from app.utils import fig_to_base64


def check_linearity(
    X: pd.Series, y: pd.Series, return_plot: bool = False
) -> AssumptionResult:
    """
    Perform a linearity check using residuals vs fitted plot and R².

    Args:
        X (pd.Series): Predictor (1D)
        y (pd.Series): Response (1D)
        return_plot (bool, optional): Whether to return base64-encoded
            PNG of the plot. Defaults to False.

    Returns:
        AssumptionResult
    """
    X_reshaped = X.values.reshape(-1, 1)
    model = LinearRegression().fit(X_reshaped, y)
    y_pred = model.predict(X_reshaped)
    residuals = y - y_pred
    r2 = r2_score(y, y_pred)

    if return_plot:
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, alpha=0.7)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")
        encoded = fig_to_base64(fig)

    return AssumptionResult(
        name="linearity",
        passed=r2 > 0.7,
        summary=f"R² = {r2:.2f} → {'Pass' if r2 > 0.7 else 'Fail'}",
        details={"r_squared": r2},
        residuals=residuals,
        fitted=y_pred,
        plot_base64=encoded if return_plot else None,
    )
