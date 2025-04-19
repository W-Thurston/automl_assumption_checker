import base64
import io

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def check_linearity(X: pd.Series, y: pd.Series, return_plot: bool = False) -> dict:
    """
    Perform a linearity check using residuals vs fitted plot and R².

    Args:
        X (pd.Series): Predictor (1D)
        y (pd.Series): Response (1D)
        return_plot (bool, optional): Whether to return base64-encoded
            PNG of the plot. Defaults to False.

    Returns:
        dict: R², residuals, optional plot, and a basic pass/fail flag
    """
    X_reshaped = X.values.reshape(-1, 1)
    model = LinearRegression().fit(X_reshaped, y)
    y_pred = model.predict(X_reshaped)
    residuals = y - y_pred
    r2 = r2_score(y, y_pred)

    result = {
        "r_squared": r2,
        "residuals": residuals,
        "fitted": y_pred,
        "pass": r2 > 0.7,  # basic heuristic for now
    }

    if return_plot:
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, alpha=0.7)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        result["plot_base64"] = encoded

    return result
