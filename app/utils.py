# app/utils.py
import base64
import io
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from app.core.types import AssumptionResult

__all__ = ["fig_to_base64", "build_result", "classify_severity"]


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_result(
    name: str,
    passed: bool,
    summary: str,
    details: Dict[str, Any],
    residuals: Optional[np.ndarray] = None,
    fitted: Optional[np.ndarray] = None,
    plot_base64: Optional[str] = None,
    plots: Optional[List[Dict[str, str]]] = None,
    severity: Optional[str] = None,
    recommendation: Optional[str] = None,
    flag: Optional[str] = None,
) -> AssumptionResult:
    """
    Helper to construct a standardized AssumptionResult object.

    Args:
        name (str): Name of the assumption.
        passed (bool): Whether the assumption check passed.
        summary (str): Short summary of result.
        details (Dict[str, Any]): Test statistics, p-values, etc.
        residuals (Optional[np.ndarray], optional): Residuals from model.
            Defaults to None.
        fitted (Optional[np.ndarray], optional): Fitted values from model.
            Defaults to None.
        plot_base64 (Optional[str], optional): Base64 image string for primary plot.
            Defaults to None.
        plots (Optional[List[Dict[str, str]]], optional): Additional plots with titles
            and image data. Defaults to None.
        severity (Optional[str], optional): Severity level ('low', 'moderate', 'high').
            Defaults to None.
        recommendation (Optional[str], optional): Suggested next step if the check fails
            Defaults to None.
        flag (Optional[str], optional): Status flag (e.g. 'info', 'warning', 'critical')
            Defaults to None.

    Returns:
        AssumptionResult: Complete diagnostic output
    """
    return AssumptionResult(
        name=name,
        passed=passed,
        summary=summary,
        details=details,
        residuals=residuals,
        fitted=fitted,
        plot_base64=plot_base64,
        plots=plots,
        severity=severity,
        recommendation=recommendation,
        flag=flag,
    )


def classify_severity(value: float, thresholds: dict) -> str:
    """
    Classify a numeric value into 'high', 'moderate', or 'low' based on thresholds.

    Args:
        value (float): The numeric metric (e.g., R², p-value, VIF).
        thresholds (dict): Dict with keys 'high', 'moderate', 'low'.

    Returns:
        str: One of 'high', 'moderate', or 'low'
    """
    if value >= thresholds["high"]:
        return "high"
    elif value >= thresholds["moderate"]:
        return "moderate"
    else:
        return "low"
