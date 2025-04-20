# app/core/types.py
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["AssumptionResult"]


@dataclass
class AssumptionResult:
    """Represents the result of a single statistical assumption check."""

    name: str
    passed: bool
    summary: str  # One-liner for report
    details: dict  # Raw test stats, R², VIF, etc.
    residuals: Optional[np.ndarray] = None
    fitted: Optional[np.ndarray] = None
    plot_base64: Optional[str] = None  # For simple assumptions
    plots: Optional[list[dict]] = None  # For rich visual outputs
    severity: Optional[str] = None  # "low", "moderate", "high"
    recommendation: Optional[str] = None  # e.g., "Try log-transforming Y"
    flag: Optional[str] = None  # "warning", "critical", "info"
