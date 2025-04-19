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
    summary: str
    details: dict
    residuals: Optional[np.ndarray] = None
    fitted: Optional[np.ndarray] = None
    plot_base64: Optional[str] = None
