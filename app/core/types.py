from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AssumptionResult:
    name: str
    passed: bool
    summary: str
    details: dict
    residuals: Optional[np.ndarray] = None
    fitted: Optional[np.ndarray] = None
    plot_base64: Optional[str] = None
