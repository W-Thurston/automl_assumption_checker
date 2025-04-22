# app/core/registry.py
from typing import Callable, Dict

import pandas as pd

from app.core.types import AssumptionResult

__all__ = ["ASSUMPTION_CHECKS", "register_assumption"]

ASSUMPTION_CHECKS: Dict[
    str, Callable[[pd.Series, pd.Series, bool], AssumptionResult]
] = {}

AssumptionCheck = Callable[[pd.Series, pd.Series, bool], AssumptionResult]


def register_assumption(
    name: str, model_types: list = ["linear"]
) -> Callable[[AssumptionCheck], AssumptionCheck]:
    """
    Decorator to register an assumption check function under a given name.

    Args:
        name (str): Assumption check name.

    Returns:
        Callable: A decorator that registers the function and returns it unchanged.
    """

    def decorator(func: AssumptionCheck) -> AssumptionCheck:
        func._assumption_name = name
        func._model_types = model_types
        ASSUMPTION_CHECKS[name] = func
        return func

    return decorator
