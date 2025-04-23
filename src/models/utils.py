# src/models/utils.py
from src.models.base_model_wrapper import BaseModelWrapper
from src.models.linear_model_wrapper import LinearModelWrapper


def get_model_wrapper(model_type: str, X, y) -> BaseModelWrapper:
    if model_type == "linear":
        return LinearModelWrapper(X, y).fit()
    elif model_type == "PLACEHOLDER":
        ...
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
