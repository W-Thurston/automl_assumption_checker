# tests/test_model_utils.py
import numpy as np
import pandas as pd
import pytest

from src.models.linear_model_wrapper import LinearModelWrapper
from src.models.utils import get_model_wrapper


def test_get_model_wrapper_linear():
    """
    Verify get_model_wrapper returns a LinearModelWrapper for 'linear' input.
    """
    X = pd.DataFrame({"x1": np.random.randn(30)})
    y = 2 * X["x1"] + np.random.randn(30)

    wrapper = get_model_wrapper("linear", X, y)
    assert isinstance(wrapper, LinearModelWrapper)
    assert hasattr(wrapper, "predict")
    assert hasattr(wrapper, "residuals")


def test_get_model_wrapper_invalid_type():
    """
    Confirm get_model_wrapper raises ValueError for unknown model_type.
    """
    X = pd.DataFrame({"x1": np.random.randn(30)})
    y = 2 * X["x1"] + np.random.randn(30)

    with pytest.raises(ValueError, match="Unsupported model type"):
        get_model_wrapper("invalid_type", X, y)
