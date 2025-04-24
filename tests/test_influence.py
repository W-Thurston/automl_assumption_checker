# tests/test_influence.py

from src.core.influence import check_influence
from src.data.simulated_data import generate_passing_influence_data
from src.models.linear_model_wrapper import LinearModelWrapper


def test_influence_passes_on_clean_data():
    df = generate_passing_influence_data(n_samples=250, seed=42)
    X = df[["x"]]
    y = df["y"]

    wrapper = LinearModelWrapper(X, y).fit()
    result = check_influence(X, y, model_wrapper=wrapper)

    assert (
        result.details["max_cooks_distance"]
        <= result.details["cooks_distance_threshold"]
    )


def test_influence_plot_returns():
    df = generate_passing_influence_data(n_samples=150, seed=123)
    X = df[["x"]]
    y = df["y"]

    result = check_influence(X, y, return_plot=True)
    assert isinstance(result.plots, list)
    assert any(
        "cooks" in p["type"] or "influence" in p["title"].lower() for p in result.plots
    )
