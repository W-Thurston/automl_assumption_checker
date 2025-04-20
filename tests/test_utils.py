from app.utils import build_result


def test_build_result_returns_valid_assumption_result():
    result = build_result(
        name="test_assumption",
        passed=True,
        summary="This is a test.",
        details={"mock_stat": 0.99},
        residuals=None,
        fitted=None,
        plot_base64="fake_base64",
        plots=[{"title": "Sample Plot", "image": "base64"}],
        severity="low",
        recommendation="Consider doing something.",
        flag="info",
    )
    assert result.name == "test_assumption"
    assert result.passed is True
    assert result.summary == "This is a test."
    assert result.details["mock_stat"] == 0.99
    assert result.plots[0]["title"] == "Sample Plot"
