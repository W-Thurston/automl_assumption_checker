# tests/test_main.py
import subprocess


def test_main_script_runs_and_prints_expected_output():
    """
    Tests that the CLI entry point script runs and
    prints the expected message.
    """
    result = subprocess.run(
        ["python", "src/main.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Welcome to the AutoML Assumption Checker" in result.stdout
