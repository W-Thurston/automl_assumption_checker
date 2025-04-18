import subprocess


def test_main_script_runs_and_prints_expected_output():
    result = subprocess.run(
        ["python", "app/main.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Welcome to the AutoML Assumption Checker" in result.stdout
