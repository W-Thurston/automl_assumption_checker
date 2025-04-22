# Contributing to AutoML Assumption Checker

Thank you for taking the time to contribute!

We welcome improvements, bug fixes, assumption modules, and documentation updates.

---

## 🔧 Getting Started

1. **Clone the repo**

```bash
git clone https://github.com/W-Thurston/automl_assumption_checker.git
cd automl_assumption_checker
```

2. Create a virtual environment

```bash
python -m venv automl-env
source automl-env/bin/activate  # or automl-env\Scripts\activate on Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 🧪 Running Tests

```bash
pytest
```

## 🧼 Code Formatting & Style

This project uses: - black for formatting - flake8 for linting - isort for import sorting

All of this is handled via pre-commit. To install:

```bash
pre-commit install
pre-commit run --all-files
```

These hooks will automatically run before each commit. CI will fail if code style isn’t followed.

## ✅ Submitting a Pull Request

    - Make your changes on a `feature/*` branch.
    - Write or update tests.
    - Ensure all pre-commit checks and CI pass.
    - Submit a pull request using the [PR template](https://github.com/W-Thurston/automl_assumption_checker/blob/main/.github/pull_request_template.md) to `dev`.
    - Link to related issues if applicable.

## 🙏 Thanks

## Every contribution — big or small — helps improve this project.
