## [0.2.0] - 2025-04-20 [🔗](https://github.com/W-Thurston/automl_assumption_checker/releases/tag/v0.2.0)

### Added

- Introduced `ModelWrapper` abstraction layer:
  - Base interface `BaseModelWrapper`
  - `LinearModelWrapper` implementation for `statsmodels.OLS`
  - Shared `get_model_wrapper()` factory in `models/utils.py`
- Shared model now fit once and reused across all assumption checks
- `--model-type` CLI flag in `report.py` (currently supports `"linear"`)
- Console report now prints model metadata using `model_wrapper.summary()`
- Assumption check registry (`register_assumption`) extended to support model filtering via `model_types`
- Centralized `run_all_checks()` returns both results and model instance
- Updated `linearity.py` to use shared model wrapper for residuals and fitted values
- Auto-skips linearity check with multivariate input but shows informative output panel

### Changed

- Function parameter order across assumption checks standardized:
  - Optional arguments like `model_wrapper` now follow `return_plot` for readability
- Assumption checks now accept `model_wrapper` as a shared input
- Dispatcher updated to enforce model compatibility filtering by `model_type`

### Fixed

- CLI crash when `model_type` was not explicitly passed
- Alignment inconsistencies in verbose mode console report

### Upcoming

- Refactor `normality`, `homoscedasticity`, and `multicollinearity` to use `model_wrapper`
- Add new assumptions: `influence`, `outliers`, `independence`, `missingness`
- Enable diagnostics for multivariate inputs
- Full v1.0.0 milestone to mark linear regression support completion

## [0.1.0] - 2025-04-19 [🔗](https://github.com/W-Thurston/automl_assumption_checker/releases/tag/v0.1.0)

### Added

- Three core assumption modules:
  - Linearity (`R²` with threshold and visual)
  - Homoscedasticity (Breusch-Pagan test)
  - Normality (Shapiro, D’Agostino, Anderson)
- `report.py` with:
  - Rich console summary (color-coded, threshold-aware)
  - Markdown and JSON export support
  - Verbose mode with diagnostic detail alignment
- Unified `AssumptionResult` dataclass for structured output
- Registry-based dispatcher system
- Plot support using base64-encoded figures
- Simulated data generators for testing + diagnostics

### Changed

- Upgraded Python to 3.12.10
- Normality test now uses majority rule logic for combined verdicts

### Fixed

- Docker DNS resolution on WSL2
- CLI pre-commit conflict between `black` and `isort`

### Upcoming

- Planned `multicollinearity.py` assumption checker
- Planned `independence.py` assumption checker
- Planned `influence.py` assumption checker
- Planned `outliers.py` assumption checker
- Planned `missingness.py` assumption checker
- Streamlit-based visual report interface
- CLI wrapper with CSV input support
