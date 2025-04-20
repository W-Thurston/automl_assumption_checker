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

- Add `multicollinearity.py` assumption checker
- Add `independence.py` assumption checker
- Add `influence.py` assumption checker
- Add `outliers.py` assumption checker
- Add `missingness.py` assumption checker
- Streamlit-based visual report interface
- CLI wrapper with CSV input support
