# Thresholds for pass/fail logic
LINEARITY_R2_THRESHOLD = 0.7
HOMOSCEDASTICITY_PVAL_THRESHOLD = 0.05
NORMALITY_PVAL_THRESHOLD = 0.05
VIF_THRESHOLD = 5

# Thresholds for diagnostic severity (optional, used for display or flagging)
R2_SEVERITY_THRESHOLDS = {"high": 0.9, "moderate": 0.7, "low": 0.5}
PVAL_SEVERITY_THRESHOLDS = {"high": 0.01, "moderate": 0.05, "low": 0.1}
VIF_SEVERITY_THRESHOLDS = {"high": 10, "moderate": 5, "low": 0}
