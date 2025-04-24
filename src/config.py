# src/config.py
import numpy as np

###
# Thresholds for pass/fail logic
###

# Linearity R_squared
LINEARITY_R2_THRESHOLD = 0.7

# Homoscedasticity p-val
HOMOSCEDASTICITY_PVAL_THRESHOLD = 0.05

# normality p-val
NORMALITY_PVAL_THRESHOLD = 0.05

# multicollinearity Variance Inflation Factor
VIF_THRESHOLD = 5

# Independence Durbin Watson
INDEPENDENCE_DW_THRESHOLDS = (1.5, 2.5)


def influence_cooks_threshold(n):
    # min(4 / n, 0.1)  # standard rule of thumb
    return 0.1


def influence_leverage_threshold(n, p):
    # p = number of predictors
    return 2 * p / n


def influence_dfbeta_threshold(n):
    return 2 / np.sqrt(n)


# Influence thresholds (dynamic)
INFLUENCE_COOKS_THRESHOLD = influence_cooks_threshold
INFLUENCE_LEVERAGE_THRESHOLD = influence_leverage_threshold
INFLUENCE_DFBETA_THRESHOLD = influence_dfbeta_threshold

###
# Thresholds for diagnostic severity (optional, used for display or flagging)
###

# Linearity R_squared severity thresholds
R2_SEVERITY_THRESHOLDS = {"high": 0.9, "moderate": 0.7, "low": 0.5}

# p-val severity thresholds
PVAL_SEVERITY_THRESHOLDS = {"high": 0.01, "moderate": 0.05, "low": 0.1}

# Variance Inflation Factor severity thresholds
VIF_SEVERITY_THRESHOLDS = {"high": 10, "moderate": 5, "low": 0}

# Cook's Distance severity thresholds
COOKS_SEVERITY_THRESHOLDS = {"high": 0.5, "moderate": 0.2, "low": 0.1}
