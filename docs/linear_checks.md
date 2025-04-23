# ✅ Linear Model Assumption Checks

---

## 🔖 Severity Levels

### 🔴 High — Must fix before trusting model.

### 🟠 Moderate — Should fix for valid inference or cleaner prediction.

### 🟡 Low/Informative — Worth noting; often fixable with robust stats or larger data.

---

## ✅ 1. Normality of Residuals

### 📌 Why It Matters:

Normal residuals allow for valid confidence intervals and hypothesis testing. If this assumption fails, inference may be misleading.

### 🔍 How to Check It:

**Visual:**

- Q-Q Plot (Quantile-Quantile)
- Histogram of residuals

**Statistical:**

- Shapiro-Wilk Test
- D’Agostino and Pearson’s Test (`scipy.stats.normaltest`)
- Anderson-Darling Test (stricter, emphasizes tails)

### ✅ Pass Criteria:

- p-value > 0.05 → fail to reject normality
- Q-Q plot: residuals should fall roughly along the diagonal

### ❌ If Test Fails:

**Severity:**

- 🟡 Moderate
- Mainly impacts inference, not predictions. Critical for confidence intervals & p-values.

**Recommendation:**

- Consider applying transformations to the target (e.g., log, square root)
- Use robust regression models or non-parametric methods

---

## ✅ 2. Multicollinearity

### 📌 Why It Matters:

Multicollinearity inflates coefficient variance, making estimates unstable and hard to interpret.

### 🔍 How to Check It:

- Variance Inflation Factor (VIF)
- Feature correlation matrix (visual)

### ✅ Pass Criteria:

- VIF < 5 (acceptable), < 10 (tolerable)

### ❌ If Test Fails:

**Severity:**

- 🟠 Moderate to High
- Inflates variance, can destabilize coefficients, reduce interpretability, and increase overfitting risk.

**Recommendation:**

- Remove or combine correlated predictors
- Use PCA or regularization (Ridge/Lasso)

---

## ✅ 3. Linearity

### 📌 Why It Matters:

Linear models assume a linear relationship between predictors and the target variable.

### 🔍 How to Check It:

- Residuals vs Fitted Plot
- Add non-linear features (e.g., polynomials) and evaluate performance

### ✅ Pass Criteria:

- Residuals scattered randomly around zero
- No visible patterns or curves in residual plot

### ❌ If Test Fails:

**Severity:**

- 🔴 High
- Violates the foundational assumption. Predictions & inference both become unreliable.

**Recommendation:**

- Add polynomial or interaction terms
- Switch to non-linear models (e.g., trees, splines)

---

## ✅ 4. Homoscedasticity

### 📌 Why It Matters:

Linear regression assumes constant variance of residuals across levels of fitted values. Violation implies inefficient estimates.

### 🔍 How to Check It:

- Residuals vs Fitted Plot
- Breusch-Pagan Test

### ✅ Pass Criteria:

- Breusch-Pagan p-value > 0.05
- Residuals show no funnel shape

### ❌ If Test Fails:

**Severity:**

- 🟡 Moderate
- Affects efficiency of estimates. Standard errors may be inaccurate or inefficient. Inference may be misleading.

**Recommendation:**

- Use heteroscedasticity-robust standard errors
- Transform the dependent variable

---

## ✅ 5. Independence

### 📌 Why It Matters:

Residuals should be independent. Violation (especially in time series) leads to biased inference.

### 🔍 How to Check It:

- Durbin-Watson Test
- Autocorrelation Function (ACF) Plot

### ✅ Pass Criteria:

- Durbin-Watson ∈ [1.5, 2.5]

### ❌ If Test Fails:

**Severity:**

- 🔴 High (in time/sequence data)
- Causes misleading standard errors and invalid inference. Often overlooked but severe in time series.

**Recommendation:**

- Model residual correlation (e.g., use ARIMA or GLS)
- Add lagged predictors or use time series models

---

# 🧪 Additional Assumption Checks (Planned / In Progress)

## 🔍 6. Influence

### 📌 Why It Matters:

Influential points disproportionately affect model coefficients. They're not just outliers — they can bend the model to their will.

### 🔍 How to Check It:

- Cook’s Distance
- Hat Values / Leverage Scores
- DFBETAs (change in coefficients if point is removed)

### ✅ Pass Criteria:

- No points with Cook's Distance > 4/n or > 1 (rule of thumb)
- Leverage values within reasonable bounds (typically < 2p/n)

### ❌ If Test Fails:

**Severity:**

- 🔴 High
- Can distort the entire model. A few points can shift coefficients dramatically.

**Recommendation:**

- Investigate high-leverage observations
- Consider model re-fit without those points
- Use robust regression (e.g., Huber, RANSAC)

---

## 🔍 7. Outliers

### 📌 Why It Matters:

Extreme residuals can distort model fit and invalidate inference — even if they're not high-leverage points.

### 🔍 How to Check It:

- Standardized or studentized residuals
- Bonferroni-adjusted p-values for residual tests
- Influence plots (residual vs leverage)

### ✅ Pass Criteria:

- No residuals beyond ±3 standard deviations
- Bonferroni-corrected outlier test not significant

### ❌ If Test Fails:

**Severity:**

- 🟠 Moderate to High
- Impacts fit, error metrics, and interpretability. Context-dependent — extreme values may be valid. More serious in small data.

**Recommendation:**

- Investigate if outliers are data entry errors or valid extremes
- Consider transformation or robust methods
- Flag for domain expert review

---

## 🔍 8. Missingness

### 📌 Why It Matters:

Patterns of missing data can introduce bias, especially if not missing completely at random (MCAR).

### 🔍 How to Check It:

- Count and percentage of missing values
- Heatmap of missingness
- Little’s MCAR test (advanced)
- Missingness vs target correlation

### ✅ Pass Criteria:

- <5% missing overall, no strong pattern by feature
- MCAR or ignorable MAR assumed

### ❌ If Test Fails:

**Severity:**

- 🟠 Moderate to High
- Can bias results, especially if not Missing Completely at Random (MCAR). The impact depends on extent & pattern.

**Recommendation:**

- Impute missing data using appropriate method (mean, MICE, KNN)
- Use models that handle missingness (e.g., tree-based, Bayesian)
- Avoid dropping rows unless missingness is random and rare
