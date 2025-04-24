# src/models/linear_model_wrapper.py
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

from src.models.base_model_wrapper import BaseModelWrapper


class LinearModelWrapper(BaseModelWrapper):
    def fit(self):
        self.model = sm.OLS(self.y, sm.add_constant(self.X)).fit()
        return self

    def predict(self):
        return self.model.predict(sm.add_constant(self.X))

    def residuals(self):
        return self.model.resid

    def fitted(self):
        return self.model.fittedvalues

    def get_influence(self) -> OLSInfluence:
        return OLSInfluence(self.model)

    def summary(self):
        return {"model_type": "Linear Regression", "r_squared": self.model.rsquared}
