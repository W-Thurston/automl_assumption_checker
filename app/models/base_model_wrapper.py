from abc import ABC, abstractmethod


class BaseModelWrapper(ABC):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    @abstractmethod
    def fit(self): ...

    @abstractmethod
    def predict(self): ...

    @abstractmethod
    def residuals(self): ...

    @abstractmethod
    def fitted(self): ...

    def summary(self):
        return {}
