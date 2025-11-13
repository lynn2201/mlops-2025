import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel


class SimpleRandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
