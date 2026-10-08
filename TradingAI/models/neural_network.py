from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


class NeuralNetworkModel:
    def __init__(self, random_state: int = 42, **kwargs) -> None:
        self.model = MLPClassifier(random_state=random_state, max_iter=500, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NeuralNetworkModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict_proba(X), index=X.index)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        predictions = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, zero_division=0),
            "recall": recall_score(y, predictions, zero_division=0),
            "f1": f1_score(y, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y, predictions),
        }
