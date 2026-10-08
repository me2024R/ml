from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from TradingAI.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel:
    def __init__(self, n_estimators: int = 200, random_state: int = 42, **kwargs) -> None:
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
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

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
