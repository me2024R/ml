from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


class LightGBMModel:
    def __init__(self, random_state: int = 42, **kwargs) -> None:
        if lgb is None:
            raise ImportError("lightgbm is not installed")
        self.model = lgb.LGBMClassifier(random_state=random_state, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
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
