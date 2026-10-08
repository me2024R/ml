from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


class CatBoostModel:
    def __init__(self, random_state: int = 42, **kwargs) -> None:
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed")
        self.model = CatBoostClassifier(random_state=random_state, verbose=False, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostModel":
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
