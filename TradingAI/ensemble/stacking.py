from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression


class StackingEnsemble:
    def __init__(self, models: list[object], meta_model=None) -> None:
        self.models = models
        self.meta_model = meta_model or LogisticRegression(max_iter=1000)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingEnsemble":
        level1 = pd.concat([model.predict_proba(X) for model in self.models], axis=1)
        self.meta_model.fit(level1, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        level1 = pd.concat([model.predict_proba(X) for model in self.models], axis=1)
        return pd.DataFrame(self.meta_model.predict_proba(level1), index=X.index)
