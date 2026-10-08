from __future__ import annotations

import pandas as pd


class SoftVotingEnsemble:
    def __init__(self, models: list[object]) -> None:
        self.models = models

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probabilities = [model.predict_proba(X) for model in self.models]
        stacked = pd.concat(probabilities, axis=1)
        return stacked.groupby(level=0, axis=1).mean()
