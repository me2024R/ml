from __future__ import annotations

import pandas as pd


class HardVotingEnsemble:
    def __init__(self, models: list[object]) -> None:
        self.models = models

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = [model.model.predict(X) for model in self.models]
        return pd.Series(pd.DataFrame(predictions).T.mode(axis=1)[0], index=X.index)
