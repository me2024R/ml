from __future__ import annotations

import pandas as pd


class WeightedVotingEnsemble:
    def __init__(self, models: list[object], weights: list[float]) -> None:
        self.models = models
        self.weights = weights

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probabilities = [model.predict_proba(X) for model in self.models]
        weighted = []
        for idx, prob in enumerate(probabilities):
            weighted.append(prob * self.weights[idx])
        return sum(weighted) / sum(self.weights)
