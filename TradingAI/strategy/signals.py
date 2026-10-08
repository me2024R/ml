from __future__ import annotations

import pandas as pd


def generate_signal(probability: float, threshold: float = 0.75) -> str:
    if probability >= threshold:
        return "BUY"
    if probability <= 1 - threshold:
        return "SELL"
    return "HOLD"


def signal_from_probabilities(probabilities: pd.DataFrame, threshold: float = 0.75) -> pd.Series:
    positive = probabilities.iloc[:, 1] if probabilities.shape[1] > 1 else probabilities.iloc[:, 0]
    return positive.apply(lambda p: generate_signal(p, threshold))
