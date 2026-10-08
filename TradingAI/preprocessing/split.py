from __future__ import annotations

import pandas as pd


def train_test_split_by_time(features: pd.DataFrame, target: pd.Series, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data by chronological order using a time-based ratio."""
    split_index = int(len(features) * train_ratio)
    X_train = features.iloc[:split_index]
    X_test = features.iloc[split_index:]
    y_train = target.iloc[:split_index]
    y_test = target.iloc[split_index:]
    return X_train, X_test, y_train, y_test
