from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar and session features to the dataframe."""
    features = df.copy()
    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features["london_session"] = features.index.hour.between(8, 16).astype(int)
    features["new_york_session"] = features.index.hour.between(13, 21).astype(int)
    features["asian_session"] = features.index.hour.between(0, 8).astype(int)
    return features
