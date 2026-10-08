from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_features(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize feature columns for model training."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return pd.DataFrame(scaled, index=features.index, columns=features.columns), scaler
