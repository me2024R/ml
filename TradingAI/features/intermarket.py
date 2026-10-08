from __future__ import annotations

import pandas as pd


def load_intermarket_series() -> dict[str, pd.Series]:
    """Return placeholder intermarket series keyed by name."""
    return {
        "DXY": pd.Series(dtype="float64"),
        "Gold": pd.Series(dtype="float64"),
        "Oil": pd.Series(dtype="float64"),
        "BondYields": pd.Series(dtype="float64"),
        "VIX": pd.Series(dtype="float64"),
    }
