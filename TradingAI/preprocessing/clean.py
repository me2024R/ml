from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def clean_dataframe(df: pd.DataFrame, tz: Optional[str] = None) -> pd.DataFrame:
    """Clean a market data dataframe by handling missing values, duplicates, timezone, and resampling."""
    if df.empty:
        return df.copy()

    cleaned = df.copy().sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]

    if tz is not None and cleaned.index.tz is None:
        cleaned.index = cleaned.index.tz_localize(tz)
    elif tz is not None and cleaned.index.tz is not None:
        cleaned.index = cleaned.index.tz_convert(tz)

    cleaned = cleaned.astype({col: "float64" for col in cleaned.select_dtypes(include=["float64", "int64"]).columns})
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.ffill().bfill()
    cleaned = cleaned.dropna(how="any")

    return cleaned


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input data for expected OHLC columns and reasonable values."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if (df["High"] < df["Low"]).any():
        raise ValueError("Found rows where High < Low")
    if (df["Open"] < 0).any() or (df["Close"] < 0).any() or (df["Volume"] < 0).any():
        raise ValueError("Found negative prices or volume")

    return df


def resample_ohlc(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """Resample OHLC data to the requested timeframe."""
    rule = timeframe.lower()
    if rule.endswith("d"):
        freq = rule
    elif rule.endswith("h"):
        freq = rule
    else:
        freq = rule

    return df.resample(freq).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
