from __future__ import annotations

import pandas as pd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def bollinger_bands(series: pd.Series, window: int = 20, std: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window).mean()
    dev = std * series.rolling(window).std()
    upper = middle + dev
    lower = middle - dev
    return middle, upper, lower


def bollinger_width(upper: pd.Series, lower: pd.Series) -> pd.Series:
    return (upper - lower) / ((upper + lower) / 2)


def rolling_std(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).std()
