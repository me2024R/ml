from __future__ import annotations

import pandas as pd


def swing_highs(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window, center=True).max()


def swing_lows(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window, center=True).min()


def higher_high(series: pd.Series, window: int = 2) -> pd.Series:
    return series > series.shift(window)


def lower_low(series: pd.Series, window: int = 2) -> pd.Series:
    return series < series.shift(window)


def support_distance(series: pd.Series, lookback: int = 20) -> pd.Series:
    support = series.rolling(lookback).min()
    return (series - support) / support.replace(0, pd.NA)


def resistance_distance(series: pd.Series, lookback: int = 20) -> pd.Series:
    resistance = series.rolling(lookback).max()
    return (resistance - series) / resistance.replace(0, pd.NA)
