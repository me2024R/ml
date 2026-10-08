from __future__ import annotations

import pandas as pd


def rolling_mean(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).mean()


def rolling_median(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).median()


def rolling_variance(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).var()


def rolling_skewness(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).skew()


def rolling_kurtosis(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).kurt()


def z_score(series: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std.replace(0, pd.NA)
