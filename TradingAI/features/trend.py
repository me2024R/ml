from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, window: int = 20) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window).mean()


def wma(series: pd.Series, window: int = 20) -> pd.Series:
    weights = np.arange(1, window + 1)
    return series.rolling(window=window).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)


def hma(series: pd.Series, window: int = 20) -> pd.Series:
    half = max(1, window // 2)
    sqrt_window = max(1, int(np.sqrt(window)))
    wma_half = series.rolling(half).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
    wma_full = series.rolling(window).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
    return 2 * wma_half - wma_full


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = low.diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(window).mean() / pd.Series(np.abs(up_move), index=high.index).rolling(window).mean()
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(window).mean() / pd.Series(np.abs(down_move), index=high.index).rolling(window).mean()
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window).mean()
