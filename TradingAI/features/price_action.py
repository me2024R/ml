from __future__ import annotations

import pandas as pd


def candle_body_size(open_price: pd.Series, close_price: pd.Series) -> pd.Series:
    return (close_price - open_price).abs()


def upper_wick(high: pd.Series, open_price: pd.Series, close_price: pd.Series) -> pd.Series:
    return high - pd.concat([open_price, close_price], axis=1).max(axis=1)


def lower_wick(low: pd.Series, open_price: pd.Series, close_price: pd.Series) -> pd.Series:
    return pd.concat([open_price, close_price], axis=1).min(axis=1) - low


def doji_detection(open_price: pd.Series, close_price: pd.Series, threshold: float = 0.001) -> pd.Series:
    return (close_price - open_price).abs() <= threshold


def engulfing_pattern(open_price: pd.Series, close_price: pd.Series, prev_open: pd.Series, prev_close: pd.Series) -> pd.Series:
    bullish = (close_price > open_price) & (prev_close < prev_open) & (close_price > prev_open) & (open_price < prev_close)
    bearish = (close_price < open_price) & (prev_close > prev_open) & (close_price < prev_open) & (open_price > prev_close)
    return bullish | bearish


def pin_bar(high: pd.Series, low: pd.Series, open_price: pd.Series, close_price: pd.Series) -> pd.Series:
    body = (close_price - open_price).abs()
    wick = (high - pd.concat([open_price, close_price], axis=1).max(axis=1)).clip(lower=0)
    return body < (wick * 0.5)
