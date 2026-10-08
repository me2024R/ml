from __future__ import annotations

import pandas as pd


class MarketRegimeDetector:
    def __init__(self, trend_window: int = 20, volatility_window: int = 20) -> None:
        self.trend_window = trend_window
        self.volatility_window = volatility_window

    def detect(self, df: pd.DataFrame) -> pd.Series:
        trend = df["Close"].pct_change().rolling(self.trend_window).mean()
        volatility = df["Close"].pct_change().rolling(self.volatility_window).std()
        regimes = []
        for trend_value, vol_value in zip(trend, volatility):
            if pd.isna(trend_value) or pd.isna(vol_value):
                regimes.append("Unknown")
            elif abs(trend_value) > 0.01 and vol_value > 0.02:
                regimes.append("Trending")
            elif abs(trend_value) <= 0.01 and vol_value > 0.02:
                regimes.append("High Volatility")
            elif abs(trend_value) > 0.01 and vol_value <= 0.02:
                regimes.append("Trending")
            else:
                regimes.append("Ranging")
        return pd.Series(regimes, index=df.index)
