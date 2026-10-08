from __future__ import annotations

from typing import Any

import pandas as pd


class BacktestEngine:
    def __init__(self, config: Any) -> None:
        self.config = config

    def run(self, signals: pd.Series, prices: pd.Series) -> pd.DataFrame:
        trades = []
        for timestamp, signal in signals.items():
            if signal == "BUY":
                trades.append({"timestamp": timestamp, "signal": signal, "price": prices.loc[timestamp]})
        return pd.DataFrame(trades)
