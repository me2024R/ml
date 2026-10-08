from __future__ import annotations

from typing import Any

import pandas as pd


class ExecutionEngine:
    def __init__(self, config: Any) -> None:
        self.config = config

    def execute(self, signals: pd.Series, prices: pd.Series) -> list[dict[str, Any]]:
        trades = []
        for idx, signal in signals.items():
            if signal == "BUY":
                trades.append({"timestamp": idx, "signal": signal, "price": prices.loc[idx]})
        return trades
