from __future__ import annotations

import pandas as pd


class RiskManager:
    def __init__(self, config) -> None:
        self.config = config

    def should_enter(self, signal: str, probability: float) -> bool:
        if signal == "BUY" and probability >= self.config.confidence_threshold:
            return True
        return False

    def should_exit(self, current_equity: float, peak_equity: float) -> bool:
        drawdown = (peak_equity - current_equity) / peak_equity if peak_equity else 0.0
        return drawdown >= 0.2
