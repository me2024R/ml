from __future__ import annotations

import pandas as pd


def atr_take_profit(atr: pd.Series, multiplier: float = 4.0) -> pd.Series:
    return atr * multiplier
