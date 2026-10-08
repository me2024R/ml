from __future__ import annotations

import pandas as pd


def atr_stop_loss(atr: pd.Series, multiplier: float = 2.0) -> pd.Series:
    return atr * multiplier
