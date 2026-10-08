from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def save_equity_curve(equity: pd.Series, path: str | Path) -> None:
    plt.figure(figsize=(10, 4))
    equity.plot()
    plt.title("Equity Curve")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_feature_importance(importances: pd.Series, path: str | Path) -> None:
    plt.figure(figsize=(10, 4))
    importances.sort_values().plot(kind="barh")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
