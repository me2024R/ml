from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TradingConfig:
    """Configuration object for the trading research pipeline."""

    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    raw_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "processed")
    model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "models")
    results_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "results")

    assets: List[str] = field(default_factory=lambda: ["AAPL"])
    timeframe: str = "1d"
    lookback: int = 252
    prediction_horizon: int = 5
    train_ratio: float = 0.8
    random_state: int = 42

    commission: float = 0.0005
    spread: float = 0.0005
    slippage: float = 0.0002
    leverage: float = 1.0
    risk_per_trade: float = 0.01
    max_daily_loss: float = 0.05
    max_open_trades: int = 3
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 4.0
    confidence_threshold: float = 0.75

    model_weights: dict = field(
        default_factory=lambda: {
            "RandomForest": 0.25,
            "XGBoost": 0.30,
            "LightGBM": 0.20,
            "CatBoost": 0.15,
            "NeuralNetwork": 0.10,
        }
    )

    def ensure_dirs(self) -> None:
        for directory in [self.raw_dir, self.processed_dir, self.model_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
