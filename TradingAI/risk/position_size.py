from __future__ import annotations


def calculate_position_size(balance: float, risk_per_trade: float, stop_loss_distance: float) -> float:
    if stop_loss_distance <= 0:
        return 0.0
    return balance * risk_per_trade / stop_loss_distance
