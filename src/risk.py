"""Position sizing and daily loss guardrails."""
from __future__ import annotations

import math


def position_size(
    balance: float,
    entry: float,
    stop: float,
    risk_fraction: float,
    max_position_fraction: float,
) -> float:
    """Return a safe long-only quantity, or zero for invalid risk inputs."""
    values = (balance, entry, stop, risk_fraction, max_position_fraction)
    if not all(math.isfinite(float(value)) for value in values):
        return 0.0
    if balance <= 0 or entry <= 0 or stop <= 0 or stop >= entry:
        return 0.0
    if not 0 < risk_fraction <= 1 or not 0 < max_position_fraction <= 1:
        return 0.0

    risk_cash = balance * risk_fraction
    units_by_risk = risk_cash / (entry - stop)
    units_by_cap = (balance * max_position_fraction) / entry
    quantity = min(units_by_risk, units_by_cap)
    return quantity if math.isfinite(quantity) and quantity > 0 else 0.0


def daily_loss_limit_hit(start_balance: float, current_balance: float, max_daily_loss: float) -> bool:
    """Fail closed when account values or loss-limit configuration are invalid."""
    values = (start_balance, current_balance, max_daily_loss)
    if not all(math.isfinite(float(value)) for value in values):
        return True
    if start_balance <= 0 or current_balance < 0 or not 0 < max_daily_loss <= 1:
        return True
    return (start_balance - current_balance) / start_balance >= max_daily_loss
