"""Position sizing and daily loss guardrails."""
from __future__ import annotations


def position_size(balance: float, entry: float, stop: float, risk_fraction: float, max_position_fraction: float) -> float:
    if balance <= 0 or entry <= 0 or stop <= 0 or stop >= entry:
        return 0.0
    risk_cash = balance * risk_fraction
    units_by_risk = risk_cash / (entry - stop)
    units_by_cap = (balance * max_position_fraction) / entry
    return max(0.0, min(units_by_risk, units_by_cap))


def daily_loss_limit_hit(start_balance: float, current_balance: float, max_daily_loss: float) -> bool:
    if start_balance <= 0:
        return True
    return (start_balance - current_balance) / start_balance >= max_daily_loss
