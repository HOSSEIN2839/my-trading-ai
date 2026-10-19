"""Offline, long-only reinforcement-learning environment for paper research.

This environment never connects to an exchange and is not imported by the live bot.
An action chosen after candle i is filled at candle i+1 open and marked at its close.
Rewards are changes in marked equity after simulated fees; this is a research baseline,
not a claim of profitability or a production-ready RL agent.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}


class PaperTradingEnv:
    """Minimal Gym-like offline environment using only historical OHLC candles."""

    def __init__(
        self,
        candles: pd.DataFrame,
        *,
        initial_balance: float = 1_000.0,
        fee_rate: float = 0.001,
        max_exposure: float = 0.25,
    ) -> None:
        required = {"open", "high", "low", "close"}
        if not required.issubset(candles.columns):
            raise ValueError(f"candles must contain columns: {sorted(required)}")
        if len(candles) < 2:
            raise ValueError("at least two candles are required")
        if not math.isfinite(float(initial_balance)) or initial_balance <= 0:
            raise ValueError("initial_balance must be finite and positive")
        if not math.isfinite(float(fee_rate)) or not 0 <= fee_rate < 1:
            raise ValueError("fee_rate must be finite and in [0, 1)")
        if not math.isfinite(float(max_exposure)) or not 0 < max_exposure <= 1:
            raise ValueError("max_exposure must be finite and in (0, 1]")

        frame = candles.loc[:, ["open", "high", "low", "close"]].copy()
        frame = frame.apply(pd.to_numeric, errors="coerce")
        if frame.isna().any().any() or not all(
            math.isfinite(float(value)) and float(value) > 0
            for value in frame.to_numpy().ravel()
        ):
            raise ValueError("OHLC data must contain finite positive numbers")
        if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
            raise ValueError("high must be at least open, close, and low")
        if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
            raise ValueError("low must not exceed open, close, and high")

        self.candles = frame.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.fee_rate = float(fee_rate)
        self.max_exposure = float(max_exposure)
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.index = 0
        self.cash = self.initial_balance
        self.quantity = 0.0
        self.equity = self.initial_balance
        self.done = False
        return self._observation()

    def _observation(self) -> dict[str, Any]:
        row = self.candles.iloc[self.index]
        return {
            "index": self.index,
            "close": float(row["close"]),
            "position_fraction": (
                self.quantity * float(row["close"]) / self.equity
                if self.equity > 0 else 0.0
            ),
            "equity": self.equity,
        }

    def step(self, action: int | str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode is finished; call reset() before stepping")
        if isinstance(action, str):
            normalized = action.strip().upper()
            reverse = {name: code for code, name in ACTIONS.items()}
            if normalized not in reverse:
                raise ValueError("action must be HOLD, BUY, SELL, 0, 1, or 2")
            action_code = reverse[normalized]
        elif isinstance(action, int) and not isinstance(action, bool) and action in ACTIONS:
            action_code = action
        else:
            raise ValueError("action must be HOLD, BUY, SELL, 0, 1, or 2")

        if self.index >= len(self.candles) - 1:
            self.done = True
            raise RuntimeError("no next candle is available")

        previous_equity = self.equity
        next_row = self.candles.iloc[self.index + 1]
        fill_price = float(next_row["open"])
        fees_paid = 0.0
        executed = "HOLD"

        if action_code == 1 and self.quantity == 0.0:
            budget = self.cash * self.max_exposure
            quantity = budget / (fill_price * (1 + self.fee_rate))
            cost = quantity * fill_price
            fees_paid = cost * self.fee_rate
            self.cash -= cost + fees_paid
            self.quantity = quantity
            executed = "BUY"
        elif action_code == 2 and self.quantity > 0.0:
            proceeds = self.quantity * fill_price
            fees_paid = proceeds * self.fee_rate
            self.cash += proceeds - fees_paid
            self.quantity = 0.0
            executed = "SELL"

        self.index += 1
        mark_price = float(next_row["close"])
        self.equity = self.cash + self.quantity * mark_price
        reward = (self.equity - previous_equity) / self.initial_balance
        self.done = self.index >= len(self.candles) - 1
        return self._observation(), float(reward), self.done, {
            "requested_action": ACTIONS[action_code],
            "executed_action": executed,
            "fill_price": fill_price,
            "mark_price": mark_price,
            "fees_paid": fees_paid,
            "cash": self.cash,
            "quantity": self.quantity,
            "equity": self.equity,
        }
