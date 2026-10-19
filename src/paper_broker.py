"""Persistent, long-only spot paper broker (not a real exchange client)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timezone


class PaperBroker:
    def __init__(self, initial_balance: float, state_path="data/paper_state.json", trades_path="data/trades.csv", fee_rate=0.001):
        self.state_path = Path(state_path)
        self.trades_path = Path(trades_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.fee_rate = fee_rate
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {"balance": float(initial_balance), "start_balance": float(initial_balance),
                          "day": datetime.now(timezone.utc).date().isoformat(), "position": None}
            self._save()
        if not self.trades_path.exists():
            with self.trades_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["time_utc", "side", "price", "quantity", "fee", "balance", "reason"])

    def _save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def equity(self, mark_price: float) -> float:
        """Cash plus marked-to-market value of any open simulated position."""
        pos = self.state["position"]
        return float(self.state["balance"]) + (float(pos["quantity"]) * mark_price if pos else 0.0)

    def roll_day(self, mark_price: float | None = None):
        """Start a new UTC daily-loss window using current marked-to-market equity.

        When mark_price is omitted, entry price is used as a compatibility fallback.
        The trading loop should pass the latest closed candle price.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        if self.state["day"] != today:
            pos = self.state["position"]
            position_value = 0.0
            if pos:
                valuation_price = float(mark_price) if mark_price is not None else float(pos["entry"])
                position_value = float(pos["quantity"]) * valuation_price
            baseline = float(self.state["balance"]) + position_value
            self.state["day"] = today
            self.state["start_balance"] = baseline
            self._save()

    def _log(self, side, price, qty, fee, reason):
        with self.trades_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), side, price, qty, fee, self.state["balance"], reason])

    def buy(self, price: float, quantity: float, stop: float, target: float, reason: str, metadata: dict | None = None) -> bool:
        cost = price * quantity
        fee = cost * self.fee_rate
        if self.state["position"] is not None or quantity <= 0 or cost + fee > self.state["balance"]:
            return False
        if price <= 0 or stop <= 0 or stop >= price or target <= price:
            return False
        self.state["balance"] -= cost + fee
        self.state["position"] = {"entry": price, "quantity": quantity, "stop": stop, "target": target, **(metadata or {})}
        self._log("BUY", price, quantity, fee, reason)
        self._save()
        return True

    def sell_all(self, price: float, reason: str) -> bool:
        pos = self.state["position"]
        if not pos or price <= 0:
            return False
        proceeds = price * pos["quantity"]
        fee = proceeds * self.fee_rate
        self.state["balance"] += proceeds - fee
        self._log("SELL", price, pos["quantity"], fee, reason)
        self.state["position"] = None
        self._save()
        return True

    def check_exit(self, low: float, high: float, close: float) -> str | None:
        pos = self.state["position"]
        if not pos:
            return None
        # If both levels fall inside one candle, assume stop is hit first (conservative).
        if low <= pos["stop"]:
            self.sell_all(pos["stop"], "stop_loss")
            return "stop_loss"
        if high >= pos["target"]:
            self.sell_all(pos["target"], "take_profit")
            return "take_profit"
        return None
