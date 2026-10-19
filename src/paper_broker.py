"""Persistent, long-only spot paper broker (not a real exchange client)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from datetime import datetime, timezone

from src.paths import project_path


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class PaperBroker:
    def __init__(self, initial_balance: float, state_path="data/paper_state.json", trades_path="data/trades.csv", fee_rate=0.001):
        if not _finite(initial_balance) or float(initial_balance) <= 0:
            raise ValueError("initial_balance must be finite and positive")
        if not _finite(fee_rate) or not 0 <= float(fee_rate) < 1:
            raise ValueError("fee_rate must be finite and in [0, 1)")
        self.state_path = project_path(state_path)
        self.trades_path = project_path(trades_path)
        self.transaction_path = self.state_path.with_suffix(".transaction.json")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.fee_rate = float(fee_rate)
        self._recover_pending_transaction()
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._validate_state()
        else:
            self.state = {"balance": float(initial_balance), "start_balance": float(initial_balance),
                          "day": datetime.now(timezone.utc).date().isoformat(), "position": None}
            self._save()
        if not self.trades_path.exists():
            with self.trades_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["time_utc", "side", "price", "quantity", "fee", "balance", "reason"])

    def _validate_state(self):
        if not isinstance(self.state, dict):
            raise ValueError("paper state must be a JSON object")
        for key in ("balance", "start_balance"):
            if key not in self.state or not _finite(self.state[key]) or float(self.state[key]) < 0:
                raise ValueError(f"paper state has invalid {key}")
        if float(self.state["start_balance"]) <= 0:
            raise ValueError("paper state has invalid start_balance")
        day = self.state.get("day")
        if not isinstance(day, str):
            raise ValueError("paper state has invalid day")
        try:
            if datetime.strptime(day, "%Y-%m-%d").date().isoformat() != day:
                raise ValueError
        except ValueError as exc:
            raise ValueError("paper state has invalid day") from exc
        if "last_processed_candle" in self.state:
            candle_id = self.state["last_processed_candle"]
            if not isinstance(candle_id, str) or not candle_id.strip():
                raise ValueError("paper state has invalid last_processed_candle")
        pos = self.state.get("position")
        if pos is not None:
            if not isinstance(pos, dict):
                raise ValueError("paper position must be an object or null")
            for key in ("entry", "quantity", "stop", "target"):
                if key not in pos or not _finite(pos[key]) or float(pos[key]) <= 0:
                    raise ValueError(f"paper position has invalid {key}")
            if not (float(pos["stop"]) < float(pos["entry"]) < float(pos["target"])):
                raise ValueError("paper position stop/entry/target are inconsistent")

    def _write_state_snapshot(self, state):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def _save(self):
        self._write_state_snapshot(self.state)

    def _recover_pending_transaction(self):
        """Roll back an interrupted cross-file trade transaction before loading state."""
        if not self.transaction_path.exists():
            return
        try:
            journal = json.loads(self.transaction_path.read_text(encoding="utf-8"))
            old_state = journal["old_state"]
            csv_existed = journal["csv_existed"]
            csv_size = journal["csv_size"]
            if (
                not isinstance(old_state, dict)
                or not isinstance(csv_existed, bool)
                or not isinstance(csv_size, int)
                or csv_size < 0
            ):
                raise ValueError("invalid transaction journal fields")
            json.dumps(old_state, allow_nan=False)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "paper trade recovery journal is corrupt; refusing to start with uncertain state"
            ) from exc

        if csv_existed:
            if not self.trades_path.exists():
                raise RuntimeError(
                    "paper trade recovery needs the previous trades CSV, but it is missing"
                )
            with self.trades_path.open("r+b") as log_file:
                log_file.truncate(csv_size)
        elif self.trades_path.exists():
            self.trades_path.unlink()

        self._write_state_snapshot(old_state)
        self.transaction_path.unlink()


    def equity(self, mark_price: float) -> float:
        """Cash plus marked-to-market value of any open simulated position."""
        if not _finite(mark_price) or float(mark_price) <= 0:
            raise ValueError("mark_price must be finite and positive")
        pos = self.state["position"]
        return float(self.state["balance"]) + (float(pos["quantity"]) * float(mark_price) if pos else 0.0)

    def record_market_price(self, price: float, symbol: str, timeframe: str, timestamp: str | None = None):
        """Persist the latest closed-candle snapshot for monitoring and dashboard display."""
        if not _finite(price) or float(price) <= 0:
            raise ValueError("market price must be finite and positive")
        self.state["last_market"] = {
            "price": float(price), "symbol": str(symbol), "timeframe": str(timeframe),
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def mark_candle_processed(self, candle_id: str):
        """Persist the last fully processed closed-candle ID across restarts."""
        if not isinstance(candle_id, str) or not candle_id.strip():
            raise ValueError("candle_id must be a non-empty string")
        self.state["last_processed_candle"] = candle_id
        self._save()

    def roll_day(self, mark_price: float | None = None):
        """Start a new UTC daily-loss window using current marked-to-market equity."""
        if mark_price is not None and (not _finite(mark_price) or float(mark_price) <= 0):
            raise ValueError("mark_price must be finite and positive")
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

    def _commit_trade(self, side, price, quantity, fee, reason, *, balance, position):
        """Persist a simulated trade or roll back in-memory and CSV changes on I/O failure.

        The state file is replaced atomically by _save(). If logging or state persistence
        raises, restore the previous in-memory state and truncate any partially appended
        CSV record so a caught exception cannot leave the running process inconsistent.
        """
        if self.transaction_path.exists():
            raise RuntimeError("an unresolved paper trade transaction exists; restart to recover it")
        previous_state = self.state.copy()
        csv_existed = self.trades_path.exists()
        csv_size = self.trades_path.stat().st_size if csv_existed else 0
        journal = {
            "old_state": previous_state,
            "csv_existed": csv_existed,
            "csv_size": csv_size,
        }
        journal_tmp = self.transaction_path.with_suffix(".tmp")
        journal_tmp.write_text(json.dumps(journal, indent=2, allow_nan=False), encoding="utf-8")
        journal_tmp.replace(self.transaction_path)
        self.state["balance"] = balance
        self.state["position"] = position
        try:
            self._log(side, price, quantity, fee, reason)
            self._save()
            self.transaction_path.unlink()
        except BaseException:
            self.state = previous_state
            try:
                if csv_existed:
                    with self.trades_path.open("r+b") as log_file:
                        log_file.truncate(csv_size)
                elif self.trades_path.exists():
                    self.trades_path.unlink()
                self._write_state_snapshot(previous_state)
                if self.transaction_path.exists():
                    self.transaction_path.unlink()
            except OSError as rollback_error:
                raise RuntimeError(
                    "paper trade persistence failed and rollback also failed; restart to recover from the transaction journal"
                ) from rollback_error
            raise

    def buy(self, price: float, quantity: float, stop: float, target: float, reason: str, metadata: dict | None = None) -> bool:
        if not all(_finite(v) for v in (price, quantity, stop, target)):
            return False
        price, quantity, stop, target = map(float, (price, quantity, stop, target))
        if price <= 0 or quantity <= 0 or stop <= 0 or stop >= price or target <= price:
            return False
        cost = price * quantity
        fee = cost * self.fee_rate
        if not _finite(cost + fee) or self.state["position"] is not None or cost + fee > self.state["balance"]:
            return False
        if metadata is not None:
            if not isinstance(metadata, dict):
                return False
            # Metadata is supplemental only; it must never override accounting
            # fields that determine position size, risk, or exit behavior.
            reserved = {"entry", "quantity", "stop", "target"}
            if reserved.intersection(metadata):
                return False
            # Validate serializability before mutating state or appending a CSV row.
            try:
                json.dumps(metadata, allow_nan=False)
            except (TypeError, ValueError):
                return False
        self._commit_trade(
            "BUY", price, quantity, fee, reason,
            balance=float(self.state["balance"]) - cost - fee,
            position={"entry": price, "quantity": quantity, "stop": stop, "target": target, **(metadata or {})},
        )
        return True

    def sell_all(self, price: float, reason: str) -> bool:
        if not _finite(price) or float(price) <= 0:
            return False
        pos = self.state["position"]
        if not pos:
            return False
        price = float(price)
        proceeds = price * float(pos["quantity"])
        fee = proceeds * self.fee_rate
        if not _finite(proceeds - fee):
            return False
        self._commit_trade(
            "SELL", price, float(pos["quantity"]), fee, reason,
            balance=float(self.state["balance"]) + proceeds - fee,
            position=None,
        )
        return True

    def check_exit(self, low: float, high: float, close: float) -> str | None:
        if not all(_finite(v) and float(v) > 0 for v in (low, high, close)) or float(low) > float(high):
            return None
        pos = self.state["position"]
        if not pos:
            return None
        # If both levels fall inside one candle, assume stop is hit first (conservative).
        if float(low) <= pos["stop"]:
            self.sell_all(pos["stop"], "stop_loss")
            return "stop_loss"
        if float(high) >= pos["target"]:
            self.sell_all(pos["target"], "take_profit")
            return "take_profit"
        return None
