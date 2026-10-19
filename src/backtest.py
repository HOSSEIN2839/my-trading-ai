"""Cost-aware historical backtest for long-only spot strategies.

Signals are computed after a candle closes and executed at the next candle open.
This is research tooling only; it never sends exchange orders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import ccxt
import numpy as np
import pandas as pd

from src.market import add_indicators, technical_signal
from src.paths import project_path


SignalFunction = Callable[[pd.Series], str]


def _valid_candles(candles: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume", "atr"}
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"missing required candle columns: {sorted(missing)}")
    frame = candles.copy().reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume", "atr"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume", "atr"]
    if frame[numeric_columns].isna().any().any() or not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("candles contain missing, non-numeric, or non-finite values")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("candles contain invalid timestamps")
    frame["timestamp"] = timestamps
    if not timestamps.is_monotonic_increasing:
        raise ValueError("candles must be ordered by ascending timestamp")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (frame["volume"] < 0).any() or (frame["atr"] < 0).any():
        raise ValueError("volume and ATR cannot be negative")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("candle high is inconsistent with OHLC values")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("candle low is inconsistent with OHLC values")
    if frame["timestamp"].duplicated().any():
        raise ValueError("duplicate candle timestamps are not allowed")
    return frame


def run_backtest(
    candles: pd.DataFrame,
    signal_fn: SignalFunction = technical_signal,
    *,
    initial_balance: float = 1000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    risk_fraction: float = 0.005,
    max_position_fraction: float = 0.25,
    stop_atr_multiplier: float = 2.0,
    reward_risk: float = 2.0,
) -> dict:
    """Run a conservative long-only backtest and return metrics plus ledgers.

    BUY/SELL/HOLD signals are evaluated on a candle close and acted on at the
    following candle open. Stops/targets use candle high/low; if both are hit,
    the stop is assumed first. Fees and adverse slippage are charged on both legs.
    """
    if initial_balance <= 0:
        raise ValueError("initial_balance must be positive")
    if not 0 <= fee_rate < 1 or not 0 <= slippage_rate < 1:
        raise ValueError("fee_rate and slippage_rate must be in [0, 1)")
    if not 0 < risk_fraction <= 1 or not 0 < max_position_fraction <= 1:
        raise ValueError("risk and position fractions must be in (0, 1]")
    if stop_atr_multiplier <= 0 or reward_risk <= 0:
        raise ValueError("stop multiplier and reward/risk must be positive")

    frame = _valid_candles(candles)
    if len(frame) < 3:
        raise ValueError("at least 3 candles are required")

    cash = float(initial_balance)
    position = None
    pending = None
    trades = []
    equity_curve = []
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    def execute_exit(raw_price: float, timestamp, reason: str):
        nonlocal cash, position, wins, gross_profit, gross_loss
        if position is None:
            return
        fill = raw_price * (1 - slippage_rate)
        qty = position["quantity"]
        exit_fee = fill * qty * fee_rate
        proceeds = fill * qty - exit_fee
        cash += proceeds
        pnl = proceeds - position["entry_cost"]
        trades.append({
            "entry_time": str(position["entry_time"]),
            "exit_time": str(timestamp),
            "entry_price": position["entry_price"],
            "exit_price": fill,
            "quantity": qty,
            "entry_fee": position["entry_fee"],
            "exit_fee": exit_fee,
            "net_pnl": pnl,
            "return_pct": pnl / position["entry_cost"] if position["entry_cost"] else 0.0,
            "exit_reason": reason,
        })
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += abs(pnl)
        position = None

    for i, row in frame.iterrows():
        timestamp = row["timestamp"]
        open_price = float(row["open"])

        # Execute decisions generated only after the previous candle closed.
        if pending is not None:
            action = pending["action"]
            if action == "SELL" and position is not None:
                execute_exit(open_price, timestamp, "signal")
            elif action == "BUY" and position is None:
                atr = pending["atr"]
                fill = open_price * (1 + slippage_rate)
                stop = fill - stop_atr_multiplier * atr
                target = fill + reward_risk * (fill - stop)
                if atr > 0 and stop > 0 and cash > 0:
                    # Include adverse stop slippage and both-side fees in the risk budget.
                    risk_cash = cash * risk_fraction
                    stop_fill = stop * (1 - slippage_rate)
                    risk_per_unit = (fill - stop_fill) + fill * fee_rate + stop_fill * fee_rate
                    qty_risk = risk_cash / risk_per_unit if risk_per_unit > 0 else 0.0
                    qty_cap = cash * max_position_fraction / (fill * (1 + fee_rate))
                    qty = min(qty_risk, qty_cap)
                    entry_fee = fill * qty * fee_rate
                    total_cost = fill * qty + entry_fee
                    if qty > 0 and total_cost <= cash:
                        cash -= total_cost
                        position = {
                            "entry_time": timestamp,
                            "entry_price": fill,
                            "entry_fee": entry_fee,
                            "entry_cost": total_cost,
                            "quantity": qty,
                            "stop": stop,
                            "target": target,
                        }
            pending = None

        # Existing positions may exit within this candle. Stop wins ambiguous bars.
        if position is not None:
            if float(row["low"]) <= position["stop"]:
                execute_exit(position["stop"], timestamp, "stop_loss")
            elif float(row["high"]) >= position["target"]:
                execute_exit(position["target"], timestamp, "take_profit")

        mark_price = float(row["close"])
        equity = cash + (position["quantity"] * mark_price if position else 0.0)
        equity_curve.append({"timestamp": str(timestamp), "equity": equity})

        # A signal from this close can only execute at the next candle's open.
        if i < len(frame) - 1:
            action = str(signal_fn(row)).upper()
            if action not in {"BUY", "SELL", "HOLD"}:
                action = "HOLD"
            pending = {"action": action, "atr": float(row["atr"])}

    final_price = float(frame.iloc[-1]["close"])
    if position is not None:
        execute_exit(final_price, frame.iloc[-1]["timestamp"], "end_of_data")
        equity_curve[-1]["equity"] = cash

    final_equity = cash
    peak = float(initial_balance)
    max_drawdown = 0.0
    for point in equity_curve:
        peak = max(peak, float(point["equity"]))
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - float(point["equity"])) / peak)

    net_profit = final_equity - initial_balance
    count = len(trades)
    gross_loss = float(gross_loss)
    metrics = {
        "initial_balance": float(initial_balance),
        "final_equity": final_equity,
        "net_profit": net_profit,
        "net_return_pct": net_profit / initial_balance * 100,
        "trades": count,
        "win_rate_pct": wins / count * 100 if count else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (None if gross_profit else 0.0),
        "max_drawdown_pct": max_drawdown * 100,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "assumptions": [
            "Signals execute at the next candle open.",
            "Fees and adverse slippage are charged on entry and exit.",
            "If stop and target are both hit in one candle, stop is assumed first.",
            "No short positions, funding, spread model, or exchange-specific lot-size rules.",
        ],
    }
    return {"metrics": metrics, "trades": trades, "equity_curve": equity_curve}


def main() -> None:
    symbol = "BTC/USDT"
    timeframe = "15m"
    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)
    if len(rows) < 100:
        raise SystemExit("تعداد کندل‌های تاریخی برای بک‌تست کافی نیست.")
    candles = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], unit="ms", utc=True)
    candles = add_indicators(candles).iloc[:-1].reset_index(drop=True)  # exclude potentially open candle
    result = run_backtest(candles)
    output = project_path("data/backtest_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))
    print(f"گزارش کامل در {output} ذخیره شد.")
    print("این نتیجه مربوط به گذشته است و تضمین‌کننده سود آینده نیست.")


if __name__ == "__main__":
    main()
