"""Public market data and deterministic technical indicators."""
from __future__ import annotations

import ccxt
import pandas as pd


def fetch_candles(symbol: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise RuntimeError("صرافی کندلی برنگرداند.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    return add_indicators(df)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    out["rsi"] = 100 - (100 / (1 + rs))
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return out


def technical_signal(row: pd.Series) -> str:
    """Conservative long-only spot signal. Never opens shorts."""
    values = [row.get(k) for k in ("close", "ema50", "ema200", "rsi", "atr")]
    if any(pd.isna(v) for v in values):
        return "HOLD"
    if row["close"] > row["ema200"] and row["ema50"] > row["ema200"] and 45 <= row["rsi"] <= 68:
        return "BUY"
    if row["close"] < row["ema50"] or row["rsi"] >= 75:
        return "SELL"
    return "HOLD"
