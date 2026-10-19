"""Public market data and deterministic technical indicators."""
from __future__ import annotations

import ccxt
import pandas as pd


def fetch_candles(symbol: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
    if not isinstance(limit, int) or limit < 3:
        raise ValueError("limit must be an integer of at least 3")
    if not symbol or not timeframe:
        raise ValueError("symbol and timeframe are required")
    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise RuntimeError("صرافی کندلی برنگرداند.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    numeric = df[["open", "high", "low", "close", "volume"]]
    if not numeric.map(lambda value: pd.notna(value) and float("-inf") < float(value) < float("inf")).all().all():
        raise RuntimeError("صرافی دادهٔ غیرعددی یا نامتناهی برگرداند.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df["volume"] < 0).any():
        raise RuntimeError("قیمت یا حجم کندل نامعتبر است.")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise RuntimeError("ساختار OHLC کندل نامعتبر است.")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise RuntimeError("ساختار OHLC کندل نامعتبر است.")
    if df["timestamp"].duplicated().any() or not df["timestamp"].is_monotonic_increasing:
        raise RuntimeError("کندل‌ها تکراری یا خارج از ترتیب زمانی هستند.")
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
    out.loc[(loss == 0) & (gain > 0), "rsi"] = 100.0
    out.loc[(gain == 0) & (loss > 0), "rsi"] = 0.0
    out.loc[(gain == 0) & (loss == 0), "rsi"] = 50.0
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
