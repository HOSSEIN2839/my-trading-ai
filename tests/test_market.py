import pandas as pd
from src.market import add_indicators, technical_signal


def test_indicators_are_added():
    n = 260
    close = pd.Series([100 + i * 0.1 for i in range(n)])
    df = pd.DataFrame({"close": close, "high": close + 1, "low": close - 1, "open": close, "volume": 10})
    out = add_indicators(df)
    assert {"ema50", "ema200", "rsi", "atr"}.issubset(out.columns)
    assert pd.notna(out.iloc[-1]["ema200"])


def test_signal_handles_missing_values():
    row = pd.Series({"close": 100, "ema50": float("nan"), "ema200": 90, "rsi": 50, "atr": 1})
    assert technical_signal(row) == "HOLD"
