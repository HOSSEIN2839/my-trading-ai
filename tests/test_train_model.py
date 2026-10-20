import numpy as np
import pandas as pd
import pytest

from src.market import add_indicators
from src.train_model import (
    HORIZON,
    _build_supervised_dataset,
    _load_local_candles,
)


def _sample_candles(rows: int = 420) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01T00:00:00Z",
        periods=rows,
        freq="15min",
    )
    step = np.arange(rows, dtype=float)
    close = 30000 + step * 0.3 + np.sin(step / 4) * 80
    open_price = close - np.sin(step) * 4
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "interval": "15m",
            # Force nanosecond resolution before converting to Unix milliseconds;
            # pandas 3 may otherwise use a different datetime resolution by default.
            "timestamp": (timestamps.astype("datetime64[ns]").astype("int64") // 1_000_000),
            "open": open_price,
            "high": np.maximum(open_price, close) + 10,
            "low": np.minimum(open_price, close) - 10,
            "close": close,
            "volume": 100 + step,
        }
    )


def test_load_local_candles_recalculates_indicators_and_ignores_csv_indicators(tmp_path):
    raw = _sample_candles()
    raw["ema200"] = -999999  # Deliberately invalid precomputed column; it must be ignored.
    path = tmp_path / "candles.csv"
    raw.to_csv(path, index=False)

    loaded = _load_local_candles(str(path), "BTC/USDT", "15m")

    assert len(loaded) == len(raw)
    assert loaded["timestamp"].is_monotonic_increasing
    assert loaded["ema200"].iloc[-1] > 0
    assert loaded["ema200"].iloc[-1] != -999999
    assert {"ema50", "ema200", "rsi", "atr"}.issubset(loaded.columns)


def test_load_local_candles_rejects_gaps(tmp_path):
    raw = _sample_candles()
    raw = raw.drop(index=100).reset_index(drop=True)
    path = tmp_path / "gapped.csv"
    raw.to_csv(path, index=False)

    with pytest.raises(SystemExit, match="فاصلهٔ زمانی"):
        _load_local_candles(str(path), "BTC/USDT", "15m")


def test_load_local_candles_rejects_wrong_symbol(tmp_path):
    raw = _sample_candles()
    raw["symbol"] = "ETHUSDT"
    path = tmp_path / "wrong_symbol.csv"
    raw.to_csv(path, index=False)

    with pytest.raises(SystemExit, match="نماد CSV"):
        _load_local_candles(str(path), "BTC/USDT", "15m")


def test_supervised_dataset_drops_rows_without_future_label():
    raw = _sample_candles(600)
    candles = raw[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], unit="ms", utc=True)
    featured_candles = add_indicators(candles)

    dataset = _build_supervised_dataset(featured_candles)

    assert dataset["label"].notna().all()
    assert dataset.index.max() == len(featured_candles) - HORIZON - 1
    row_index = 200
    expected_return = (
        featured_candles.loc[row_index + HORIZON, "close"]
        / featured_candles.loc[row_index, "close"]
        - 1
    )
    if row_index in dataset.index:
        if expected_return > 0.004:
            assert dataset.loc[row_index, "label"] == "BUY"
        elif expected_return < -0.004:
            assert dataset.loc[row_index, "label"] == "SELL"
        else:
            assert dataset.loc[row_index, "label"] == "HOLD"
