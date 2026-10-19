import pandas as pd
import pytest

from src.backtest import run_backtest


def candles(rows):
    return pd.DataFrame([
        {
            "timestamp": pd.Timestamp(f"2026-01-01T00:{i * 15:02d}:00Z"),
            "open": item[0], "high": item[1], "low": item[2], "close": item[3],
            "volume": 100.0, "atr": 1.0, "signal": item[4],
        }
        for i, item in enumerate(rows)
    ])


def signal(row):
    return row["signal"]


def test_signal_executes_on_next_open_and_charges_costs():
    frame = candles([
        (100, 101, 99, 100, "BUY"),
        (100, 102, 99, 101, "HOLD"),
        (102, 105, 101, 104, "HOLD"),
        (104, 105, 103, 104, "HOLD"),
    ])
    result = run_backtest(frame, signal, initial_balance=1000, fee_rate=0.001, slippage_rate=0)

    assert result["metrics"]["trades"] == 1
    trade = result["trades"][0]
    assert trade["entry_price"] == pytest.approx(100)
    assert trade["exit_price"] == pytest.approx(104)
    assert trade["exit_reason"] == "take_profit"
    assert trade["net_pnl"] > 0
    assert result["metrics"]["final_equity"] > 1000


def test_stop_is_assumed_first_when_stop_and_target_hit_same_candle():
    frame = candles([
        (100, 101, 99, 100, "BUY"),
        (100, 101, 99, 100, "HOLD"),
        (100, 105, 97, 101, "HOLD"),
        (100, 101, 99, 100, "HOLD"),
    ])
    result = run_backtest(frame, signal, initial_balance=1000, fee_rate=0, slippage_rate=0)

    assert result["metrics"]["trades"] == 1
    assert result["trades"][0]["exit_reason"] == "stop_loss"
    assert result["trades"][0]["net_pnl"] < 0


def test_backtest_rejects_duplicate_timestamps():
    frame = candles([
        (100, 101, 99, 100, "HOLD"),
        (100, 101, 99, 100, "HOLD"),
        (100, 101, 99, 100, "HOLD"),
    ])
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="duplicate"):
        run_backtest(frame, signal)


def test_no_trade_metrics_are_finite_and_json_friendly():
    frame = candles([
        (100, 101, 99, 100, "HOLD"),
        (100, 101, 99, 100, "HOLD"),
        (100, 101, 99, 100, "HOLD"),
    ])
    result = run_backtest(frame, signal)
    assert result["metrics"]["trades"] == 0
    assert result["metrics"]["profit_factor"] == 0.0
    assert result["metrics"]["net_return_pct"] == 0.0


def test_backtest_rejects_unordered_timestamps_and_non_finite_values():
    frame = candles([
        (100, 101, 99, 100, "HOLD"),
        (100, 101, 99, 100, "HOLD"),
        (100, 101, 99, 100, "HOLD"),
    ])
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"] - pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="ordered"):
        run_backtest(frame, signal)

    frame = candles([
        (100, 101, 99, 100, "HOLD"),
        (100, 101, 99, 100, "HOLD"),
        (100, 101, 99, 100, "HOLD"),
    ])
    frame.loc[1, "close"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        run_backtest(frame, signal)
