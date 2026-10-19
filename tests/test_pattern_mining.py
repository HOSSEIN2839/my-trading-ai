import numpy as np
import pandas as pd
import pytest

from src.market import add_indicators
from src.pattern_mining import _wilson_lower, mine_patterns


def sample_candles(n=900):
    # Deterministic regime-shifted series: stable fixture, no network dependency.
    t = np.arange(n, dtype=float)
    returns = np.where((t.astype(int) // 120) % 2 == 0, 0.0015, -0.0011)
    returns += 0.001 * np.sin(t / 7.0)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = close * 0.002
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": 1000 + 100 * np.sin(t / 11.0) + t % 17,
    })
    return add_indicators(frame)


def test_pattern_mining_returns_auditable_chronological_report():
    report = mine_patterns(
        sample_candles(), threshold=0.001, min_train_support=20,
        min_test_support=8, max_patterns=10,
    )
    assert report["schema_version"] == 1
    assert report["train_rows"] + report["purged_rows"] + report["test_rows"] <= report["complete_rows"]
    assert report["purged_rows"] == 3
    assert report["patterns"]
    assert len(report["patterns"]) <= 10
    for pattern in report["patterns"]:
        assert pattern["direction"] in {"BUY", "SELL"}
        assert pattern["train_support"] >= 20
        assert pattern["test_support"] >= 8
        assert 0 <= pattern["test_directional_hit_rate"] <= 1
        assert 0 <= pattern["test_hit_rate_wilson_95_lower"] <= 1
        assert pattern["research_only"] is True
    assert any("not automatically used" in warning for warning in report["warnings"])


def test_pattern_mining_rejects_bad_inputs():
    with pytest.raises(ValueError, match="at least 100"):
        mine_patterns(pd.DataFrame({"close": [1.0] * 50}))
    frame = sample_candles()
    frame.loc[10, "close"] = np.nan
    with pytest.raises(ValueError, match="finite and positive"):
        mine_patterns(frame)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"bins": 1},
        {"min_train_support": 1},
        {"min_test_support": 0},
        {"max_patterns": 0},
        {"threshold": float("nan")},
        {"train_fraction": 0.4},
    ],
)
def test_pattern_mining_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        mine_patterns(sample_candles(), **kwargs)


def test_pattern_mining_rejects_bad_timestamp_order_and_duplicates():
    frame = sample_candles()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="timestamps"):
        mine_patterns(frame)


def test_wilson_lower_bound_is_bounded_and_conservative():
    assert _wilson_lower(0, 10) == 0
    assert 0 < _wilson_lower(8, 10) < 0.8
    assert _wilson_lower(10, 10) < 1
