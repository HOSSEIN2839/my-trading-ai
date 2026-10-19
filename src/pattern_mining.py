"""Leakage-aware exploratory mining of simple market-regime patterns.

Patterns are research hypotheses only. This module does not place orders, alter
the trading model, or promote patterns into the live decision path.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.learning import FEATURES, make_features


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    """Two-sided 95% Wilson lower confidence bound for a binomial rate."""
    if total <= 0:
        return 0.0
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return max(0.0, (center - margin) / denominator)


def mine_patterns(
    candles: pd.DataFrame,
    *,
    horizon: int = 3,
    threshold: float = 0.004,
    train_fraction: float = 0.70,
    bins: int = 4,
    min_train_support: int = 25,
    min_test_support: int = 10,
    max_patterns: int = 25,
) -> dict[str, Any]:
    """Mine single-feature quantile regimes and evaluate them on a later holdout.

    Quantile cut points and candidate directions are learned from the training
    segment only. A purge gap equal to horizon prevents training labels from
    consuming holdout-period prices. Returns are close-to-close and exclude fees,
    slippage, spread and execution effects, so these results are not trade P&L.
    """
    if not isinstance(candles, pd.DataFrame):
        raise ValueError("candles must be a pandas DataFrame")
    if "close" not in candles or len(candles) < 100:
        raise ValueError("at least 100 candles with close prices are required")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least 2")
    if not isinstance(min_train_support, int) or min_train_support < 2:
        raise ValueError("min_train_support must be an integer of at least 2")
    if not isinstance(min_test_support, int) or min_test_support < 1:
        raise ValueError("min_test_support must be a positive integer")
    if not isinstance(max_patterns, int) or max_patterns < 1:
        raise ValueError("max_patterns must be a positive integer")
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if not math.isfinite(float(train_fraction)) or not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")

    frame = candles.copy().reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("close prices must be finite and positive")
    if "timestamp" in frame:
        times = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
            raise ValueError("timestamps must be valid, unique, and ascending")

    features = make_features(frame).replace([np.inf, -np.inf], np.nan)
    forward_return = pd.Series(close).shift(-horizon).div(pd.Series(close)).sub(1.0)
    data = features.copy()
    data["_forward_return"] = forward_return
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(data) < 80:
        raise ValueError("not enough complete feature rows after cleaning")

    split = int(len(data) * train_fraction)
    train = data.iloc[: max(0, split - horizon)].copy()
    test = data.iloc[split:].copy()
    if len(train) < min_train_support * 2 or len(test) < min_test_support:
        raise ValueError("training or holdout segment is too small for requested support")

    candidates: list[dict[str, Any]] = []
    for feature in FEATURES:
        train_values = train[feature].to_numpy(dtype=float)
        finite_train = train_values[np.isfinite(train_values)]
        if len(finite_train) < min_train_support:
            continue
        edges = np.unique(np.quantile(finite_train, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 3:
            continue
        internal = edges[1:-1]
        train_bin = np.searchsorted(internal, train_values, side="right")
        test_values = test[feature].to_numpy(dtype=float)
        test_bin = np.searchsorted(internal, test_values, side="right")

        for bin_id in range(len(edges) - 1):
            train_mask = train_bin == bin_id
            support = int(train_mask.sum())
            if support < min_train_support:
                continue
            train_returns = train.loc[train_mask, "_forward_return"].to_numpy(dtype=float)
            mean_return = float(np.mean(train_returns))
            if mean_return >= threshold:
                direction = "BUY"
            elif mean_return <= -threshold:
                direction = "SELL"
            else:
                continue

            test_mask = test_bin == bin_id
            test_support = int(test_mask.sum())
            if test_support < min_test_support:
                continue
            test_returns = test.loc[test_mask, "_forward_return"].to_numpy(dtype=float)
            oriented = test_returns if direction == "BUY" else -test_returns
            wins = int(np.count_nonzero(oriented > 0))
            candidates.append({
                "feature": feature,
                "bin": int(bin_id),
                "lower_bound": float(edges[bin_id]),
                "upper_bound": float(edges[bin_id + 1]),
                "direction": direction,
                "train_support": support,
                "train_mean_forward_return": mean_return,
                "test_support": test_support,
                "test_mean_forward_return": float(np.mean(test_returns)),
                "test_directional_hit_rate": wins / test_support,
                "test_hit_rate_wilson_95_lower": _wilson_lower(wins, test_support),
                "test_mean_oriented_return": float(np.mean(oriented)),
                "research_only": True,
            })

    # Also mine two-feature interactions. All bin boundaries are derived from
    # training data only; a stricter support floor limits tiny subgroups.
    feature_bins: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for feature in FEATURES:
        values = train[feature].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 3:
            continue
        internal = edges[1:-1]
        feature_bins[feature] = (
            edges,
            np.searchsorted(internal, values, side="right"),
            np.searchsorted(internal, test[feature].to_numpy(dtype=float), side="right"),
        )

    interaction_support = max(min_train_support, 30)
    feature_names = list(feature_bins)
    for left_index, left_name in enumerate(feature_names):
        left_edges, left_train_bin, left_test_bin = feature_bins[left_name]
        for right_name in feature_names[left_index + 1:]:
            right_edges, right_train_bin, right_test_bin = feature_bins[right_name]
            for left_bin in range(len(left_edges) - 1):
                for right_bin in range(len(right_edges) - 1):
                    train_mask = (left_train_bin == left_bin) & (right_train_bin == right_bin)
                    support = int(train_mask.sum())
                    if support < interaction_support:
                        continue
                    train_returns = train.loc[train_mask, "_forward_return"].to_numpy(dtype=float)
                    mean_return = float(np.mean(train_returns))
                    if mean_return >= threshold:
                        direction = "BUY"
                    elif mean_return <= -threshold:
                        direction = "SELL"
                    else:
                        continue
                    test_mask = (left_test_bin == left_bin) & (right_test_bin == right_bin)
                    test_support = int(test_mask.sum())
                    if test_support < min_test_support:
                        continue
                    test_returns = test.loc[test_mask, "_forward_return"].to_numpy(dtype=float)
                    oriented = test_returns if direction == "BUY" else -test_returns
                    wins = int(np.count_nonzero(oriented > 0))
                    candidates.append({
                        "feature": f"{left_name} + {right_name}",
                        "features": [left_name, right_name],
                        "bins": [int(left_bin), int(right_bin)],
                        "bounds": [
                            [float(left_edges[left_bin]), float(left_edges[left_bin + 1])],
                            [float(right_edges[right_bin]), float(right_edges[right_bin + 1])],
                        ],
                        "direction": direction,
                        "train_support": support,
                        "train_mean_forward_return": mean_return,
                        "test_support": test_support,
                        "test_mean_forward_return": float(np.mean(test_returns)),
                        "test_directional_hit_rate": wins / test_support,
                        "test_hit_rate_wilson_95_lower": _wilson_lower(wins, test_support),
                        "test_mean_oriented_return": float(np.mean(oriented)),
                        "research_only": True,
                    })

    candidates.sort(
        key=lambda item: (item["test_mean_oriented_return"], item["test_support"]),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "method": "single- and pairwise-feature quantile regimes; chronological holdout; purged forward labels",
        "candles_received": int(len(frame)),
        "complete_rows": int(len(data)),
        "train_rows": int(len(train)),
        "purged_rows": int(horizon),
        "test_rows": int(len(test)),
        "horizon_candles": horizon,
        "threshold": float(threshold),
        "candidate_count": len(candidates),
        "patterns": candidates[:max_patterns],
        "warnings": [
            "Exploratory patterns are not automatically used by the bot.",
            "Holdout results can be optimistic when many patterns are tested; validate on untouched periods.",
            "Forward close returns omit fees, spread, slippage, liquidity and executable fills.",
            "No pattern or metric guarantees future profitability.",
        ],
    }


def main() -> None:
    """Download a bounded public sample and save an exploratory pattern report."""
    import json
    import os
    import ccxt

    from dotenv import load_dotenv
    from src.market import add_indicators
    from src.paths import project_path

    load_dotenv()
    symbol = os.getenv("SYMBOL", "BTC/USDT")
    timeframe = os.getenv("TIMEFRAME", "15m")
    requested_limit = int(os.getenv("PATTERN_CANDLE_LIMIT", "1000"))
    if not 100 <= requested_limit <= 1000:
        raise SystemExit("PATTERN_CANDLE_LIMIT must be between 100 and 1000 for this one-request mode.")
    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=requested_limit)
    if len(rows) < 100:
        raise SystemExit("Not enough public historical candles for pattern mining.")
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # Exclude the potentially unfinished candle.
    frame = frame.iloc[:-1].reset_index(drop=True)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise SystemExit("Public market data contains invalid prices or volume.")
    frame = add_indicators(frame)
    report = mine_patterns(frame)
    report["symbol"] = symbol
    report["timeframe"] = timeframe
    report_path = project_path(os.getenv("PATTERN_REPORT_PATH", "data/pattern_report.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary_path.replace(report_path)
    print(json.dumps({key: value for key, value in report.items() if key != "patterns"}, indent=2))
    print(f"Pattern report saved to: {report_path}")
    print("Research-only: discovered patterns do not change bot decisions or place orders.")


if __name__ == "__main__":
    main()
