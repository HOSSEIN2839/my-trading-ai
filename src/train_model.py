"""Train and chronologically evaluate the ML model on public historical candles.

Run: python -m src.train_model
This is research tooling, not evidence of future profitability.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import ccxt
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score

from src.market import add_indicators
from src.learning import FEATURES, make_features, DEFAULT_MODEL_PATH
from src.paths import project_path

load_dotenv()

SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
CANDLE_LIMIT = max(500, int(os.getenv("TRAIN_CANDLE_LIMIT", "1000")))
HORIZON = 3
THRESHOLD = 0.004


def _new_model():
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=8,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )


def main():
    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    raw = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=min(CANDLE_LIMIT, 1000))
    if len(raw) < 400:
        raise SystemExit("برای آموزش امن‌تر، دادهٔ تاریخی کافی دریافت نشد.")

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # Exclude the currently forming candle to keep training labels and features stable.
    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)
    df = add_indicators(df)

    X = make_features(df)
    future_return = df["close"].shift(-HORIZON) / df["close"] - 1
    y = pd.Series("HOLD", index=df.index, dtype="object")
    y.loc[future_return > THRESHOLD] = "BUY"
    y.loc[future_return < -THRESHOLD] = "SELL"
    y.loc[future_return.isna()] = None

    dataset = X.copy()
    dataset["label"] = y
    dataset = dataset.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(dataset) <= 300:
        raise SystemExit("پس از پاک‌سازی، دادهٔ کافی برای آموزش باقی نماند.")

    split = int(len(dataset) * 0.8)
    # Purge the final HORIZON rows from training: their forward-return labels
    # would otherwise consume prices from the chronological holdout period.
    train = dataset.iloc[:split - HORIZON]
    test = dataset.iloc[split:]
    if len(train) < 200 or len(test) < 50:
        raise SystemExit("اندازهٔ بخش آموزش یا آزمون برای ارزیابی قابل‌اتکا کافی نیست.")

    evaluation_model = _new_model()
    evaluation_model.fit(train[FEATURES], train["label"])
    pred = evaluation_model.predict(test[FEATURES])
    report = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "candles": len(dataset),
        "train_rows": len(train),
        "purged_rows": HORIZON,
        "test_rows": len(test),
        "accuracy": float(accuracy_score(test["label"], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test["label"], pred)),
        "classification_report": classification_report(
            test["label"], pred, zero_division=0, output_dict=True
        ),
        "note": "Chronological holdout with a purged label horizon; not a backtest of net trading returns.",
    }

    # Evaluate on a held-out period first, then fit the deployable artifact on all
    # available labelled rows. Keep the write atomic so an interrupted run cannot
    # leave a partially written model file.
    final_model = _new_model()
    final_model.fit(dataset[FEATURES], dataset["label"])
    model_path = project_path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump({
        "model": final_model,
        "features": FEATURES,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "trained_through": str(df["timestamp"].iloc[-1]),
        "label_horizon": HORIZON,
        "label_threshold": THRESHOLD,
        "holdout_metrics": {
            "accuracy": report["accuracy"],
            "balanced_accuracy": report["balanced_accuracy"],
        },
    }, temporary_path)
    temporary_path.replace(model_path)

    report_path = project_path("data/training_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "classification_report"}, indent=2))
    print(f"مدل ذخیره شد: {model_path}")
    print(f"گزارش جزئی کلاس‌ها در {report_path} ذخیره شد.")
    print("این معیارها تضمین سود نیستند؛ پیش از اتکا باید بک‌تست هزینه‌محور انجام شود.")


if __name__ == "__main__":
    main()
