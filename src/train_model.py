"""Train and chronologically evaluate the ML model on public historical candles.

Run: python -m src.train_model
This is research tooling, not evidence of future profitability.
"""
from __future__ import annotations

import json
from pathlib import Path
import ccxt
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score

from src.market import add_indicators
from src.learning import FEATURES, make_features, DEFAULT_MODEL_PATH

SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
CANDLE_LIMIT = 1000
HORIZON = 3
THRESHOLD = 0.004  # 0.4% forward return; transaction costs still need careful validation.


def main():
    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    raw = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
    if len(raw) < 400:
        raise SystemExit("برای آموزش امن‌تر، دادهٔ تاریخی کافی دریافت نشد.")
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = add_indicators(df)
    X = make_features(df)
    future_return = df["close"].shift(-HORIZON) / df["close"] - 1
    y = pd.Series("HOLD", index=df.index)
    y.loc[future_return > THRESHOLD] = "BUY"
    y.loc[future_return < -THRESHOLD] = "SELL"
    dataset = X.copy()
    dataset["label"] = y
    dataset = dataset.replace([float("inf"), float("-inf")], float("nan")).dropna()
    # Keep the final horizon out because its future outcome is unknown.
    if len(dataset) <= 300:
        raise SystemExit("پس از پاک‌سازی، دادهٔ کافی برای آموزش باقی نماند.")
    split = int(len(dataset) * 0.8)
    train, test = dataset.iloc[:split], dataset.iloc[split:]
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=8,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    model.fit(train[FEATURES], train["label"])
    pred = model.predict(test[FEATURES])
    report = {
        "symbol": SYMBOL, "timeframe": TIMEFRAME, "candles": len(dataset),
        "train_rows": len(train), "test_rows": len(test),
        "accuracy": float(accuracy_score(test["label"], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test["label"], pred)),
        "classification_report": classification_report(test["label"], pred, zero_division=0, output_dict=True),
        "note": "Chronological holdout only; not a backtest of net trading returns. Revalidate on unseen periods before use."
    }
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "symbol": SYMBOL, "timeframe": TIMEFRAME}, DEFAULT_MODEL_PATH)
    Path("data/training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "classification_report"}, indent=2))
    print("\nگزارش جزئی کلاس‌ها در data/training_report.json ذخیره شد.")
    print("این معیارها تضمین سود نیستند؛ قبل از استفاده باید بک‌تست هزینه‌محور انجام شود.")


if __name__ == "__main__":
    main()
