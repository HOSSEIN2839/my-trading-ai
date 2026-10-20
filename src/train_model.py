"""Train and chronologically evaluate the ML model using a local historical CSV.

Run: python -m src.train_model
This is research tooling, not evidence of future profitability. It does not run the bot.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from src.learning import DEFAULT_MODEL_PATH, FEATURES, make_features
from src.market import add_indicators
from src.paths import project_path

load_dotenv()

SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "data/candles_BTCUSDT_15m.csv")
HORIZON = 3
THRESHOLD = 0.004
CLASS_LABELS = ["BUY", "HOLD", "SELL"]
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _new_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def _load_local_candles(path: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load and validate OHLCV data; never use precomputed indicator columns."""
    csv_path = project_path(path)
    if not csv_path.is_file():
        raise SystemExit(
            f"فایل دیتاست پیدا نشد: {csv_path}\n"
            "مسیر را با TRAIN_DATA_PATH تنظیم کن یا CSV را در پوشه data قرار بده."
        )

    available_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [column for column in OHLCV_COLUMNS if column not in available_columns]
    if missing:
        raise SystemExit(f"ستون‌های ضروری در CSV وجود ندارند: {', '.join(missing)}")

    optional_metadata = [column for column in ("symbol", "interval") if column in available_columns]
    df = pd.read_csv(csv_path, usecols=OHLCV_COLUMNS + optional_metadata, low_memory=False)
    if df.empty:
        raise SystemExit("فایل CSV هیچ کندلی ندارد.")

    expected_symbol = symbol.replace("/", "").strip().upper()
    if "symbol" in df.columns:
        symbols = set(df["symbol"].dropna().astype(str).str.replace("/", "", regex=False).str.strip().str.upper())
        if symbols != {expected_symbol}:
            raise SystemExit(
                f"نماد CSV با تنظیمات پروژه هماهنگ نیست. انتظار: {expected_symbol}؛ "
                f"موجود در فایل: {sorted(symbols)}"
            )

    if "interval" in df.columns:
        intervals = set(df["interval"].dropna().astype(str).str.strip())
        if intervals != {timeframe}:
            raise SystemExit(
                f"تایم‌فریم CSV با تنظیمات پروژه هماهنگ نیست. انتظار: {timeframe}؛ "
                f"موجود در فایل: {sorted(intervals)}"
            )

    numeric_timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
    if numeric_timestamps.notna().all():
        df["timestamp"] = pd.to_datetime(numeric_timestamps, unit="ms", utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise SystemExit("ستون timestamp دارای زمان نامعتبر است.")

    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    numeric = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise SystemExit("در قیمت‌ها یا حجم، مقدار خالی، غیرعددی یا نامتناهی پیدا شد.")

    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df["volume"] < 0).any():
        raise SystemExit("قیمت‌ها باید مثبت و حجم نباید منفی باشد.")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise SystemExit("دادهٔ نامعتبر OHLC پیدا شد: high از یکی از قیمت‌های کندل کمتر است.")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise SystemExit("دادهٔ نامعتبر OHLC پیدا شد: low از یکی از قیمت‌های کندل بیشتر است.")
    if df["timestamp"].duplicated().any():
        raise SystemExit("در دیتاست زمان تکراری وجود دارد.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    try:
        # Pandas 3 interprets the legacy suffix "m" differently; normalize minute
        # timeframes (for example "15m") to the unambiguous "15min" spelling.
        duration_text = timeframe[:-1] + "min" if timeframe.endswith("m") and not timeframe.endswith("min") else timeframe
        interval = pd.Timedelta(duration_text)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"تایم‌فریم قابل‌فهم نیست: {timeframe}") from exc
    if interval <= pd.Timedelta(0):
        raise SystemExit("تایم‌فریم باید مثبت باشد.")

    gaps = df["timestamp"].diff().dropna()
    irregular = gaps != interval
    if irregular.any():
        raise SystemExit(
            f"در دیتاست {int(irregular.sum())} فاصلهٔ زمانی غیر از {timeframe} پیدا شد. "
            "برای جلوگیری از برچسب‌گذاری اشتباه، آموزش متوقف شد."
        )

    # Exclude the latest candle only when it is still forming.
    if len(df) and df["timestamp"].iloc[-1] + interval > pd.Timestamp.now(tz="UTC"):
        df = df.iloc[:-1].reset_index(drop=True)

    if len(df) < 400:
        raise SystemExit(f"برای آموزش حداقل ۴۰۰ کندل بسته‌شده لازم است؛ تعداد موجود: {len(df)}")

    # Recalculate indicators from OHLCV. Any indicator columns in the CSV are ignored.
    return add_indicators(df)


def _build_supervised_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create features and forward-return labels without using future data as features."""
    features = make_features(df)
    future_return = df["close"].shift(-HORIZON) / df["close"] - 1

    labels = pd.Series("HOLD", index=df.index, dtype="object")
    labels.loc[future_return > THRESHOLD] = "BUY"
    labels.loc[future_return < -THRESHOLD] = "SELL"
    labels.loc[future_return.isna()] = None

    dataset = features.copy()
    dataset["label"] = labels
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    if len(dataset) <= 300:
        raise SystemExit("پس از پاک‌سازی ویژگی‌ها و برچسب‌ها، دادهٔ کافی برای آموزش باقی نماند.")
    return dataset


def main() -> None:
    df = _load_local_candles(TRAIN_DATA_PATH, SYMBOL, TIMEFRAME)
    dataset = _build_supervised_dataset(df)

    split = int(len(dataset) * 0.8)
    # Purge labels near the split because each label uses the next HORIZON closes.
    train = dataset.iloc[: split - HORIZON]
    test = dataset.iloc[split:]
    if len(train) < 200 or len(test) < 50:
        raise SystemExit("اندازهٔ بخش آموزش یا آزمون برای ارزیابی کافی نیست.")

    evaluation_model = _new_model()
    evaluation_model.fit(train[FEATURES], train["label"])
    predictions = evaluation_model.predict(test[FEATURES])

    report = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "data_source": str(TRAIN_DATA_PATH),
        "data_start": df["timestamp"].iloc[0].isoformat(),
        "data_end": df["timestamp"].iloc[-1].isoformat(),
        "candles_after_feature_cleanup": int(len(dataset)),
        "train_rows": int(len(train)),
        "purged_rows": HORIZON,
        "test_rows": int(len(test)),
        "label_horizon_candles": HORIZON,
        "label_threshold": THRESHOLD,
        "label_distribution": {
            str(key): int(value) for key, value in dataset["label"].value_counts().items()
        },
        "accuracy": float(accuracy_score(test["label"], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test["label"], predictions)),
        "classification_report": classification_report(
            test["label"],
            predictions,
            labels=CLASS_LABELS,
            target_names=CLASS_LABELS,
            zero_division=0,
            output_dict=True,
        ),
        "note": (
            "Chronological holdout with a purged forward-label horizon. "
            "Classification metrics are not a backtest of net trading returns. "
            "The dataset ends at the reported data_end timestamp."
        ),
    }

    # Evaluate first, then fit the saved artifact on all labelled rows.
    final_model = _new_model()
    final_model.fit(dataset[FEATURES], dataset["label"])

    model_path = project_path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model_path = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "trained_through": str(df["timestamp"].iloc[-1]),
            "label_horizon": HORIZON,
            "label_threshold": THRESHOLD,
            "data_source": str(TRAIN_DATA_PATH),
            "holdout_metrics": {
                "accuracy": report["accuracy"],
                "balanced_accuracy": report["balanced_accuracy"],
            },
        },
        temporary_model_path,
    )
    temporary_model_path.replace(model_path)

    report_path = project_path("data/training_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary_report_path.replace(report_path)

    summary = {key: value for key, value in report.items() if key != "classification_report"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"مدل ذخیره شد: {model_path}")
    print(f"گزارش ارزیابی ذخیره شد: {report_path}")
    print("این ارزیابی تضمین سود نیست؛ ربات اجرا نشده و هیچ معامله‌ای انجام نشده است.")


if __name__ == "__main__":
    main()
