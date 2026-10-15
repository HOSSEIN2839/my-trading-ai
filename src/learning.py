"""A small supervised model trained on historical candles; not a profit guarantee."""
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

FEATURES = ["close_return_1", "close_return_3", "ema_gap", "rsi", "atr_pct", "volume_change"]
DEFAULT_MODEL_PATH = Path("data/model.joblib")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["close_return_1"] = df["close"].pct_change()
    out["close_return_3"] = df["close"].pct_change(3)
    out["ema_gap"] = (df["ema50"] - df["ema200"]) / df["close"]
    out["rsi"] = df["rsi"] / 100.0
    out["atr_pct"] = df["atr"] / df["close"]
    out["volume_change"] = df["volume"].pct_change().replace([float("inf"), float("-inf")], float("nan"))
    return out.replace([float("inf"), float("-inf")], float("nan"))


def load_model(path: Path = DEFAULT_MODEL_PATH):
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        if payload.get("features") != FEATURES:
            return None
        return payload["model"]
    except Exception:
        return None


def predict(row: pd.Series, model) -> dict:
    if model is None:
        return {"decision": "HOLD", "confidence": 0.0, "reason": "مدل یادگیری هنوز آموزش داده نشده است."}
    frame = pd.DataFrame([{key: row.get(key) for key in FEATURES}])
    if frame.isna().any().any():
        return {"decision": "HOLD", "confidence": 0.0, "reason": "ویژگی‌های کافی برای پیش‌بینی وجود ندارد."}
    try:
        probs = model.predict_proba(frame[FEATURES])[0]
        idx = int(probs.argmax())
        decision = str(model.classes_[idx])
        confidence = float(probs[idx])
        # Low-confidence predictions deliberately become HOLD.
        if confidence < 0.60:
            decision = "HOLD"
        return {"decision": decision, "confidence": confidence, "reason": "پیش‌بینی مدل یادگیری ماشین"}
    except Exception:
        return {"decision": "HOLD", "confidence": 0.0, "reason": "پیش‌بینی نامعتبر؛ توقف امن."}
