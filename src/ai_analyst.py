"""Optional Gemini market-analysis gate for paper trading."""
from __future__ import annotations

import json
import os
import requests


def analyze_market(symbol: str, timeframe: str, row, technical: str) -> dict:
    """Return an optional AI opinion. No API key means the local system can still operate."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "enabled": False, "decision": "HOLD", "confidence": 0,
            "reason": "Gemini تنظیم نشده؛ تصمیم با مدل محلی و سیگنال تکنیکال گرفته می‌شود.",
        }

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text":
            "You are a cautious crypto market analyst, not a financial adviser. "
            "Return ONLY JSON with keys decision (BUY, SELL, HOLD), confidence (integer 0-100), reason (short string). "
            "Use only supplied data. Do not invent news or prices. This is long-only spot trading; never recommend shorting. "
            "If uncertain, output HOLD. Do not override risk controls."
        }]},
        "contents": [{"parts": [{"text": json.dumps({
            "symbol": symbol, "timeframe": timeframe, "technical_signal": technical,
            "close": float(row["close"]), "ema50": float(row["ema50"]),
            "ema200": float(row["ema200"]), "rsi": float(row["rsi"]),
            "atr": float(row["atr"]),
        })}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    try:
        response = requests.post(url, headers={"x-goog-api-key": api_key}, json=payload, timeout=25)
        response.raise_for_status()
        body = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(body)
        decision = str(result.get("decision", "HOLD")).upper()
        confidence = int(result.get("confidence", 0))
        if decision not in {"BUY", "SELL", "HOLD"} or not 0 <= confidence <= 100:
            raise ValueError("AI returned invalid decision")
        return {
            "enabled": True, "decision": decision, "confidence": confidence,
            "reason": str(result.get("reason", "بدون توضیح"))[:300],
        }
    except Exception as exc:
        # If configured AI fails, fail closed rather than silently bypassing it.
        return {
            "enabled": True, "decision": "HOLD", "confidence": 0,
            "reason": f"خطای تحلیل AI؛ توقف امن: {type(exc).__name__}",
        }


def combine_signal(technical: str, ai: dict, ml: dict) -> str:
    """Use local technical+ML agreement when Gemini is absent; otherwise require all three."""
    ml_decision = ml.get("decision", "HOLD")
    ml_confidence = float(ml.get("confidence", 0))
    ai_enabled = bool(ai.get("enabled", True))

    if technical == "SELL" or ml_decision == "SELL" or (
        ai_enabled and ai.get("decision") == "SELL"
    ):
        return "SELL"

    local_buy = technical == "BUY" and ml_decision == "BUY" and ml_confidence >= 0.60
    if not local_buy:
        return "HOLD"

    if not ai_enabled:
        return "BUY"

    if ai.get("decision") == "BUY" and int(ai.get("confidence", 0)) >= 65:
        return "BUY"
    return "HOLD"
