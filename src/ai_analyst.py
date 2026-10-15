"""Gemini market-analysis gate. The model cannot bypass risk rules."""
from __future__ import annotations

import json
import os
import requests


def analyze_market(symbol: str, timeframe: str, row, technical: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"decision": "HOLD", "confidence": 0, "reason": "GEMINI_API_KEY تنظیم نشده؛ برای ایمنی معامله‌ای انجام نشد."}

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
        response = requests.post(url, params={"key": api_key}, json=payload, timeout=25)
        response.raise_for_status()
        body = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(body)
        decision = str(result.get("decision", "HOLD")).upper()
        confidence = int(result.get("confidence", 0))
        if decision not in {"BUY", "SELL", "HOLD"} or not 0 <= confidence <= 100:
            raise ValueError("AI returned invalid decision")
        reason = str(result.get("reason", "بدون توضیح"))[:300]
        return {"decision": decision, "confidence": confidence, "reason": reason}
    except Exception as exc:
        return {"decision": "HOLD", "confidence": 0, "reason": f"خطای تحلیل AI؛ توقف امن: {type(exc).__name__}"}


def combine_signal(technical: str, ai: dict) -> str:
    """Require agreement for entries; AI/technical sell can close a paper position."""
    decision = ai.get("decision", "HOLD")
    confidence = ai.get("confidence", 0)
    if technical == "BUY" and decision == "BUY" and confidence >= 65:
        return "BUY"
    if technical == "SELL" or decision == "SELL":
        return "SELL"
    return "HOLD"
