from src.ai_analyst import analyze_market, combine_signal


def test_gemini_is_optional_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = analyze_market("BTC/USDT", "15m", {}, "BUY")
    assert result["enabled"] is False
    assert result["decision"] == "HOLD"


def test_local_consensus_can_buy_without_gemini():
    ai = {"enabled": False, "decision": "HOLD", "confidence": 0}
    ml = {"decision": "BUY", "confidence": 0.75}
    assert combine_signal("BUY", ai, ml) == "BUY"


def test_local_disagreement_does_not_buy_without_gemini():
    ai = {"enabled": False, "decision": "HOLD", "confidence": 0}
    ml = {"decision": "HOLD", "confidence": 0.75}
    assert combine_signal("BUY", ai, ml) == "HOLD"


def test_enabled_gemini_must_agree_for_buy():
    ai = {"enabled": True, "decision": "HOLD", "confidence": 0}
    ml = {"decision": "BUY", "confidence": 0.75}
    assert combine_signal("BUY", ai, ml) == "HOLD"


def test_sell_signal_still_closes_paper_position():
    ai = {"enabled": False, "decision": "HOLD", "confidence": 0}
    ml = {"decision": "HOLD", "confidence": 0.5}
    assert combine_signal("SELL", ai, ml) == "SELL"
