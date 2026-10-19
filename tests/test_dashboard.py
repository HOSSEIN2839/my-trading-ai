import json

from src.dashboard import create_app
from src.memory import TradingMemory


def test_dashboard_is_read_only_and_exposes_paper_status(tmp_path):
    state_path = tmp_path / "paper_state.json"
    memory_path = tmp_path / "memory.sqlite3"
    state_path.write_text(json.dumps({
        "balance": 800.0,
        "start_balance": 1000.0,
        "day": "2026-10-18",
        "position": {"entry": 100.0, "quantity": 2.0, "stop": 90.0, "target": 120.0},
        "last_market": {"price": 110.0, "symbol": "BTC/USDT", "timeframe": "15m", "timestamp": "2026-10-18T10:00:00+00:00"},
    }), encoding="utf-8")
    with TradingMemory(memory_path) as memory:
        memory.record_decision(symbol="BTC/USDT", timeframe="15m", decision="BUY", price=100, confidence=0.8)
        memory.record_trade(symbol="BTC/USDT", side="BUY", price=100, quantity=2, fee=0.2)

    app = create_app(state_path=state_path, memory_path=memory_path)
    app.config["TESTING"] = True
    client = app.test_client()

    health = client.get("/api/health").get_json()
    status = client.get("/api/status").get_json()

    assert health == {"ok": True, "mode": "paper", "read_only": True}
    assert status["mode"] == "paper"
    assert status["equity"] == 1020.0
    assert status["market"]["symbol"] == "BTC/USDT"
    assert status["memory"]["decisions"] == 1
    assert status["memory"]["trades"] == 1
    assert client.get("/").status_code == 200
    assert client.post("/api/status").status_code == 405


def test_dashboard_returns_empty_lists_when_memory_is_missing(tmp_path):
    app = create_app(state_path=tmp_path / "missing.json", memory_path=tmp_path / "missing.sqlite3")
    client = app.test_client()
    assert client.get("/api/trades").get_json() == {"items": []}
    assert client.get("/api/decisions").get_json() == {"items": []}
