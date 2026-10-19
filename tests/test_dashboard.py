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
    assert client.get("/api/decisions?limit=invalid").get_json() == {"items": []}
    assert client.get("/api/backtest").get_json()["available"] is False



def test_dashboard_survives_malformed_persisted_state(tmp_path):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(json.dumps({
        "balance": "not-a-number",
        "position": {"entry": "invalid", "quantity": "NaN", "stop": 90, "target": 120},
        "last_market": [],
    }), encoding="utf-8")
    app = create_app(state_path=state_path, memory_path=tmp_path / "missing.sqlite3")
    response = app.test_client().get("/api/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state_available"] is True
    assert payload["balance"] == 0.0
    assert payload["equity"] == 0.0
    assert payload["market"] == {}


def test_dashboard_exposes_saved_backtest_metrics(tmp_path, monkeypatch):
    report_path = tmp_path / "backtest_report.json"
    report_path.write_text(json.dumps({
        "metrics": {
            "net_return_pct": 4.25,
            "max_drawdown_pct": 2.5,
            "win_rate_pct": 60.0,
            "trades": 10,
        },
        "trades": [],
        "equity_curve": [],
    }), encoding="utf-8")
    monkeypatch.setenv("BACKTEST_REPORT_PATH", str(report_path))
    app = create_app(state_path=tmp_path / "missing.json", memory_path=tmp_path / "missing.sqlite3")

    payload = app.test_client().get("/api/backtest").get_json()
    assert payload["available"] is True
    assert payload["metrics"]["net_return_pct"] == 4.25
    assert payload["metrics"]["max_drawdown_pct"] == 2.5
    assert payload["metrics"]["win_rate_pct"] == 60.0
    assert payload["metrics"]["trades"] == 10


def test_integration_diagnostics_report_ready_components(tmp_path, monkeypatch):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(json.dumps({"balance": 1000, "start_balance": 1000, "position": None}), encoding="utf-8")
    memory_path = tmp_path / "memory.sqlite3"
    with TradingMemory(memory_path):
        pass
    trades_path = tmp_path / "trades.csv"
    trades_path.write_text("time_utc,side,price,quantity,fee,balance,reason\n", encoding="utf-8")
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"test-artifact")
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setenv("PAPER_TRADES_PATH", str(trades_path))
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    app = create_app(state_path=state_path, memory_path=memory_path)
    payload = app.test_client().get("/api/integration").get_json()

    assert payload["overall"] == "ok"
    assert {item["id"]: item["status"] for item in payload["checks"]} == {
        "mode": "ok", "state": "ok", "memory": "ok", "ledger": "ok", "ledger_sync": "ok", "model": "ok"
    }


def test_integration_diagnostics_warn_when_runtime_artifacts_are_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setenv("PAPER_TRADES_PATH", str(tmp_path / "missing.csv"))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    app = create_app(state_path=tmp_path / "missing.json", memory_path=tmp_path / "missing.sqlite3")

    payload = app.test_client().get("/api/integration").get_json()

    assert payload["overall"] == "warning"
    checks = {item["id"]: item["status"] for item in payload["checks"]}
    assert checks["mode"] == "ok"
    assert checks["state"] == "missing"
    assert checks["memory"] == "missing"
    assert checks["ledger"] == "missing"
    assert checks["ledger_sync"] == "missing"
    assert checks["model"] == "missing"


def test_integration_diagnostics_flag_corrupt_state_as_error(tmp_path, monkeypatch):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("MODE", "paper")
    app = create_app(state_path=state_path, memory_path=tmp_path / "missing.sqlite3")

    payload = app.test_client().get("/api/integration").get_json()
    checks = {item["id"]: item for item in payload["checks"]}

    assert payload["overall"] == "error"
    assert checks["state"]["status"] == "error"
    assert "JSON" in checks["state"]["detail"]


def test_integration_diagnostics_detect_trade_ledger_mismatch(tmp_path, monkeypatch):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(json.dumps({"balance": 1000, "start_balance": 1000, "position": None}), encoding="utf-8")
    memory_path = tmp_path / "memory.sqlite3"
    with TradingMemory(memory_path):
        pass
    trades_path = tmp_path / "trades.csv"
    trades_path.write_text(
        "time_utc,side,price,quantity,fee,balance,reason\\n"
        "2026-10-19T10:00:00+00:00,BUY,100,1,0.1,899.9,test\\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setenv("PAPER_TRADES_PATH", str(trades_path))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))

    app = create_app(state_path=state_path, memory_path=memory_path)
    payload = app.test_client().get("/api/integration").get_json()
    checks = {item["id"]: item for item in payload["checks"]}

    assert payload["overall"] == "warning"
    assert checks["ledger_sync"]["status"] == "warning"
    assert "CSV" in checks["ledger_sync"]["detail"]
    assert "SQLite" in checks["ledger_sync"]["detail"]
