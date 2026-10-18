import pytest

from src.memory import TradingMemory


def test_memory_persists_decisions_and_context(tmp_path):
    db = tmp_path / "memory.sqlite3"
    with TradingMemory(db) as memory:
        decision_id = memory.record_decision(
            symbol="BTC/USDT", timeframe="15m", decision="BUY",
            price=65000, confidence=0.82, context={"rsi": 54, "source": "paper"},
        )
        memory.annotate_decision(decision_id, {"return_pct": 0.7, "closed": True})

    with TradingMemory(db) as memory:
        rows = memory.recent_decisions()
        assert len(rows) == 1
        assert rows[0]["decision"] == "BUY"
        assert rows[0]["context"]["rsi"] == 54
        assert rows[0]["outcome"]["closed"] is True


def test_memory_records_trades_patterns_and_model_evaluations(tmp_path):
    with TradingMemory(tmp_path / "memory.sqlite3") as memory:
        memory.record_trade(symbol="BTC/USDT", side="BUY", price=100, quantity=0.1, fee=0.01)
        memory.remember_pattern(
            pattern_key="pattern-001", description="test pattern",
            evidence_count=10, success_count=6, average_return=0.002,
        )
        memory.save_model_version(
            version="v1", artifact_path="data/model-v1.joblib", metrics={"net_return": 0.03}
        )
        memory.record_evaluation(
            model_version="v1", metrics={"max_drawdown": 0.05}, accepted=True
        )
        assert memory.summary() == {
            "decisions": 0, "trades": 1, "patterns": 1, "evaluations": 1, "model_versions": 1
        }


def test_memory_rejects_invalid_decision_and_pattern_counts(tmp_path):
    with TradingMemory(tmp_path / "memory.sqlite3") as memory:
        with pytest.raises(ValueError):
            memory.record_decision(symbol="BTC/USDT", timeframe="15m", decision="MOON")
        with pytest.raises(ValueError):
            memory.remember_pattern(
                pattern_key="bad", description="bad", evidence_count=2, success_count=3
            )


def test_memory_requires_positive_trade_size(tmp_path):
    with TradingMemory(tmp_path / "memory.sqlite3") as memory:
        with pytest.raises(ValueError):
            memory.record_trade(symbol="BTC/USDT", side="BUY", price=0, quantity=1)
