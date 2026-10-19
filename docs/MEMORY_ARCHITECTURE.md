# Persistent trading memory

The `src.memory.TradingMemory` module provides local SQLite persistence for the
paper-trading research agent. It is independent of Gemini, ChatGPT, and other
hosted AI services.

It stores:
- timestamped BUY/SELL/HOLD decisions and their context/outcomes;
- simulated trades, fees and optional realized P&L;
- candidate patterns with evidence counts and observed returns;
- model evaluation records and model-version metadata.

Example:

```python
from src.memory import TradingMemory

with TradingMemory("data/memory.sqlite3") as memory:
    decision_id = memory.record_decision(
        symbol="BTC/USDT", timeframe="15m", decision="HOLD",
        price=65000, confidence=0.72, context={"volatility": 0.015},
    )
    memory.annotate_decision(decision_id, {"evaluated": False})
    print(memory.summary())
```

The database is created automatically and survives restarts. Keep it out of
public commits if it contains private information. SQLite is suitable for a
single-process prototype; concurrent multi-worker deployments may need a
dedicated database.

Important: this is the persistence layer, not a complete learning algorithm.
Recording a pattern does not prove that it predicts future returns. Candidate
models still need chronological, out-of-sample evaluation with trading fees and
slippage before promotion. No live order execution is implemented here.
