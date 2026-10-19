# Architecture and integration contracts

## Goals

- Keep the trading system paper-only. No module in this architecture sends real exchange orders.
- Separate market data, decisions, risk controls, simulated execution, durable memory, and presentation.
- Keep the dashboard read-only: it observes state through HTTP endpoints and never imports the trading loop or mutates the broker.
- Make changes testable at module boundaries before adding more autonomous behavior.

## Runtime boundaries

| Component | Responsibility | Must not do |
|---|---|---|
| `src/market.py` | Fetch public candles and calculate deterministic indicators | Place orders or persist model decisions |
| `src/learning.py` | Feature creation and model inference | Fetch market data or change broker state |
| `src/backtest.py` | Cost-aware historical simulation and financial metrics | Send real orders or silently claim ML/live performance |
| `src/ai_analyst.py` | Optional external analysis and signal combination | Become a required dependency for basic operation |
| `src/risk.py` | Position sizing and daily-loss guard | Execute orders or call external services |
| `src/paper_broker.py` | Simulate long-only spot orders, fees, stops/targets, account state | Connect to a live exchange |
| `src/memory.py` | Persist decisions, trade records, outcomes, patterns and model evaluations | Decide whether an order is allowed |
| `src/bot.py` | Orchestrate the runtime and connect components | Bypass risk checks or enable live mode |
| `src/dashboard.py` | Expose read-only JSON views over persisted state | Change account state or place orders |
| `dashboard/index.html` | Render dashboard data from the API | Implement trading rules or write directly to SQLite |

## Integration flow

1. The bot fetches candles and derives features from the last closed candle.
2. It persists the market snapshot so monitoring and daily-risk accounting share the same price reference.
3. The broker rolls the UTC daily-loss window using marked-to-market equity.
4. The bot evaluates existing stop/target exits, checks the daily loss guard, and combines technical, ML, and optional AI signals.
5. Approved paper actions go through `PaperBroker`; decisions and simulated trades are written to `TradingMemory`.
6. The dashboard reads the persisted broker state and SQLite memory via `/api/status`, `/api/trades`, and `/api/decisions`.
7. The historical backtest evaluates a signal at candle close and executes it at the next candle open; it applies fees and adverse slippage on both legs and resolves stop/target ambiguity conservatively.

## State and data contracts

- Broker state defaults to `data/paper_state.json`; override with `PAPER_STATE_PATH`.
- Broker CSV log defaults to `data/trades.csv`; override with `PAPER_TRADES_PATH`.
- SQLite memory defaults to `data/memory.sqlite3`; override with `MEMORY_DB_PATH`.
- Dashboard listens on loopback `127.0.0.1:8765` by default. `DASHBOARD_PORT` can change the local port.
- The dashboard's `mode` and `read_only` fields are descriptive safety indicators, not a substitute for the bot's own `MODE=paper` guard.

## Change rules

1. If a function signature or persisted state shape changes, update all callers and tests in the same change.
2. Add or update unit tests for accounting, boundary conditions, and failure paths.
3. Keep API responses stable or version them before breaking the dashboard.
4. Do not auto-promote a newly trained model without out-of-sample evaluation and explicit acceptance criteria.
5. Keep the dashboard read-only and bound to loopback unless a separately reviewed authentication and deployment design is implemented.
6. A passing unit test suite is necessary but not proof of profitability. Backtesting must account for fees, slippage, candle timing, and lookahead leakage.

## Current known limitations

- The model does not yet retrain automatically from realized trade outcomes.
- The first cost-aware backtest is a technical-signal baseline; the exact combined ML/AI runtime policy and walk-forward evaluation are not yet simulated.
- The current historical data fetch is limited to one exchange API request and is not yet a paginated, gap-audited multi-year dataset.
- The dashboard reports persisted runtime state; it does not control or guarantee that the bot process is running.
- When no market snapshot exists, equity can only be estimated using the entry price for an open position.
