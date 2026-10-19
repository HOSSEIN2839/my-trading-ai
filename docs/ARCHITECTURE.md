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
| `src/pattern_mining.py` | Mine simple market-regime hypotheses and evaluate on a later holdout | Change bot decisions, promote patterns, or place orders |
| `src/backtest.py` | Cost-aware historical simulation and financial metrics | Send real orders or silently claim ML/live performance |
| `src/ai_analyst.py` | Optional external analysis and signal combination | Become a required dependency for basic operation |
| `src/risk.py` | Position sizing and daily-loss guard | Execute orders or call external services |
| `src/paper_broker.py` | Simulate long-only spot orders, fees, stops/targets, account state | Connect to a live exchange |
| `src/memory.py` | Persist decisions, trade records, outcomes, patterns and model evaluations | Decide whether an order is allowed |
| `src/bot.py` | Orchestrate the runtime and connect components | Bypass risk checks or enable live mode |
| `src/dashboard.py` | Expose read-only JSON views and integration diagnostics over persisted state | Change account state or place orders |
| `src/paths.py` | Resolve relative data/model/report paths from the repository root | Hold business logic or trading state |
| `dashboard/index.html` | Render dashboard data from the API | Implement trading rules or write directly to SQLite |

## Integration flow

1. The bot fetches candles and derives features from the last closed candle.
2. It persists the market snapshot so monitoring and daily-risk accounting share the same price reference.
3. The broker rolls the UTC daily-loss window using marked-to-market equity and persists the last processed closed-candle ID to reduce duplicate decisions after restarts.
4. The bot evaluates exits from the last closed candle, checks the daily loss guard, and combines technical, ML, and optional AI signals. Paper entries and signal exits use the next candle open, not the signal candle close.
5. Approved paper actions go through `PaperBroker`; decisions and simulated trades are written to `TradingMemory`. Exploratory pattern mining is an offline, separate command and cannot modify the runtime signal.
6. The dashboard reads the persisted broker state and SQLite memory via `/api/status`, `/api/trades`, and `/api/decisions`; `/api/integration` checks the paper-mode flag and the presence/basic structure of the state file, SQLite tables, CSV ledger, and ML artifact.
7. The historical backtest evaluates a signal at candle close and executes it at the next candle open; it applies fees and adverse slippage on both legs, includes costs in risk sizing, and resolves stop/target ambiguity conservatively.

## State and data contracts

- Relative paths are resolved from the repository root by `src/paths.py`, independent of the caller's working directory.
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
- `src/pattern_mining.py` discovers single-feature quantile regimes using a chronological holdout and purged forward labels. Its reports are exploratory only, use a bounded one-request candle sample, omit transaction costs and do not promote patterns into bot decisions. Multiple-pattern selection can overfit; independent walk-forward periods and net-return validation remain necessary.
- Model training purges the forward-label horizon at the chronological split, evaluates a held-out period, then trains the saved artifact on all labelled history. This evaluation is classification-based, not proof of net trading profitability.
- Model artifacts are rejected if their feature list, symbol, or timeframe does not match the runtime configuration.
- Cross-file persistence (JSON broker state, CSV log, and SQLite memory) is not yet a single atomic transaction; a process crash between writes can leave the ledger and memory temporarily inconsistent.
- The first cost-aware backtest is a technical-signal baseline; the exact combined ML/AI runtime policy and walk-forward evaluation are not yet simulated.
- The current historical data fetch is limited to one exchange API request and is not yet a paginated, gap-audited multi-year dataset.
- The dashboard reports persisted runtime state and the latest saved backtest report; it does not control or guarantee that the bot process is running. `/api/integration` checks expected artifacts and basic schema, but does not prove that the bot is currently running or that a model is semantically valid.
- When no market snapshot exists, equity can only be estimated using the entry price for an open position.
