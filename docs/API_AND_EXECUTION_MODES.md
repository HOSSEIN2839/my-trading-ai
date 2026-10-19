# API providers and trading-mode boundary

## Purpose

Reserve clean integration points for external AI providers and exchange APIs without weakening the current paper-only guarantee. This document is a design contract, not a claim that live trading is implemented.

## Separate three independent capabilities

1. **AI provider API** — optional analysis/inference (for example Gemini or another provider). It may return a constrained opinion, but it never receives exchange credentials and never places orders.
2. **Market-data API** — public/read-only candle and ticker access. It must validate timestamps, OHLCV values, gaps, duplicates, and stale data before signals can be evaluated.
3. **Execution adapter** — paper execution today. A future exchange execution adapter would be a separate module with separate credentials and explicit approval gates. It must not be added by turning a string or UI toggle into an unrestricted order path.

A model output is untrusted input. Normalize it to a strict schema, reject invalid/ambiguous outputs, and run every candidate action through deterministic risk checks.

## Mode policy

- The only supported runtime mode today is `paper`. Missing mode defaults to paper.
- Unknown mode values and `live` must fail closed before model loading or any exchange setup.
- The dashboard is observational; it must not be able to change the process mode or submit orders.
- A future UI may show a mode selector, but selecting Live must not immediately send orders. It should require a separate server-side enablement flag, explicit confirmation, valid live credentials, a successful account/permissions check, and a fresh risk preflight. If any check fails, remain in paper mode.
- Switching to paper must stop new live orders immediately; open live positions must be reported and handled by a separately specified safe shutdown policy, never silently assumed closed.
- Live execution must remain unavailable until its own reviewed implementation and tests exist. Do not add exchange order methods to the current paper broker.

## API configuration and secrets

- Keep provider selection separate from credentials: e.g. `AI_PROVIDER=none|gemini|openai_compatible`, `AI_BASE_URL`, `AI_MODEL`, and a provider-specific secret stored only in the hosting environment.
- Keep exchange configuration separate: `EXCHANGE_ID`, public market-data settings, and future execution credentials must be distinct.
- Never store real API keys in Git, dashboard HTML, browser JavaScript, logs, error messages, SQLite, CSV, or trading decision records.
- The dashboard must never return secret values. At most, it can report configured/not configured and a sanitized connection status.
- For future exchange keys, use least privilege: disable withdrawals, grant only necessary trading permissions, use IP allowlisting where supported, and preferably use a dedicated account/subaccount.
- AI provider failure must be visible and fail closed for any signal path that requires AI; it must not silently bypass configured AI gates.

## Proposed future UI

The dashboard may eventually include an API settings view with:
- AI provider/model and connectivity status;
- exchange market-data connectivity status;
- separate status for execution credentials (never reveal the key);
- current mode clearly labelled PAPER;
- a disabled Live control with the reason it is unavailable;
- risk limits and a preflight checklist.

Settings and secrets must be handled by an authenticated server-side endpoint, not static HTML. The existing read-only dashboard contract remains in force until authentication, CSRF protection, secret storage, and authorization are designed and tested.

## Required tests before any future Live capability

- Default mode is paper; `live`, unknown, missing, malformed, or conflicting mode configuration cannot initialize live execution.
- No live order can be sent by the paper broker or by the AI provider.
- API keys never appear in responses, logs, exceptions, or persisted records.
- Mode changes are server-side, authenticated, audited, and resistant to replay/CSRF.
- Live enablement fails when any preflight check is stale, missing, or fails.
- Risk limits are enforced by deterministic code independent of model/provider output.
- Network timeout, rate limiting, invalid responses, duplicate requests, partial fills, uncertain order status, process restart, and reconciliation are covered.
- Paper and live adapters are tested separately; the live adapter is tested against an exchange sandbox/testnet before any real account is considered.
- A green test suite is necessary but does not demonstrate profitability or make real trading risk-free.

## Current status

The repository currently supports paper trading only. Gemini is optional; when its key is absent, the AI analyst reports itself disabled. The bot's startup guard rejects non-paper modes. No live execution adapter or authenticated mutable settings endpoint exists.
