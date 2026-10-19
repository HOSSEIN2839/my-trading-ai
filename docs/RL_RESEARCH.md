# Reinforcement-learning research boundary

`src/rl_environment.py` is an offline, long-only, Gym-like environment for research. It uses historical OHLC data only, fills an action selected after candle i at candle i+1 open, marks equity at that candle's close, and charges simulated fees. The reward is the change in marked equity divided by initial balance.

- Actions: HOLD, BUY, SELL; no shorting or leverage.
- Position size is capped by `max_exposure`; no action can contact an exchange.
- The environment is intentionally not imported by `src/bot.py` and does not train, deploy, or promote an agent.
- This baseline does not yet model spread, slippage, stop/target execution, gaps in candle history, funding, or exchange lot-size rules. Results must not be treated as comparable to the cost-aware backtest until these assumptions are aligned.
- Before training any RL policy, use chronological train/validation/test splits, a holdout untouched by hyperparameter tuning, transaction-cost sensitivity analysis, multiple random seeds, and comparison to a simple baseline. Do not auto-promote a learned policy into the paper bot without a separate reviewed integration and tests.
