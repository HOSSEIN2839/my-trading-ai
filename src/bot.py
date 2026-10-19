"""AI-assisted paper-trading loop. This version cannot place live orders."""
from __future__ import annotations

import logging
import math
import os
import time
from dotenv import load_dotenv

from src.market import fetch_candles, technical_signal
from src.ai_analyst import analyze_market, combine_signal
from src.learning import load_model, make_features, predict, FEATURES
from src.risk import position_size, daily_loss_limit_hit
from src.paper_broker import PaperBroker
from src.memory import TradingMemory

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
LIMIT = int(os.getenv("CANDLE_LIMIT", "250"))
POLL_SECONDS = max(15, int(os.getenv("POLL_SECONDS", "60")))
RISK = float(os.getenv("RISK_PER_TRADE", "0.005"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "0.02"))
MAX_POSITION = float(os.getenv("MAX_POSITION_FRACTION", "0.25"))
STOP_ATR = float(os.getenv("ATR_STOP_MULTIPLIER", "2.0"))
RR = float(os.getenv("REWARD_RISK", "2.0"))
FEE = float(os.getenv("TAKER_FEE", "0.001"))


def main():
    if os.getenv("MODE", "paper").lower() != "paper":
        raise SystemExit("این نسخه فقط MODE=paper را پشتیبانی می‌کند؛ معامله واقعی غیرفعال است.")
    config_values = (RISK, MAX_DAILY_LOSS, MAX_POSITION, STOP_ATR, RR, FEE, float(os.getenv("INITIAL_BALANCE", "1000")))
    if not all(math.isfinite(value) for value in config_values):
        raise SystemExit("تنظیمات عددی باید مقادیر محدود و معتبر داشته باشند.")
    if not 0 < RISK <= 1 or not 0 < MAX_POSITION <= 1 or not 0 < MAX_DAILY_LOSS <= 1:
        raise SystemExit("RISK_PER_TRADE، MAX_POSITION_FRACTION و MAX_DAILY_LOSS باید بین صفر و یک باشند.")
    if STOP_ATR <= 0 or RR <= 0 or not 0 <= FEE < 1 or config_values[-1] <= 0:
        raise SystemExit("تنظیمات حد ضرر، نسبت سود به ریسک، کارمزد یا موجودی اولیه نامعتبر است.")
    model = load_model(symbol=SYMBOL, timeframe=TIMEFRAME)
    if model is None:
        raise SystemExit("مدل یادگیری پیدا نشد. ابتدا دستور python -m src.train_model را اجرا کن.")
    broker = PaperBroker(
        float(os.getenv("INITIAL_BALANCE", "1000")),
        state_path=os.getenv("PAPER_STATE_PATH", "data/paper_state.json"),
        trades_path=os.getenv("PAPER_TRADES_PATH", "data/trades.csv"),
        fee_rate=FEE,
    )
    memory = TradingMemory(os.getenv("MEMORY_DB_PATH", "data/memory.sqlite3"))
    last_candle = broker.state.get("last_processed_candle")
    logging.info("شروع ربات | %s | %s | حالت شبیه‌سازی؛ هیچ سفارش واقعی ارسال نمی‌شود", SYMBOL, TIMEFRAME)

    while True:
        try:
            candles = fetch_candles(SYMBOL, TIMEFRAME, LIMIT)
            features = make_features(candles)
            row = candles.iloc[-2].copy()
            # Signals use the last closed candle; paper orders use the next candle open,
            # matching the backtest execution contract and avoiding close-price lookahead.
            execution_price = float(candles.iloc[-1]["open"])
            for feature in FEATURES:
                row[feature] = features.iloc[-2][feature]
            price = float(row["close"])
            broker.record_market_price(price, SYMBOL, TIMEFRAME, str(row["timestamp"]))
            # Roll the UTC daily-loss window against current marked-to-market equity,
            # including an open position valued at the latest closed candle.
            broker.roll_day(price)
            candle_id = str(row["timestamp"])
            position_before_exit = dict(broker.state["position"]) if broker.state["position"] else None
            exit_reason = broker.check_exit(float(row["low"]), float(row["high"]), price)
            if exit_reason:
                logging.info("خروج شبیه‌سازی‌شده: %s", exit_reason)
                if position_before_exit:
                    exit_price = float(position_before_exit["stop"] if exit_reason == "stop_loss" else position_before_exit["target"])
                    exit_qty = float(position_before_exit["quantity"])
                    entry_price = float(position_before_exit["entry"])
                    realized_pnl = (exit_price - entry_price) * exit_qty - (entry_price * exit_qty + exit_price * exit_qty) * FEE
                    memory.record_trade(symbol=SYMBOL, side="SELL", price=exit_price, quantity=exit_qty,
                                        fee=exit_price * exit_qty * FEE, realized_pnl=realized_pnl, reason=exit_reason,
                                        details={"entry": entry_price, "source": "paper_broker"})
                    if position_before_exit.get("decision_id") is not None:
                        memory.annotate_decision(int(position_before_exit["decision_id"]),
                                                 {"exit_reason": exit_reason, "exit_price": exit_price,
                                                  "realized_pnl_after_fees": realized_pnl,
                                                  "return_pct_after_fees": realized_pnl / (entry_price * exit_qty)})

            # Exit checks run even for an already-seen candle so a position left open
            # during downtime can still be closed if its latest completed candle hit a stop.
            if candle_id == last_candle:
                time.sleep(POLL_SECONDS)
                continue
            if last_candle is None:
                # On first launch, do not retroactively fill a signal at a candle open
                # that may already be minutes old. Begin entries from the next cycle.
                broker.mark_candle_processed(candle_id)
                last_candle = candle_id
                logging.info("شروع امن: کندل فعلی ثبت شد؛ تصمیم جدید از کندل بسته‌شدهٔ بعدی.")
                time.sleep(POLL_SECONDS)
                continue

            equity = broker.equity(price)
            if daily_loss_limit_hit(float(broker.state["start_balance"]), equity, MAX_DAILY_LOSS):
                logging.warning("حد زیان روزانه فعال شد؛ تا روز بعد معامله جدید انجام نمی‌شود.")
                broker.mark_candle_processed(candle_id)
                last_candle = candle_id
                time.sleep(POLL_SECONDS)
                continue

            tech = technical_signal(row)
            ai = analyze_market(SYMBOL, TIMEFRAME, row, tech)
            ml = predict(row, model)
            signal = combine_signal(tech, ai, ml)
            decision_id = memory.record_decision(
                symbol=SYMBOL, timeframe=TIMEFRAME, decision=signal, price=price,
                confidence=float(ml.get("confidence", 0.0)),
                context={"technical": tech, "ml": ml, "ai_enabled": bool(ai.get("enabled", False)),
                         "ai_decision": ai.get("decision", "HOLD"), "features": {k: row.get(k) for k in FEATURES}},
                timestamp=row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
            )
            ai_status = f"Gemini={ai['decision']}({ai['confidence']}%)" if ai.get("enabled") else "Gemini=خاموش (اختیاری)"
            logging.info("قیمت=%.4f | تکنیکال=%s | %s | ML=%s(%.0f%%) | نهایی=%s",
                         price, tech, ai_status, ml["decision"], ml["confidence"] * 100, signal)

            pos = broker.state["position"]
            atr = float(row["atr"])
            if pos and signal == "SELL":
                closed_position = dict(pos)
                if broker.sell_all(execution_price, "signal_exit"):
                    exit_qty = float(closed_position["quantity"])
                    entry_price = float(closed_position["entry"])
                    realized_pnl = (execution_price - entry_price) * exit_qty - (entry_price * exit_qty + execution_price * exit_qty) * FEE
                    memory.record_trade(symbol=SYMBOL, side="SELL", price=execution_price,
                                        quantity=exit_qty, fee=execution_price * exit_qty * FEE, realized_pnl=realized_pnl,
                                        reason="signal_exit", details={"entry": entry_price, "source": "paper_broker"})
                    if closed_position.get("decision_id") is not None:
                        memory.annotate_decision(int(closed_position["decision_id"]),
                                                 {"exit_reason": "signal_exit", "exit_price": execution_price,
                                                  "realized_pnl_after_fees": realized_pnl,
                                                  "return_pct_after_fees": realized_pnl / (entry_price * exit_qty)})
                logging.info("پوزیشن مجازی بسته شد.")
            elif not pos and signal == "BUY" and atr > 0:
                cash_balance = float(broker.state["balance"])
                stop = execution_price - STOP_ATR * atr
                target = execution_price + RR * (execution_price - stop)
                qty = position_size(cash_balance, execution_price, stop, RISK, MAX_POSITION)
                if broker.buy(execution_price, qty, stop, target, f"AI_enabled:{ai.get('enabled', True)} ML:{ml['confidence']:.2f}", metadata={"decision_id": decision_id}):
                    memory.record_trade(symbol=SYMBOL, side="BUY", price=execution_price, quantity=qty,
                                        fee=execution_price * qty * FEE, reason="model_signal",
                                        details={"stop": stop, "target": target, "ml_confidence": ml["confidence"], "source": "paper_broker"})
                    logging.info("خرید مجازی: fill=%.4f qty=%.8f stop=%.4f target=%.4f", execution_price, qty, stop, target)
                else:
                    logging.info("خرید مجازی رد شد (موجودی/اندازه/پوزیشن).")
        except KeyboardInterrupt:
            logging.info("ربات متوقف شد.")
            break
        except Exception:
            logging.exception("خطای چرخه؛ ربات پس از وقفه دوباره تلاش می‌کند.")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
