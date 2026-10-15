"""AI-assisted paper-trading loop. This version cannot place live orders."""
from __future__ import annotations

import logging
import os
import time
from dotenv import load_dotenv

from src.market import fetch_candles, technical_signal
from src.ai_analyst import analyze_market, combine_signal
from src.learning import load_model, make_features, predict
from src.risk import position_size, daily_loss_limit_hit
from src.paper_broker import PaperBroker

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
    model = load_model()
    if model is None:
        raise SystemExit("مدل یادگیری پیدا نشد. ابتدا دستور python -m src.train_model را اجرا کن.")
    broker = PaperBroker(float(os.getenv("INITIAL_BALANCE", "1000")), fee_rate=FEE)
    last_candle = None
    logging.info("شروع ربات | %s | %s | حالت شبیه‌سازی؛ هیچ سفارش واقعی ارسال نمی‌شود", SYMBOL, TIMEFRAME)

    while True:
        try:
            broker.roll_day()
            candles = fetch_candles(SYMBOL, TIMEFRAME, LIMIT)
            # Do not trade on the still-forming candle.
            row = candles.iloc[-2]
            candle_id = str(row["timestamp"])
            if candle_id == last_candle:
                time.sleep(POLL_SECONDS)
                continue
            last_candle = candle_id

            price = float(row["close"])
            exit_reason = broker.check_exit(float(row["low"]), float(row["high"]), price)
            if exit_reason:
                logging.info("خروج شبیه‌سازی‌شده: %s", exit_reason)

            equity = broker.equity(price)
            if daily_loss_limit_hit(float(broker.state["start_balance"]), equity, MAX_DAILY_LOSS):
                logging.warning("حد زیان روزانه فعال شد؛ تا روز بعد معامله جدید انجام نمی‌شود.")
                time.sleep(POLL_SECONDS)
                continue

            tech = technical_signal(row)
            ai = analyze_market(SYMBOL, TIMEFRAME, row, tech)
            ml = predict(row, model)
            signal = combine_signal(tech, ai, ml)
            logging.info("قیمت=%.4f | تکنیکال=%s | Gemini=%s(%s%%) | ML=%s(%.0f%%) | نهایی=%s",
                         price, tech, ai["decision"], ai["confidence"], ml["decision"], ml["confidence"] * 100, signal)

            pos = broker.state["position"]
            atr = float(row["atr"])
            if pos and signal == "SELL":
                broker.sell_all(price, "signal_exit")
                logging.info("پوزیشن مجازی بسته شد.")
            elif not pos and signal == "BUY" and atr > 0:
                cash_balance = float(broker.state["balance"])
                stop = price - STOP_ATR * atr
                target = price + RR * (price - stop)
                qty = position_size(cash_balance, price, stop, RISK, MAX_POSITION)
                if broker.buy(price, qty, stop, target, f"Gemini:{ai['confidence']} ML:{ml['confidence']:.2f}"):
                    logging.info("خرید مجازی: qty=%.8f stop=%.4f target=%.4f", qty, stop, target)
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
