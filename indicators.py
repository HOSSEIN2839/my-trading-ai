import pandas as pd
import pandas_ta as ta

def calculate_indicators(df, timeframe="5m"):
    """
    محاسبه شاخص‌ها با بررسی تایم‌فریم
    df: DataFrame با ستون‌های 'open','high','low','close','volume'
    timeframe: رشته، مثل "1m", "5m", "1h", "1d"
    """

    # تبدیل ستون‌ها به float برای جلوگیری از خطا
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    # حداقل داده مورد نیاز بسته به شاخص‌ها
    min_length = 30
    if len(df) < min_length:
        print(f"تعداد کندل‌ها ({len(df)}) کم است برای تایم‌فریم {timeframe}")
        return df

    print(f"محاسبه شاخص‌ها روی تایم‌فریم {timeframe} شروع شد...")

    # --- روند ---
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
    else:
        df["macd"] = None
        df["macd_signal"] = None

    # --- مومنتوم ---
    df["rsi"] = ta.rsi(df["close"], length=14)
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None:
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]
    df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    # --- نوسان ---
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    bb = ta.bbands(df["close"])
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]

    # --- حجم ---
    df["obv"] = ta.obv(df["close"], df["volume"])
    df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

    print("محاسبه شاخص‌ها تمام شد ✅")
    return df

# ---------- تست با داده واقعی (می‌تونی تغییر بدی) ----------
if __name__ == "__main__":
    import ccxt
    import datetime

    # صرافی Binance
    exchange = ccxt.binance()
    symbol = "BTC/USDT"
    timeframe = "5m"
    limit = 100  # تعداد کندل

    since = exchange.parse8601(str(datetime.datetime.utcnow() - datetime.timedelta(days=1)))
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)

    # محاسبه شاخص‌ها
    df = calculate_indicators(df, timeframe=timeframe)
    print(df.tail())
