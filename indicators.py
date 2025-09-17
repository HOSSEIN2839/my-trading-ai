import pandas as pd
import pandas_ta as ta

def calculate_indicators(df):
    """
    محاسبه شاخص‌های تکنیکال
    df: DataFrame با ستون‌های 'open','high','low','close','volume'
    """
    # روند
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]

    # مومنتوم
    df["rsi"] = ta.rsi(df["close"], length=14)
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]
    df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    # نوسان
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    bb = ta.bbands(df["close"])
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]

    # حجم
    df["obv"] = ta.obv(df["close"], df["volume"])
    df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

    return df

# تست Skeleton با DataFrame نمونه
if __name__ == "__main__":
    data = {
        "open": [100,101,102,103,104],
        "high": [101,102,103,104,105],
        "low": [99,100,101,102,103],
        "close": [100,101,102,103,104],
        "volume": [10,15,12,13,11]
    }
    df = pd.DataFrame(data)
    df = calculate_indicators(df)
    print(df)
