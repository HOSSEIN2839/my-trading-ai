"""Network-isolated tests for exchange OHLCV ingestion and validation."""
import pytest

from src import market


def valid_rows():
    base = 1_797_000_000_000
    return [
        [base, 100, 102, 99, 101, 10],
        [base + 900_000, 101, 103, 100, 102, 11],
        [base + 1_800_000, 102, 104, 101, 103, 12],
    ]


class FakeExchange:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, limit):
        self.calls.append((symbol, timeframe, limit))
        return self.rows


def test_fetch_candles_validates_arguments_before_network(monkeypatch):
    def forbidden_exchange(*args, **kwargs):
        raise AssertionError("invalid arguments must be rejected before network setup")

    monkeypatch.setattr(market.ccxt, "binance", forbidden_exchange)
    with pytest.raises(ValueError, match="at least 3"):
        market.fetch_candles("BTC/USDT", "15m", 2)
    with pytest.raises(ValueError, match="required"):
        market.fetch_candles("", "15m", 10)


def test_fetch_candles_returns_validated_ohlcv_and_indicators(monkeypatch):
    exchange = FakeExchange(valid_rows())
    monkeypatch.setattr(market.ccxt, "binance", lambda config: exchange)

    result = market.fetch_candles("BTC/USDT", "15m", 3)

    assert exchange.calls == [("BTC/USDT", "15m", 3)]
    assert result["timestamp"].is_monotonic_increasing
    assert {"ema50", "ema200", "rsi", "atr"}.issubset(result.columns)
    assert result.iloc[-1]["close"] == 103


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: [], "کندلی برنگرداند"),
        (lambda rows: [[*rows[0][:4], float("nan"), rows[0][5]], *rows[1:]], "غیرعددی یا نامتناهی"),
        (lambda rows: [[rows[0][0], 0, *rows[0][2:]], *rows[1:]], "قیمت یا حجم"),
        (lambda rows: [[*rows[0][:5], -1], *rows[1:]], "قیمت یا حجم"),
        (lambda rows: [[rows[0][0], 100, 99, 98, 100, 10], *rows[1:]], "ساختار OHLC"),
        (lambda rows: [[rows[0][0], 100, 102, 101, 100, 10], *rows[1:]], "ساختار OHLC"),
        (lambda rows: [rows[0], rows[0], rows[2]], "تکراری"),
        (lambda rows: [rows[1], rows[0], rows[2]], "خارج از ترتیب"),
    ],
)
def test_fetch_candles_rejects_invalid_exchange_payload(monkeypatch, mutate, message):
    rows = mutate(valid_rows())
    exchange = FakeExchange(rows)
    monkeypatch.setattr(market.ccxt, "binance", lambda config: exchange)

    with pytest.raises(RuntimeError, match=message):
        market.fetch_candles("BTC/USDT", "15m", 3)
