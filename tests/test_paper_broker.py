from datetime import datetime, timedelta, timezone

import pytest

from src.paper_broker import PaperBroker


def make_broker(tmp_path, balance=1000, fee=0.001):
    return PaperBroker(
        balance,
        state_path=tmp_path / "paper_state.json",
        trades_path=tmp_path / "trades.csv",
        fee_rate=fee,
    )


def test_equity_marks_open_position_to_market(tmp_path):
    broker = make_broker(tmp_path)
    assert broker.buy(100, 2, 90, 120, "test")
    assert broker.equity(110) == pytest.approx(1019.8)


def test_daily_roll_uses_current_mark_price_for_open_position(tmp_path):
    broker = make_broker(tmp_path)
    assert broker.buy(100, 2, 90, 120, "test")
    broker.state["day"] = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    broker._save()

    broker.roll_day(mark_price=80)

    assert broker.state["day"] == datetime.now(timezone.utc).date().isoformat()
    # Cash after buying (799.8) plus current marked value (160), not entry value (200).
    assert broker.state["start_balance"] == pytest.approx(959.8)


def test_rejects_invalid_or_unaffordable_orders(tmp_path):
    broker = make_broker(tmp_path)
    assert not broker.buy(100, 0, 90, 120, "zero quantity")
    assert not broker.buy(100, 1, 101, 120, "stop above entry")
    assert not broker.buy(100, 1, 90, 100, "target not above entry")
    assert not broker.buy(100, 20, 90, 120, "too expensive")


def test_stop_loss_takes_priority_when_both_levels_hit(tmp_path):
    broker = make_broker(tmp_path)
    assert broker.buy(100, 1, 90, 110, "test")
    reason = broker.check_exit(low=89, high=111, close=100)
    assert reason == "stop_loss"
    assert broker.state["position"] is None
    assert broker.state["balance"] == pytest.approx(989.81)


def test_sell_all_clears_position_and_accounts_for_fee(tmp_path):
    broker = make_broker(tmp_path)
    assert broker.buy(100, 1, 90, 110, "test")
    assert broker.sell_all(110, "manual-test")
    assert broker.state["position"] is None
    assert round(broker.state["balance"], 4) == 1009.79
    assert not broker.sell_all(110, "already closed")


def test_buy_rolls_back_state_and_csv_when_state_save_fails(tmp_path, monkeypatch):
    broker = make_broker(tmp_path)
    original_balance = broker.state["balance"]
    original_csv = broker.trades_path.read_bytes()

    def fail_save():
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(broker, "_save", fail_save)
    with pytest.raises(OSError, match="simulated disk write failure"):
        broker.buy(100, 1, 90, 120, "write failure")

    assert broker.state["balance"] == original_balance
    assert broker.state["position"] is None
    assert broker.trades_path.read_bytes() == original_csv


def test_sell_rolls_back_state_and_csv_when_state_save_fails(tmp_path, monkeypatch):
    broker = make_broker(tmp_path)
    assert broker.buy(100, 1, 90, 120, "open position")
    original_balance = broker.state["balance"]
    original_position = broker.state["position"].copy()
    original_csv = broker.trades_path.read_bytes()

    def fail_save():
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(broker, "_save", fail_save)
    with pytest.raises(OSError, match="simulated disk write failure"):
        broker.sell_all(110, "write failure")

    assert broker.state["balance"] == original_balance
    assert broker.state["position"] == original_position
    assert broker.trades_path.read_bytes() == original_csv


def test_broker_rejects_non_finite_orders_and_market_prices(tmp_path):
    broker = make_broker(tmp_path)
    assert not broker.buy(float("nan"), 1, 90, 120, "nan price")
    assert not broker.buy(100, float("inf"), 90, 120, "infinite quantity")
    with pytest.raises(ValueError):
        broker.record_market_price(float("nan"), "BTC/USDT", "15m")
    with pytest.raises(ValueError):
        broker.equity(float("inf"))


def test_buy_rejects_metadata_that_overrides_accounting_fields(tmp_path):
    broker = make_broker(tmp_path)
    original_balance = broker.state["balance"]

    assert not broker.buy(
        100, 1, 90, 120, "metadata override",
        metadata={"quantity": 1000},
    )
    assert broker.state["balance"] == original_balance
    assert broker.state["position"] is None


def test_buy_rejects_non_json_or_non_finite_metadata_before_mutation(tmp_path):
    broker = make_broker(tmp_path)
    original_balance = broker.state["balance"]

    assert not broker.buy(100, 1, 90, 120, "nan metadata", metadata={"confidence": float("nan")})
    assert not broker.buy(100, 1, 90, 120, "object metadata", metadata={"bad": object()})
    assert broker.state["balance"] == original_balance
    assert broker.state["position"] is None


@pytest.mark.parametrize(
    "state",
    [
        {"balance": 1000, "start_balance": 0, "day": "2026-10-18", "position": None},
        {"balance": 1000, "start_balance": 1000, "day": "yesterday", "position": None},
        {"balance": 1000, "start_balance": 1000, "position": None},
        {"balance": 1000, "start_balance": 1000, "day": "2026-10-18",
         "position": None, "last_processed_candle": ""},
    ],
)
def test_broker_rejects_invalid_persisted_state_metadata(tmp_path, state):
    import json

    state_path = tmp_path / "paper_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError):
        PaperBroker(1000, state_path=state_path, trades_path=tmp_path / "trades.csv")


def test_broker_rejects_corrupt_persisted_position(tmp_path):
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(
        '{"balance": 1000, "start_balance": 1000, "day": "2026-10-18", '
        '"position": {"entry": 100, "quantity": NaN, "stop": 90, "target": 120}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        PaperBroker(1000, state_path=state_path, trades_path=tmp_path / "trades.csv")


def test_last_processed_candle_survives_restart(tmp_path):
    broker = make_broker(tmp_path)
    candle_id = "2026-10-19T10:15:00+00:00"
    broker.mark_candle_processed(candle_id)

    reloaded = PaperBroker(
        1000,
        state_path=tmp_path / "paper_state.json",
        trades_path=tmp_path / "trades.csv",
    )
    assert reloaded.state["last_processed_candle"] == candle_id
