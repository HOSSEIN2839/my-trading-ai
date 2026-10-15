from src.risk import position_size, daily_loss_limit_hit


def test_position_size_respects_risk_and_position_cap():
    qty = position_size(1000, 100, 90, 0.01, 0.25)
    assert qty > 0
    assert qty <= 2.5
    assert qty * (100 - 90) <= 10


def test_invalid_stop_returns_zero():
    assert position_size(1000, 100, 101, 0.01, 0.25) == 0


def test_daily_loss_guard():
    assert daily_loss_limit_hit(1000, 970, 0.02)
    assert not daily_loss_limit_hit(1000, 990, 0.02)
