"""Safety regression tests for the orchestration entry point.

These tests deliberately stop before the network/polling loop; they verify that
unsafe modes and invalid configuration fail closed before model or broker setup.
"""
import pytest

import src.bot as bot


def test_bot_refuses_live_mode_before_loading_model(monkeypatch):
    monkeypatch.setenv("MODE", "live")

    def forbidden_model_load(**kwargs):
        raise AssertionError("model loading must not occur in live mode")

    monkeypatch.setattr(bot, "load_model", forbidden_model_load)

    with pytest.raises(SystemExit, match="فقط MODE=paper"):
        bot.main()


def test_bot_refuses_unknown_mode_before_loading_model(monkeypatch):
    monkeypatch.setenv("MODE", "production")

    def forbidden_model_load(**kwargs):
        raise AssertionError("model loading must not occur for unknown mode")

    monkeypatch.setattr(bot, "load_model", forbidden_model_load)

    with pytest.raises(SystemExit, match="فقط MODE=paper"):
        bot.main()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("RISK_PER_TRADE", "nan"),
        ("MAX_DAILY_LOSS", "inf"),
        ("MAX_POSITION_FRACTION", "nan"),
        ("ATR_STOP_MULTIPLIER", "inf"),
        ("REWARD_RISK", "nan"),
        ("TAKER_FEE", "inf"),
        ("INITIAL_BALANCE", "nan"),
    ],
)
def test_bot_rejects_non_finite_risk_configuration_before_model_load(
    monkeypatch, variable, value
):
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setenv(variable, value)

    def forbidden_model_load(**kwargs):
        raise AssertionError("invalid configuration must fail before model loading")

    monkeypatch.setattr(bot, "load_model", forbidden_model_load)

    with pytest.raises(SystemExit, match="مقادیر محدود و معتبر"):
        bot.main()


def test_bot_paper_mode_fails_closed_when_model_is_missing(monkeypatch):
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setattr(bot, "load_model", lambda **kwargs: None)

    with pytest.raises(SystemExit, match="ابتدا دستور"):
        bot.main()
