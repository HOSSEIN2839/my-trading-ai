import pandas as pd
import pytest

from src.rl_environment import PaperTradingEnv


def candles():
    return pd.DataFrame({
        "open": [100, 100, 110],
        "high": [102, 112, 113],
        "low": [98, 99, 108],
        "close": [100, 110, 111],
    })


def test_buy_fills_next_open_and_rewards_mark_to_market():
    env = PaperTradingEnv(candles(), initial_balance=1000, fee_rate=0, max_exposure=0.5)
    obs, reward, done, info = env.step("BUY")
    assert info["fill_price"] == 100
    assert info["mark_price"] == 110
    assert info["executed_action"] == "BUY"
    assert obs["equity"] == pytest.approx(1050)
    assert reward == pytest.approx(0.05)
    assert not done


def test_fee_is_charged_on_entry_and_exit():
    env = PaperTradingEnv(candles(), initial_balance=1000, fee_rate=0.01, max_exposure=0.5)
    _, _, _, buy = env.step("BUY")
    assert buy["fees_paid"] > 0
    _, _, done, sell = env.step("SELL")
    assert done
    assert sell["fees_paid"] > 0
    assert sell["quantity"] == 0
    assert sell["equity"] < 1050


def test_environment_rejects_invalid_data_and_config():
    with pytest.raises(ValueError):
        PaperTradingEnv(candles().iloc[:1])
    invalid = candles()
    invalid.loc[0, "close"] = float("nan")
    with pytest.raises(ValueError):
        PaperTradingEnv(invalid)
    with pytest.raises(ValueError):
        PaperTradingEnv(candles(), fee_rate=float("inf"))


def test_invalid_actions_and_step_after_end_are_rejected():
    env = PaperTradingEnv(candles())
    with pytest.raises(ValueError):
        env.step("SHORT")
    env.step("HOLD")
    _, _, done, _ = env.step("HOLD")
    assert done
    with pytest.raises(RuntimeError):
        env.step("HOLD")


def test_buy_then_sell_returns_to_cash_and_has_finite_reward():
    env = PaperTradingEnv(candles(), initial_balance=1000, fee_rate=0.001)
    _, r1, _, _ = env.step(1)
    obs, r2, _, info = env.step(2)
    assert obs["position_fraction"] == 0
    assert info["quantity"] == 0
    assert all(map(lambda x: abs(x) < float("inf"), [r1, r2]))



def test_position_exposure_cap_prevents_leverage_and_duplicate_entries():
    env = PaperTradingEnv(candles(), initial_balance=1000, fee_rate=0, max_exposure=0.25)
    _, _, _, first = env.step("BUY")
    assert first["quantity"] * first["fill_price"] <= 250
    held_quantity = first["quantity"]

    _, _, _, second = env.step("BUY")
    assert second["executed_action"] == "HOLD"
    assert second["quantity"] == pytest.approx(held_quantity)
    assert second["cash"] >= 0


def test_reset_restores_initial_account_and_position():
    env = PaperTradingEnv(candles(), initial_balance=1000, fee_rate=0, max_exposure=0.5)
    env.step("BUY")
    observation = env.reset()

    assert observation["index"] == 0
    assert observation["equity"] == pytest.approx(1000)
    assert observation["position_fraction"] == 0
    assert env.cash == pytest.approx(1000)
    assert env.quantity == 0
    assert not env.done


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("open", 0),
        ("high", 99),
        ("low", 101),
        ("close", float("inf")),
    ],
)
def test_environment_rejects_invalid_ohlc_structure(column, value):
    invalid = candles()
    if value == float("inf"):
        invalid[column] = invalid[column].astype(float)
    invalid.loc[0, column] = value
    with pytest.raises(ValueError):
        PaperTradingEnv(invalid)


def test_boolean_is_not_accepted_as_numeric_action():
    env = PaperTradingEnv(candles())
    with pytest.raises(ValueError):
        env.step(True)
