import pandas as pd
from src.learning import FEATURES, make_features, predict


class DummyModel:
    classes_ = ["BUY", "HOLD", "SELL"]

    def predict_proba(self, frame):
        return [[0.8, 0.1, 0.1]]


def test_make_features_has_expected_columns():
    n = 260
    close = pd.Series([100 + i * 0.1 for i in range(n)])
    from src.market import add_indicators
    df = pd.DataFrame({
        "close": close, "high": close + 1, "low": close - 1,
        "open": close, "volume": pd.Series([100 + i for i in range(n)])
    })
    featured = make_features(add_indicators(df))
    assert list(featured.columns) == FEATURES
    assert featured.iloc[-1].notna().all()


def test_model_prediction_uses_confidence_threshold():
    row = pd.Series({
        "close_return_1": 0.01, "close_return_3": 0.02,
        "ema_gap": 0.01, "rsi": 0.6, "atr_pct": 0.01, "volume_change": 0.1
    })
    assert predict(row, DummyModel())["decision"] == "BUY"
    assert predict(row, None)["decision"] == "HOLD"
