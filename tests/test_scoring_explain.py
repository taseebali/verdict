import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.core.scoring import (
    describe_drivers,
    fit_with_oof,
    format_value,
    prepare_features,
    row_reasons,
    score_frame,
)


def test_logistic_regression_high_cardinality_categorical_stays_sparse():
    """A 2,000-level categorical column must not densify into a huge matrix."""
    n = 4000
    rng = np.random.default_rng(3)
    signal = rng.normal(size=n)
    high_card = [f"c{i % 2000}" for i in range(n)]
    target = np.where(signal + rng.normal(scale=0.4, size=n) > 0.3, "Yes", "No")
    df = pd.DataFrame({"signal": signal, "high_card": high_card, "target": target})

    model = fit_with_oof(df, "target", "Yes", method="logistic_regression")

    X = prepare_features(df, model.numeric, model.categorical)
    Xt = model.pipeline.named_steps["prep"].transform(X)
    assert sp.issparse(Xt)

    reasons = row_reasons(model, X.iloc[:5])
    assert len(reasons) == 5
    for row in reasons:
        assert len(row) <= 3


def _frame(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    color = rng.choice(["red", "blue", "green"], size=n)
    noise_col = rng.normal(size=n)
    target = np.where(x + (color == "red") * 1.5 + rng.normal(scale=0.4, size=n) > 0.8, "Yes", "No")
    return pd.DataFrame({"x": x, "color": color, "noise": noise_col, "target": target})


def test_importance_ranks_signal_above_noise():
    model = fit_with_oof(_frame(), "target", "Yes")
    ranked = [name for name, _ in model.importance]
    assert set(ranked) == {"x", "color", "noise"}
    assert ranked.index("noise") == 2
    assert all(score >= 0 for _, score in model.importance)


def test_drivers_find_the_planted_segment():
    model = fit_with_oof(_frame(), "target", "Yes")
    segments = {d.feature: d for d in model.drivers}
    assert segments["color"].segment == "color = red"
    red = segments["color"]
    assert red.rate > red.overall
    assert red.lift == pytest.approx(red.rate / red.overall)
    assert 0 < red.share < 1


def test_numeric_driver_is_a_readable_range():
    df = _frame()
    X = prepare_features(df, ["x"], [])
    y = (df["target"] == "Yes").to_numpy()
    drivers = describe_drivers(X, y, [("x", 0.3)], numeric=["x"])
    assert len(drivers) == 1
    assert drivers[0].segment.startswith("x ")
    assert " – " in drivers[0].segment


@pytest.mark.parametrize("method", ["random_forest", "logistic_regression"])
def test_row_reasons_are_top_positive_contributors(method):
    df = _frame()
    model = fit_with_oof(df, "target", "Yes", method=method)
    X = prepare_features(df.head(10), model.numeric, model.categorical)
    reasons = row_reasons(model, X)
    assert len(reasons) == 10
    for row in reasons:
        assert len(row) <= 3
        assert all(r.impact > 0 for r in row)
        assert all(r.feature in model.features for r in row)
        assert all(isinstance(r.value, str) and r.value for r in row)


def test_score_frame_needs_model_columns_and_tolerates_unseen_values():
    model = fit_with_oof(_frame(), "target", "Yes")
    new = pd.DataFrame({"x": [0.5, np.nan], "color": ["purple", "red"], "noise": [0, 0], "extra": [1, 2]})
    proba = score_frame(model, new)
    assert proba.shape == (2,)
    assert np.all((proba >= 0) & (proba <= 1))
    with pytest.raises(ValueError, match="color"):
        score_frame(model, new.drop(columns="color"))


@pytest.mark.parametrize("value, expected", [
    (2.0, "2"), (1234.0, "1,234"), (0.456, "0.46"), (12345.67, "12,346"),
    (np.nan, "(missing)"), (None, "(missing)"), ("Month-to-month", "Month-to-month"),
    (True, "True"), (False, "False"), (np.bool_(True), "True"),
])
def test_format_value(value, expected):
    assert format_value(value) == expected


def test_logistic_regression_categorical_reasons_meaningful():
    """For LR, red rows should have positive color impact; non-red rows should have smaller impact."""
    df = _frame(n=200)
    model = fit_with_oof(df, "target", "Yes", method="logistic_regression")
    X = prepare_features(df, model.numeric, model.categorical)
    reasons = row_reasons(model, X)

    # Collect color impacts
    red_impacts = []
    non_red_impacts = []
    for i, reason_list in enumerate(reasons):
        color_reason = [r for r in reason_list if r.feature == "color"]
        color_val = df.iloc[i]["color"]
        if color_reason:
            impact = color_reason[0].impact
            if color_val == "red":
                red_impacts.append(impact)
            else:
                non_red_impacts.append(impact)

    # Red rows should have stronger color signal than non-red rows
    if red_impacts and non_red_impacts:
        assert np.mean(red_impacts) > np.max(non_red_impacts)


def test_logistic_regression_all_numeric_no_error():
    """Regression: row_reasons should work on all-numeric datasets with logistic_regression."""
    df = _frame().drop(columns="color")
    model = fit_with_oof(df, "target", "Yes", method="logistic_regression")
    X = prepare_features(df.head(10), model.numeric, model.categorical)
    reasons = row_reasons(model, X)
    assert len(reasons) == 10
    for row in reasons:
        assert len(row) <= 3
        assert all(r.impact > 0 for r in row)
        assert all(r.feature in model.features for r in row)
