import numpy as np
import pandas as pd
import pytest

from src.core.scoring import (
    build_pipeline,
    fit_with_oof,
    infer_roles,
    prepare_features,
    target_suggestions,
)


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """x and color=red push the outcome to Yes; customer_id is an identifier."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    color = rng.choice(["red", "blue", "green"], size=n)
    noise = rng.normal(scale=0.5, size=n)
    target = np.where(x + (color == "red") * 1.0 + noise > 0.5, "Yes", "No")
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "x": x,
        "color": color,
        "target": target,
    })


# --- roles & suggestions ----------------------------------------------------

def test_infer_roles_marks_text_and_integer_ids():
    df = _frame()
    df["account_no"] = np.arange(1000, 1000 + len(df))
    df["flag"] = [True, False] * (len(df) // 2)
    roles = infer_roles(df, target="target")
    assert roles.identifiers == ["customer_id", "account_no"]
    assert roles.numeric == ["x"]
    assert roles.categorical == ["color", "flag"]


def test_small_frames_have_no_identifiers():
    df = _frame(n=20)
    roles = infer_roles(df, target="target")
    assert roles.identifiers == []
    assert "customer_id" in roles.categorical


def test_target_suggestions_prefers_outcome_like_names():
    df = pd.DataFrame({
        "a_flag": [0, 1] * 30,
        "churned": ["y", "n"] * 30,
        "tier": ["a", "b", "c"] * 20,
        "amount": np.linspace(0, 1, 60),
    })
    assert target_suggestions(df) == ["churned", "a_flag", "tier"]


# --- pipeline -----------------------------------------------------------------

def test_pipeline_handles_missing_values_and_unseen_categories():
    df = _frame()
    df.loc[0, "x"] = np.nan
    df.loc[1, "color"] = np.nan
    X = prepare_features(df, ["x"], ["color"])
    y = (df["target"] == "Yes").to_numpy()
    pipe = build_pipeline(["x"], ["color"], "random_forest").fit(X, y)
    new = prepare_features(pd.DataFrame({"x": [0.1], "color": ["purple"]}), ["x"], ["color"])
    proba = pipe.predict_proba(new)[:, 1]
    assert 0.0 <= proba[0] <= 1.0


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        build_pipeline(["x"], [], "xgboost")


# --- fit_with_oof -------------------------------------------------------------

def test_every_row_gets_an_out_of_fold_score():
    df = _frame()
    model = fit_with_oof(df, "target", "Yes")
    assert len(model.oof_proba) == len(df)
    assert np.all((model.oof_proba >= 0) & (model.oof_proba <= 1))
    assert model.roc_auc > 0.7
    assert model.base_rate == pytest.approx((df["target"] == "Yes").mean())
    assert model.identifiers == ["customer_id"]
    assert model.features == ["x", "color"]
    assert list(model.row_ids) == list(df.index)
    assert 2 <= len(model.roc_points) <= 101


def test_positive_class_choice_flips_the_framing():
    df = _frame()
    yes = fit_with_oof(df, "target", "Yes")
    no = fit_with_oof(df, "target", "No")
    assert no.base_rate == pytest.approx(1 - yes.base_rate)
    assert np.array_equal(no.actual, ~yes.actual)


def test_multiclass_target_becomes_one_vs_rest():
    df = _frame()
    df["tier"] = np.where(df["x"] > 0.8, "gold", np.where(df["x"] > -0.5, "silver", "bronze"))
    model = fit_with_oof(df.drop(columns="target"), "tier", "gold")
    assert np.array_equal(model.actual, (df["tier"] == "gold").to_numpy())


def test_rows_with_blank_target_are_skipped():
    df = _frame()
    df.loc[:4, "target"] = np.nan
    model = fit_with_oof(df, "target", "Yes")
    assert model.rows_skipped == 5
    assert len(model.oof_proba) == len(df) - 5


def test_excluded_features_are_not_used():
    model = fit_with_oof(_frame(), "target", "Yes", excluded=["color"])
    assert model.features == ["x"]


def test_logistic_regression_method_works():
    model = fit_with_oof(_frame(), "target", "Yes", method="logistic_regression")
    assert model.method == "logistic_regression"
    assert model.roc_auc > 0.7


@pytest.mark.parametrize("target, positive, message", [
    ("x", "1", "distinct values"),
    ("target", "Maybe", "does not appear"),
    ("missing_col", "Yes", "not found"),
])
def test_validation_errors(target, positive, message):
    with pytest.raises(ValueError, match=message):
        fit_with_oof(_frame(), target, positive)


def test_too_few_positives_is_rejected():
    df = _frame()
    df["target"] = "No"
    df.loc[:2, "target"] = "Yes"
    with pytest.raises(ValueError, match="at least 5"):
        fit_with_oof(df, "target", "Yes")


def test_no_features_left_is_rejected():
    with pytest.raises(ValueError, match="No usable feature"):
        fit_with_oof(_frame(), "target", "Yes", excluded=["x", "color"])
