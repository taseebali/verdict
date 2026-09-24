from pathlib import Path

import pandas as pd
import pytest


def _sample_from_row(state, row_index: int = 0) -> dict:
    """Build a feature dict from an actual row of the currently loaded raw
    dataframe, using real category values (not placeholders)."""
    sample = {}
    for f in state.model_features:
        val = state.df[f].iloc[row_index]
        sample[f] = float(val) if state.df[f].dtype.kind in "if" else str(val)
    return sample


def test_predict_after_training(client):
    client.post("/api/datasets/demo")
    client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    from app.state import get_state
    state = get_state()
    sample = _sample_from_row(state)

    response = client.post("/api/predict", json={"features": sample})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_without_trained_model_returns_400(client):
    response = client.post("/api/predict", json={"features": {}})
    assert response.status_code == 400


def test_predict_unknown_categorical_value_returns_400(client):
    client.post("/api/datasets/demo")
    client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    from app.state import get_state
    state = get_state()
    sample = _sample_from_row(state)
    sample["contract_type"] = "definitely-not-a-real-contract-type"

    response = client.post("/api/predict", json={"features": sample})
    assert response.status_code == 400
    assert "contract_type" in response.json()["detail"]


def test_predict_without_predict_proba_reports_valid_probability(client):
    """The API contract requires trained_model_name to name a model type that
    supports predict_proba (only logistic_regression/random_forest are
    reachable via /api/train). A model without predict_proba is still
    reachable directly (e.g. via the auto-generated /docs UI hitting
    state.trained_model), so the fallback path must still report a valid
    0-1 probability instead of leaking the raw prediction value.
    """
    client.post("/api/datasets/demo")
    client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )

    from app.state import get_state
    state = get_state()

    class _HardVoteStub:
        """A model with no predict_proba, whose prediction is deliberately
        outside [0, 1] to prove the old bug (probability = raw prediction)
        is gone."""

        def predict(self, X):
            return [42]

    state.trained_model = _HardVoteStub()
    sample = _sample_from_row(state)

    response = client.post("/api/predict", json={"features": sample})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 42
    assert body["probability"] == 0.0
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_matches_pipeline_transform_directly(client):
    """Equivalence test for the Critical-1 fix: the /api/predict endpoint must
    route features through the SAME preprocessing transform the model was
    trained on. Build the feature dict from an actual training-data row, then
    verify the endpoint's prediction matches calling
    pipeline.model_manager.predict() directly on the corresponding transformed
    test row — i.e. the raw feature dict and the already-transformed row must
    yield identical predictions once both go through the real pipeline.
    """
    client.post("/api/datasets/demo")
    train_resp = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    assert train_resp.status_code == 200

    from app.state import get_state
    state = get_state()
    pipeline = state.pipeline

    # Take a row straight out of the transformed test set (X_test) — this is
    # exactly the feature space the model was trained/evaluated on.
    X_test = pipeline.X_test
    test_row = X_test.iloc[[0]]
    direct_prediction = int(pipeline.model_manager.predict(state.trained_model_name, test_row.to_numpy())[0])
    direct_proba = pipeline.model_manager.predict_proba(state.trained_model_name, test_row.to_numpy())[0]

    # Recover the RAW (untransformed) feature values for that same row by
    # inverse-transforming through the fitted encoders/scaler, then send that
    # raw dict through the API exactly as a real client would.
    preprocessor = pipeline.preprocessor
    raw_row = test_row.copy()
    numeric_cols = [c for c in preprocessor.numeric_cols if c in raw_row.columns]
    if numeric_cols:
        raw_row[numeric_cols] = preprocessor.scaler.inverse_transform(raw_row[numeric_cols])
    for col, encoder in preprocessor.label_encoders.items():
        if col not in raw_row.columns:
            continue
        raw_row[col] = encoder.inverse_transform(raw_row[col].astype(int))

    raw_features = {}
    for col in state.model_features:
        val = raw_row.iloc[0][col]
        raw_features[col] = float(val) if col in preprocessor.numeric_cols else str(val)

    response = client.post("/api/predict", json={"features": raw_features})
    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] == direct_prediction
    expected_probability = float(direct_proba[1]) if len(direct_proba) > 1 else float(direct_proba[0])
    assert body["probability"] == pytest.approx(expected_probability, abs=1e-6)
    assert body["confidence"] == pytest.approx(float(max(direct_proba)), abs=1e-6)


DEMO_CSV = Path(__file__).parents[2] / "data" / "verdict_demo.csv"


def test_predict_returns_original_label_for_text_target(client):
    df = pd.read_csv(DEMO_CSV).head(500)
    df["churn"] = df["churn"].map({0: "No", 1: "Yes"})
    client.post(
        "/api/datasets/upload",
        files={"file": ("d.csv", df.to_csv(index=False).encode(), "text/csv")},
    )
    train = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    assert train.status_code == 200

    from app.state import get_state
    sample = _sample_from_row(get_state())
    response = client.post("/api/predict", json={"features": sample})
    assert response.status_code == 200
    assert response.json()["prediction"] in ("No", "Yes")

    audit = client.get("/api/audit-logs").json()
    assert audit[-1]["prediction"] in ("No", "Yes")


def test_multiclass_probability_agrees_with_prediction(client):
    client.post("/api/datasets/demo")
    client.post(
        "/api/train",
        json={"target": "satisfaction", "features": None, "method": "random_forest"},
    )
    from app.state import get_state
    sample = _sample_from_row(get_state())
    body = client.post("/api/predict", json={"features": sample}).json()
    assert body["probability"] == pytest.approx(body["confidence"])


def test_whatif_delta_compares_same_class_multiclass(client):
    """The whatif delta must compare the SAME class's probability across
    baseline and scenario, namely the baseline's predicted class — not each
    row's own top-probability class, which can differ once the scenario
    flips the prediction."""
    client.post("/api/datasets/demo")
    client.post(
        "/api/train",
        json={"target": "satisfaction", "features": None, "method": "random_forest"},
    )
    from app.state import get_state
    from app.routers.predictions import _transform_features

    state = get_state()
    baseline_features = _sample_from_row(state, 0)
    scenario_features = dict(baseline_features)
    # Push every numeric feature to its minimum — for this demo data/model,
    # this reliably flips the predicted class, which is what exposes the bug
    # (comparing two different classes' probabilities).
    for f in state.model_features:
        if state.df[f].dtype.kind in "if":
            scenario_features[f] = float(state.df[f].min())

    response = client.post(
        "/api/whatif",
        json={"baseline_features": baseline_features, "scenario_features": scenario_features},
    )
    assert response.status_code == 200
    body = response.json()

    baseline_row = _transform_features(baseline_features)
    scenario_row = _transform_features(scenario_features)
    baseline_proba = state.trained_model.predict_proba(baseline_row)[0]
    scenario_proba = state.trained_model.predict_proba(scenario_row)[0]
    baseline_raw = state.trained_model.predict(baseline_row)[0]
    idx = list(state.trained_model.classes_).index(baseline_raw)

    expected_delta = scenario_proba[idx] - baseline_proba[idx]
    assert body["delta_probability"] == pytest.approx(expected_delta, abs=1e-4)


def test_whatif_delta_binary_matches_probability_difference(client):
    client.post("/api/datasets/demo")
    client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    from app.state import get_state
    state = get_state()
    baseline_features = _sample_from_row(state, 0)
    scenario_features = dict(baseline_features)
    for f in state.model_features:
        if state.df[f].dtype.kind in "if":
            scenario_features[f] = float(state.df[f].max())

    response = client.post(
        "/api/whatif",
        json={"baseline_features": baseline_features, "scenario_features": scenario_features},
    )
    assert response.status_code == 200
    body = response.json()
    expected = body["scenario"]["probability"] - body["baseline"]["probability"]
    assert body["delta_probability"] == pytest.approx(expected, abs=1e-4)
