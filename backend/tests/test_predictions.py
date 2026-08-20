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

