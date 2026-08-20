def test_predict_after_training(client):
    client.post("/api/datasets/demo")
    train_resp = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    features = train_resp.json()
    from app.state import get_state
    state = get_state()
    sample = {f: float(state.df[f].iloc[0]) if state.df[f].dtype.kind in "if" else 0 for f in state.model_features}

    response = client.post("/api/predict", json={"features": sample})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_without_trained_model_returns_400(client):
    response = client.post("/api/predict", json={"features": {}})
    assert response.status_code == 400
