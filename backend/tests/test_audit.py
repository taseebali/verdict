def test_audit_logs_after_prediction(client):
    client.post("/api/datasets/demo")
    client.post("/api/train", json={"target": "churn", "features": None, "method": "random_forest"})
    from app.state import get_state
    state = get_state()
    sample = {f: float(state.df[f].iloc[0]) if state.df[f].dtype.kind in "if" else 0 for f in state.model_features}
    client.post("/api/predict", json={"features": sample})

    response = client.get("/api/audit-logs")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["prediction"] in (0, 1)


def test_audit_logs_empty_before_predictions(client):
    response = client.get("/api/audit-logs")
    assert response.status_code == 200
    assert response.json() == []


def test_model_persisted_after_training(client):
    from src.artifacts.model_serializer import ModelSerializer
    client.post("/api/datasets/demo")
    response = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    assert response.status_code == 200
    # Verify model was saved to disk
    assert ModelSerializer.model_exists("random_forest")


def test_download_model_after_training(client):
    client.post("/api/datasets/demo")
    response = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    assert response.status_code == 200

    # Download the trained model
    response = client.get("/api/models/random_forest/download")
    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith("random_forest.joblib\"")
    # Verify response has content
    assert len(response.content) > 0


def test_download_nonexistent_model_returns_404(client):
    response = client.get("/api/models/nonexistent/download")
    assert response.status_code == 404
