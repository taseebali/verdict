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
