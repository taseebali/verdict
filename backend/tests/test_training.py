def test_train_random_forest(client):
    client.post("/api/datasets/demo")
    response = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["metrics"]["accuracy"] <= 1.0
    assert isinstance(body["feature_importance"], dict)
    assert len(body["feature_importance"]) > 0


def test_train_without_dataset_returns_400(client):
    response = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    assert response.status_code == 400
