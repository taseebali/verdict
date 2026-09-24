from pathlib import Path

import pandas as pd

DEMO_CSV = Path(__file__).parents[2] / "data" / "verdict_demo.csv"


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


def test_train_rejects_continuous_target(client):
    client.post("/api/datasets/demo")
    response = client.post(
        "/api/train",
        json={"target": "customer_lifetime_value", "features": None, "method": "random_forest"},
    )
    assert response.status_code == 400
    assert "customer_lifetime_value" in response.json()["detail"]


def test_train_rejects_unknown_method(client):
    client.post("/api/datasets/demo")
    response = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "xgboost"},
    )
    assert response.status_code == 422


def test_train_reports_dropped_identifier_columns(client):
    df = pd.read_csv(DEMO_CSV).head(500)
    df["customer_id"] = [f"C{i:05d}" for i in range(len(df))]
    client.post(
        "/api/datasets/upload",
        files={"file": ("d.csv", df.to_csv(index=False).encode(), "text/csv")},
    )
    response = client.post(
        "/api/train",
        json={"target": "churn", "features": None, "method": "random_forest"},
    )
    assert response.status_code == 200
    assert response.json()["dropped_features"] == ["customer_id"]
