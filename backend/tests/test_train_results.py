import csv
import io
from pathlib import Path

TELCO_CSV = Path(__file__).parents[2] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def _train(client, target="churn", positive="1", **extra):
    return client.post("/api/train", json={"target": target, "positive_class": positive, **extra})


def _demo_trained(client):
    client.post("/api/datasets/demo")
    response = _train(client)
    assert response.status_code == 200, response.text
    return response.json()


def test_train_returns_an_honest_summary(client):
    body = _demo_trained(client)
    assert body["dataset_name"] == "verdict_demo.csv"
    assert body["target"] == "churn" and body["positive_class"] == "1"
    assert body["rows_scored"] == 5000
    assert 0.8 < body["roc_auc"] <= 1.0
    assert 0 < body["base_rate"] < 1
    assert "churn" not in body["features"]
    scores = [i["score"] for i in body["importance"]]
    assert scores == sorted(scores, reverse=True)
    assert 1 <= len(body["drivers"]) <= 5
    assert {"feature", "segment", "rate", "overall", "share", "lift"} <= set(body["drivers"][0])


def test_train_errors(client):
    assert _train(client).status_code == 404
    client.post("/api/datasets/demo")
    assert _train(client, target="customer_lifetime_value", positive="1").status_code == 400
    assert _train(client, positive="7").status_code == 400
    assert _train(client, method="xgboost").status_code == 422


def test_summary_requires_a_model_and_is_per_visitor(client, other_client):
    client.post("/api/datasets/demo")
    assert client.get("/api/results/summary").status_code == 404
    _train(client)
    assert client.get("/api/results/summary").json()["rows_scored"] == 5000
    assert other_client.get("/api/results/summary").status_code == 404


def test_new_dataset_clears_the_model(client):
    _demo_trained(client)
    client.post("/api/datasets/demo")
    assert client.get("/api/results/summary").status_code == 404


def test_decision_curve_and_recommendation(client):
    _demo_trained(client)
    body = client.post("/api/results/decision", json={}).json()
    assert len(body["curve"]) == 101
    assert body["recommended"]["net"] >= 0
    best = max(p["net"] for p in body["curve"])
    assert body["recommended"]["net"] == best or best <= 0
    assert client.post("/api/results/decision", json={"success_rate": 0}).status_code == 422
    assert client.post("/api/results/decision", json={"saved_value": 0}).status_code == 422
    assert client.post("/api/results/decision", json={"action_cost": -1}).status_code == 422


def test_rows_are_ranked_with_reasons(client):
    _demo_trained(client)
    first = client.get("/api/results/rows?limit=25").json()
    assert first["total"] == 5000 and first["source"] == "training"
    probs = [r["probability"] for r in first["rows"]]
    assert probs == sorted(probs, reverse=True)
    assert all(len(r["reasons"]) <= 3 for r in first["rows"])
    assert all(r["label"].startswith("#") for r in first["rows"])
    assert all(isinstance(r["actual"], bool) for r in first["rows"])
    second = client.get("/api/results/rows?offset=25&limit=25").json()
    assert second["rows"][0]["probability"] <= probs[-1]
    assert client.get("/api/results/rows?limit=101").status_code == 422
    assert client.get("/api/results/rows?source=new").status_code == 404


def test_rows_use_the_identifier_column_as_label(client):
    with open(TELCO_CSV, "rb") as f:
        client.post("/api/datasets/upload", files={"file": ("telco.csv", f, "text/csv")})
    assert _train(client, target="Churn", positive="Yes").status_code == 200
    rows = client.get("/api/results/rows?limit=5").json()["rows"]
    assert all(not r["label"].startswith("#") for r in rows)
    assert all(len(r["label"]) > 4 for r in rows)


def test_export_csv(client):
    _demo_trained(client)
    response = client.get("/api/results/export.csv?threshold=0.5")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 5000
    probs = [float(r["verdict_probability"]) for r in rows]
    assert probs == sorted(probs, reverse=True)
    for r in rows[:50]:
        assert (r["verdict_flag"] == "True") == (float(r["verdict_probability"]) > 0.5)
    flagged = [r for r in rows if r["verdict_flag"] == "True"]
    assert flagged and " = " in flagged[0]["verdict_reasons"]
