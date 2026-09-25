from pathlib import Path

import pandas as pd
import pytest

from app.sessions import store

DEMO_CSV = Path(__file__).parents[2] / "data" / "verdict_demo.csv"


def _trained(client):
    client.post("/api/datasets/demo")
    assert client.post("/api/train", json={"target": "churn", "positive_class": "1"}).status_code == 200


def _upload_new(client, df: pd.DataFrame):
    return client.post("/api/results/score",
                       files={"file": ("new.csv", df.to_csv(index=False).encode(), "text/csv")})


def test_whatif_returns_baseline_scenario_and_row_values(client):
    _trained(client)
    top = client.get("/api/results/rows?limit=1").json()["rows"][0]
    base = client.post("/api/results/whatif", json={"row_id": top["row_id"], "changes": {}}).json()
    assert base["delta"] == pytest.approx(0)
    assert "tenure_months" in base["features"]
    changed = client.post("/api/results/whatif", json={
        "row_id": top["row_id"],
        "changes": {"tenure_months": 70, "contract_type": "Two year", "complaint_count": 0},
    }).json()
    assert changed["baseline"] == pytest.approx(base["baseline"])
    assert changed["delta"] == pytest.approx(changed["scenario"] - changed["baseline"])
    assert changed["scenario"] < changed["baseline"]


def test_whatif_errors(client):
    _trained(client)
    assert client.post("/api/results/whatif", json={"row_id": 999999, "changes": {}}).status_code == 404
    response = client.post("/api/results/whatif", json={"row_id": 0, "changes": {"nope": 1}})
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_score_new_file_and_page_it(client):
    _trained(client)
    new = pd.read_csv(DEMO_CSV).head(100).drop(columns="churn")
    new.loc[0, "contract_type"] = "Brand-new plan"
    response = _upload_new(client, new)
    assert response.status_code == 200
    assert response.json() == {"rows_scored": 100, "source": "new", "name": "new.csv"}
    page = client.get("/api/results/rows?source=new&limit=100").json()
    assert page["total"] == 100
    assert all(r["actual"] is None for r in page["rows"])
    probs = [r["probability"] for r in page["rows"]]
    assert probs == sorted(probs, reverse=True)
    export = client.get("/api/results/export.csv?source=new&threshold=0.5")
    assert export.status_code == 200
    assert len(export.text.strip().splitlines()) == 101


def test_score_new_file_missing_columns(client):
    _trained(client)
    new = pd.read_csv(DEMO_CSV).head(10).drop(columns=["churn", "tenure_months"])
    response = _upload_new(client, new)
    assert response.status_code == 400
    assert "tenure_months" in response.json()["detail"]


def test_new_file_decision(client):
    _trained(client)
    body = {"threshold": 0.5, "action_cost": 20, "saved_value": 500, "success_rate": 0.3}
    assert client.post("/api/results/new/decision", json=body).status_code == 404
    _upload_new(client, pd.read_csv(DEMO_CSV).head(200).drop(columns="churn"))
    result = client.post("/api/results/new/decision", json=body).json()
    probs = [r["probability"] for r in client.get("/api/results/rows?source=new&limit=100").json()["rows"]]
    probs += [r["probability"] for r in client.get("/api/results/rows?source=new&offset=100&limit=100").json()["rows"]]
    assert result["flagged"] == sum(p > 0.5 for p in probs)
    expected = sum(p * 0.3 * 500 - 20 for p in probs if p > 0.5)
    assert result["expected_net"] == pytest.approx(expected, rel=1e-3)
    bad = {**body, "threshold": 1.5}
    assert client.post("/api/results/new/decision", json=bad).status_code == 422


def test_score_new_file_drops_stale_reasons_cache(client):
    """Scoring a second file must not let its rows reuse the first file's cached
    reasons: /rows computes reasons keyed by ("new", row_id), and row ids in the
    new file's positional index (0..N-1) collide with the previous file's ids.
    """
    _trained(client)
    demo = pd.read_csv(DEMO_CSV).drop(columns="churn")
    file_a = demo.head(50)
    file_b = demo.iloc[2000:2050].reset_index(drop=True)

    assert _upload_new(client, file_a).status_code == 200
    page_a = client.get("/api/results/rows?source=new&limit=1").json()
    assert page_a["rows"][0]["probability"] == pytest.approx(max(
        r["probability"] for r in client.get("/api/results/rows?source=new&limit=50").json()["rows"]
    ))

    assert _upload_new(client, file_b).status_code == 200

    # No cache entries for "new" should survive the second scoring.
    session = next(iter(store._sessions.values()))
    assert not any(key[0] == "new" for key in session.reasons_cache)

    page_b = client.get("/api/results/rows?source=new&limit=50").json()
    top_b = page_b["rows"][0]
    assert top_b["probability"] == pytest.approx(max(r["probability"] for r in page_b["rows"]))
    # The top row's reasons must match a fresh row_reasons call on B's data, not A's cache.
    from app.sessions import require_model
    from src.core.scoring import prepare_features, row_reasons

    model = require_model(session)
    row_id = top_b["row_id"]
    X = prepare_features(file_b.loc[[row_id]], model.numeric, model.categorical)
    fresh_reasons = row_reasons(model, X)[0]
    assert [r["feature"] for r in top_b["reasons"]] == [r.feature for r in fresh_reasons]
    assert [pytest.approx(r["impact"]) for r in top_b["reasons"]] == [r.impact for r in fresh_reasons]
