def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_load_demo_dataset(client):
    response = client.post("/api/datasets/demo")
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == 5000
    assert body["columns"] == 25
    assert "churn" in body["numeric_columns"] or "churn" in body["categorical_columns"]


def test_upload_dataset(client):
    csv_content = b"a,b,target\n1,2,0\n3,4,1\n5,6,0\n"
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == 3
    assert body["columns"] == 3


def test_current_summary_before_load_returns_404(client):
    response = client.get("/api/datasets/current")
    assert response.status_code == 404
