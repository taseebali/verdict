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


def test_sample_row_returns_a_real_row(client):
    client.post("/api/datasets/demo")
    response = client.get("/api/datasets/sample-row")
    assert response.status_code == 200
    features = response.json()["features"]
    assert "churn" in features
    assert "contract_type" in features
    assert features["contract_type"] in ["Month-to-month", "One year", "Two year"]


def test_sample_row_before_load_returns_404(client):
    response = client.get("/api/datasets/sample-row")
    assert response.status_code == 404


def test_categories_returns_known_values(client):
    client.post("/api/datasets/demo")
    response = client.get("/api/datasets/categories")
    assert response.status_code == 200
    categories = response.json()["categories"]
    assert set(categories["contract_type"]) == {"Month-to-month", "One year", "Two year"}


def test_sample_row_with_blank_rows_returns_no_nan(client):
    # 15 of 20 rows have a blank "charges" cell (like Telco's TotalCharges) —
    # state.df.sample(n=1) can land on one of those and fail to JSON-encode
    # the resulting NaN.
    rows = []
    for i in range(20):
        charges = "" if i % 4 != 0 else f"{i}.5"
        rows.append(f"{i},{charges}\n")
    csv = "id,charges\n" + "".join(rows)
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("blanks.csv", csv.encode(), "text/csv")},
    )
    assert response.status_code == 200

    for _ in range(20):
        response = client.get("/api/datasets/sample-row")
        assert response.status_code == 200
        features = response.json()["features"]
        assert all(v is not None for v in features.values())


def test_upload_converts_numeric_text_columns(client):
    rows = "".join(f"{i}.5,{'a' if i % 2 else 'b'}\n" for i in range(40))
    csv = "amount,label\n" + " ,a\n" + rows
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("d.csv", csv.encode(), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "amount" in body["numeric_columns"]
    assert "amount" not in body["categorical_columns"]
