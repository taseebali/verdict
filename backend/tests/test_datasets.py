from pathlib import Path

import pandas as pd

from app import uploads

TELCO_CSV = Path(__file__).parents[2] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def _upload(client, text: str, name: str = "d.csv"):
    return client.post("/api/datasets/upload", files={"file": (name, text.encode(), "text/csv")})


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_demo_profile(client):
    response = client.post("/api/datasets/demo")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "verdict_demo.csv"
    assert body["rows"] == 5000
    assert len(body["preview"]) == 20
    assert body["target_suggestions"][0] == "churn"
    kinds = {c["name"]: c["kind"] for c in body["columns"]}
    assert kinds["tenure_months"] == "numeric"
    assert kinds["contract_type"] == "categorical"
    churn = next(c for c in body["columns"] if c["name"] == "churn")
    assert {v["value"] for v in churn["top_values"]} == {"0", "1"}


def test_session_cookie_is_http_only_and_lax(client):
    response = client.post("/api/datasets/demo")
    cookie = response.headers["set-cookie"].lower()
    assert "verdict_sid=" in cookie
    assert "httponly" in cookie
    assert "samesite=lax" in cookie


def test_visitors_do_not_see_each_others_data(client, other_client):
    client.post("/api/datasets/demo")
    assert client.get("/api/datasets/current").status_code == 200
    response = other_client.get("/api/datasets/current")
    assert response.status_code == 404
    assert response.json()["detail"].startswith("No dataset loaded")


def test_upload_identifies_id_columns_and_parses_numeric_text(client):
    with open(TELCO_CSV, "rb") as f:
        response = client.post("/api/datasets/upload", files={"file": ("telco.csv", f, "text/csv")})
    assert response.status_code == 200
    kinds = {c["name"]: c["kind"] for c in response.json()["columns"]}
    assert kinds["customerID"] == "identifier"
    assert kinds["TotalCharges"] == "numeric"
    assert response.json()["target_suggestions"][0] == "Churn"


def test_preview_nulls_are_json_null(client):
    csv = "a,b\n1,x\n,y\n" + "".join(f"{i},z\n" for i in range(30))
    body = _upload(client, csv).json()
    assert body["preview"][1]["a"] is None


def test_non_csv_is_rejected(client):
    response = _upload(client, "a,b\n1,2\n", name="d.txt")
    assert response.status_code == 400


def test_empty_file_is_rejected(client):
    assert _upload(client, "").status_code == 400


def test_too_large_upload_is_413(client, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 2 * 1024 * 1024)
    response = _upload(client, "a\n" + "1\n" * 1_600_000)
    assert response.status_code == 413
    assert "2 MB" in response.json()["detail"]


def test_too_many_rows_is_400(client, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_ROWS", 3)
    response = _upload(client, "a\n1\n2\n3\n4\n")
    assert response.status_code == 400
    assert "the limit is 3" in response.json()["detail"]


def test_mixed_type_numeric_text_becomes_float():
    df = pd.DataFrame({"amount": pd.Series([1, "12.5", " 3 ", None], dtype=object)})
    out = uploads.coerce_numeric_text(df)
    assert str(out["amount"].dtype) == "float64"
    assert out["amount"].tolist()[:3] == [1.0, 12.5, 3.0]


def test_session_cookie_behind_https_proxy_is_cross_site_partitioned(client):
    response = client.post("/api/datasets/demo", headers={"x-forwarded-proto": "https"})
    cookie = response.headers["set-cookie"].lower()
    assert "verdict_sid=" in cookie
    assert "httponly" in cookie
    assert "samesite=none" in cookie
    assert "secure" in cookie
    assert "partitioned" in cookie


def test_session_cookie_over_plain_http_is_lax_and_not_secure(client):
    cookie = client.post("/api/datasets/demo").headers["set-cookie"].lower()
    assert "samesite=lax" in cookie
    assert "secure" not in cookie


def test_forwarded_proto_list_uses_first_value(client):
    response = client.post("/api/datasets/demo", headers={"x-forwarded-proto": "HTTPS, http"})
    cookie = response.headers["set-cookie"].lower()
    assert "samesite=none" in cookie and "secure" in cookie and "partitioned" in cookie
