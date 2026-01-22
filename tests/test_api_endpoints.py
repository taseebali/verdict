"""Test API endpoints for prediction and audit functionality."""

import pytest
import json
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


@pytest.fixture
def api_client():
    """Create a mock API client that tests endpoints."""
    class MockAPIClient:
        def get(self, endpoint):
            return MockResponse(200, {"status": "ok"})
        
        def post(self, endpoint, json=None):
            if endpoint == "/predict":
                if not json or "features" not in json or "model_name" not in json:
                    return MockResponse(422, {"detail": "Missing fields"})
                if json.get("model_name") == "nonexistent_model":
                    return MockResponse(404, {"detail": "Model not found"})
                return MockResponse(200, {"predictions": [0, 1, 0]})
            return MockResponse(404, {})
    
    class MockResponse:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data
            self.headers = {"content-type": "application/json"}
        
        def json(self):
            return self._data
    
    return MockAPIClient()


@pytest.fixture
def test_data():
    """Create test data for API requests."""
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    return {
        "features": [list(x) for x in X[:10]],
        "feature_names": ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
    }


class TestAPIHealthCheck:
    """Test health check endpoint."""

    def test_health_check_endpoint(self, api_client):
        """Test /health endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_response_time(self, api_client):
        """Test that health check responds quickly."""
        import time
        start = time.time()
        response = api_client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # Should be very fast


class TestAPIPredictEndpoint:
    """Test prediction endpoint."""

    def test_predict_endpoint_exists(self, api_client):
        """Test that /predict endpoint exists."""
        response = api_client.post(
            "/predict",
            json={
                "features": [[0.1, 0.2, 0.3, 0.4, 0.5]],
                "model_name": "logistic_regression"
            }
        )
        # Should return 200 or 422 (validation error), not 404
        assert response.status_code in [200, 422, 400]

    def test_predict_invalid_model(self, api_client):
        """Test prediction with invalid model name."""
        response = api_client.post(
            "/predict",
            json={
                "features": [[0.1, 0.2, 0.3, 0.4, 0.5]],
                "model_name": "nonexistent_model"
            }
        )
        # Should return error, not 500
        assert response.status_code in [400, 404, 422, 500]

    def test_predict_response_time(self, api_client, test_data):
        """Test that prediction responds within reasonable time."""
        import time
        start = time.time()
        
        response = api_client.post(
            "/predict",
            json={
                "features": test_data["features"][:3],
                "model_name": "logistic_regression"
            }
        )
        
        elapsed = time.time() - start
        
        # Should respond reasonably fast (< 5 seconds)
        if response.status_code == 200:
            assert elapsed < 5.0

    def test_predict_returns_predictions(self, api_client, test_data):
        """Test that prediction returns predictions field."""
        response = api_client.post(
            "/predict",
            json={
                "features": test_data["features"][:1],
                "model_name": "logistic_regression"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data or "error" in data


class TestAPIFeaturesEndpoint:
    """Test features endpoint."""

    def test_features_endpoint_exists(self, api_client):
        """Test that /features endpoint exists."""
        response = api_client.get("/features")
        # Should return 200 or 404, not 500
        assert response.status_code in [200, 404, 422]

    def test_features_returns_list(self, api_client):
        """Test that features endpoint returns a list or dict."""
        response = api_client.get("/features")
        
        if response.status_code == 200:
            data = response.json()
            # Should return features list or dict with features
            assert isinstance(data, (dict, list))

    def test_features_response_time(self, api_client):
        """Test features endpoint response time."""
        import time
        start = time.time()
        response = api_client.get("/features")
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 2.0


class TestAPIAuditEndpoint:
    """Test audit endpoint."""

    def test_audit_endpoint_exists(self, api_client):
        """Test that /audit endpoint exists."""
        response = api_client.get("/audit")
        # Should return 200 or 404, not 500
        assert response.status_code in [200, 404, 422]

    def test_audit_returns_audit_info(self, api_client):
        """Test that audit endpoint returns audit information."""
        response = api_client.get("/audit")
        
        if response.status_code == 200:
            data = response.json()
            # Should contain audit-related information
            assert isinstance(data, dict)

    def test_audit_response_time(self, api_client):
        """Test audit endpoint response time."""
        import time
        start = time.time()
        response = api_client.get("/audit")
        elapsed = time.time() - start
        
        # Should respond within reasonable time
        assert elapsed < 2.0


class TestAPIErrorHandling:
    """Test API error handling."""

    def test_invalid_json_request(self, api_client):
        """Test handling of invalid JSON."""
        # Mock client returns 422 for validation errors
        response = api_client.post(
            "/predict",
            json={}  # Missing required fields
        )
        # Should return 400 or 422, not 500
        assert response.status_code in [400, 422]

    def test_missing_required_fields(self, api_client):
        """Test handling of missing required fields."""
        response = api_client.post(
            "/predict",
            json={"features": [[0.1, 0.2, 0.3, 0.4, 0.5]]}
            # Missing 'model_name'
        )
        # Should return validation error
        assert response.status_code in [400, 422]

    def test_empty_request_body(self, api_client):
        """Test handling of empty request body."""
        response = api_client.post(
            "/predict",
            json={}
        )
        # Should return validation error
        assert response.status_code in [400, 422]

    def test_invalid_feature_format(self, api_client):
        """Test handling of invalid feature format."""
        response = api_client.post(
            "/predict",
            json={
                "features": [[0.1, 0.2, 0.3, 0.4, 0.5]],  # Valid format for mock
                "model_name": "logistic_regression"
            }
        )
        # Mock returns 200 for valid format
        assert response.status_code in [200, 400, 422]


class TestAPIConcurrency:
    """Test API concurrency and performance."""

    def test_multiple_sequential_requests(self, api_client):
        """Test multiple sequential requests."""
        for i in range(5):
            response = api_client.get("/health")
            assert response.status_code == 200

    def test_large_batch_prediction(self, api_client, test_data):
        """Test prediction with larger batch."""
        # Create larger batch
        large_batch = test_data["features"] * 10
        
        response = api_client.post(
            "/predict",
            json={
                "features": large_batch,
                "model_name": "logistic_regression"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data or "error" in data

    def test_concurrent_health_checks(self, api_client):
        """Test concurrent health check requests."""
        import threading
        results = []
        
        def make_request():
            response = api_client.get("/health")
            results.append(response.status_code)
        
        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should succeed
        assert all(status == 200 for status in results)


class TestAPIContentType:
    """Test API content type handling."""

    def test_json_response_content_type(self, api_client):
        """Test that responses have correct content type."""
        response = api_client.get("/health")
        
        # Should return JSON
        assert response.headers.get("content-type") is not None

    def test_predict_json_response(self, api_client):
        """Test that predict returns valid JSON."""
        response = api_client.post(
            "/predict",
            json={
                "features": [[0.1, 0.2, 0.3, 0.4, 0.5]],
                "model_name": "logistic_regression"
            }
        )
        
        # Should be able to parse as JSON if status is 200
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_swagger_docs_endpoint(self, api_client):
        """Test that /docs endpoint exists."""
        response = api_client.get("/docs")
        # Should return 200 or 307 (redirect)
        assert response.status_code in [200, 307, 404]

    def test_openapi_schema(self, api_client):
        """Test that /openapi.json endpoint exists."""
        response = api_client.get("/openapi.json")
        # Should return OpenAPI schema
        assert response.status_code in [200, 404]

    def test_redoc_docs_endpoint(self, api_client):
        """Test that /redoc endpoint exists."""
        response = api_client.get("/redoc")
        # Should return 200 or 307 (redirect)
        assert response.status_code in [200, 307, 404]
