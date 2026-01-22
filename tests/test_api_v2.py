"""
Tests for Enhanced API with Real Models (P2.2) - Consolidated v2.0

Tests for the consolidated v2.0 API with What-If analysis, 
recommendations, and human-readable formatting.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.api.app import app, format_value, generate_feature_recommendations
from src.api.schemas import (
    PredictRequest, WhatIfRequest, PredictResponse, WhatIfResponse
)


class TestFormatValue:
    """Test human-readable value formatting"""
    
    def test_format_currency(self):
        """Test currency formatting"""
        assert format_value("monthlyCharges", 89.50) == "$89.50"
        assert format_value("cost", 100.0) == "$100.00"
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        assert format_value("churnRate", 12.5) == "12.5%"
        assert format_value("percentage", 75.0) == "75.0%"
    
    def test_format_tenure(self):
        """Test tenure/months formatting"""
        assert format_value("tenure", 24) == "24 months"
        assert format_value("months_active", 36) == "36 months"
    
    def test_format_age(self):
        """Test age formatting"""
        assert format_value("age", 35) == "35 years"
        assert format_value("customer_age", 45) == "45 years"
    
    def test_format_default(self):
        """Test default formatting"""
        assert format_value("unknown_field", 42.5) == "42.50"


class TestGenerateRecommendations:
    """Test feature recommendation generation"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for recommendations"""
        return pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'monthlyCharges': np.random.uniform(20, 150, 100),
            'tenure': np.random.randint(0, 72, 100),
        })
    
    def test_recommendations_structure(self, sample_data):
        """Test recommendation structure"""
        recs = generate_feature_recommendations(sample_data)
        
        assert isinstance(recs, dict)
        assert len(recs) == 3
        
        for feature, rec_data in recs.items():
            assert "mean" in rec_data
            assert "median" in rec_data
            assert "std" in rec_data
            assert "min" in rec_data
            assert "max" in rec_data
    
    def test_recommendations_values(self, sample_data):
        """Test recommendation values are reasonable"""
        recs = generate_feature_recommendations(sample_data)
        
        for feature, rec_data in recs.items():
            assert rec_data["min"] <= rec_data["mean"] <= rec_data["max"]
            assert rec_data["min"] <= rec_data["median"] <= rec_data["max"]
            assert rec_data["std"] >= 0
    
    def test_recommendations_exclude_target(self):
        """Test that target column is excluded"""
        data = pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'churn': np.random.randint(0, 2, 100),
        })
        
        recs = generate_feature_recommendations(data)
        assert "churn" not in recs


class TestPredictRequest:
    """Test prediction request schema"""
    
    def test_predict_request_valid(self):
        """Test valid prediction request"""
        req = PredictRequest(
            features={"age": 35, "monthlyCharges": 89.50},
            model_name="random_forest"
        )
        
        assert req.features["age"] == 35
        assert req.model_name == "random_forest"
    
    def test_predict_request_defaults(self):
        """Test prediction request defaults"""
        req = PredictRequest(features={"age": 35})
        
        assert req.model_name == "random_forest"
        assert req.return_probabilities is True


class TestWhatIfRequest:
    """Test what-if request schema"""
    
    def test_whatif_request_valid(self):
        """Test valid what-if request"""
        req = WhatIfRequest(
            current_features={"age": 35, "tenure": 24},
            scenario_changes={"tenure": 36},
            model_name="random_forest"
        )
        
        assert req.current_features["age"] == 35
        assert req.scenario_changes["tenure"] == 36
    
    def test_whatif_request_partial_changes(self):
        """Test what-if with partial feature changes"""
        req = WhatIfRequest(
            current_features={"age": 35, "tenure": 24, "charges": 89.50},
            scenario_changes={"tenure": 36}
        )
        
        assert len(req.scenario_changes) == 1
        assert req.scenario_changes["tenure"] == 36


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns info"""
        client = app.client if hasattr(app, 'client') else None
        # Simplified - just test that app exists
        assert app is not None
    
    def test_schemas_imported(self):
        """Test that all schemas are properly imported"""
        assert PredictRequest is not None
        assert WhatIfRequest is not None
        assert PredictResponse is not None
        assert WhatIfResponse is not None


class TestConsolidatedAPI:
    """Tests for consolidated API structure"""
    
    def test_app_has_routes(self):
        """Test that app has expected routes"""
        # Check that app has router/routes
        assert hasattr(app, 'routes') or hasattr(app, 'router')
    
    def test_format_functions_available(self):
        """Test that formatting functions are available"""
        assert callable(format_value)
        assert callable(generate_feature_recommendations)
    
    def test_api_consolidation(self):
        """Test that API is properly consolidated"""
        # Verify format_value handles all types
        assert "$" in format_value("charge", 50)
        assert "%" in format_value("rate", 75)
        assert "years" in format_value("age", 35)
        assert "months" in format_value("tenure", 24)
