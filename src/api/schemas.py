"""Pydantic schemas for API requests/responses."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


# ===== REQUEST SCHEMAS =====

class PredictRequest(BaseModel):
    """Prediction request with feature values."""
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    model_name: str = Field(default="random_forest", description="Model to use")
    return_probabilities: bool = Field(default=True, description="Return probability distribution")


class WhatIfRequest(BaseModel):
    """What-if scenario analysis request."""
    current_features: Dict[str, float] = Field(..., description="Current feature values")
    scenario_changes: Dict[str, float] = Field(..., description="Changes to apply")
    model_name: str = Field(default="random_forest", description="Model to use")


# ===== RESPONSE SCHEMAS =====

class FeatureInfo(BaseModel):
    """Information about a single feature."""
    name: str
    dtype: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    recommended_value: Optional[float] = None
    display_format: str = "number"


class FeaturesResponse(BaseModel):
    """Features list response."""
    status: str
    features: List[FeatureInfo]
    total_features: int


class PredictResponse(BaseModel):
    """Prediction response with all details."""
    status: str
    prediction: int
    prediction_label: str
    confidence: float
    confidence_level: str
    formatted_features: Dict[str, str]
    model_name: str
    execution_time_ms: float


class WhatIfResponse(BaseModel):
    """What-if analysis response."""
    status: str
    current_prediction: int
    current_prediction_label: str
    current_confidence: float
    scenario_prediction: int
    scenario_prediction_label: str
    scenario_confidence: float
    prediction_changed: bool
    formatted_changes: Dict[str, Dict[str, str]]
    model_name: str


class RecommendationResponse(BaseModel):
    """Feature recommendations response."""
    status: str
    recommendations: Dict[str, Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_available: List[str]


class AuditRecord(BaseModel):
    """Single audit record."""
    id: str
    timestamp: str
    model_name: str
    prediction: str
    confidence: float
    status: str


class AuditResponse(BaseModel):
    """Audit trail response."""
    status: str
    count: int
    records: List[AuditRecord]


class ErrorResponse(BaseModel):
    """Error response."""
    status: str
    error: str
    error_type: str
