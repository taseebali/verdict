from typing import Any, Literal, Optional, Union

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    rows: int
    columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_pct: float
    warnings: list[str]


class TrainRequest(BaseModel):
    target: str
    features: Optional[list[str]] = None
    method: Literal["random_forest", "logistic_regression"] = "random_forest"


class TrainResponse(BaseModel):
    model_name: str
    metrics: dict[str, float]
    feature_importance: dict[str, float]
    dropped_features: list[str] = []


class SampleRowResponse(BaseModel):
    features: dict[str, Any]


class CategoriesResponse(BaseModel):
    categories: dict[str, list[str]]


class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: Union[int, float, str]
    probability: float
    confidence: float


class WhatIfRequest(BaseModel):
    baseline_features: dict[str, Any]
    scenario_features: dict[str, Any]


class WhatIfResponse(BaseModel):
    baseline: PredictResponse
    scenario: PredictResponse
    delta_probability: float
