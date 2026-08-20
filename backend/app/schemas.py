from typing import Any, Optional

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
    method: str = "random_forest"


class TrainResponse(BaseModel):
    model_name: str
    metrics: dict[str, float]
    feature_importance: dict[str, float]
