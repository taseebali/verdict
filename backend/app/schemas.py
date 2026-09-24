from typing import Any, Literal

from pydantic import BaseModel, Field


class ValueCount(BaseModel):
    value: str
    count: int


class ColumnProfile(BaseModel):
    name: str
    kind: Literal["numeric", "categorical", "identifier"]
    missing_pct: float
    unique: int
    top_values: list[ValueCount]


class DatasetProfile(BaseModel):
    name: str
    rows: int
    columns: list[ColumnProfile]
    preview: list[dict[str, Any]]
    target_suggestions: list[str]


class TrainRequest(BaseModel):
    target: str
    positive_class: str
    method: Literal["random_forest", "logistic_regression"] = "random_forest"
    excluded: list[str] = []


class ImportanceOut(BaseModel):
    feature: str
    score: float


class DriverOut(BaseModel):
    feature: str
    segment: str
    rate: float
    overall: float
    share: float
    lift: float


class TrainSummary(BaseModel):
    dataset_name: str
    target: str
    positive_class: str
    method: str
    rows_scored: int
    rows_skipped: int
    base_rate: float
    roc_auc: float
    roc_points: list[tuple[float, float]]
    features: list[str]
    identifiers: list[str]
    importance: list[ImportanceOut]
    drivers: list[DriverOut]


class DecisionRequest(BaseModel):
    action_cost: float = Field(20, ge=0)
    saved_value: float = Field(500, gt=0)
    success_rate: float = Field(0.3, gt=0, le=1)


class CurvePointOut(BaseModel):
    threshold: float
    flagged: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    net: float


class DecisionResponse(BaseModel):
    curve: list[CurvePointOut]
    recommended: CurvePointOut


class ReasonOut(BaseModel):
    feature: str
    value: str
    impact: float


class ScoredRow(BaseModel):
    row_id: int
    label: str
    probability: float
    actual: bool | None
    reasons: list[ReasonOut]


class RowsResponse(BaseModel):
    total: int
    source: Literal["training", "new"]
    rows: list[ScoredRow]
