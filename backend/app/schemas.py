from typing import Any, Literal

from pydantic import BaseModel


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
