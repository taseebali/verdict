from typing import Any, Optional

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    rows: int
    columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_pct: float
    warnings: list[str]
