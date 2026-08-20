import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile

from app.schemas import DatasetSummary
from app.state import get_state

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DEMO_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "verdict_demo.csv"


def _summarize(df: pd.DataFrame) -> DatasetSummary:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    missing_pct = round(float(df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100, 2) if df.size else 0.0

    warnings: list[str] = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        high_corr_pairs = 0
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if corr.iloc[i, j] > 0.9:
                    high_corr_pairs += 1
        if high_corr_pairs:
            warnings.append(f"High correlation detected: {high_corr_pairs} feature pairs > 0.9")

    return DatasetSummary(
        rows=df.shape[0],
        columns=df.shape[1],
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        missing_pct=missing_pct,
        warnings=warnings,
    )


@router.post("/demo", response_model=DatasetSummary)
def load_demo_dataset():
    state = get_state()
    if not DEMO_DATA_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Demo dataset not found at {DEMO_DATA_PATH}")
    state.df = pd.read_csv(DEMO_DATA_PATH)
    state.pipeline = None
    return _summarize(state.df)


@router.post("/upload", response_model=DatasetSummary)
async def upload_dataset(file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    state = get_state()
    state.df = df
    state.pipeline = None
    return _summarize(state.df)


@router.get("/current", response_model=DatasetSummary)
def get_current_dataset():
    state = get_state()
    if state.df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet")
    return _summarize(state.df)
