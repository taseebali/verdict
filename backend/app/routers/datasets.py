import io
import re
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile

from app.schemas import DatasetSummary
from app.state import get_state
from src.core.data_handler import DataHandler

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DEMO_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "verdict_demo.csv"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB

# DataHandler prefixes quality warnings with an emoji marker (e.g. "⚠️  ...");
# strip any leading non-ASCII/whitespace characters before they reach the UI.
_LEADING_EMOJI_RE = re.compile(r"^[^\w(]+\s*")


def _clean_warning(warning: str) -> str:
    return _LEADING_EMOJI_RE.sub("", warning).strip()


def _summarize(df: pd.DataFrame) -> DatasetSummary:
    # Use DataHandler for data quality checks
    handler = DataHandler(df)
    summary = handler.get_data_summary()

    numeric_cols = summary["numeric_columns"]
    categorical_cols = summary["categorical_columns"]

    # Calculate overall missing percentage
    missing_pct = round(float(df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100, 2) if df.size else 0.0

    # Get quality warnings from DataHandler
    warnings: list[str] = []
    try:
        _, quality_warnings, _ = handler.validate_data_quality()
        warnings.extend(_clean_warning(w) for w in quality_warnings)
    except Exception:
        # If validation fails, still return basic summary with no warnings
        pass

    # Add correlation check
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
    state.reset_model()
    state.dataset_summary = _summarize(state.df)
    return state.dataset_summary


@router.post("/upload", response_model=DatasetSummary)
async def upload_dataset(file: UploadFile):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large — maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
        )
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV — check the file is valid CSV format")
    state = get_state()
    state.df = df
    state.reset_model()
    state.dataset_summary = _summarize(state.df)
    return state.dataset_summary


@router.get("/current", response_model=DatasetSummary)
def get_current_dataset():
    state = get_state()
    if state.df is None or state.dataset_summary is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet")
    return state.dataset_summary
