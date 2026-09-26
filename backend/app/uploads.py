"""Reading uploaded CSVs with size/row limits and light cleaning."""
import io
import os

import pandas as pd
from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = int(os.getenv("VERDICT_MAX_ROWS", "100000"))


def coerce_numeric_text(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that are really numbers (e.g. "29.85" with a few
    blank cells) to float, so they aren't treated as huge categories."""
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col].astype("string").str.strip(), errors="coerce")
        if converted.notna().sum() >= 0.95 * df[col].notna().sum():
            df[col] = converted.astype("float64")
    return df


def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large — the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that file as CSV.")
    if df.empty:
        raise HTTPException(status_code=400, detail="The file has no data rows.")
    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"The file has {len(df):,} rows — the limit is {MAX_ROWS:,}.",
        )
    return coerce_numeric_text(df)
