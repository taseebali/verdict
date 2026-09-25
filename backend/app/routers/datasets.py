import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, UploadFile

from app.schemas import ColumnProfile, DatasetProfile, ValueCount
from app.sessions import Session, get_session, require_dataset
from app.uploads import read_csv_upload
from src.core.scoring import infer_roles, target_suggestions

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DEMO_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "verdict_demo.csv"
PREVIEW_ROWS = 20
MAX_TOP_VALUES = 20


def build_profile(name: str, df: pd.DataFrame) -> DatasetProfile:
    roles = infer_roles(df)
    kinds = {**{c: "numeric" for c in roles.numeric},
             **{c: "categorical" for c in roles.categorical},
             **{c: "identifier" for c in roles.identifiers}}
    columns = []
    for col in df.columns:
        series = df[col]
        unique = int(series.nunique())
        top = []
        if unique <= MAX_TOP_VALUES:
            counts = series.dropna().astype(str).value_counts()
            top = [ValueCount(value=str(v), count=int(n)) for v, n in counts.items()]
        columns.append(ColumnProfile(
            name=str(col),
            kind=kinds[col],
            missing_pct=round(float(series.isna().mean() * 100), 1),
            unique=unique,
            top_values=top,
        ))
    preview = json.loads(df.head(PREVIEW_ROWS).to_json(orient="records"))
    return DatasetProfile(name=name, rows=len(df), columns=columns, preview=preview,
                          target_suggestions=target_suggestions(df))


def _store(session: Session, name: str, df: pd.DataFrame) -> DatasetProfile:
    with session.lock:
        session.df = df
        session.dataset_name = name
        session.reset_model()
    return build_profile(name, df)


@router.post("/demo", response_model=DatasetProfile)
def load_demo(session: Session = Depends(get_session)):
    return _store(session, DEMO_DATA_PATH.name, pd.read_csv(DEMO_DATA_PATH))


@router.post("/upload", response_model=DatasetProfile)
def upload(file: UploadFile, session: Session = Depends(get_session)):
    df = read_csv_upload(file)
    return _store(session, file.filename, df)


@router.get("/current", response_model=DatasetProfile)
def current(session: Session = Depends(get_session)):
    df = require_dataset(session)
    return build_profile(session.dataset_name, df)
