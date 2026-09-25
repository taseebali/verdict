from dataclasses import asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.schemas import (
    CurvePointOut,
    DecisionRequest,
    DecisionResponse,
    DriverOut,
    ImportanceOut,
    NewDecisionRequest,
    NewDecisionResponse,
    ReasonOut,
    RowsResponse,
    ScoreResponse,
    ScoredRow,
    TrainSummary,
    WhatIfRequest,
    WhatIfResponse,
)
from app.sessions import NewScores, Session, existing_session, require_model
from app.uploads import read_csv_upload
from src.core.scoring import prepare_features, row_reasons, score_frame
from src.decision.decision_curve import decision_curve, expected_net_for_new, recommend

router = APIRouter(prefix="/api/results", tags=["results"])

Source = Literal["training", "new"]
EXPORT_REASON_LIMIT = 1000


def summarize(session: Session) -> TrainSummary:
    m = session.model
    return TrainSummary(
        dataset_name=session.dataset_name or "",
        target=m.target,
        positive_class=m.positive_class,
        method=m.method,
        rows_scored=len(m.oof_proba),
        rows_skipped=m.rows_skipped,
        base_rate=m.base_rate,
        roc_auc=m.roc_auc,
        roc_points=m.roc_points,
        features=m.features,
        identifiers=m.identifiers,
        importance=[ImportanceOut(feature=f, score=s) for f, s in m.importance],
        drivers=[DriverOut(feature=d.feature, segment=d.segment, rate=d.rate,
                           overall=d.overall, share=d.share, lift=d.lift) for d in m.drivers],
    )


def source_view(session: Session, source: Source) -> tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """(rows, probabilities, known outcomes or None), positionally aligned."""
    model = require_model(session)
    if source == "training":
        return session.df.loc[model.row_ids], model.oof_proba, model.actual
    if session.new_scores is None:
        raise HTTPException(status_code=404, detail="No new file scored yet.")
    return session.new_scores.df, session.new_scores.proba, None


def reasons_for(session: Session, source: Source, frame: pd.DataFrame) -> list:
    """Per-row reasons, computed once per (source, row id) and cached on the session."""
    model = session.model
    keys = [(source, int(i)) for i in frame.index]
    missing = [k for k in keys if k not in session.reasons_cache]
    if missing:
        subset = frame.loc[[i for _, i in missing]]
        X = prepare_features(subset, model.numeric, model.categorical)
        for key, reasons in zip(missing, row_reasons(model, X)):
            session.reasons_cache[key] = reasons
    return [session.reasons_cache[k] for k in keys]


def row_labels(frame: pd.DataFrame, identifiers: list[str]) -> list[str]:
    id_col = next((c for c in identifiers if c in frame.columns), None)
    if id_col is not None:
        return [f"#{int(i) + 1}" if pd.isna(v) else str(v) for i, v in zip(frame.index, frame[id_col])]
    return [f"#{int(i) + 1}" for i in frame.index]


@router.get("/summary", response_model=TrainSummary)
def summary(session: Session = Depends(existing_session)):
    with session.lock:
        require_model(session)
        return summarize(session)


@router.post("/decision", response_model=DecisionResponse)
def decision(request: DecisionRequest, session: Session = Depends(existing_session)):
    model = require_model(session)
    curve = decision_curve(model.oof_proba, model.actual, request.action_cost,
                           request.saved_value, request.success_rate)
    return DecisionResponse(curve=[CurvePointOut(**asdict(p)) for p in curve],
                            recommended=CurvePointOut(**asdict(recommend(curve))))


@router.get("/rows", response_model=RowsResponse)
def rows(offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100),
         source: Source = "training", session: Session = Depends(existing_session)):
    with session.lock:
        frame, proba, actual = source_view(session, source)
        order = np.argsort(-proba, kind="stable")[offset:offset + limit]
        page = frame.iloc[order]
        reasons = reasons_for(session, source, page)
        labels = row_labels(page, session.model.identifiers)
        return RowsResponse(total=len(proba), source=source, rows=[
            ScoredRow(
                row_id=int(row_id),
                label=labels[k],
                probability=float(proba[order[k]]),
                actual=None if actual is None else bool(actual[order[k]]),
                reasons=[ReasonOut(**asdict(r)) for r in reasons[k]],
            )
            for k, row_id in enumerate(page.index)
        ])


@router.get("/export.csv")
def export(threshold: float = Query(0.5, ge=0, le=1), source: Source = "training",
           session: Session = Depends(existing_session)):
    with session.lock:
        frame, proba, _ = source_view(session, source)
        order = np.argsort(-proba, kind="stable")
        out = frame.iloc[order].copy()
        p = proba[order]
        out["verdict_probability"] = np.round(p, 4)
        out["verdict_flag"] = p > threshold
        texts = [""] * len(out)
        flagged = np.flatnonzero(p > threshold)[:EXPORT_REASON_LIMIT]
        if len(flagged):
            reasons = reasons_for(session, source, out.iloc[flagged])
            for pos, row in zip(flagged, reasons):
                texts[pos] = "; ".join(f"{r.feature} = {r.value}" for r in row)
        out["verdict_reasons"] = texts
        return Response(
            content=out.to_csv(index=False),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="verdict_{source}_scores.csv"'},
        )


def _jsonable(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


@router.post("/whatif", response_model=WhatIfResponse)
def whatif(request: WhatIfRequest, session: Session = Depends(existing_session)):
    with session.lock:
        model = require_model(session)
        if request.row_id not in session.df.index:
            raise HTTPException(status_code=404, detail="Row not found.")
        unknown = sorted(set(request.changes) - set(model.features))
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown feature(s): {', '.join(unknown)}")
        base = session.df.loc[[request.row_id]].astype(object)
        scenario = base.copy()
        for feature, value in request.changes.items():
            scenario.at[request.row_id, feature] = value
        baseline = float(score_frame(model, base)[0])
        changed = float(score_frame(model, scenario)[0])
        return WhatIfResponse(
            baseline=baseline,
            scenario=changed,
            delta=changed - baseline,
            features={c: _jsonable(base.at[request.row_id, c]) for c in model.features},
        )


@router.post("/score", response_model=ScoreResponse)
def score(file: UploadFile, session: Session = Depends(existing_session)):
    df = read_csv_upload(file)
    with session.lock:
        model = require_model(session)
        try:
            proba = score_frame(model, df)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        session.new_scores = NewScores(name=file.filename, df=df, proba=proba)
        session.reasons_cache = {k: v for k, v in session.reasons_cache.items() if k[0] != "new"}
    return ScoreResponse(rows_scored=len(df), source="new", name=file.filename)


@router.post("/new/decision", response_model=NewDecisionResponse)
def new_decision(request: NewDecisionRequest, session: Session = Depends(existing_session)):
    with session.lock:
        _, proba, _ = source_view(session, "new")
        flagged, net = expected_net_for_new(proba, request.threshold, request.action_cost,
                                            request.saved_value, request.success_rate)
    return NewDecisionResponse(flagged=flagged, expected_net=net)
