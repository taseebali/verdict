from fastapi import APIRouter, Depends, HTTPException

from app.routers.results import summarize
from app.schemas import TrainRequest, TrainSummary
from app.sessions import Session, existing_session, require_dataset
from src.core.scoring import fit_with_oof

router = APIRouter(prefix="/api", tags=["training"])


@router.post("/train", response_model=TrainSummary)
def train(request: TrainRequest, session: Session = Depends(existing_session)):
    with session.lock:
        df = require_dataset(session)
        try:
            model = fit_with_oof(df, request.target, request.positive_class,
                                 request.method, request.excluded)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        session.reset_model()
        session.model = model
        return summarize(session)
