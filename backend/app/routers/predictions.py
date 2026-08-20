import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas import PredictRequest, PredictResponse, WhatIfRequest, WhatIfResponse
from app.state import get_state

router = APIRouter(prefix="/api", tags=["predictions"])


def _predict_one(features: dict) -> PredictResponse:
    state = get_state()
    if state.trained_model is None:
        raise HTTPException(status_code=400, detail="No trained model — call /api/train first")

    row = np.array([[features.get(f, 0) for f in state.model_features]])
    prediction = int(state.trained_model.predict(row)[0])

    if hasattr(state.trained_model, "predict_proba"):
        proba = state.trained_model.predict_proba(row)[0]
        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = float(max(proba))
    else:
        probability = float(prediction)
        confidence = 1.0

    state.audit_logger.log_prediction(
        prediction=prediction,
        probability=probability,
        confidence=confidence,
        model_name=state.trained_model_name,
        feature_values=features,
    )

    return PredictResponse(prediction=prediction, probability=probability, confidence=confidence)


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return _predict_one(request.features)


@router.post("/whatif", response_model=WhatIfResponse)
def whatif(request: WhatIfRequest):
    baseline = _predict_one(request.baseline_features)
    scenario = _predict_one(request.scenario_features)
    return WhatIfResponse(
        baseline=baseline,
        scenario=scenario,
        delta_probability=round(scenario.probability - baseline.probability, 4),
    )
