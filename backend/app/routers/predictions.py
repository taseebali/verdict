import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas import PredictRequest, PredictResponse, WhatIfRequest, WhatIfResponse
from app.state import get_state

router = APIRouter(prefix="/api", tags=["predictions"])


def _transform_features(features: dict) -> np.ndarray:
    """Route a raw feature dict through the same label-encoding + scaling the
    training data went through, so it lands in the feature space the model
    was actually trained on."""
    state = get_state()
    preprocessor = state.pipeline.preprocessor

    row = {col: features.get(col, 0) for col in state.model_features}
    df = pd.DataFrame([row], columns=state.model_features)

    for col, encoder in preprocessor.label_encoders.items():
        if col not in df.columns:
            continue
        value = str(df.at[0, col])
        try:
            df[col] = encoder.transform([value])
        except ValueError:
            known = ", ".join(map(str, encoder.classes_))
            raise HTTPException(
                status_code=400,
                detail=f"Unknown value '{value}' for feature '{col}'. Known values: {known}",
            )

    numeric_cols = [c for c in preprocessor.numeric_cols if c in df.columns]
    if numeric_cols:
        df[numeric_cols] = preprocessor.scaler.transform(df[numeric_cols])

    return df[state.model_features].to_numpy()


def _predict_one(features: dict) -> PredictResponse:
    state = get_state()
    if state.trained_model is None or state.pipeline is None:
        raise HTTPException(status_code=400, detail="No trained model — call /api/train first")

    row = _transform_features(features)
    prediction = int(state.trained_model.predict(row)[0])

    if hasattr(state.trained_model, "predict_proba"):
        proba = state.trained_model.predict_proba(row)[0]
        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = float(max(proba))
    else:
        # Model has no predict_proba (e.g. a hard-voting ensemble) — there's
        # no real probability distribution to report, so represent the hard
        # decision as a valid 0-1 value rather than leaking the raw class
        # label (which can be any integer, not a probability).
        probability = 1.0 if prediction == 1 else 0.0
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
