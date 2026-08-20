from fastapi import APIRouter, HTTPException

from app.schemas import TrainRequest, TrainResponse
from app.state import get_state
from src.core.pipeline import MLPipeline
from src.explain.explainability import ExplainabilityAnalyzer
from src.artifacts.model_serializer import ModelSerializer

router = APIRouter(prefix="/api/train", tags=["training"])


@router.post("", response_model=TrainResponse)
def train_model(request: TrainRequest):
    state = get_state()
    if state.df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded — call /api/datasets/demo or /api/datasets/upload first")
    if request.target not in state.df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{request.target}' not found in dataset")

    df = state.df
    if request.features == []:
        raise HTTPException(status_code=400, detail="Select at least one feature")
    if request.features is not None:
        missing = [f for f in request.features if f not in state.df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown feature column(s): {', '.join(missing)}",
            )
        df = df[request.features + [request.target]]

    pipeline = MLPipeline(df, target_col=request.target)
    is_valid, message = pipeline.validate()
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    pipeline.preprocess()
    train_results = pipeline.train([request.method])
    if train_results[request.method].get("status") == "failed":
        raise HTTPException(status_code=500, detail=train_results[request.method]["error"])

    eval_results = pipeline.evaluate([request.method])
    model = pipeline.model_manager.get_models()[request.method]

    X_train, X_test = pipeline.X_train, pipeline.X_test
    y_test = pipeline.y_test
    analyzer = ExplainabilityAnalyzer(model, X_train, X_test, pipeline.preprocessor.get_feature_names())
    importance = analyzer.get_feature_importance(use_cache=False, y_test=y_test)

    state.pipeline = pipeline
    state.trained_model = model
    state.trained_model_name = request.method
    state.model_features = pipeline.preprocessor.get_feature_names()
    state.target_column = request.target

    # Persist the model
    ModelSerializer.save_model(
        model,
        model_name=request.method,
        metadata={"target": request.target, "features": state.model_features},
        overwrite=True,
    )

    return TrainResponse(
        model_name=request.method,
        metrics=eval_results[request.method],
        feature_importance=importance,
    )
