from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.artifacts.model_serializer import ModelSerializer

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/{name}/download")
def download_model(name: str):
    if not ModelSerializer.model_exists(name):
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    model_path = ModelSerializer.MODELS_DIR / f"{name}.joblib"
    return FileResponse(str(model_path), filename=f"{name}.joblib")
