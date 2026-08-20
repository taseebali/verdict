import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.state import get_state
from src.artifacts.model_serializer import ModelSerializer


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    state = get_state()
    state.df = None
    state.dataset_summary = None
    state.pipeline = None
    state.trained_model = None
    state.trained_model_name = None
    state.model_features = None
    state.target_column = None
    state.audit_logger.clear_logs()
    yield


@pytest.fixture(autouse=True)
def isolate_models_dir(tmp_path, monkeypatch):
    """Redirect model persistence to a temp directory so tests don't leave
    .joblib artifacts in the working tree."""
    monkeypatch.setattr(ModelSerializer, "MODELS_DIR", tmp_path / "models")
    yield
