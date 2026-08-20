import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.state import get_state


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    state = get_state()
    state.df = None
    state.pipeline = None
    state.trained_model = None
    state.trained_model_name = None
    state.model_features = None
    state.target_column = None
    yield
