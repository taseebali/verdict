import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.sessions import store


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def other_client():
    """A second visitor with its own cookie jar."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    store.clear()
    yield
    store.clear()
