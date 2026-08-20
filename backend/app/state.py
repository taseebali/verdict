"""In-memory application state — single active session, no auth/DB."""
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.pipeline import MLPipeline
from src.decision.decision_audit_logger import DecisionAuditLogger


class AppState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.pipeline: Optional[MLPipeline] = None
        self.trained_model: Optional[Any] = None
        self.trained_model_name: Optional[str] = None
        self.model_features: Optional[list[str]] = None
        self.target_column: Optional[str] = None
        self.audit_logger = DecisionAuditLogger()


_state = AppState()


def get_state() -> AppState:
    return _state
