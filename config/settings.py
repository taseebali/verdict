"""Project configuration - imports from core constants for centralization."""

# Import core constants to maintain single source of truth
from src.core.constants import (
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
    CV_FOLDS,
    MISSING_VALUE_THRESHOLD,
    NUMERIC_FEATURES_DTYPE,
    CATEGORICAL_FEATURES_DTYPE,
    MAX_FILE_SIZE_MB,
    ALLOWED_FILE_EXTENSIONS,
    CACHE_DURATION_SECONDS,
    API_PORT,
    API_HOST,
    API_DEBUG,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
)

# Model configurations (backward compatible with existing code)
MODEL_CONFIGS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "params": LOGISTIC_REGRESSION_PARAMS,
    },
    "random_forest": {
        "name": "Random Forest",
        "params": RANDOM_FOREST_PARAMS,
    },
}

# Backward compatibility aliases
ALLOWED_EXTENSIONS = ALLOWED_FILE_EXTENSIONS
CACHE_DURATION = CACHE_DURATION_SECONDS
CV_RANDOM_STATE = RANDOM_SEED

# API Authentication (Phase 3)
API_AUTH_TOKEN = None  # Set to enable token-based auth
API_AUTH_ENABLED = False
