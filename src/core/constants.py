"""Shared constants and thresholds for the VERDICT platform."""

# ============================================================================
# Data Handling Constants
# ============================================================================

# Minimum dataset requirements
MIN_DATASET_ROWS = 10
MIN_DATASET_COLUMNS = 2

# Missing value handling
MISSING_VALUE_THRESHOLD = 0.5  # Drop columns with >50% missing values
MISSING_VALUE_WARNING_THRESHOLD = 5.0  # Warn if >5% missing

# Data quality thresholds
SMALL_DATASET_ROW_COUNT = 100  # Datasets with <100 rows are considered small
DUPLICATE_ROW_WARNING_THRESHOLD = 1
CLASS_IMBALANCE_WARNING_THRESHOLD = 10.0  # Warn if any class <10%
HIGH_CORRELATION_THRESHOLD = 0.9  # Warn if feature correlation > 0.9

# ============================================================================
# Model Training Constants
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Train/test split
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Cross-validation
CV_FOLDS = 5

# ============================================================================
# Problem Type Detection
# ============================================================================

# Multiclass detection threshold
MULTICLASS_UNIQUE_VALUES_THRESHOLD = 10  # If >10 unique values, consider multiclass
REGRESSION_UNIQUE_VALUES_THRESHOLD = 100  # If >100 unique continuous values, regression
MIN_SAMPLES_PER_CLASS = 5  # Minimum samples per class for valid multiclass

# ============================================================================
# Feature Processing
# ============================================================================

NUMERIC_FEATURES_DTYPE = ["int64", "float64"]
CATEGORICAL_FEATURES_DTYPE = ["object", "category"]

# ============================================================================
# Model Configurations
# ============================================================================

# Logistic Regression parameters
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

# Random Forest parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ============================================================================
# Evaluation & Metrics
# ============================================================================

CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REGRESSION_METRICS = ["r2", "mae", "rmse", "mape"]

# ============================================================================
# File Handling
# ============================================================================

MAX_FILE_SIZE_MB = 100
ALLOWED_FILE_EXTENSIONS = ["csv"]

# ============================================================================
# Cache Configuration
# ============================================================================

CACHE_DURATION_SECONDS = 3600  # 1 hour

# ============================================================================
# API Configuration
# ============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = False

# ============================================================================
# Confidence & Decision Thresholds
# ============================================================================

DEFAULT_DECISION_THRESHOLD = 0.5  # Default threshold for binary classification
MIN_CONFIDENCE_FOR_DECISION = 0.6  # Minimum confidence to make a decision
HIGH_CONFIDENCE_THRESHOLD = 0.8  # Threshold for "high confidence" decisions

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# UI Configuration
# ============================================================================

STREAMLIT_PAGE_TITLE = "Verdict ML Platform"
STREAMLIT_PAGE_ICON = "🚀"
STREAMLIT_LAYOUT = "wide"

# ============================================================================
# Error Messages & Validation
# ============================================================================

ERROR_MESSAGES = {
    "empty_dataset": "Dataset is empty.",
    "insufficient_rows": f"Dataset must have at least {MIN_DATASET_ROWS} rows.",
    "insufficient_columns": f"Dataset must have at least {MIN_DATASET_COLUMNS} columns.",
    "invalid_target_column": "Target column not found in dataset.",
    "no_numeric_features": "No numeric features found for model training.",
    "class_too_small": "Smallest class has too few samples for training.",
}
