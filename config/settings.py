"""Configuration settings for AI Decision Copilot."""

# Model hyperparameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Model configurations
MODEL_CONFIGS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "params": {
            "max_iter": 1000,
            "random_state": RANDOM_SEED,
            "solver": "lbfgs",
        },
    },
    "random_forest": {
        "name": "Random Forest",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        },
    },
}

# Data preprocessing
MISSING_VALUE_THRESHOLD = 0.5  # Drop columns with >50% missing values
NUMERIC_FEATURES_DTYPE = ["int64", "float64"]
CATEGORICAL_FEATURES_DTYPE = ["object", "category"]

# File upload
MAX_FILE_SIZE_MB = 100
ALLOWED_EXTENSIONS = ["csv"]

# Evaluation
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REGRESSION_METRICS = ["r2", "mae", "rmse", "mape"]

# Cache
CACHE_DURATION = 3600  # 1 hour in seconds
