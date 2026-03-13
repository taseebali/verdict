"""Centralized error messages with actionable guidance."""

# Comprehensive error message templates covering 20+ common scenarios
ERROR_MESSAGES = {
    # Data Validation Errors
    "missing_values": {
        "message": "Your dataset has missing values",
        "suggestion": "Missing values can cause training to fail. You can:",
        "actions": [
            "Remove rows with missing values (Pandas: df.dropna())",
            "Remove columns with >50% missing values",
            "Fill missing with mean/median (numeric) or mode (categorical)",
        ],
        "link": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html",
    },
    "class_imbalance": {
        "message": "Classes are severely imbalanced (>80/20 ratio)",
        "suggestion": "Class imbalance can bias models towards majority class:",
        "actions": [
            "Use SMOTE or random oversampling (imbalanced-learn library)",
            "Weight classes inversely in training",
            "Use appropriate metrics (F1, ROC-AUC) instead of accuracy",
            "Collect more data for minority class",
        ],
        "link": "https://imbalanced-learn.org/",
    },
    "too_few_samples": {
        "message": "Dataset is too small for reliable models",
        "suggestion": "Minimum 50 samples recommended (you have {count}):",
        "actions": [
            "Collect more data or find a larger dataset",
            "Try simpler models (Logistic Regression, Decision Trees)",
            "Use stratified cross-validation carefully",
            "Consider using pre-trained models (transfer learning)",
        ],
        "link": "https://scikit-learn.org/stable/modules/cross_validation.html",
    },
    "too_many_features": {
        "message": "Too many features relative to samples",
        "suggestion": "High-dimensional data with few samples causes overfitting:",
        "actions": [
            "Remove low-variance features",
            "Use feature selection (SelectKBest, RFE)",
            "Apply dimensionality reduction (PCA, UMAP)",
            "Collect more data",
        ],
        "link": "https://scikit-learn.org/stable/modules/feature_selection.html",
    },
    "non_numeric_features": {
        "message": "Non-numeric columns detected in features",
        "suggestion": "Models require numeric input. You need to encode categorical features:",
        "actions": [
            "Use one-hot encoding for categorical features",
            "Use label encoding for ordinal features",
            "Remove text columns that cannot be encoded meaningfully",
            "Use feature extraction techniques (e.g., TF-IDF for text)",
        ],
        "link": "https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features",
    },
    "mixed_data_types": {
        "message": "Dataset contains mixed data types",
        "suggestion": "Models work best with consistent data types:",
        "actions": [
            "Check each column's dtype (df.dtypes)",
            "Convert string numbers to numeric (pd.to_numeric())",
            "Handle date columns (pd.to_datetime())",
            "Remove or encode non-standard types",
        ],
        "link": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html",
    },
    "duplicate_rows": {
        "message": "Duplicate rows detected in dataset",
        "suggestion": "Duplicates can inflate performance metrics and waste training data:",
        "actions": [
            "Remove exact duplicates (df.drop_duplicates())",
            "Investigate why duplicates exist",
            "Check if duplicates are legitimate data points",
        ],
        "link": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html",
    },
    "target_imbalance": {
        "message": "Target variable is imbalanced",
        "suggestion": "Imbalanced targets can lead to biased predictions:",
        "actions": [
            "Check class distribution (df.target.value_counts())",
            "Consider stratified train/test split",
            "Use class weights in model training",
            "Adjust decision threshold if using probability predictions",
        ],
        "link": "https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics",
    },
    "infinite_values": {
        "message": "Dataset contains infinite values",
        "suggestion": "Infinite values will break model training:",
        "actions": [
            "Remove rows with infinite values",
            "Replace infinity with NaN, then impute",
            "Investigate source of infinite values",
        ],
        "link": "https://pandas.pydata.org/docs/reference/api/pandas.isnull.html",
    },
    "zero_variance": {
        "message": "One or more features have zero variance",
        "suggestion": "Constant features provide no predictive value:",
        "actions": [
            "Remove features with zero variance",
            "Check for columns with single unique value",
            "Verify data preprocessing didn't collapse values",
        ],
        "link": "https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling",
    },
    # Model Training Errors
    "model_not_found": {
        "message": "Requested model not found",
        "suggestion": "The model '{model}' is not available in VERDICT.",
        "actions": [
            "Check available models in the Models section",
            "Verify model name is spelled correctly",
            "Contact support if model should be available",
        ],
        "link": "https://verdict.example.com/docs/models",
    },
    "training_failed": {
        "message": "Model training failed",
        "suggestion": "This usually indicates a data or parameter issue:",
        "actions": [
            "Check data types are correct",
            "Verify target column exists",
            "Reduce feature count or sample size for testing",
            "Check model parameters are valid",
        ],
        "link": "https://verdict.example.com/docs/troubleshooting",
    },
    "insufficient_memory": {
        "message": "Out of memory during training",
        "suggestion": "Dataset is too large for available memory:",
        "actions": [
            "Use smaller sample for initial testing",
            "Reduce number of features",
            "Sample data before loading",
            "Use batch processing or incremental learning",
        ],
        "link": "https://scikit-learn.org/stable/modules/partial_fit.html",
    },
    "invalid_parameters": {
        "message": "Invalid model parameters provided",
        "suggestion": "One or more parameter values are invalid:",
        "actions": [
            "Check parameter types match expected (e.g., int vs float)",
            "Verify parameter ranges (e.g., 0 < alpha < 1)",
            "Review parameter documentation",
        ],
        "link": "https://scikit-learn.org/stable/modules/classes.html",
    },
    "convergence_warning": {
        "message": "Model did not converge during training",
        "suggestion": "The optimizer stopped before fully converging:",
        "actions": [
            "Increase max_iter parameter",
            "Scale features using StandardScaler",
            "Reduce learning rate",
            "Try different solver algorithm",
        ],
        "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    },
    # Prediction Errors
    "prediction_shape_mismatch": {
        "message": "Input features don't match training data",
        "suggestion": "Number/names of features must match original data:",
        "actions": [
            "Check feature count matches training data",
            "Verify feature names are identical",
            "Apply same preprocessing as training",
        ],
        "link": "https://verdict.example.com/docs/predictions",
    },
    "model_not_trained": {
        "message": "Model has not been trained yet",
        "suggestion": "You must train a model before making predictions:",
        "actions": [
            "Go to Model Training page",
            "Select features and target variable",
            "Click Train Model button",
            "Wait for training to complete",
        ],
        "link": "https://verdict.example.com/docs/workflow",
    },
    # Data Preprocessing Errors
    "split_too_small": {
        "message": "Train/test split resulted in insufficient data",
        "suggestion": "Split ratio leaves too few samples in one set:",
        "actions": [
            "Increase test_size (use larger test set)",
            "Decrease test_size (use smaller test set)",
            "Use k-fold cross-validation instead",
            "Collect more data",
        ],
        "link": "https://scikit-learn.org/stable/modules/cross_validation.html",
    },
    "feature_scaling_failed": {
        "message": "Feature scaling encountered issues",
        "suggestion": "Some features could not be properly scaled:",
        "actions": [
            "Check for infinite or NaN values",
            "Verify all features are numeric",
            "Try different scaling method (StandardScaler vs MinMaxScaler)",
        ],
        "link": "https://scikit-learn.org/stable/modules/preprocessing.html",
    },
    "encoding_failed": {
        "message": "Categorical encoding failed",
        "suggestion": "Could not encode categorical features:",
        "actions": [
            "Check for unexpected values in categorical columns",
            "Use higher cardinality encoding for many unique values",
            "Investigate data quality issues",
        ],
        "link": "https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features",
    },
    # File I/O Errors
    "file_not_found": {
        "message": "File not found: '{filename}'",
        "suggestion": "VERDICT could not locate the requested file:",
        "actions": [
            "Verify file path is correct",
            "Check file exists in the correct location",
            "Upload file using the Data Explorer",
        ],
        "link": "https://verdict.example.com/docs/data-loading",
    },
    "invalid_file_format": {
        "message": "Unsupported file format",
        "suggestion": "VERDICT supports CSV, Excel, and Parquet files:",
        "actions": [
            "Convert file to CSV, XLSX, or PARQUET format",
            "Check file extension is correct",
            "Verify file is not corrupted",
        ],
        "link": "https://verdict.example.com/docs/data-loading",
    },
    "csv_parse_error": {
        "message": "Error parsing CSV file",
        "suggestion": "The CSV file format is invalid:",
        "actions": [
            "Check CSV delimiter is correct (comma, semicolon, etc.)",
            "Verify header row exists",
            "Check for encoding issues (try UTF-8)",
            "Use CSV validator tool to debug",
        ],
        "link": "https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html",
    },
    # General Errors
    "permission_denied": {
        "message": "Permission denied",
        "suggestion": "You don't have permission to access this resource:",
        "actions": [
            "Check your account permissions",
            "Contact administrator",
        ],
        "link": "https://verdict.example.com/docs/security",
    },
    "timeout": {
        "message": "Operation timed out",
        "suggestion": "The operation took too long to complete:",
        "actions": [
            "Try with smaller dataset",
            "Reduce model complexity",
            "Try again later if server is busy",
        ],
        "link": "https://verdict.example.com/docs/performance",
    },
}


def get_error_message(error_key: str, **format_args) -> dict:
    """
    Get error message template and format with provided arguments.

    Args:
        error_key: Key in ERROR_MESSAGES dict
        **format_args: Arguments to format the message

    Returns:
        Dictionary with 'message', 'suggestion', 'actions', 'link'
    """
    if error_key not in ERROR_MESSAGES:
        return {
            "message": "An unexpected error occurred",
            "suggestion": error_key,
            "actions": ["Check VERDICT documentation", "Contact support"],
            "link": "https://verdict.example.com/docs",
        }

    error_def = ERROR_MESSAGES[error_key].copy()

    # Format strings with provided arguments
    if format_args:
        error_def["message"] = error_def["message"].format(**format_args)
        error_def["suggestion"] = error_def["suggestion"].format(**format_args)

    return error_def


def format_error_for_ui(error_key: str, **format_args) -> str:
    """
    Format error message for display in Streamlit UI.

    Args:
        error_key: Key in ERROR_MESSAGES dict
        **format_args: Arguments to format the message

    Returns:
        Formatted string ready for st.error() or st.warning()
    """
    error_def = get_error_message(error_key, **format_args)

    message = error_def["message"]
    suggestion = error_def["suggestion"]
    actions = "\n".join(f"• {action}" for action in error_def["actions"])
    link = error_def.get("link", "")

    formatted = f"""
### {message}

{suggestion}

**Recommended Actions:**
{actions}

---
📚 [Learn More]({link})
"""

    return formatted.strip()


def format_error_for_log(error_key: str, exception: Exception = None, **format_args) -> str:
    """
    Format error message for logging.

    Args:
        error_key: Key in ERROR_MESSAGES dict
        exception: Original exception if available
        **format_args: Arguments to format the message

    Returns:
        Formatted string for logging
    """
    error_def = get_error_message(error_key, **format_args)

    message = error_def["message"]
    suggestion = error_def["suggestion"]
    actions = ", ".join(error_def["actions"])

    log_msg = f"{message} | {suggestion} | Actions: {actions}"

    if exception:
        log_msg += f" | Exception: {str(exception)}"

    return log_msg
