"""Centralized error handling with user guidance."""

import logging
from typing import Dict, Optional, Any
from .error_messages import (
    get_error_message,
    format_error_for_ui,
    format_error_for_log,
)

logger = logging.getLogger(__name__)


class VerdictErrorHandler:
    """Centralized error handling with actionable user guidance."""

    # Error severity levels
    SEVERITY_LEVELS = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌",
        "critical": "🚨",
    }

    @staticmethod
    def handle_validation_error(
        error_type: str,
        details: Optional[str] = None,
        severity: str = "error",
        **format_args,
    ) -> Dict[str, Any]:
        """
        Generate user-friendly error response for validation errors.

        Args:
            error_type: Key in ERROR_MESSAGES dict
            details: Additional error details
            severity: 'info', 'warning', 'error', or 'critical'
            **format_args: Arguments to format error messages

        Returns:
            Dictionary with 'ui_message', 'log_message', 'severity', 'is_critical'
        """
        error_def = get_error_message(error_type, **format_args)
        ui_message = format_error_for_ui(error_type, **format_args)
        log_message = format_error_for_log(error_type, exception=None, **format_args)

        # Add details if provided
        if details:
            log_message += f" | Details: {details}"

        # Log the error
        log_func = getattr(logger, severity, logger.error)
        log_func(log_message)

        return {
            "ui_message": ui_message,
            "log_message": log_message,
            "severity": severity,
            "is_critical": severity in ("error", "critical"),
            "error_type": error_type,
            "details": details or "",
        }

    @staticmethod
    def handle_exception(
        exception: Exception,
        error_type: str,
        user_message: str = "An error occurred",
        severity: str = "error",
        **format_args,
    ) -> Dict[str, Any]:
        """
        Generate user-friendly error response for exceptions.

        Args:
            exception: The exception that occurred
            error_type: Key in ERROR_MESSAGES dict
            user_message: Custom message for user
            severity: 'info', 'warning', 'error', or 'critical'
            **format_args: Arguments to format error messages

        Returns:
            Dictionary with error information
        """
        error_def = get_error_message(error_type, **format_args)
        ui_message = format_error_for_ui(error_type, **format_args)
        log_message = format_error_for_log(
            error_type, exception=exception, **format_args
        )

        # Log the exception
        log_func = getattr(logger, severity, logger.error)
        log_func(log_message, exc_info=True)

        return {
            "ui_message": ui_message,
            "log_message": log_message,
            "severity": severity,
            "is_critical": severity in ("error", "critical"),
            "error_type": error_type,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
        }

    @staticmethod
    def validate_dataframe(
        df,
        min_rows: int = 10,
        min_cols: int = 1,
        allow_missing: bool = False,
        allow_categorical: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate DataFrame and return structured error if invalid.

        Args:
            df: pandas DataFrame to validate
            min_rows: Minimum required rows
            min_cols: Minimum required columns
            allow_missing: Whether to allow missing values
            allow_categorical: Whether to allow categorical columns

        Returns:
            Dictionary with 'valid', 'errors' (list of error responses)
        """
        errors = []

        # Check if DataFrame
        try:
            if not hasattr(df, "shape"):
                errors.append(
                    VerdictErrorHandler.handle_validation_error(
                        "invalid_parameters", details="Input is not a DataFrame"
                    )
                )
                return {"valid": False, "errors": errors}
        except Exception as e:
            errors.append(
                VerdictErrorHandler.handle_exception(
                    e, "training_failed", details="Could not validate input"
                )
            )
            return {"valid": False, "errors": errors}

        # Check size
        if df.shape[0] < min_rows:
            errors.append(
                VerdictErrorHandler.handle_validation_error(
                    "too_few_samples", count=df.shape[0]
                )
            )

        if df.shape[1] < min_cols:
            errors.append(
                VerdictErrorHandler.handle_validation_error(
                    "invalid_parameters", details=f"Need at least {min_cols} columns"
                )
            )

        # Check missing values
        if not allow_missing and df.isnull().any().any():
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            errors.append(
                VerdictErrorHandler.handle_validation_error(
                    "missing_values",
                    details=f"{missing_pct:.1f}% of data is missing",
                )
            )

        # Check data types
        if not allow_categorical:
            non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
            if len(non_numeric_cols) > 0:
                errors.append(
                    VerdictErrorHandler.handle_validation_error(
                        "non_numeric_features",
                        details=f"Found categorical columns: {list(non_numeric_cols)}",
                    )
                )

        # Check for infinite values
        try:
            import numpy as np

            numeric_cols = df.select_dtypes(include=["number"]).columns
            has_inf = np.isinf(df[numeric_cols]).any().any()
            if has_inf:
                errors.append(
                    VerdictErrorHandler.handle_validation_error("infinite_values")
                )
        except Exception:
            pass

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            errors.append(
                VerdictErrorHandler.handle_validation_error(
                    "duplicate_rows",
                    details=f"Found {duplicate_count} duplicate rows",
                    severity="warning",
                )
            )

        return {"valid": len(errors) == 0, "errors": errors}

    @staticmethod
    def validate_column_exists(
        df, column: str, error_if_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Check if a column exists in DataFrame.

        Args:
            df: pandas DataFrame
            column: Column name to check
            error_if_missing: Whether to return error dict if missing

        Returns:
            Dictionary with 'exists' bool and optional 'error'
        """
        exists = column in df.columns if hasattr(df, "columns") else False

        result = {"exists": exists}

        if not exists and error_if_missing:
            result["error"] = VerdictErrorHandler.handle_validation_error(
                "invalid_parameters",
                details=f"Column '{column}' not found. Available columns: {list(df.columns)}",
            )

        return result

    @staticmethod
    def validate_numeric_column(
        df, column: str, allow_missing: bool = False
    ) -> Dict[str, Any]:
        """
        Check if a column is numeric.

        Args:
            df: pandas DataFrame
            column: Column name to check
            allow_missing: Whether missing values are acceptable

        Returns:
            Dictionary with 'valid' bool and optional 'error'
        """
        # Check existence first
        if column not in df.columns:
            return {
                "valid": False,
                "error": VerdictErrorHandler.handle_validation_error(
                    "invalid_parameters", details=f"Column '{column}' not found"
                ),
            }

        # Check if numeric
        try:
            import pandas as pd

            if not pd.api.types.is_numeric_dtype(df[column]):
                return {
                    "valid": False,
                    "error": VerdictErrorHandler.handle_validation_error(
                        "non_numeric_features",
                        details=f"Column '{column}' is {df[column].dtype}, not numeric",
                    ),
                }
        except Exception as e:
            return {
                "valid": False,
                "error": VerdictErrorHandler.handle_exception(
                    e, "invalid_parameters", details=f"Could not check type of '{column}'"
                ),
            }

        # Check for missing if not allowed
        if not allow_missing and df[column].isnull().any():
            missing_pct = (df[column].isnull().sum() / len(df)) * 100
            return {
                "valid": False,
                "error": VerdictErrorHandler.handle_validation_error(
                    "missing_values",
                    details=f"Column '{column}' has {missing_pct:.1f}% missing values",
                    severity="warning",
                ),
            }

        return {"valid": True}

    @staticmethod
    def get_error_summary(errors: list) -> str:
        """
        Generate a summary of multiple errors.

        Args:
            errors: List of error dictionaries

        Returns:
            Formatted string summarizing all errors
        """
        if not errors:
            return "No errors"

        critical = [e for e in errors if e.get("is_critical")]
        warnings = [e for e in errors if not e.get("is_critical")]

        summary = []
        if critical:
            summary.append(f"🚨 **{len(critical)} Critical Error(s):**")
            for e in critical:
                summary.append(f"  • {e.get('error_type', 'Unknown')}")

        if warnings:
            summary.append(f"⚠️ **{len(warnings)} Warning(s):**")
            for e in warnings:
                summary.append(f"  • {e.get('error_type', 'Unknown')}")

        return "\n".join(summary)
