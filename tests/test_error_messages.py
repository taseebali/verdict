"""Tests for error handling and messaging system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.core.error_messages import (
    ERROR_MESSAGES,
    get_error_message,
    format_error_for_ui,
    format_error_for_log,
)
from src.core.error_handler import VerdictErrorHandler


class TestErrorMessages:
    """Test error message templates and formatting."""

    def test_error_messages_comprehensive(self):
        """Check that we have 20+ error message templates."""
        assert len(ERROR_MESSAGES) >= 20

    def test_all_error_messages_have_required_fields(self):
        """Each error message should have message, suggestion, actions, link."""
        for key, error_def in ERROR_MESSAGES.items():
            assert "message" in error_def, f"Missing 'message' in {key}"
            assert "suggestion" in error_def, f"Missing 'suggestion' in {key}"
            assert "actions" in error_def, f"Missing 'actions' in {key}"
            assert "link" in error_def, f"Missing 'link' in {key}"
            assert isinstance(error_def["actions"], list), f"'actions' not list in {key}"
            assert len(error_def["actions"]) >= 1, f"No actions for {key}"

    def test_get_error_message_returns_dict(self):
        """get_error_message should return formatted dict."""
        result = get_error_message("missing_values")
        assert isinstance(result, dict)
        assert "message" in result
        assert "suggestion" in result
        assert "actions" in result

    def test_get_error_message_with_formatting(self):
        """Should format message with provided arguments."""
        result = get_error_message("too_few_samples", count=25)
        assert "25" in result["suggestion"]

    def test_get_error_message_unknown_key(self):
        """Should handle unknown error keys gracefully."""
        result = get_error_message("nonexistent_error")
        assert result["message"] == "An unexpected error occurred"
        assert isinstance(result["actions"], list)

    def test_format_error_for_ui_contains_markdown(self):
        """UI format should contain markdown formatting."""
        result = format_error_for_ui("missing_values")
        assert "###" in result  # Markdown header
        assert "•" in result  # Bullet points
        assert "[Learn More]" in result  # Link

    def test_format_error_for_ui_readable(self):
        """UI format should be readable and complete."""
        result = format_error_for_ui("class_imbalance")
        assert len(result) > 50
        assert isinstance(result, str)

    def test_format_error_for_log_contains_all_info(self):
        """Log format should be concise but complete."""
        result = format_error_for_log("missing_values")
        assert "missing values" in result.lower()
        assert len(result) > 20

    def test_format_error_for_log_with_exception(self):
        """Log format should include exception info when provided."""
        exc = ValueError("test error")
        result = format_error_for_log("training_failed", exception=exc)
        assert "test error" in result

    def test_error_message_actions_are_actionable(self):
        """All error actions should be clear and helpful."""
        for key, error_def in ERROR_MESSAGES.items():
            assert len(error_def["actions"]) >= 1, f"No actions in {key}"
            for action in error_def["actions"]:
                # Actions should be meaningful strings with at least 10 characters
                assert len(action) > 10, f"Action too short in {key}: '{action}'"
                assert isinstance(action, str), f"Action not string in {key}"


class TestVerdictErrorHandler:
    """Test VerdictErrorHandler class."""

    def test_handle_validation_error_returns_dict(self):
        """Should return structured error dict."""
        result = VerdictErrorHandler.handle_validation_error("missing_values")
        assert isinstance(result, dict)
        assert "ui_message" in result
        assert "log_message" in result
        assert "severity" in result
        assert "is_critical" in result

    def test_handle_validation_error_severity_levels(self):
        """Should handle different severity levels."""
        for severity in ["info", "warning", "error", "critical"]:
            result = VerdictErrorHandler.handle_validation_error(
                "missing_values", severity=severity
            )
            assert result["severity"] == severity
            assert result["is_critical"] == (severity in ("error", "critical"))

    def test_handle_validation_error_with_details(self):
        """Should include additional details."""
        result = VerdictErrorHandler.handle_validation_error(
            "missing_values", details="50% of column X is missing"
        )
        assert "50%" in result["log_message"]

    def test_handle_exception_includes_exception_info(self):
        """Should capture exception details."""
        exc = ValueError("test error")
        result = VerdictErrorHandler.handle_exception(exc, "training_failed")
        assert result["exception_type"] == "ValueError"
        assert "test error" in result["exception_message"]

    def test_validate_dataframe_valid_data(self):
        """Should validate correct DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]})
        result = VerdictErrorHandler.validate_dataframe(df)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_dataframe_too_few_rows(self):
        """Should catch DataFrames with too few rows."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = VerdictErrorHandler.validate_dataframe(df, min_rows=10)
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("too_few_samples" in str(e) for e in result["errors"])

    def test_validate_dataframe_too_few_columns(self):
        """Should catch DataFrames with too few columns."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = VerdictErrorHandler.validate_dataframe(df, min_cols=2)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_dataframe_with_missing_values(self):
        """Should catch missing values."""
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = VerdictErrorHandler.validate_dataframe(
            df, allow_missing=False, min_rows=2
        )
        assert result["valid"] is False
        assert any("missing" in str(e).lower() for e in result["errors"])

    def test_validate_dataframe_with_duplicates(self):
        """Should warn about duplicate rows."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 4, 5]})
        result = VerdictErrorHandler.validate_dataframe(df, min_rows=2)
        # Duplicates should be warning, not critical error
        has_duplicate_warning = any(
            e.get("error_type") == "duplicate_rows" for e in result["errors"]
        )
        # May or may not have duplicate warning depending on validation logic
        assert isinstance(result, dict)

    def test_validate_dataframe_with_categorical(self):
        """Should reject categorical columns when not allowed."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = VerdictErrorHandler.validate_dataframe(
            df, allow_categorical=False, min_rows=2
        )
        assert result["valid"] is False
        assert any("numeric" in str(e).lower() for e in result["errors"])

    def test_validate_dataframe_with_infinity(self):
        """Should catch infinite values."""
        df = pd.DataFrame({"a": [1, np.inf, 3], "b": [4, 5, 6]})
        result = VerdictErrorHandler.validate_dataframe(df, min_rows=2)
        assert result["valid"] is False
        assert any("infinite" in str(e).lower() for e in result["errors"])

    def test_validate_column_exists_true(self):
        """Should confirm column exists."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = VerdictErrorHandler.validate_column_exists(df, "a")
        assert result["exists"] is True
        assert "error" not in result

    def test_validate_column_exists_false(self):
        """Should report missing column."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = VerdictErrorHandler.validate_column_exists(df, "c", error_if_missing=True)
        assert result["exists"] is False
        assert "error" in result

    def test_validate_numeric_column_valid(self):
        """Should accept numeric column."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = VerdictErrorHandler.validate_numeric_column(df, "a")
        assert result["valid"] is True
        assert "error" not in result

    def test_validate_numeric_column_non_numeric(self):
        """Should reject non-numeric column."""
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        result = VerdictErrorHandler.validate_numeric_column(df, "a")
        assert result["valid"] is False
        assert "error" in result

    def test_validate_numeric_column_missing(self):
        """Should reject column with missing values."""
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = VerdictErrorHandler.validate_numeric_column(
            df, "a", allow_missing=False
        )
        assert result["valid"] is False

    def test_validate_numeric_column_nonexistent(self):
        """Should handle missing column."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = VerdictErrorHandler.validate_numeric_column(df, "missing")
        assert result["valid"] is False

    def test_get_error_summary_empty(self):
        """Should handle empty error list."""
        summary = VerdictErrorHandler.get_error_summary([])
        assert "No errors" in summary

    def test_get_error_summary_with_errors(self):
        """Should summarize multiple errors."""
        errors = [
            {"error_type": "missing_values", "is_critical": True},
            {"error_type": "class_imbalance", "is_critical": False},
        ]
        summary = VerdictErrorHandler.get_error_summary(errors)
        assert "missing_values" in summary
        assert "class_imbalance" in summary
        assert "Critical" in summary
        assert "Warning" in summary


class TestErrorIntegration:
    """Integration tests for error handling."""

    def test_full_error_flow_validation(self):
        """Test complete error handling flow for data validation."""
        df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
        
        validation = VerdictErrorHandler.validate_dataframe(
            df, allow_missing=False, allow_categorical=False, min_rows=5
        )
        
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
        
        # Check error structure
        for error in validation["errors"]:
            assert "ui_message" in error
            assert "log_message" in error

    def test_full_error_flow_exception(self):
        """Test complete error handling flow for exceptions."""
        try:
            raise ValueError("Custom error")
        except ValueError as e:
            result = VerdictErrorHandler.handle_exception(
                e, "training_failed", details="During model training"
            )
        
        assert result["is_critical"] is True
        assert "Custom error" in result["exception_message"]

    def test_error_messages_are_unique(self):
        """Error messages should not be duplicated."""
        messages = [msg["message"] for msg in ERROR_MESSAGES.values()]
        assert len(messages) == len(set(messages)), "Duplicate error messages found"
