"""Tests for validators module."""

import pytest
import pandas as pd
import numpy as np
from src.core.validators import DataValidator


class TestDataValidatorFormat:
    """Test format validation."""

    def test_validate_format_valid_dataframe(self):
        """Test validation passes for valid dataframe."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        is_valid, msg = DataValidator.validate_format(df)
        assert is_valid is True

    def test_validate_format_none_input(self):
        """Test validation fails for None input."""
        is_valid, msg = DataValidator.validate_format(None)
        assert is_valid is False
        assert 'None' in msg

    def test_validate_format_empty_dataframe(self):
        """Test validation fails for empty dataframe."""
        df = pd.DataFrame()
        is_valid, msg = DataValidator.validate_format(df)
        assert is_valid is False
        assert 'empty' in msg.lower()


class TestDataValidatorStructure:
    """Test structure validation."""

    @pytest.fixture
    def valid_df(self):
        """Create a valid test dataframe."""
        return pd.DataFrame({
            'A': range(20),
            'B': range(20, 40),
            'C': range(40, 60)
        })

    def test_validate_structure_valid(self, valid_df):
        """Test validation passes for valid structure."""
        is_valid, msg = DataValidator.validate_structure(valid_df)
        assert is_valid is True

    def test_validate_structure_insufficient_rows(self):
        """Test validation fails for insufficient rows."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        is_valid, msg = DataValidator.validate_structure(df)
        assert is_valid is False
        assert 'rows' in msg.lower()

    def test_validate_structure_insufficient_columns(self):
        """Test validation fails for insufficient columns."""
        df = pd.DataFrame({'A': range(20)})
        is_valid, msg = DataValidator.validate_structure(df)
        assert is_valid is False
        assert 'column' in msg.lower()

    def test_validate_basic(self, valid_df):
        """Test basic validation combines format and structure."""
        is_valid, msg = DataValidator.validate_basic(valid_df)
        assert is_valid is True


class TestDataValidatorQuality:
    """Test quality validation."""

    @pytest.fixture
    def quality_df(self):
        """Create test dataframe for quality checks."""
        return pd.DataFrame({
            'feature_1': range(100),
            'feature_2': range(100, 200),
            'target': [0, 1] * 50
        })

    def test_validate_quality_clean_data(self, quality_df):
        """Test quality check on clean data."""
        result = DataValidator.validate_quality(quality_df, 'target')
        assert isinstance(result, dict)
        assert 'warnings' in result
        assert 'rows' in result
        assert 'columns' in result

    def test_validate_quality_missing_values(self):
        """Test detection of missing values."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4, 5] * 20,
            'B': range(100),
            'target': [0, 1] * 50
        })
        result = DataValidator.validate_quality(df, 'target')
        assert result['missing_percentage'] > 0
        # Should have a warning about missing data
        assert any('missing' in str(w).lower() for w in result['warnings'])

    def test_validate_quality_duplicates(self):
        """Test detection of duplicate rows."""
        # Create a dataframe with explicitly duplicated rows
        base_df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})
        df = pd.concat([base_df] * 20, ignore_index=True)  # Repeat the 3 rows 20 times
        df['target'] = [i % 2 for i in range(len(df))]
        result = DataValidator.validate_quality(df, 'target')
        # Should have duplicates since we repeated rows
        assert result['duplicate_rows'] > 0

    def test_validate_quality_small_dataset(self):
        """Test warning for small datasets."""
        df = pd.DataFrame({
            'A': range(50),
            'B': range(50, 100),
            'target': [0, 1] * 25
        })
        result = DataValidator.validate_quality(df, 'target')
        # Should warn about small dataset
        assert any('small' in str(w).lower() for w in result['warnings'])

    def test_validate_quality_class_imbalance(self):
        """Test detection of class imbalance."""
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100, 200),
            'target': [1] * 95 + [0] * 5  # 95% vs 5% imbalance
        })
        result = DataValidator.validate_quality(df, 'target')
        # Should have class distribution
        assert 'class_distribution' in result


class TestDataValidatorDetectDuplicates:
    """Test duplicate detection."""

    def test_detect_duplicates_no_duplicates(self):
        """Test detection when no duplicates exist."""
        df = pd.DataFrame({
            'A': range(10),
            'B': range(10, 20)
        })
        result = DataValidator.detect_duplicates(df)
        assert result['exact_duplicate_count'] == 0

    def test_detect_duplicates_with_duplicates(self):
        """Test detection when duplicates exist."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 1, 2],
            'B': [10, 20, 30, 10, 20]
        })
        result = DataValidator.detect_duplicates(df)
        assert result['exact_duplicate_count'] == 2


class TestDataValidatorDetectMismatches:
    """Test type mismatch detection."""

    def test_detect_type_mismatches_no_mismatches(self):
        """Test when no type mismatches exist."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.5, 2.5, 3.5, 4.5, 5.5]
        })
        result = DataValidator.detect_type_mismatches(df)
        assert result['mismatches_found'] == 0


class TestDataValidatorRangeIssues:
    """Test value range issue detection."""

    def test_detect_range_issues_normal_data(self):
        """Test detection on normal range data."""
        df = pd.DataFrame({
            'A': range(100),
            'B': np.random.normal(100, 10, 100)
        })
        result = DataValidator.detect_value_range_issues(df)
        assert result['range_issues_found'] >= 0
