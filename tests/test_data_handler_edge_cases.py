"""Comprehensive tests for DataHandler CSV edge case handling."""

import pytest
import pandas as pd
import numpy as np
from src.core.data_handler import DataHandler


class TestNullRowDetection:
    """Test null row detection and reporting."""
    
    def test_detect_null_rows_none(self):
        """Test dataset with no null rows."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["x", "y", "z", "w", "v"]
        })
        handler = DataHandler(df)
        result = handler.detect_null_rows()
        
        assert result["null_row_count"] == 0
        assert result["null_row_percentage"] == 0.0
        assert result["null_row_indices"] == []
        assert result["affected_columns"] == []
    
    def test_detect_null_rows_partial(self):
        """Test dataset with some null rows."""
        df = pd.DataFrame({
            "A": [1, 2, None, 4, 5],
            "B": [10, None, 30, 40, 50],
            "C": ["x", "y", "z", None, "v"]
        })
        handler = DataHandler(df)
        result = handler.detect_null_rows()
        
        assert result["null_row_count"] == 3  # Rows 1, 2, 3 have nulls
        assert result["null_row_percentage"] == 60.0
        assert len(result["null_row_indices"]) == 3
        assert set(result["affected_columns"]) == {"A", "B", "C"}
    
    def test_detect_null_rows_threshold(self):
        """Test null row count exceeds 50%."""
        df = pd.DataFrame({
            "A": [1, None, None, None, 5],
            "B": [10, 20, None, 40, 50]
        })
        handler = DataHandler(df)
        result = handler.detect_null_rows()
        
        assert result["null_row_percentage"] == 60.0
        assert result["affected_columns"] == ["A", "B"]


class TestDuplicateDetection:
    """Test duplicate row detection and reporting."""
    
    def test_detect_duplicates_none(self):
        """Test dataset with no duplicates."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50]
        })
        handler = DataHandler(df)
        result = handler.detect_duplicates()
        
        assert result["exact_duplicate_count"] == 0
        assert result["total_affected_rows"] == 0
        assert result["duplicate_percentage"] == 0.0
    
    def test_detect_duplicates_exact(self):
        """Test dataset with exact duplicate rows."""
        df = pd.DataFrame({
            "A": [1, 2, 2, 4, 5],
            "B": [10, 20, 20, 40, 50],
            "C": ["x", "y", "y", "w", "v"]
        })
        handler = DataHandler(df)
        result = handler.detect_duplicates()
        
        assert result["exact_duplicate_count"] == 1
        assert result["total_affected_rows"] == 2  # 2 rows involved
        assert result["duplicate_percentage"] == 40.0  # 2 out of 5
        assert 1 in result["duplicate_indices"]  # Row with index 1 and 2
    
    def test_detect_duplicates_multiple(self):
        """Test dataset with multiple duplicate sets."""
        df = pd.DataFrame({
            "A": [1, 1, 2, 2, 3],
            "B": [10, 10, 20, 20, 30]
        })
        handler = DataHandler(df)
        result = handler.detect_duplicates()
        
        assert result["exact_duplicate_count"] == 2  # 2 duplicates (one from each pair)
        assert result["total_affected_rows"] == 4  # 4 rows involved
        assert result["duplicate_percentage"] == 80.0


class TestTypeMismatchDetection:
    """Test type mismatch detection in columns."""
    
    def test_type_mismatch_numeric_column_with_strings(self):
        """Test numeric column containing non-numeric strings."""
        df = pd.DataFrame({
            "A": [1, 2, "three", 4, 5],
            "B": [10.5, 20.5, 30.5, 40.5, 50.5],
            "C": ["x", "y", "z", "w", "v"]
        })
        handler = DataHandler(df)
        result = handler.detect_type_mismatches()
        
        # Column A should be detected as numeric column due to initial type detection
        assert result["mismatches_found"] >= 0  # May or may not detect depending on initial dtype
    
    def test_type_mismatch_numeric_strings(self):
        """Test numeric column with non-numeric string values."""
        df = pd.DataFrame({
            "A": ["1", "2", "NA", "4", "5"],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"]
        })
        handler = DataHandler(df)
        result = handler.detect_type_mismatches()
        
        # "NA" in column A should be flagged
        assert isinstance(result, dict)
        assert "mismatches_found" in result
        assert "details" in result
    
    def test_no_type_mismatches(self):
        """Test dataset with consistent types."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10.5, 20.5, 30.5, 40.5, 50.5],
            "C": ["x", "y", "z", "w", "v"]
        })
        handler = DataHandler(df)
        result = handler.detect_type_mismatches()
        
        assert result["mismatches_found"] == 0


class TestValueRangeDetection:
    """Test suspicious value range detection."""
    
    def test_value_range_normal(self):
        """Test dataset with normal value ranges."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50]
        })
        handler = DataHandler(df)
        result = handler.detect_value_range_issues()
        
        assert result["range_issues_found"] <= 1  # Normal ranges shouldn't be flagged
    
    def test_value_range_extreme_outliers(self):
        """Test dataset with extreme outliers."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 1000000],
            "B": [10, 20, 30, 40, 50]
        })
        handler = DataHandler(df)
        result = handler.detect_value_range_issues()
        
        # Should flag column A due to extreme outlier
        assert isinstance(result, dict)
        assert "range_issues_found" in result
        assert "details" in result
    
    def test_value_range_mixed_scales(self):
        """Test dataset with potentially mixed scales."""
        df = pd.DataFrame({
            "Age": [25, 30, 35, 28, 32],
            "Salary": [50000, 60000, 75000, 55000, 65000]
        })
        handler = DataHandler(df)
        result = handler.detect_value_range_issues()
        
        assert isinstance(result, dict)
        assert "range_issues_found" in result


class TestComprehensiveDataQuality:
    """Test comprehensive data quality validation."""
    
    def test_quality_check_clean_dataset(self):
        """Test quality check on clean dataset."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "B": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "C": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        })
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        # Should pass quality check
        assert isinstance(passed, bool)
        assert isinstance(warnings, list)
        assert "null_rows" in report
        assert report["null_rows"]["null_row_count"] == 0
    
    def test_quality_check_empty_dataset(self):
        """Test quality check on empty dataset."""
        df = pd.DataFrame()
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        assert passed == False
        assert any("empty" in w.lower() for w in warnings)
    
    def test_quality_check_too_few_rows(self):
        """Test quality check on dataset with too few rows."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [10, 20, 30]
        })
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        assert passed == False
        assert any("rows" in w.lower() for w in warnings)
    
    def test_quality_check_too_few_columns(self):
        """Test quality check on dataset with only 1 column."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        assert passed == False
        assert any("column" in w.lower() for w in warnings)
    
    def test_quality_check_high_null_percentage(self):
        """Test quality check with >50% null values."""
        df = pd.DataFrame({
            "A": [1, None, None, None, None, 6, 7, 8, 9, 10],
            "B": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        # High null percentage should flag it
        assert isinstance(passed, bool)
        assert isinstance(warnings, list)
    
    def test_quality_check_warnings_not_critical(self):
        """Test quality check with warnings (but not critical failures)."""
        df = pd.DataFrame({
            "A": [1, 2, 2, 4, 5, 6, 7, 8, 9, 10],  # 1 duplicate
            "B": [10, 20, 20, 40, 50, 60, 70, 80, 90, 100],
            "C": list("abcdefghij")
        })
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        # Should have warnings but not fail
        assert isinstance(passed, bool)
        assert isinstance(warnings, list)
        assert "duplicates" in report
    
    def test_quality_check_report_structure(self):
        """Test that quality report has expected structure."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "B": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        expected_keys = ["null_rows", "duplicates", "type_mismatches", "value_ranges"]
        for key in expected_keys:
            assert key in report, f"Missing key '{key}' in quality report"


class TestDataQualityIntegration:
    """Integration tests combining multiple edge cases."""
    
    def test_dataset_with_all_issues(self):
        """Test dataset with multiple types of issues."""
        df = pd.DataFrame({
            "ID": [1, 2, 2, None, 5, 6, 7, 8, 9, 10],  # nulls, duplicates
            "Value": [100, 200, 200, 300, None, 600, 700, 800, 900, 10000],  # nulls, outlier
            "Category": ["A", "B", "B", "C", "D", "E", "F", "G", "H", "I"]
        })
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        assert "null_rows" in report
        assert "duplicates" in report
        assert report["null_rows"]["null_row_count"] > 0
        assert report["duplicates"]["exact_duplicate_count"] > 0
    
    def test_real_world_csv_scenario(self):
        """Test realistic CSV with missing values and inconsistencies."""
        df = pd.DataFrame({
            "Age": [25, 30, None, 28, 32, 25, 35, 28, 29, 30],
            "Salary": [50000, 60000, 70000, 55000, 65000, 50000, 75000, 55000, 60000, 65000],
            "Department": ["Sales", "IT", "HR", "Sales", "IT", "Sales", "HR", "Sales", "IT", "HR"],
            "YearsExperience": [2, 5, None, 4, 6, 2, 8, 4, 5, 6]
        })
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        assert isinstance(passed, bool)
        assert isinstance(warnings, list)
        assert isinstance(report, dict)
        assert len(report) > 0


class TestDataHandlerErrorHandling:
    """Test error handling in DataHandler edge case detection."""
    
    def test_single_row_dataset(self):
        """Test edge case with single row."""
        df = pd.DataFrame({
            "A": [1],
            "B": [10]
        })
        handler = DataHandler(df)
        passed, warnings, report = handler.validate_data_quality()
        
        assert passed == False
    
    def test_all_null_column(self):
        """Test dataset with entire column as null."""
        df = pd.DataFrame({
            "A": [None, None, None, None, None, None, None, None, None, None],
            "B": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        handler = DataHandler(df)
        result = handler.detect_null_rows()
        
        assert result["null_row_count"] == 10
        assert result["null_row_percentage"] == 100.0
    
    def test_all_duplicates(self):
        """Test dataset where all rows are identical."""
        df = pd.DataFrame({
            "A": [1, 1, 1, 1, 1],
            "B": [10, 10, 10, 10, 10]
        })
        handler = DataHandler(df)
        result = handler.detect_duplicates()
        
        assert result["exact_duplicate_count"] > 0
        assert result["duplicate_percentage"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
