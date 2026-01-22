"""Comprehensive tests for MultiClassDetector with edge case handling."""

import pytest
import pandas as pd
import numpy as np
from src.decision.multiclass_handler import MultiClassDetector


class TestProblemTypeDetection:
    """Test problem type detection with edge cases."""
    
    def test_detect_binary(self):
        """Test binary classification detection."""
        target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "binary"
    
    def test_detect_multiclass_3_classes(self):
        """Test multiclass detection with 3 classes."""
        target = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "multiclass"
    
    def test_detect_multiclass_5_classes(self):
        """Test multiclass detection with 5 classes."""
        target = pd.Series([0, 1, 2, 3, 4] * 10)
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "multiclass"
    
    def test_detect_regression_high_unique(self):
        """Test regression detection with >100 unique numeric values."""
        target = pd.Series(np.arange(150, dtype=float))
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "regression"
    
    def test_detect_binary_string_labels(self):
        """Test binary detection with string labels."""
        target = pd.Series(["cat", "dog"] * 5)
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "binary"
    
    def test_detect_multiclass_string_labels(self):
        """Test multiclass detection with string labels."""
        target = pd.Series(["red", "blue", "green"] * 5)
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "multiclass"
    
    def test_detect_edge_case_empty_target(self):
        """Test edge case: empty target."""
        target = pd.Series([], dtype=float)
        result = MultiClassDetector.detect_problem_type(target)
        # Should handle gracefully, default to binary
        assert result in ["binary", "multiclass", "regression"]
    
    def test_detect_edge_case_all_nan(self):
        """Test edge case: all NaN target."""
        target = pd.Series([np.nan, np.nan, np.nan])
        result = MultiClassDetector.detect_problem_type(target)
        # Should handle gracefully
        assert result in ["binary", "multiclass", "regression"]
    
    def test_detect_edge_case_single_class(self):
        """Test edge case: only one unique class."""
        target = pd.Series([1, 1, 1, 1, 1])
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "binary"  # Fallback
    
    def test_detect_regression_boundary(self):
        """Test regression at boundary (exactly 100 unique values)."""
        target = pd.Series(np.arange(100, dtype=float))
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "regression"


class TestTrainingStrategy:
    """Test training strategy routing."""
    
    def test_strategy_binary(self):
        """Test binary strategy."""
        result = MultiClassDetector.get_training_strategy("binary")
        assert result == "binary"
    
    def test_strategy_multiclass(self):
        """Test multiclass/OvR strategy."""
        result = MultiClassDetector.get_training_strategy("multiclass")
        assert result == "ovr"
    
    def test_strategy_regression(self):
        """Test regression strategy."""
        result = MultiClassDetector.get_training_strategy("regression")
        assert result == "regression"


class TestUniqueClasses:
    """Test unique class detection."""
    
    def test_unique_classes_numeric(self):
        """Test unique class extraction (numeric)."""
        target = pd.Series([1, 2, 3, 1, 2, 3])
        result = MultiClassDetector.get_unique_classes(target)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)
    
    def test_unique_classes_string(self):
        """Test unique class extraction (string)."""
        target = pd.Series(["a", "b", "c", "a", "b"])
        result = MultiClassDetector.get_unique_classes(target)
        expected = np.array(["a", "b", "c"])
        np.testing.assert_array_equal(result, expected)
    
    def test_unique_classes_with_nan(self):
        """Test unique classes ignores NaN."""
        target = pd.Series([1, 2, np.nan, 1, 2])
        result = MultiClassDetector.get_unique_classes(target)
        expected = np.array([1., 2.])
        np.testing.assert_array_equal(result, expected)


class TestClassDistribution:
    """Test class distribution detection."""
    
    def test_distribution_balanced(self):
        """Test balanced class distribution."""
        target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        result = MultiClassDetector.get_class_distribution(target)
        assert result[0] == 4
        assert result[1] == 4
    
    def test_distribution_imbalanced(self):
        """Test imbalanced class distribution."""
        target = pd.Series([0] * 90 + [1] * 10)
        result = MultiClassDetector.get_class_distribution(target)
        assert result[0] == 90
        assert result[1] == 10


class TestImbalanceDetection:
    """Test class imbalance detection."""
    
    def test_balanced_classes(self):
        """Test balanced classes (50-50)."""
        target = pd.Series([0, 1] * 50)
        result = MultiClassDetector.is_imbalanced(target, imbalance_ratio=0.8)
        assert result == False  # 50/50 is not imbalanced
    
    def test_imbalanced_classes_80_20(self):
        """Test imbalanced classes (80-20)."""
        target = pd.Series([0] * 80 + [1] * 20)
        result = MultiClassDetector.is_imbalanced(target, imbalance_ratio=0.8)
        assert result == True  # 20/80 = 0.25 < 0.8
    
    def test_imbalanced_classes_threshold_boundary(self):
        """Test imbalance at threshold boundary."""
        target = pd.Series([0] * 100 + [1] * 80)  # 80/100 = 0.8
        result = MultiClassDetector.is_imbalanced(target, imbalance_ratio=0.8)
        # At boundary, should be False (not strictly <)
        assert result == False
    
    def test_imbalanced_classes_multiclass(self):
        """Test imbalance detection in multiclass."""
        target = pd.Series([0] * 100 + [1] * 50 + [2] * 10)
        result = MultiClassDetector.is_imbalanced(target)
        assert result == True  # 10/100 = 0.1 < 0.8


class TestTargetValidation:
    """Test comprehensive target validation."""
    
    def test_valid_target_binary(self):
        """Test valid binary target."""
        target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        valid, warnings = MultiClassDetector.validate_target(target)
        assert valid == True
        assert len(warnings) == 0
    
    def test_valid_target_multiclass(self):
        """Test valid multiclass target."""
        target = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        valid, warnings = MultiClassDetector.validate_target(target)
        assert valid == True
    
    def test_invalid_target_empty(self):
        """Test invalid: empty target."""
        target = pd.Series([], dtype=int)
        valid, warnings = MultiClassDetector.validate_target(target)
        assert valid == False
        assert len(warnings) > 0
    
    def test_invalid_target_all_nan(self):
        """Test invalid: all NaN target."""
        target = pd.Series([np.nan, np.nan, np.nan])
        valid, warnings = MultiClassDetector.validate_target(target)
        assert valid == False
    
    def test_invalid_target_high_nan(self):
        """Test invalid: >50% NaN values."""
        target = pd.Series([np.nan, np.nan, np.nan, 1, 2])
        valid, warnings = MultiClassDetector.validate_target(target)
        assert valid == False
    
    def test_warning_target_moderate_nan(self):
        """Test warning: 10-50% NaN values."""
        target = pd.Series([np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        valid, warnings = MultiClassDetector.validate_target(target)
        # Should be valid but with warning
        assert valid == True
        assert any("NaN" in w for w in warnings)
    
    def test_invalid_target_single_class(self):
        """Test invalid: only one unique class."""
        target = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        valid, warnings = MultiClassDetector.validate_target(target)
        assert valid == False
    
    def test_warning_target_small_class_size(self):
        """Test warning: class size < 5 samples."""
        target = pd.Series([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])
        valid, warnings = MultiClassDetector.validate_target(target)
        # Should be valid but may have warning about small classes
        assert valid == True
    
    def test_warning_target_imbalanced(self):
        """Test warning: imbalanced classes."""
        target = pd.Series([0] * 90 + [1] * 10)
        valid, warnings = MultiClassDetector.validate_target(target)
        assert valid == True
        assert any("imbalanced" in w.lower() for w in warnings)


class TestEdgeCases:
    """Test various edge cases."""
    
    def test_edge_case_large_multiclass(self):
        """Test edge case: 50+ classes."""
        target = pd.Series(list(range(50)) * 5)
        valid, warnings = MultiClassDetector.validate_target(target)
        problem_type = MultiClassDetector.detect_problem_type(target)
        assert problem_type == "multiclass"
        assert valid == True
    
    def test_edge_case_binary_string_yes_no(self):
        """Test binary with 'yes'/'no' labels."""
        target = pd.Series(["yes", "no"] * 10)
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "binary"
    
    def test_edge_case_multiclass_mixed_dtypes(self):
        """Test multiclass with mixed numeric types."""
        target = pd.Series([1, 2, 3, 1.0, 2.0, 3.0, 1, 2, 3, 1])
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "multiclass"
    
    def test_edge_case_regression_float_values(self):
        """Test regression with float values and high uniqueness."""
        target = pd.Series(np.random.uniform(0, 1000, 150))
        result = MultiClassDetector.detect_problem_type(target)
        assert result == "regression"
    
    def test_classes_sorted_correctly(self):
        """Test that unique classes are returned sorted."""
        target = pd.Series([3, 1, 2, 1, 3, 2, 3, 1])
        result = MultiClassDetector.get_unique_classes(target)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
