"""Multi-class problem detection and training strategy routing."""

import pandas as pd
import numpy as np
from typing import Literal, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiClassDetector:
    """Detect problem type and route to appropriate training strategy."""

    UNIQUE_THRESHOLD_MULTICLASS = 10  # If >10 unique values, consider multiclass
    UNIQUE_THRESHOLD_REGRESSION = 100  # If >100 unique continuous values, regression
    MIN_SAMPLES_PER_CLASS = 5  # Minimum samples per class for valid multiclass

    @staticmethod
    def detect_problem_type(target_series: pd.Series) -> Literal["binary", "multiclass", "regression"]:
        """Detect if problem is binary classification, multiclass, or regression.
        
        Args:
            target_series: Target column values
            
        Returns:
            "binary", "multiclass", or "regression"
        """
        # Remove NaN values
        y = target_series.dropna()
        
        # Edge case: empty target
        if len(y) == 0:
            logger.warning("Empty target series. Defaulting to binary.")
            return "binary"
        
        # Count unique values
        n_unique = y.nunique()
        
        # Edge case: single unique value
        if n_unique == 1:
            logger.warning(f"Only 1 unique class found. Not a valid classification problem.")
            return "binary"  # Fallback
        
        # Check data type
        is_numeric = pd.api.types.is_numeric_dtype(y)
        
        logger.info(f"Target analysis: {n_unique} unique values, dtype={y.dtype}, numeric={is_numeric}")
        
        # Regression: numeric with many unique values
        if is_numeric and n_unique >= MultiClassDetector.UNIQUE_THRESHOLD_REGRESSION:
            logger.info("Detected REGRESSION problem type")
            return "regression"
        
        # Binary: exactly 2 unique values
        if n_unique == 2:
            logger.info("Detected BINARY CLASSIFICATION problem type")
            return "binary"
        
        # Multiclass: 3-99 unique values (or categorical)
        if 3 <= n_unique < MultiClassDetector.UNIQUE_THRESHOLD_REGRESSION:
            logger.info(f"Detected MULTICLASS CLASSIFICATION problem type ({n_unique} classes)")
            return "multiclass"
        
        # Default to multiclass for edge cases
        logger.warning(f"Ambiguous target: {n_unique} unique values. Defaulting to multiclass.")
        return "multiclass"

    @staticmethod
    def get_training_strategy(problem_type: Literal["binary", "multiclass", "regression"]) -> str:
        """Get the training strategy for the detected problem type.
        
        Args:
            problem_type: Output from detect_problem_type()
            
        Returns:
            "binary" (standard binary classifier)
            "ovr" (One-vs-Rest for multiclass)
            "regression" (for continuous targets)
        """
        if problem_type == "binary":
            logger.info("Using BINARY classification strategy")
            return "binary"
        elif problem_type == "multiclass":
            logger.info("Using ONE-VS-REST (OvR) multiclass strategy")
            return "ovr"
        elif problem_type == "regression":
            logger.info("Using REGRESSION strategy")
            return "regression"
        else:
            logger.error(f"Unknown problem type: {problem_type}")
            return "binary"  # Fallback

    @staticmethod
    def get_unique_classes(target_series: pd.Series) -> np.ndarray:
        """Get unique class labels from target.
        
        Args:
            target_series: Target column
            
        Returns:
            Sorted array of unique class labels
        """
        return np.sort(target_series.dropna().unique())

    @staticmethod
    def get_class_distribution(target_series: pd.Series) -> pd.Series:
        """Get class distribution (counts and percentages).
        
        Args:
            target_series: Target column
            
        Returns:
            Series with class counts and percentages
        """
        value_counts = target_series.value_counts()
        logger.info(f"Class distribution:\n{value_counts}")
        return value_counts

    @staticmethod
    def is_imbalanced(target_series: pd.Series, imbalance_ratio: float = 0.8) -> bool:
        """Check if target is imbalanced (minority class <80% of majority).
        
        Args:
            target_series: Target column
            imbalance_ratio: Threshold for imbalance detection (default: 0.8)
            
        Returns:
            True if imbalanced, False otherwise
        """
        value_counts = target_series.value_counts()
        min_count = value_counts.min()
        max_count = value_counts.max()
        ratio = min_count / max_count
        
        is_imbal = ratio < imbalance_ratio
        logger.info(f"Imbalance ratio: {ratio:.2f}. Imbalanced: {is_imbal}")
        
        return is_imbal

    @staticmethod
    def validate_target(target_series: pd.Series) -> Tuple[bool, List[str]]:
        """Comprehensive validation of target variable.
        
        Args:
            target_series: Target column to validate
            
        Returns:
            Tuple[valid: bool, warnings: List[str]]
        """
        warnings = []
        
        # Check 1: Empty target
        if len(target_series) == 0:
            return False, ["Target series is empty"]
        
        # Check 2: All NaN
        if target_series.isnull().all():
            return False, ["All target values are NaN"]
        
        # Check 3: >50% NaN
        nan_pct = target_series.isnull().sum() / len(target_series) * 100
        if nan_pct > 50:
            return False, [f"More than 50% NaN values ({nan_pct:.1f}%)"]
        elif nan_pct > 10:
            warnings.append(f"⚠️  {nan_pct:.1f}% NaN values in target")
        
        # Check 4: Single unique value
        y = target_series.dropna()
        n_unique = y.nunique()
        if n_unique == 1:
            return False, [f"Only 1 unique class found - not a valid classification/regression target"]
        
        # Check 5: Sample size per class (for multiclass)
        problem_type = MultiClassDetector.detect_problem_type(target_series)
        if problem_type in ["binary", "multiclass"]:
            value_counts = target_series.value_counts()
            min_class_size = value_counts.min()
            
            if min_class_size < MultiClassDetector.MIN_SAMPLES_PER_CLASS:
                warnings.append(f"⚠️  Minimum class size is {min_class_size} (recommended: {MultiClassDetector.MIN_SAMPLES_PER_CLASS}+)")
        
        # Check 6: Class imbalance
        if MultiClassDetector.is_imbalanced(target_series):
            warnings.append(f"⚠️  Target is imbalanced (minority <80% of majority)")
        
        return True, warnings
