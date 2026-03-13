"""Data validation module - unified validation logic."""

from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np


class DataValidator:
    """Centralized data validation for format, structure, and quality checks."""

    # Validation thresholds
    MIN_ROWS = 10
    MIN_COLUMNS = 2
    MIN_DATASET_SIZE_FOR_WARNING = 100
    MISSING_VALUE_WARNING_THRESHOLD = 5.0  # percentage
    DUPLICATE_WARNING_THRESHOLD = 0
    CLASS_IMBALANCE_WARNING_THRESHOLD = 10.0  # percentage
    HIGH_CORRELATION_THRESHOLD = 0.9
    CORRELATION_WARNING_THRESHOLD = 0  # warn if any pair > 0.9

    @staticmethod
    def validate_format(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate basic dataframe format and structure.
        
        Args:
            df: Input dataframe
            
        Returns:
            (is_valid, message) tuple
        """
        if df is None:
            return False, "Dataset is None."
        
        if not isinstance(df, pd.DataFrame):
            return False, "Input must be a pandas DataFrame."
        
        if df.empty:
            return False, "Dataset is empty."
        
        return True, "Format validation passed."

    @staticmethod
    def validate_structure(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate dataframe structure (rows, columns).
        
        Args:
            df: Input dataframe
            
        Returns:
            (is_valid, message) tuple
        """
        if len(df) < DataValidator.MIN_ROWS:
            return False, f"Dataset must have at least {DataValidator.MIN_ROWS} rows. Found {len(df)}."
        
        if len(df.columns) < DataValidator.MIN_COLUMNS:
            return False, f"Dataset must have at least {DataValidator.MIN_COLUMNS} columns. Found {len(df.columns)}."
        
        return True, "Structure validation passed."

    @staticmethod
    def validate_basic(df: pd.DataFrame) -> Tuple[bool, str]:
        """Run format and structure validations.
        
        Args:
            df: Input dataframe
            
        Returns:
            (is_valid, message) tuple
        """
        is_valid, msg = DataValidator.validate_format(df)
        if not is_valid:
            return is_valid, msg
        
        is_valid, msg = DataValidator.validate_structure(df)
        return is_valid, msg

    @staticmethod
    def validate_quality(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
        """Comprehensive data quality validation with warnings.
        
        Args:
            df: Input dataframe
            target_col: Optional target column name for class imbalance check
            
        Returns:
            Dict with warnings and metrics
        """
        warnings = []
        metrics = {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(exclude=[np.number]).columns),
        }
        
        # Check size
        if len(df) < DataValidator.MIN_DATASET_SIZE_FOR_WARNING:
            warnings.append(f"⚠️ Small dataset: {len(df)} rows (recommend ≥100)")
        
        # Missing values
        total_cells = len(df) * len(df.columns)
        missing_count = df.isnull().sum().sum()
        missing_pct = (missing_count / total_cells) * 100 if total_cells > 0 else 0
        
        if missing_pct > DataValidator.MISSING_VALUE_WARNING_THRESHOLD:
            warnings.append(f"⚠️ Missing data: {missing_pct:.1f}%")
        
        metrics['missing_percentage'] = round(missing_pct, 2)
        metrics['missing_count'] = int(missing_count)
        
        # Duplicates
        dup_count = df.duplicated().sum()
        if dup_count > DataValidator.DUPLICATE_WARNING_THRESHOLD:
            warnings.append(f"⚠️ Duplicate rows: {dup_count}")
        
        metrics['duplicate_rows'] = int(dup_count)
        
        # Zero variance columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zero_variance_cols = []
        for col in numeric_cols:
            if df[col].std() == 0:
                zero_variance_cols.append(col)
                warnings.append(f"⚠️ Zero variance column: '{col}'")
        
        metrics['zero_variance_columns'] = zero_variance_cols
        
        # Class imbalance (if target column provided)
        if target_col and target_col in df.columns:
            value_counts = df[target_col].value_counts(normalize=True)
            if len(value_counts) > 1:
                min_pct = value_counts.min() * 100
                if min_pct < DataValidator.CLASS_IMBALANCE_WARNING_THRESHOLD:
                    warnings.append(f"⚠️ Class imbalance: smallest class {min_pct:.1f}%")
                metrics['class_distribution'] = value_counts.round(3).to_dict()
        
        # High correlation between features
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > DataValidator.HIGH_CORRELATION_THRESHOLD:
                        high_corr_pairs.append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': round(float(corr_val), 3)
                        })
            
            if high_corr_pairs:
                warnings.append(f"⚠️ High correlation detected: {len(high_corr_pairs)} feature pairs > {DataValidator.HIGH_CORRELATION_THRESHOLD}")
            
            metrics['high_correlation_pairs'] = high_corr_pairs
        
        metrics['warnings'] = warnings
        metrics['passed'] = len(warnings) == 0
        
        return metrics

    @staticmethod
    def detect_duplicates(df: pd.DataFrame) -> Dict:
        """Detect and report duplicate rows (exact and partial).
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict with duplicate statistics
        """
        duplicates = df.duplicated(keep=False)
        duplicate_rows = df[duplicates].sort_values(by=list(df.columns))
        
        return {
            "exact_duplicate_count": int(df.duplicated().sum()),
            "total_affected_rows": int(duplicates.sum()),
            "duplicate_indices": df[duplicates].index.tolist(),
            "duplicate_percentage": round(duplicates.sum() / len(df) * 100, 2),
            "duplicate_rows_sample": duplicate_rows.head(5).to_dict(orient='records') if len(duplicate_rows) > 0 else [],
        }

    @staticmethod
    def detect_type_mismatches(df: pd.DataFrame) -> Dict:
        """Detect numeric columns with non-numeric values, categorical mismatches.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict with type mismatch details
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        mismatches = []
        
        for col in numeric_cols:
            # Check for non-numeric strings in numeric columns
            non_numeric = pd.to_numeric(df[col], errors='coerce')
            if non_numeric.isnull().sum() > 0:
                non_numeric_rows = df[non_numeric.isnull()]
                mismatches.append({
                    "column": col,
                    "issue_type": "non_numeric_values_in_numeric_column",
                    "sample_values": non_numeric_rows[col].unique()[:5].tolist(),
                    "affected_row_count": int(len(non_numeric_rows)),
                })
        
        return {
            "mismatches_found": len(mismatches),
            "details": mismatches,
        }

    @staticmethod
    def detect_value_range_issues(df: pd.DataFrame) -> Dict:
        """Detect numeric columns with suspicious value ranges (extreme outliers).
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict with value range issues
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        range_issues = []
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            min_val = col_data.min()
            max_val = col_data.max()
            range_val = max_val - min_val
            mean_val = col_data.mean()
            std_val = col_data.std()
            
            # Detect extreme outliers (>3 sigma)
            if std_val > 0:
                outlier_threshold = mean_val + (3 * std_val)
                outliers = col_data[col_data > outlier_threshold]
                outlier_count = len(outliers)
            else:
                outlier_count = 0
            
            # Flag if range seems suspicious (e.g., >10000x the mean)
            if mean_val != 0 and abs(range_val / mean_val) > 10000:
                range_issues.append({
                    "column": col,
                    "min_value": float(min_val),
                    "max_value": float(max_val),
                    "range": float(range_val),
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "outlier_count_3sigma": int(outlier_count),
                    "issue": "suspicious_large_range",
                })
        
        return {
            "range_issues_found": len(range_issues),
            "details": range_issues,
        }
