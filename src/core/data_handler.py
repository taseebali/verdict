"""Data validation and handling module."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from config.settings import MISSING_VALUE_THRESHOLD, NUMERIC_FEATURES_DTYPE, CATEGORICAL_FEATURES_DTYPE


class DataHandler:
    """Handles data loading, validation, and basic exploration with comprehensive edge case handling."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame."""
        self.df = df
        self.original_df = df.copy()
        self.numeric_cols = []
        self.categorical_cols = []
        self.data_quality_report = {}
        self._identify_column_types()

    def _identify_column_types(self) -> None:
        """Identify numeric and categorical columns."""
        self.numeric_cols = self.df.select_dtypes(include=NUMERIC_FEATURES_DTYPE).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=CATEGORICAL_FEATURES_DTYPE).columns.tolist()

    def get_data_summary(self) -> Dict:
        """Return summary statistics of the dataset."""
        return {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict(),
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.categorical_cols,
        }

    def validate_data(self) -> Tuple[bool, str]:
        """Validate dataset integrity."""
        if self.df.empty:
            return False, "Dataset is empty."
        
        if len(self.df) < 10:
            return False, "Dataset must have at least 10 rows."
        
        if len(self.df.columns) < 2:
            return False, "Dataset must have at least 2 columns."
        
        return True, "Data validation passed."

    def handle_missing_values(self, strategy: str = "drop_columns") -> None:
        """Handle missing values in the dataset.
        
        Args:
            strategy: "drop_columns" (default) or "drop_rows"
        """
        if strategy == "drop_columns":
            # Drop columns with >50% missing values
            missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
            cols_to_drop = missing_pct[missing_pct > MISSING_VALUE_THRESHOLD * 100].index.tolist()
            self.df.drop(columns=cols_to_drop, inplace=True)
            self._identify_column_types()
            
            # Drop rows with any remaining missing values
            self.df.dropna(inplace=True)
        
        elif strategy == "drop_rows":
            self.df.dropna(inplace=True)

    def get_columns(self) -> List[str]:
        """Return list of all columns."""
        return self.df.columns.tolist()

    def get_column_info(self, column: str) -> Dict:
        """Get detailed info about a specific column."""
        col_data = self.df[column]
        
        info = {
            "name": column,
            "dtype": str(col_data.dtype),
            "missing_count": col_data.isnull().sum(),
            "missing_percentage": round(col_data.isnull().sum() / len(self.df) * 100, 2),
        }
        
        if column in self.numeric_cols:
            info.update({
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "unique_count": col_data.nunique(),
            })
        else:
            info.update({
                "unique_count": col_data.nunique(),
                "top_values": col_data.value_counts().head(5).to_dict(),
            })
        
        return info

    def detect_null_rows(self) -> Dict:
        """Detect and report rows with null values.
        
        Returns:
            Dict with null_row_count, null_row_indices, null_row_percentage, affected_columns
        """
        null_rows = self.df[self.df.isnull().any(axis=1)]
        return {
            "null_row_count": len(null_rows),
            "null_row_indices": null_rows.index.tolist() if len(null_rows) > 0 else [],
            "null_row_percentage": round(len(null_rows) / len(self.df) * 100, 2),
            "affected_columns": self.df.columns[self.df.isnull().any()].tolist(),
        }

    def detect_duplicates(self) -> Dict:
        """Detect and report duplicate rows (exact and partial).
        
        Returns:
            Dict with duplicate_count, duplicate_indices, duplicate_percentage, duplicate_rows (sample)
        """
        duplicates = self.df.duplicated(keep=False)
        duplicate_rows = self.df[duplicates].sort_values(by=list(self.df.columns))
        
        return {
            "exact_duplicate_count": self.df.duplicated().sum(),
            "total_affected_rows": duplicates.sum(),
            "duplicate_indices": self.df[duplicates].index.tolist(),
            "duplicate_percentage": round(duplicates.sum() / len(self.df) * 100, 2),
            "duplicate_rows_sample": duplicate_rows.head(5).to_dict(orient='records') if len(duplicate_rows) > 0 else [],
        }

    def detect_type_mismatches(self) -> Dict:
        """Detect numeric columns with non-numeric values, categorical mismatches.
        
        Returns:
            Dict with column_name, issue_type, sample_values, affected_row_count
        """
        mismatches = []
        
        for col in self.numeric_cols:
            # Check for non-numeric strings in numeric columns
            non_numeric = pd.to_numeric(self.df[col], errors='coerce')
            if non_numeric.isnull().sum() > 0:
                non_numeric_rows = self.df[non_numeric.isnull()]
                mismatches.append({
                    "column": col,
                    "issue_type": "non_numeric_values_in_numeric_column",
                    "sample_values": non_numeric_rows[col].unique()[:5].tolist(),
                    "affected_row_count": len(non_numeric_rows),
                })
        
        return {
            "mismatches_found": len(mismatches),
            "details": mismatches,
        }

    def detect_value_range_issues(self) -> Dict:
        """Detect numeric columns with suspicious value ranges (extreme outliers, inconsistent scales).
        
        Returns:
            Dict with column_name, min_value, max_value, range, outlier_count
        """
        range_issues = []
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
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

    def validate_data_quality(self) -> Tuple[bool, List[str], Dict]:
        """Comprehensive data quality check combining all edge case detection.
        
        Returns:
            Tuple[passed: bool, warnings: List[str], report: Dict]
        """
        warnings = []
        report = {}
        
        # Check 1: Empty dataset
        if self.df.empty:
            return False, ["Dataset is empty."], report
        
        # Check 2: Minimum rows
        if len(self.df) < 10:
            return False, [f"Dataset has only {len(self.df)} rows; minimum 10 required."], report
        
        # Check 3: Minimum columns
        if len(self.df.columns) < 2:
            return False, [f"Dataset has only {len(self.df.columns)} columns; minimum 2 required."], report
        
        # Check 4: Null rows
        null_report = self.detect_null_rows()
        report["null_rows"] = null_report
        if null_report["null_row_count"] > 0:
            pct = null_report["null_row_percentage"]
            if pct > 50:
                return False, [f"Dataset has {pct}% null rows; >50% threshold exceeded."], report
            elif pct > 10:
                warnings.append(f"⚠️  {pct}% of rows have null values. Consider removal.")
        
        # Check 5: Duplicates
        dup_report = self.detect_duplicates()
        report["duplicates"] = dup_report
        if dup_report["exact_duplicate_count"] > 0:
            pct = dup_report["duplicate_percentage"]
            warnings.append(f"⚠️  {dup_report['exact_duplicate_count']} exact duplicate rows ({pct}%) detected. Consider deduplication.")
        
        # Check 6: Type mismatches
        mismatch_report = self.detect_type_mismatches()
        report["type_mismatches"] = mismatch_report
        if mismatch_report["mismatches_found"] > 0:
            warnings.append(f"⚠️  {mismatch_report['mismatches_found']} column(s) have type mismatches. Review sample values.")
        
        # Check 7: Value range issues
        range_report = self.detect_value_range_issues()
        report["value_ranges"] = range_report
        if range_report["range_issues_found"] > 0:
            warnings.append(f"⚠️  {range_report['range_issues_found']} numeric column(s) have suspicious value ranges.")
        
        self.data_quality_report = report
        passed = len(warnings) == 0 and not any(key in report for key in ["null_rows", "type_mismatches"])
        
        return passed, warnings, report
