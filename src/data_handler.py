"""Data validation and handling module."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from config.settings import MISSING_VALUE_THRESHOLD, NUMERIC_FEATURES_DTYPE, CATEGORICAL_FEATURES_DTYPE


class DataHandler:
    """Handles data loading, validation, and basic exploration."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame."""
        self.df = df
        self.original_df = df.copy()
        self.numeric_cols = []
        self.categorical_cols = []
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
