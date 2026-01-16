"""Data preprocessing pipeline."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from config.settings import RANDOM_SEED, TEST_SIZE, VAL_SIZE, NUMERIC_FEATURES_DTYPE, CATEGORICAL_FEATURES_DTYPE


class Preprocessor:
    """Handles data preprocessing including encoding, scaling, and splitting."""

    def __init__(self, df: pd.DataFrame, target_col: str):
        """Initialize preprocessor."""
        self.df = df.copy()
        self.target_col = target_col
        self.numeric_cols = df.select_dtypes(include=NUMERIC_FEATURES_DTYPE).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=CATEGORICAL_FEATURES_DTYPE).columns.tolist()
        
        # Remove target from feature lists
        if self.target_col in self.numeric_cols:
            self.numeric_cols.remove(self.target_col)
        if self.target_col in self.categorical_cols:
            self.categorical_cols.remove(self.target_col)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = None
        self.feature_names = []

    def encode_categorical(self) -> None:
        """Encode categorical features using LabelEncoder."""
        df_encoded = self.df.copy()
        
        # Encode categorical features
        for col in self.categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        # Check if target is categorical and determine task type
        self.is_classification = self.df[self.target_col].dtype in CATEGORICAL_FEATURES_DTYPE or \
                                  self.df[self.target_col].nunique() < 20
        
        if self.is_classification and self.target_col in self.categorical_cols:
            le = LabelEncoder()
            df_encoded[self.target_col] = le.fit_transform(df_encoded[self.target_col].astype(str))
            self.target_encoder = le
        
        self.df = df_encoded

    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Scale numeric features using StandardScaler."""
        X_scaled = X.copy()
        
        if self.numeric_cols:
            X_scaled[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])
        
        return X_scaled

    def prepare_data(self, test_size: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = TEST_SIZE
        
        # Encode categorical variables
        self.encode_categorical()
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale numeric features
        X_scaled = self.scale_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y if self.is_classification else None
        )
        
        return X_train, X_test, y_train, y_test

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names

    def get_task_type(self) -> str:
        """Return task type: 'classification' or 'regression'."""
        return "classification" if self.is_classification else "regression"

def rank_target_columns(df: pd.DataFrame) -> List[Tuple[str, int, str, int]]:
    """
    Rank DataFrame columns by suitability as target columns.
    
    Ranking system:
    1 (Best): Binary columns (exactly 2 unique values)
    2: Low cardinality (2-20 unique values)
    3: Medium cardinality (21-50 unique values)
    4: High cardinality (50+ unique values) - likely features, not targets
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of tuples: (column_name, rank, reason, unique_count)
        Sorted by rank (best first)
    """
    ranked = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        
        # Skip columns with only 1 unique value (constant columns)
        if unique_count < 2:
            continue
        
        # Determine rank and reason
        if unique_count == 2:
            rank = 1
            reason = f"✅ Binary ({unique_count} classes)"
        elif unique_count <= 20:
            rank = 2
            reason = f"✅ Low cardinality ({unique_count} classes)"
        elif unique_count <= 50:
            rank = 3
            reason = f"⚠️ Medium cardinality ({unique_count} classes)"
        else:
            rank = 4
            reason = f"❌ High cardinality ({unique_count} classes) - likely a feature"
        
        ranked.append((col, rank, reason, unique_count))
    
    # Sort by rank (best first), then by column name (alphabetically)
    ranked.sort(key=lambda x: (x[1], x[0]))
    
    return ranked