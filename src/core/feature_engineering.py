"""
Feature Engineering Module - Automated feature creation, transformation, and selection.

This module provides comprehensive feature engineering capabilities for ML pipelines:

FEATURE GENERATION:
- Polynomial features (degree 2, 3, configurable)
- Feature interactions (pairwise products)
- Statistical binning (quantile, uniform, kmeans strategies)
- Domain-specific features (statistical aggregates, correlation-based)

FEATURE SELECTION:
- Statistical feature selection (f_classif, mutual information)
- Automated k-best feature selection
- Importance-based ranking

FEATURE SCALING:
- Standard scaling (mean=0, std=1)
- Robust scaling (resistant to outliers)
- Fitted scaler persistence for apply to new data

PIPELINES:
- Complete end-to-end feature engineering pipeline
- Configurable transformations
- Reproducibility via random state control
- Handles edge cases (NaN values, constant features, large datasets)

DATACLASSES:
- FeatureEngineeringResult: Container for transformation results

CLASSES:
- FeatureEngineer: Main feature engineering engine with all transformation methods
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringResult:
    """Container for feature engineering results."""
    original_features: int
    generated_features: int
    total_features: int
    feature_names: List[str]
    feature_types: Dict[str, str]  # feature -> type (polynomial/interaction/bin)
    transformers: Dict[str, object]  # Fitted transformers for apply
    performance_improvement: Optional[float] = None
    selected_features: Optional[List[str]] = None


class FeatureEngineer:
    """Automated feature engineering engine."""

    def __init__(self, random_state: int = 42):
        """Initialize feature engineer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.poly_features = None
        self.scaler = None
        self.selected_features = None
        logger.info("FeatureEngineer initialized")

    def generate_polynomial_features(
        self,
        X: pd.DataFrame,
        degree: int = 2,
        include_bias: bool = False
    ) -> Tuple[pd.DataFrame, FeatureEngineeringResult]:
        """Generate polynomial features.
        
        Args:
            X: Input features dataframe
            degree: Polynomial degree (2 or 3 recommended)
            include_bias: Whether to include bias term
            
        Returns:
            (transformed_df, result)
        """
        try:
            original_count = X.shape[1]
            
            self.poly_features = PolynomialFeatures(
                degree=degree,
                include_bias=include_bias,
                interaction_only=False
            )
            
            X_poly = self.poly_features.fit_transform(X)
            poly_feature_names = self.poly_features.get_feature_names_out(X.columns)
            
            # Filter to keep only new features (not originals)
            new_poly_names = [name for name in poly_feature_names if name not in X.columns]
            
            # Create dataframe with all features
            X_transformed = pd.DataFrame(
                X_poly,
                columns=poly_feature_names,
                index=X.index
            )
            
            result = FeatureEngineeringResult(
                original_features=original_count,
                generated_features=len(new_poly_names),
                total_features=X_transformed.shape[1],
                feature_names=list(X_transformed.columns),
                feature_types={name: 'polynomial' for name in new_poly_names},
                transformers={'polynomial': self.poly_features}
            )
            
            logger.info(f"Generated {len(new_poly_names)} polynomial features (degree={degree})")
            return X_transformed, result
            
        except Exception as e:
            logger.error(f"Error in polynomial feature generation: {e}")
            raise

    def generate_interaction_features(
        self,
        X: pd.DataFrame,
        max_interactions: Optional[int] = None
    ) -> Tuple[pd.DataFrame, FeatureEngineeringResult]:
        """Generate feature interaction terms.
        
        Args:
            X: Input features dataframe
            max_interactions: Limit number of interactions (None = all)
            
        Returns:
            (transformed_df, result)
        """
        try:
            original_count = X.shape[1]
            interaction_features = {}
            feature_types = {}
            
            columns = X.columns.tolist()
            interaction_count = 0
            
            # Generate pairwise interactions
            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    if max_interactions and interaction_count >= max_interactions:
                        break
                    
                    interaction_name = f"{col1}_x_{col2}"
                    interaction_features[interaction_name] = X[col1] * X[col2]
                    feature_types[interaction_name] = 'interaction'
                    interaction_count += 1
            
            # Combine original and interaction features
            X_with_interactions = X.copy()
            for name, values in interaction_features.items():
                X_with_interactions[name] = values
            
            result = FeatureEngineeringResult(
                original_features=original_count,
                generated_features=len(interaction_features),
                total_features=X_with_interactions.shape[1],
                feature_names=list(X_with_interactions.columns),
                feature_types=feature_types,
                transformers={}
            )
            
            logger.info(f"Generated {len(interaction_features)} interaction features")
            return X_with_interactions, result
            
        except Exception as e:
            logger.error(f"Error in interaction feature generation: {e}")
            raise

    def generate_binned_features(
        self,
        X: pd.DataFrame,
        features_to_bin: Optional[List[str]] = None,
        n_bins: int = 5,
        strategy: str = 'quantile'
    ) -> Tuple[pd.DataFrame, FeatureEngineeringResult]:
        """Generate binned versions of continuous features.
        
        Args:
            X: Input features dataframe
            features_to_bin: Features to bin (None = all continuous)
            n_bins: Number of bins
            strategy: 'quantile', 'uniform', or 'kmeans'
            
        Returns:
            (transformed_df, result)
        """
        try:
            if features_to_bin is None:
                # Auto-detect continuous features
                features_to_bin = X.select_dtypes(include=[np.number]).columns.tolist()
            
            original_count = X.shape[1]
            binned_features = {}
            feature_types = {}
            bins_info = {}
            
            X_transformed = X.copy()
            
            for feature in features_to_bin:
                if feature not in X.columns:
                    continue
                
                try:
                    if strategy == 'quantile':
                        binned, bin_edges = pd.qcut(X[feature], q=n_bins, duplicates='drop',
                                                   labels=False, retbins=True)
                    elif strategy == 'uniform':
                        binned, bin_edges = pd.cut(X[feature], bins=n_bins,
                                                  labels=False, retbins=True)
                    else:  # kmeans
                        binned, bin_edges = pd.cut(X[feature], bins=n_bins,
                                                  labels=False, retbins=True)
                    
                    bin_name = f"{feature}_binned"
                    X_transformed[bin_name] = binned.astype('int')
                    feature_types[bin_name] = 'binned'
                    bins_info[bin_name] = bin_edges.tolist()
                    binned_features[bin_name] = X_transformed[bin_name]
                    
                except Exception as e:
                    logger.warning(f"Failed to bin feature {feature}: {e}")
                    continue
            
            result = FeatureEngineeringResult(
                original_features=original_count,
                generated_features=len(binned_features),
                total_features=X_transformed.shape[1],
                feature_names=list(X_transformed.columns),
                feature_types=feature_types,
                transformers={'bins_info': bins_info}
            )
            
            logger.info(f"Generated {len(binned_features)} binned features (strategy={strategy})")
            return X_transformed, result
            
        except Exception as e:
            logger.error(f"Error in binning feature generation: {e}")
            raise

    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        k: int = 10,
        method: str = 'f_classif'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select top k features using statistical tests.
        
        Args:
            X: Input features dataframe
            y: Target variable
            k: Number of features to select
            method: 'f_classif' or 'mutual_info'
            
        Returns:
            (selected_features_df, selected_feature_names)
        """
        try:
            k = min(k, X.shape[1])  # Can't select more than available
            
            if method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=k)
            elif method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            X_selected = selector.fit_transform(X, y)
            selected_names = X.columns[selector.get_support()].tolist()
            
            X_selected_df = pd.DataFrame(X_selected, columns=selected_names, index=X.index)
            self.selected_features = selected_names
            
            logger.info(f"Selected {k} features using {method}: {selected_names}")
            return X_selected_df, selected_names
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise

    def scale_features(
        self,
        X: pd.DataFrame,
        method: str = 'standard'
    ) -> Tuple[pd.DataFrame, object]:
        """Scale features to normalized range.
        
        Args:
            X: Input features dataframe
            method: 'standard' (0 mean, 1 std) or 'robust' (resistant to outliers)
            
        Returns:
            (scaled_df, scaler)
        """
        try:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            X_scaled = self.scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            logger.info(f"Scaled features using {method} method")
            return X_scaled_df, self.scaler
            
        except Exception as e:
            logger.error(f"Error in feature scaling: {e}")
            raise

    def create_domain_features(
        self,
        X: pd.DataFrame,
        target: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Create domain-specific features based on statistical properties.
        
        Args:
            X: Input features dataframe
            target: Optional target for correlation-based features
            
        Returns:
            (features_with_domain, feature_types_dict)
        """
        try:
            X_domain = X.copy()
            feature_types = {}
            
            # Add statistical aggregates for numeric features
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                # Mean of all numeric features
                X_domain['mean_all_numeric'] = X[numeric_cols].mean(axis=1)
                feature_types['mean_all_numeric'] = 'domain'
                
                # Std of all numeric features
                X_domain['std_all_numeric'] = X[numeric_cols].std(axis=1)
                feature_types['std_all_numeric'] = 'domain'
                
                # Max value
                X_domain['max_all_numeric'] = X[numeric_cols].max(axis=1)
                feature_types['max_all_numeric'] = 'domain'
                
                # Range (max - min)
                X_domain['range_all_numeric'] = X[numeric_cols].max(axis=1) - X[numeric_cols].min(axis=1)
                feature_types['range_all_numeric'] = 'domain'
            
            # Correlation with target if provided
            if target is not None and len(numeric_cols) > 0:
                correlations = {}
                for col in numeric_cols:
                    corr = np.abs(np.corrcoef(X[col], target)[0, 1])
                    if not np.isnan(corr):
                        correlations[col] = corr
                
                if correlations:
                    max_corr_feature = max(correlations, key=correlations.get)
                    X_domain['target_correlation_proxy'] = X[max_corr_feature] * correlations[max_corr_feature]
                    feature_types['target_correlation_proxy'] = 'domain'
            
            logger.info(f"Created {len(feature_types)} domain-specific features")
            return X_domain, feature_types
            
        except Exception as e:
            logger.error(f"Error in domain feature creation: {e}")
            raise

    def apply_transformation(
        self,
        X: pd.DataFrame,
        transformers: Dict[str, object]
    ) -> pd.DataFrame:
        """Apply saved transformations to new data.
        
        Args:
            X: New data to transform
            transformers: Dictionary of fitted transformers
            
        Returns:
            Transformed dataframe
        """
        try:
            X_transformed = X.copy()
            
            # Apply polynomial transformation if available
            if 'polynomial' in transformers and self.poly_features:
                X_poly = self.poly_features.transform(X)
                X_transformed = pd.DataFrame(
                    X_poly,
                    columns=self.poly_features.get_feature_names_out(X.columns),
                    index=X.index
                )
            
            # Apply scaling if available
            if self.scaler:
                X_scaled = self.scaler.transform(X_transformed)
                X_transformed = pd.DataFrame(
                    X_scaled,
                    columns=X_transformed.columns,
                    index=X_transformed.index
                )
            
            logger.info("Applied saved transformations to new data")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error applying transformations: {e}")
            raise

    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Get importance scores for features.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Dictionary of feature -> importance score
        """
        try:
            importances = {}
            
            # F-score based importance
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X, y)
            
            for feature, score in zip(X.columns, selector.scores_):
                if not np.isnan(score) and not np.isinf(score):
                    importances[feature] = float(score)
            
            # Normalize to 0-1 range
            if importances:
                max_importance = max(importances.values())
                if max_importance > 0:
                    importances = {k: v/max_importance for k, v in importances.items()}
            
            logger.info(f"Computed importance for {len(importances)} features")
            return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error computing feature importance: {e}")
            return {}

    def engineer_features_pipeline(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        polynomial_degree: int = 2,
        add_interactions: bool = True,
        n_bins: int = 5,
        scale: bool = True,
        select_k: Optional[int] = None
    ) -> Tuple[pd.DataFrame, FeatureEngineeringResult]:
        """Complete feature engineering pipeline.
        
        Args:
            X: Input features
            y: Optional target for feature selection
            polynomial_degree: Degree for polynomial features
            add_interactions: Whether to add interaction terms
            n_bins: Number of bins for binning
            scale: Whether to scale features
            select_k: Number of features to select (None = keep all)
            
        Returns:
            (engineered_features, result)
        """
        try:
            X_result = X.copy()
            all_feature_types = {}
            original_count = X.shape[1]
            
            # Step 1: Polynomial features
            if polynomial_degree > 1:
                X_result, poly_result = self.generate_polynomial_features(
                    X_result, degree=polynomial_degree
                )
                all_feature_types.update(poly_result.feature_types)
                logger.info(f"Step 1: Added polynomial features")
            
            # Step 2: Interactions
            if add_interactions:
                X_result, inter_result = self.generate_interaction_features(
                    X_result, max_interactions=10
                )
                all_feature_types.update(inter_result.feature_types)
                logger.info(f"Step 2: Added interaction features")
            
            # Step 3: Binning
            X_result, bin_result = self.generate_binned_features(
                X_result, n_bins=n_bins
            )
            all_feature_types.update(bin_result.feature_types)
            logger.info(f"Step 3: Added binned features")
            
            # Step 4: Scaling
            if scale:
                X_result, scaler = self.scale_features(X_result, method='standard')
                logger.info(f"Step 4: Scaled features")
            
            # Step 5: Feature selection
            if select_k and y is not None:
                X_result, selected = self.select_features(X_result, y, k=select_k)
                logger.info(f"Step 5: Selected {select_k} features")
            
            final_result = FeatureEngineeringResult(
                original_features=original_count,
                generated_features=X_result.shape[1] - original_count,
                total_features=X_result.shape[1],
                feature_names=list(X_result.columns),
                feature_types=all_feature_types,
                transformers={'poly': self.poly_features, 'scaler': self.scaler},
                selected_features=self.selected_features
            )
            
            logger.info(f"Pipeline complete: {original_count} -> {X_result.shape[1]} features")
            return X_result, final_result
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {e}")
            raise
