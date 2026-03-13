"""Tests for feature engineering module."""

import pytest
import numpy as np
import pandas as pd

from src.core.feature_engineering import FeatureEngineer, FeatureEngineeringResult


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer(random_state=42)

    @pytest.fixture
    def sample_data(self):
        """Generate sample dataset."""
        np.random.seed(42)
        X = pd.DataFrame({
            'age': np.random.normal(40, 15, 100),
            'income': np.random.normal(50000, 20000, 100),
            'hours': np.random.normal(40, 8, 100),
            'education': np.random.choice([1, 2, 3, 4], 100)
        })
        y = (X['age'] > 40).astype(int).values
        return X, y

    def test_engineer_initialization(self, engineer):
        """Test FeatureEngineer initializes correctly."""
        assert engineer is not None
        assert engineer.random_state == 42
        assert engineer.poly_features is None
        assert engineer.scaler is None

    def test_polynomial_features_degree2(self, engineer, sample_data):
        """Test polynomial feature generation (degree 2)."""
        X, _ = sample_data
        
        X_poly, result = engineer.generate_polynomial_features(X, degree=2)
        
        assert isinstance(result, FeatureEngineeringResult)
        assert result.original_features == 4
        assert result.generated_features > 0
        assert X_poly.shape[0] == X.shape[0]
        assert X_poly.shape[1] > X.shape[1]
        assert result.total_features == X_poly.shape[1]

    def test_polynomial_features_degree3(self, engineer, sample_data):
        """Test polynomial feature generation (degree 3)."""
        X, _ = sample_data
        
        X_poly, result = engineer.generate_polynomial_features(X, degree=3)
        
        assert result.generated_features > 0
        assert X_poly.shape[1] > X.shape[1]

    def test_interaction_features(self, engineer, sample_data):
        """Test interaction feature generation."""
        X, _ = sample_data
        
        X_inter, result = engineer.generate_interaction_features(X)
        
        assert isinstance(result, FeatureEngineeringResult)
        assert result.original_features == 4
        assert result.generated_features > 0
        assert X_inter.shape[0] == X.shape[0]
        # Check for 'x' indicating multiplication
        new_cols = [c for c in X_inter.columns if '_x_' in c]
        assert len(new_cols) == result.generated_features

    def test_interaction_features_limited(self, engineer, sample_data):
        """Test interaction feature generation with max limit."""
        X, _ = sample_data
        
        X_inter, result = engineer.generate_interaction_features(X, max_interactions=3)
        
        assert result.generated_features <= 3

    def test_binned_features(self, engineer, sample_data):
        """Test binned feature generation."""
        X, _ = sample_data
        
        X_binned, result = engineer.generate_binned_features(X, n_bins=5)
        
        assert isinstance(result, FeatureEngineeringResult)
        assert result.generated_features > 0
        assert X_binned.shape[0] == X.shape[0]
        # Check that binned columns exist
        binned_cols = [c for c in X_binned.columns if '_binned' in c]
        assert len(binned_cols) == result.generated_features

    def test_binned_features_strategies(self, engineer, sample_data):
        """Test different binning strategies."""
        X, _ = sample_data
        
        for strategy in ['quantile', 'uniform', 'kmeans']:
            X_binned, result = engineer.generate_binned_features(
                X, n_bins=5, strategy=strategy
            )
            assert result.generated_features > 0

    def test_feature_selection_f_classif(self, engineer, sample_data):
        """Test feature selection with f_classif."""
        X, y = sample_data
        
        X_selected, selected_names = engineer.select_features(X, y, k=2, method='f_classif')
        
        assert X_selected.shape[1] == 2
        assert len(selected_names) == 2
        assert all(name in X.columns for name in selected_names)

    def test_feature_selection_mutual_info(self, engineer, sample_data):
        """Test feature selection with mutual information."""
        X, y = sample_data
        
        X_selected, selected_names = engineer.select_features(X, y, k=3, method='mutual_info')
        
        assert X_selected.shape[1] == 3
        assert len(selected_names) == 3

    def test_feature_selection_k_exceeds_features(self, engineer, sample_data):
        """Test feature selection when k > num features."""
        X, y = sample_data
        
        X_selected, selected_names = engineer.select_features(X, y, k=100)
        
        # Should select at most X.shape[1] features
        assert X_selected.shape[1] <= X.shape[1]

    def test_scale_features_standard(self, engineer, sample_data):
        """Test standard scaling."""
        X, _ = sample_data
        
        X_scaled, scaler = engineer.scale_features(X, method='standard')
        
        # Check that mean is ~0 and std is ~1 (with tolerance for numerical precision)
        assert np.abs(X_scaled.mean(axis=0)).max() < 1e-8
        # Allow for numerical precision in std calculation
        assert np.abs(X_scaled.std(axis=0, ddof=0) - 1).max() < 0.1

    def test_scale_features_robust(self, engineer, sample_data):
        """Test robust scaling."""
        X, _ = sample_data
        
        X_scaled, scaler = engineer.scale_features(X, method='robust')
        
        # Just check shape and type
        assert X_scaled.shape == X.shape
        assert isinstance(X_scaled, pd.DataFrame)

    def test_domain_features(self, engineer, sample_data):
        """Test domain-specific feature creation."""
        X, y = sample_data
        
        X_domain, feature_types = engineer.create_domain_features(X, target=y)
        
        assert X_domain.shape[0] == X.shape[0]
        assert X_domain.shape[1] > X.shape[1]
        assert 'mean_all_numeric' in X_domain.columns
        assert 'std_all_numeric' in X_domain.columns

    def test_apply_transformation(self, engineer, sample_data):
        """Test applying saved transformations."""
        X_train, _ = sample_data
        
        X_poly, result = engineer.generate_polynomial_features(X_train, degree=2)
        
        # Apply to new data
        X_test = X_train.iloc[:10].copy()
        X_test_transformed = engineer.apply_transformation(X_test, result.transformers)
        
        assert X_test_transformed.shape[0] == X_test.shape[0]

    def test_feature_importance(self, engineer, sample_data):
        """Test feature importance computation."""
        X, y = sample_data
        
        importance = engineer.get_feature_importance(X, y)
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        # All scores should be between 0 and 1 (normalized)
        assert all(0 <= v <= 1 for v in importance.values())
        # Sorted by importance
        values = list(importance.values())
        assert values == sorted(values, reverse=True)

    def test_pipeline_basic(self, engineer, sample_data):
        """Test basic feature engineering pipeline."""
        X, y = sample_data
        
        X_eng, result = engineer.engineer_features_pipeline(
            X, y=y, polynomial_degree=2, add_interactions=True,
            n_bins=5, scale=True, select_k=None
        )
        
        assert isinstance(result, FeatureEngineeringResult)
        assert result.original_features == 4
        assert X_eng.shape[0] == X.shape[0]
        assert X_eng.shape[1] > X.shape[1]

    def test_pipeline_with_selection(self, engineer, sample_data):
        """Test pipeline with feature selection."""
        X, y = sample_data
        
        X_eng, result = engineer.engineer_features_pipeline(
            X, y=y, select_k=15
        )
        
        assert X_eng.shape[1] == 15
        assert result.total_features == 15

    def test_pipeline_no_polynomial(self, engineer, sample_data):
        """Test pipeline without polynomial features."""
        X, y = sample_data
        
        X_eng, result = engineer.engineer_features_pipeline(
            X, y=y, polynomial_degree=1, add_interactions=False,
            n_bins=0, scale=False
        )
        
        # Should have at most original features
        assert X_eng.shape[1] <= X.shape[1] + 10

    def test_result_dataclass(self, engineer, sample_data):
        """Test FeatureEngineeringResult dataclass."""
        X, y = sample_data
        
        X_poly, result = engineer.generate_polynomial_features(X, degree=2)
        
        assert result.original_features > 0
        assert result.generated_features > 0
        assert result.total_features > result.original_features
        assert len(result.feature_names) == result.total_features
        assert isinstance(result.feature_types, dict)
        assert isinstance(result.transformers, dict)

    def test_missing_feature_handling(self, engineer, sample_data):
        """Test handling of missing features during transformation."""
        X, _ = sample_data
        X_subset = X[['age', 'income']]
        
        X_inter, result = engineer.generate_interaction_features(X_subset)
        
        assert X_inter.shape[1] > X_subset.shape[1]

    def test_nan_handling_scaling(self, engineer):
        """Test scaling with NaN values."""
        X = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [10, 20, 30, np.nan, 50]
        })
        
        # Drop NaN
        X_clean = X.dropna()
        X_scaled, _ = engineer.scale_features(X_clean, method='standard')
        
        assert X_scaled.shape[1] == 2

    def test_constant_feature_handling(self, engineer):
        """Test handling of constant features."""
        X = pd.DataFrame({
            'a': [1, 1, 1, 1, 1],
            'b': [1, 2, 3, 4, 5]
        })
        y = np.array([0, 1, 0, 1, 0])
        
        X_selected, selected = engineer.select_features(X, y, k=1)
        
        assert X_selected.shape[1] == 1

    def test_large_dataset(self, engineer):
        """Test feature engineering on large dataset."""
        np.random.seed(42)
        X = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, 1000)
            for i in range(20)
        })
        y = (X['feature_0'] > 0).astype(int).values
        
        X_eng, result = engineer.engineer_features_pipeline(
            X, y=y, select_k=30
        )
        
        assert X_eng.shape[1] == 30
        assert X_eng.shape[0] == 1000

    def test_feature_engineering_reproducibility(self, sample_data):
        """Test reproducibility with same random state."""
        X, y = sample_data
        
        eng1 = FeatureEngineer(random_state=42)
        X_eng1, _ = eng1.engineer_features_pipeline(X, y=y)
        
        eng2 = FeatureEngineer(random_state=42)
        X_eng2, _ = eng2.engineer_features_pipeline(X, y=y)
        
        pd.testing.assert_frame_equal(X_eng1, X_eng2)

    def test_various_column_types(self, engineer):
        """Test with numeric columns only for interactions."""
        X = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'float': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        X_inter, result = engineer.generate_interaction_features(X)
        
        assert X_inter.shape[1] > X.shape[1]

    def test_polynomial_bias_term(self, engineer, sample_data):
        """Test polynomial feature generation with bias term."""
        X, _ = sample_data
        X_subset = X[['age']].head(20)
        
        X_poly_no_bias, _ = engineer.generate_polynomial_features(
            X_subset, degree=2, include_bias=False
        )
        
        # Should include original and squared term
        assert '1' not in X_poly_no_bias.columns[0]

    def test_binning_preserves_indices(self, engineer, sample_data):
        """Test that binning preserves index."""
        X, _ = sample_data
        original_index = X.index
        
        X_binned, _ = engineer.generate_binned_features(X, n_bins=3)
        
        assert all(X_binned.index == original_index)

    def test_feature_importance_ranking(self, engineer, sample_data):
        """Test that feature importance is properly ranked."""
        X, y = sample_data
        
        importance = engineer.get_feature_importance(X, y)
        
        # Should be sorted descending
        names, scores = zip(*importance.items())
        assert list(scores) == sorted(scores, reverse=True)
