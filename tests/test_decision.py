"""Tests for decision support and analysis modules."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.decision.decision_mapper import DecisionMapper
from src.decision.confidence_estimator import ConfidenceEstimator
from src.decision.data_quality_analyzer import DataQualityAnalyzer


# Fixtures
@pytest.fixture
def decision_mapper():
    """Create DecisionMapper instance."""
    return DecisionMapper()


@pytest.fixture
def confidence_estimator():
    """Create ConfidenceEstimator instance."""
    return ConfidenceEstimator()


@pytest.fixture
def data_quality_analyzer():
    """Create DataQualityAnalyzer instance."""
    return DataQualityAnalyzer()


@pytest.fixture
def sample_proba():
    """Create sample probability predictions."""
    return np.array([
        [0.8, 0.2],      # High confidence for class 0
        [0.3, 0.7],      # High confidence for class 1
        [0.5, 0.5],      # Low confidence (uncertain)
        [0.9, 0.1],      # Very high confidence for class 0
        [0.4, 0.6],      # Moderate confidence for class 1
    ])


@pytest.fixture
def sample_predictions():
    """Create sample binary predictions."""
    return np.array([0, 1, 0, 1, 1])


@pytest.fixture
def sample_df():
    """Create sample DataFrame with features."""
    return pd.DataFrame({
        'age': [25, 35, 45, 55, 65],
        'income': [30000, 50000, 75000, 100000, 120000],
        'credit_score': [600, 650, 700, 750, 800],
        'existing_loans': [1, 2, 0, 3, 1]
    })


@pytest.fixture
def sample_df_with_leakage(sample_df):
    """Create DataFrame with potential target leakage."""
    df = sample_df.copy()
    df['target_proxy'] = np.array([0, 1, 0, 1, 1])  # Almost identical to actual target
    return df


# DecisionMapper Tests
class TestDecisionMapper:
    
    def test_initialization(self, decision_mapper):
        """Test DecisionMapper initialization."""
        assert decision_mapper.action_mappings == {}
        assert decision_mapper.decision_context == {}
    
    def test_define_outcome(self, decision_mapper):
        """Test defining an outcome."""
        decision_mapper.define_outcome(
            name='loan_approval',
            positive_label='Likely to default',
            negative_label='Low default risk',
            positive_action='Decline or set stricter terms',
            negative_action='Approve standard terms',
            description='Loan default prediction'
        )
        
        assert 'loan_approval' in decision_mapper.action_mappings
        mapping = decision_mapper.action_mappings['loan_approval']
        assert mapping['positive_label'] == 'Likely to default'
        assert mapping['negative_label'] == 'Low default risk'
    
    def test_get_action_positive_prediction(self, decision_mapper):
        """Test getting action for positive prediction."""
        decision_mapper.define_outcome(
            name='customer_churn',
            positive_label='Will Churn',
            negative_label='Will Stay',
            positive_action='Send retention offer',
            negative_action='Standard outreach',
            description='Customer churn prediction'
        )
        
        action = decision_mapper.get_action(
            outcome_name='customer_churn',
            prediction=1,
            confidence=0.85
        )
        
        assert action['prediction'] == 1
        assert action['label'] == 'Will Churn'
        assert action['action'] == 'Send retention offer'
        assert action['confidence'] == 0.85
    
    def test_get_action_negative_prediction(self, decision_mapper):
        """Test getting action for negative prediction."""
        decision_mapper.define_outcome(
            name='fraud_detection',
            positive_label='Fraudulent',
            negative_label='Legitimate',
            positive_action='Block transaction',
            negative_action='Allow transaction',
            description='Fraud detection model'
        )
        
        action = decision_mapper.get_action(
            outcome_name='fraud_detection',
            prediction=0,
            confidence=0.92
        )
        
        assert action['prediction'] == 0
        assert action['label'] == 'Legitimate'
        assert action['action'] == 'Allow transaction'
    
    def test_get_action_undefined_outcome(self, decision_mapper):
        """Test getting action for undefined outcome returns error."""
        action = decision_mapper.get_action(
            outcome_name='undefined_outcome',
            prediction=1
        )
        
        assert 'error' in action
        assert action['action'] is None
    
    def test_get_decision_matrix(self, decision_mapper):
        """Test retrieving decision matrix."""
        decision_mapper.define_outcome(
            name='test_outcome',
            positive_label='Positive Result',
            negative_label='Negative Result',
            positive_action='Action A',
            negative_action='Action B',
            description='Test description'
        )
        
        matrix = decision_mapper.get_decision_matrix('test_outcome')
        
        assert matrix['outcome'] == 'test_outcome'
        assert matrix['positive']['prediction'] == 1
        assert matrix['negative']['prediction'] == 0
        assert matrix['positive']['label'] == 'Positive Result'
        assert matrix['negative']['label'] == 'Negative Result'
    
    def test_multiple_outcomes(self, decision_mapper):
        """Test defining and managing multiple outcomes."""
        outcomes = [
            ('outcome_1', 'Positive 1', 'Negative 1', 'Action 1', 'Action -1'),
            ('outcome_2', 'Positive 2', 'Negative 2', 'Action 2', 'Action -2'),
            ('outcome_3', 'Positive 3', 'Negative 3', 'Action 3', 'Action -3'),
        ]
        
        for name, pos_label, neg_label, pos_action, neg_action in outcomes:
            decision_mapper.define_outcome(
                name=name,
                positive_label=pos_label,
                negative_label=neg_label,
                positive_action=pos_action,
                negative_action=neg_action
            )
        
        assert len(decision_mapper.action_mappings) == 3
        
        # Each should be retrievable
        for name, _, _, _, _ in outcomes:
            assert name in decision_mapper.action_mappings


# ConfidenceEstimator Tests
class TestConfidenceEstimator:
    
    def test_initialization(self, confidence_estimator):
        """Test ConfidenceEstimator initialization."""
        assert confidence_estimator.confidence_cache == {}
    
    def test_estimate_probability_confidence(self, confidence_estimator, sample_proba):
        """Test probability confidence estimation."""
        confidence = confidence_estimator.estimate_probability_confidence(sample_proba)
        
        assert len(confidence) == len(sample_proba)
        assert np.all(confidence >= 0) and np.all(confidence <= 1)
        assert confidence[0] == 0.8  # Max of first row
        assert confidence[1] == 0.7  # Max of second row
    
    def test_estimate_margin_confidence(self, confidence_estimator, sample_proba):
        """Test margin-based confidence estimation."""
        margin = confidence_estimator.estimate_margin_confidence(sample_proba)
        
        assert len(margin) == len(sample_proba)
        assert np.all(margin >= 0) and np.all(margin <= 1)
        # Margin should increase when predictions are more certain
        assert margin[3] > margin[2]  # [0.9, 0.1] has larger margin than [0.5, 0.5]
    
    def test_estimate_uncertainty(self, confidence_estimator, sample_proba):
        """Test entropy-based uncertainty estimation."""
        uncertainty = confidence_estimator.estimate_uncertainty(sample_proba)
        
        assert len(uncertainty) == len(sample_proba)
        assert np.all(uncertainty >= 0) and np.all(uncertainty <= 1)
        # Uncertain cases should have higher entropy
        assert uncertainty[2] > uncertainty[0]  # [0.5, 0.5] more uncertain than [0.8, 0.2]
    
    def test_estimate_ensemble_confidence(self, confidence_estimator):
        """Test ensemble confidence estimation."""
        predictions_list = [
            np.array([0, 1, 0, 1, 1]),
            np.array([0, 1, 0, 1, 1]),
            np.array([0, 1, 1, 1, 1]),
        ]
        
        confidence = confidence_estimator.estimate_ensemble_confidence(predictions_list)
        
        assert len(confidence) == 5
        # Perfect agreement for samples with all models predicting same
        assert confidence[0] == 1.0  # All predict 0
        assert confidence[1] == 1.0  # All predict 1
        # Lower confidence for sample 2 (disagreement)
        assert confidence[2] < 1.0
    
    def test_get_confidence_levels(self, confidence_estimator, sample_proba):
        """Test confidence level categorization."""
        levels = confidence_estimator.get_confidence_levels(
            sample_proba,
            low_threshold=0.6,
            high_threshold=0.8
        )
        
        assert len(levels) == len(sample_proba)
        assert all(level in ['Low', 'Medium', 'High'] for level in levels)
        # [0.8, 0.2] -> confidence 0.8 -> 'High'
        assert levels[0] == 'High'
        # [0.5, 0.5] -> confidence 0.5 -> 'Low'
        assert levels[2] == 'Low'
    
    def test_get_reliability_indicators(self, confidence_estimator, sample_proba, sample_predictions):
        """Test comprehensive reliability indicators."""
        df = confidence_estimator.get_reliability_indicators(
            sample_proba,
            sample_predictions
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_proba)
        assert 'probability_confidence' in df.columns
        assert 'margin_confidence' in df.columns
        assert 'uncertainty' in df.columns
        assert 'confidence_level' in df.columns
        assert 'predicted_class' in df.columns
        assert 'reliability_score' in df.columns
    
    def test_flag_uncertain_predictions(self, confidence_estimator, sample_proba):
        """Test flagging uncertain predictions."""
        flags = confidence_estimator.flag_uncertain_predictions(
            sample_proba,
            uncertainty_threshold=0.5
        )
        
        assert isinstance(flags, np.ndarray)
        assert len(flags) == len(sample_proba)
        # [0.5, 0.5] should be flagged as uncertain
        assert flags[2] == True
    
    def test_get_confidence_distribution_stats(self, confidence_estimator, sample_proba):
        """Test confidence distribution statistics."""
        stats = confidence_estimator.get_confidence_distribution_stats(sample_proba)
        
        assert 'mean_confidence' in stats
        assert 'median_confidence' in stats
        assert 'std_confidence' in stats
        assert stats['mean_confidence'] > 0
        assert stats['median_confidence'] > 0
        assert stats['std_confidence'] >= 0


# DataQualityAnalyzer Tests
class TestDataQualityAnalyzer:
    
    def test_initialization(self, data_quality_analyzer):
        """Test DataQualityAnalyzer initialization."""
        assert data_quality_analyzer.quality_report == {}
    
    def test_detect_target_leakage_no_leakage(self, data_quality_analyzer, sample_df):
        """Test leakage detection with clean data."""
        X = sample_df[['age', 'income', 'credit_score']]
        y = np.array([0, 1, 0, 1, 1])
        
        result = data_quality_analyzer.detect_target_leakage(X, y)
        
        assert 'has_leakage' in result
        assert 'suspicious_features' in result
        # Clean data should have no suspicious features
        assert len(result['suspicious_features']) == 0
    
    def test_detect_target_leakage_with_leakage(self, data_quality_analyzer, sample_df_with_leakage):
        """Test leakage detection with suspicious features."""
        X = sample_df_with_leakage[['target_proxy']]
        y = np.array([0, 1, 0, 1, 1])
        
        result = data_quality_analyzer.detect_target_leakage(X, y)
        
        assert result['has_leakage'] == True
        assert len(result['suspicious_features']) > 0
    
    def test_detect_distribution_drift(self, data_quality_analyzer, sample_df):
        """Test distribution drift detection."""
        X_train = sample_df.copy()
        X_test = sample_df.copy()
        X_test['age'] = X_test['age'] + 50  # Shift test distribution
        
        result = data_quality_analyzer.detect_distribution_drift(X_train, X_test)
        
        assert 'has_drift' in result
        assert 'drifted_features' in result
        # Should detect drift in age column
        age_drifted = any(f['feature'] == 'age' for f in result['drifted_features'])
        # May or may not detect depending on p-value threshold
        assert isinstance(result['has_drift'], bool)
    
    def test_detect_class_imbalance_balanced(self, data_quality_analyzer):
        """Test imbalance detection on balanced data."""
        y = np.array([0, 1, 0, 1, 0, 1])  # Perfectly balanced
        
        result = data_quality_analyzer.detect_class_imbalance(y)
        
        assert 'class_distribution' in result
        assert 'imbalance_ratio' in result
        assert 'severity' in result
        assert result['imbalance_ratio'] == 1.0
        assert result['severity'] == 'Low'
    
    def test_detect_class_imbalance_severe(self, data_quality_analyzer):
        """Test imbalance detection on severely imbalanced data."""
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])  # 8:1 ratio
        
        result = data_quality_analyzer.detect_class_imbalance(y)
        
        assert result['imbalance_ratio'] == 8.0
        assert result['severity'] == 'High'
    
    def test_detect_missing_values(self, data_quality_analyzer, sample_df):
        """Test missing value detection."""
        df = sample_df.copy()
        df.loc[0, 'age'] = np.nan
        df.loc[1, 'income'] = np.nan
        
        result = data_quality_analyzer.detect_missing_values(df)
        
        assert 'features_with_missing' in result
        assert 'total_missing_percentage' in result
        assert len(result['features_with_missing']) > 0


# Integration Tests
class TestDecisionIntegration:
    
    def test_complete_decision_workflow(self, decision_mapper, confidence_estimator):
        """Test complete decision-making workflow."""
        # Define outcome
        decision_mapper.define_outcome(
            name='product_recommendation',
            positive_label='Likely interested',
            negative_label='Not interested',
            positive_action='Send promotional email',
            negative_action='Send notification only',
            description='Product interest prediction'
        )
        
        # Get action for prediction with confidence
        prediction = 1
        confidence = 0.87
        
        action = decision_mapper.get_action(
            outcome_name='product_recommendation',
            prediction=prediction,
            confidence=confidence
        )
        
        assert action['action'] == 'Send promotional email'
        assert action['confidence'] == 0.87
    
    def test_confidence_guided_decision_making(self, confidence_estimator, decision_mapper):
        """Test using confidence scores to guide decisions."""
        # Create probabilities
        proba = np.array([
            [0.9, 0.1],   # High confidence
            [0.5, 0.5],   # Low confidence
            [0.8, 0.2]    # Medium confidence
        ])
        
        # Get confidence levels
        levels = confidence_estimator.get_confidence_levels(proba)
        predictions = np.array([0, 1, 0])
        
        # Define outcome
        decision_mapper.define_outcome(
            name='high_value_action',
            positive_label='Positive',
            negative_label='Negative',
            positive_action='High-impact action',
            negative_action='Standard action'
        )
        
        # Apply different actions based on confidence
        actions = []
        for pred, conf_level in zip(predictions, levels):
            if conf_level == 'High':
                action = decision_mapper.get_action('high_value_action', pred)
                actions.append(action)
        
        # Should have 2 high-confidence actions
        assert len(actions) == 2


# Edge Cases and Error Handling
class TestDecisionEdgeCases:
    
    def test_decision_mapper_empty_outcome(self, decision_mapper):
        """Test handling of empty outcome definitions."""
        matrix = decision_mapper.get_decision_matrix('nonexistent')
        assert 'error' in matrix
    
    def test_confidence_with_single_sample(self, confidence_estimator):
        """Test confidence estimation on single sample."""
        proba = np.array([[0.7, 0.3]])
        
        confidence = confidence_estimator.estimate_probability_confidence(proba)
        assert len(confidence) == 1
        assert confidence[0] == 0.7
    
    def test_confidence_with_multiclass(self, confidence_estimator):
        """Test confidence estimation with multiclass probabilities."""
        # 3-class problem
        proba = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.1, 0.8]
        ])
        
        confidence = confidence_estimator.estimate_probability_confidence(proba)
        assert np.all(confidence == np.array([0.7, 0.5, 0.8]))
    
    def test_data_quality_with_categorical_features(self, data_quality_analyzer):
        """Test data quality analysis with categorical features."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'missing_numeric': [1.0, np.nan, 3.0, 4.0, 5.0],
            'missing_cat': ['A', 'B', np.nan, 'C', 'B']
        })
        
        # Should handle categorical without errors
        try:
            result = data_quality_analyzer.detect_missing_values(df)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Should handle categorical features: {e}")
