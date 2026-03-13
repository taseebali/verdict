"""
ML Operations Module - Comprehensive machine learning functionality for ensemble training and drift monitoring.

This consolidated module combines previously separate modules for production efficiency:

ENSEMBLE METHODS & TRAINING:
- Random Forest classifier (baseline)
- XGBoost gradient boosting (when available)
- LightGBM fast gradient boosting (when available)
- Voting ensemble (hard/soft voting with weighted estimators)
- Stacking ensemble (meta-learner based stacking)
- Hyperparameter tuning with GridSearchCV
- Feature importance extraction and analysis
- Multi-method comparison and evaluation
- Cross-validation support

DRIFT DETECTION & MONITORING:
- Kolmogorov-Smirnov (KS) test for continuous feature drift
- Chi-square test for categorical feature drift
- Multi-feature batch drift detection
- Multivariate/overall drift assessment
- Model performance degradation detection
- Distribution comparison utilities (histograms, statistics)
- Statistical severity classification (none/low/medium/high)
- Real-time drift monitoring capabilities

DATACLASSES & CONTAINERS:
- EnsembleResult: Container for ensemble training results with metrics
- DriftResult: Container for individual feature drift detection results
- OverallDriftResult: Container for overall/multivariate drift assessment

CLASSES:
- EnsembleManager: Training and comparison of ensemble learning methods
- DriftDetector: Statistical drift detection and monitoring engine

USAGE:
  # Ensemble training
  from src.core.ml_operations import EnsembleManager, EnsembleResult
  manager = EnsembleManager()
  voting_result = manager.train_voting(X_train, X_test, y_train, y_test)
  
  # Drift detection
  from src.core.ml_operations import DriftDetector, DriftResult
  detector = DriftDetector()
  feature_drift = detector.detect_feature_drift(train_df, test_df)
  overall_drift = detector.assess_overall_drift(feature_drift)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Optional imports for XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# ENSEMBLE METHODS SECTION
# ============================================================================

@dataclass
class EnsembleResult:
    """Container for ensemble training results."""
    method: str
    model: Any
    train_score: float
    test_score: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    cv_scores: Optional[Dict[str, Tuple[float, float]]]
    feature_importance: Optional[Dict[str, float]]
    params: Dict[str, Any]
    training_time: float


class EnsembleManager:
    """Manager for ensemble learning methods."""

    # Parameter grids for hyperparameter tuning
    PARAM_GRIDS = {
        'xgboost': {
            'balanced': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'scale_pos_weight': [1]
            },
            'default': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
                'gamma': [0, 1, 5],
                'scale_pos_weight': [1]
            }
        },
        'lightgbm': {
            'balanced': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05],
                'num_leaves': [31, 50],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'default': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [20, 31, 50, 100],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
                'min_child_samples': [5, 10, 20]
            }
        },
        'voting': {
            'default': {
                'voting': ['hard', 'soft'],
                'weights': [[1, 1, 1], [1, 2, 1], [2, 1, 1]]
            }
        },
        'stacking': {
            'default': {
                'final_estimator__C': [0.1, 1.0, 10.0],
                'cv': [3, 5]
            }
        }
    }

    def __init__(self):
        """Initialize ensemble manager."""
        self.available_methods = self._detect_available_methods()
        logger.info(f"Ensemble methods available: {self.available_methods}")

    def _detect_available_methods(self) -> List[str]:
        """Detect which ensemble methods are available."""
        methods = ['random_forest']  # Always available
        if XGBOOST_AVAILABLE:
            methods.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            methods.append('lightgbm')
        methods.extend(['voting', 'stacking'])
        return methods

    def train_xgboost(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        use_tuning: bool = False
    ) -> EnsembleResult:
        """Train XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        import time
        start_time = time.time()

        try:
            default_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }

            if params:
                default_params.update(params)

            model = xgb.XGBClassifier(**default_params)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            test_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None

            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)

            roc_auc = None
            if test_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    roc_auc = roc_auc_score(y_test, test_proba)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")

            feature_importance = dict(zip(
                [f'feature_{i}' for i in range(X_train.shape[1])],
                model.feature_importances_
            ))

            training_time = time.time() - start_time

            return EnsembleResult(
                method='xgboost',
                model=model,
                train_score=train_score,
                test_score=test_score,
                precision=precision,
                recall=recall,
                f1=f1,
                roc_auc=roc_auc,
                cv_scores=None,
                feature_importance=feature_importance,
                params=default_params,
                training_time=training_time
            )

        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            raise

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        use_tuning: bool = False
    ) -> EnsembleResult:
        """Train LightGBM classifier."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        import time
        start_time = time.time()

        try:
            default_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }

            if params:
                default_params.update(params)

            model = lgb.LGBMClassifier(**default_params)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            test_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None

            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)

            roc_auc = None
            if test_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    roc_auc = roc_auc_score(y_test, test_proba)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")

            feature_importance = dict(zip(
                [f'feature_{i}' for i in range(X_train.shape[1])],
                model.feature_importances_
            ))

            training_time = time.time() - start_time

            return EnsembleResult(
                method='lightgbm',
                model=model,
                train_score=train_score,
                test_score=test_score,
                precision=precision,
                recall=recall,
                f1=f1,
                roc_auc=roc_auc,
                cv_scores=None,
                feature_importance=feature_importance,
                params=default_params,
                training_time=training_time
            )

        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            raise

    def train_voting(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        voting: str = 'hard',
        weights: Optional[List[float]] = None
    ) -> EnsembleResult:
        """Train Voting Classifier using base estimators."""
        import time
        start_time = time.time()

        try:
            base_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42))
            ]

            if XGBOOST_AVAILABLE:
                base_estimators.append(('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)))
            
            if weights is not None and len(weights) != len(base_estimators):
                if len(weights) > len(base_estimators):
                    weights = weights[:len(base_estimators)]
                else:
                    weights = list(weights) + [1] * (len(base_estimators) - len(weights))

            model = VotingClassifier(
                estimators=base_estimators,
                voting=voting,
                weights=weights,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            if voting == 'soft':
                test_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
            else:
                test_proba = None

            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)

            roc_auc = None
            if test_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    roc_auc = roc_auc_score(y_test, test_proba)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")

            training_time = time.time() - start_time

            params = {
                'voting': voting,
                'weights': weights,
                'n_estimators': 100
            }

            return EnsembleResult(
                method='voting',
                model=model,
                train_score=train_score,
                test_score=test_score,
                precision=precision,
                recall=recall,
                f1=f1,
                roc_auc=roc_auc,
                cv_scores=None,
                feature_importance=None,
                params=params,
                training_time=training_time
            )

        except Exception as e:
            logger.error(f"Voting Classifier training failed: {e}")
            raise

    def train_stacking(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        cv: int = 5
    ) -> EnsembleResult:
        """Train Stacking Classifier with meta-learner."""
        import time
        start_time = time.time()

        try:
            base_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42))
            ]

            if XGBOOST_AVAILABLE:
                base_estimators.append(('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)))

            model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                cv=cv
            )

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            test_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None

            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)

            roc_auc = None
            if test_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    roc_auc = roc_auc_score(y_test, test_proba)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")

            training_time = time.time() - start_time

            params = {
                'cv': cv,
                'base_estimators': len(base_estimators),
                'final_estimator': 'LogisticRegression'
            }

            return EnsembleResult(
                method='stacking',
                model=model,
                train_score=train_score,
                test_score=test_score,
                precision=precision,
                recall=recall,
                f1=f1,
                roc_auc=roc_auc,
                cv_scores=None,
                feature_importance=None,
                params=params,
                training_time=training_time
            )

        except Exception as e:
            logger.error(f"Stacking Classifier training failed: {e}")
            raise

    def train_with_tuning(
        self,
        method: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        cv: int = 5,
        param_grid_preset: str = 'balanced'
    ) -> EnsembleResult:
        """Train ensemble method with hyperparameter tuning."""
        logger.info(f"Starting hyperparameter tuning for {method}...")

        try:
            if method == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost not available")
                
                base_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
                param_grid = self.PARAM_GRIDS['xgboost'].get(param_grid_preset, self.PARAM_GRIDS['xgboost']['default'])

            elif method == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM not available")
                
                base_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
                param_grid = self.PARAM_GRIDS['lightgbm'].get(param_grid_preset, self.PARAM_GRIDS['lightgbm']['default'])

            else:
                raise ValueError(f"Tuning not supported for {method}. Use train_{method}() directly.")

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )

            import time
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time

            best_model = grid_search.best_estimator_

            train_pred = best_model.predict(X_train)
            test_pred = best_model.predict(X_test)
            test_proba = best_model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None

            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)

            roc_auc = None
            if test_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    roc_auc = roc_auc_score(y_test, test_proba)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")

            cv_scores = {
                'accuracy': (grid_search.cv_results_['mean_test_score'].mean(), grid_search.cv_results_['std_test_score'].mean())
            }

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

            return EnsembleResult(
                method=method,
                model=best_model,
                train_score=train_score,
                test_score=test_score,
                precision=precision,
                recall=recall,
                f1=f1,
                roc_auc=roc_auc,
                cv_scores=cv_scores,
                feature_importance=dict(zip(
                    [f'feature_{i}' for i in range(X_train.shape[1])],
                    best_model.feature_importances_
                )),
                params=grid_search.best_params_,
                training_time=training_time
            )

        except Exception as e:
            logger.error(f"Ensemble tuning failed: {e}")
            raise

    def compare_methods(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        methods: Optional[List[str]] = None
    ) -> Dict[str, EnsembleResult]:
        """Compare multiple ensemble methods on same data."""
        if methods is None:
            methods = self.available_methods

        results = {}

        for method in methods:
            try:
                logger.info(f"Training {method}...")

                if method == 'xgboost':
                    results[method] = self.train_xgboost(X_train, X_test, y_train, y_test)
                elif method == 'lightgbm':
                    results[method] = self.train_lightgbm(X_train, X_test, y_train, y_test)
                elif method == 'voting':
                    results[method] = self.train_voting(X_train, X_test, y_train, y_test)
                elif method == 'stacking':
                    results[method] = self.train_stacking(X_train, X_test, y_train, y_test)
                elif method == 'random_forest':
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    test_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None

                    train_score = accuracy_score(y_train, train_pred)
                    test_score = accuracy_score(y_test, test_pred)
                    precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)

                    roc_auc = None
                    if test_proba is not None and len(np.unique(y_test)) == 2:
                        try:
                            roc_auc = roc_auc_score(y_test, test_proba)
                        except Exception:
                            pass

                    results[method] = EnsembleResult(
                        method='random_forest',
                        model=model,
                        train_score=train_score,
                        test_score=test_score,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        roc_auc=roc_auc,
                        cv_scores=None,
                        feature_importance=dict(zip(
                            [f'feature_{i}' for i in range(X_train.shape[1])],
                            model.feature_importances_
                        )),
                        params={'n_estimators': 100},
                        training_time=0.0
                    )

            except Exception as e:
                logger.error(f"Failed to train {method}: {e}")
                continue

        return results

    def get_summary_table(self, results: Dict[str, EnsembleResult]) -> pd.DataFrame:
        """Generate summary table from ensemble results."""
        data = []
        for method, result in results.items():
            data.append({
                'Method': method.upper(),
                'Train Accuracy': f"{result.train_score:.4f}",
                'Test Accuracy': f"{result.test_score:.4f}",
                'Precision': f"{result.precision:.4f}",
                'Recall': f"{result.recall:.4f}",
                'F1-Score': f"{result.f1:.4f}",
                'ROC-AUC': f"{result.roc_auc:.4f}" if result.roc_auc else "N/A",
                'Time (s)': f"{result.training_time:.2f}"
            })

        return pd.DataFrame(data)


# ============================================================================
# DRIFT DETECTION SECTION
# ============================================================================

@dataclass
class DriftResult:
    """Container for drift detection results."""
    feature: str
    feature_type: str
    drift_detected: bool
    p_value: float
    statistic: float
    threshold: float
    severity: str
    description: str
    train_dist: Optional[Dict[str, float]] = None
    test_dist: Optional[Dict[str, float]] = None


@dataclass  
class OverallDriftResult:
    """Container for overall drift assessment."""
    overall_drift_detected: bool
    num_features_drifted: int
    total_features_checked: int
    drift_percentage: float
    severity: str
    feature_results: List[DriftResult]
    timestamp: str
    description: str


class DriftDetector:
    """Detector for model and data drift."""

    DEFAULT_THRESHOLDS = {
        'ks_statistic': 0.1,
        'chi2_p_value': 0.05,
        'drift_ratio': 0.3,
        'severity_low': 0.1,
        'severity_medium': 0.3,
        'severity_high': 0.5
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """Initialize drift detector."""
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)
        logger.info(f"DriftDetector initialized with thresholds: {self.thresholds}")

    def detect_drift_ks_test(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        feature_name: str = "feature",
        threshold: Optional[float] = None
    ) -> DriftResult:
        """Detect drift in continuous feature using Kolmogorov-Smirnov test."""
        threshold = threshold or self.thresholds['ks_statistic']
        
        try:
            train_clean = train_data[~np.isnan(train_data)]
            test_clean = test_data[~np.isnan(test_data)]
            
            if len(train_clean) == 0 or len(test_clean) == 0:
                logger.warning(f"Feature {feature_name} has no valid data")
                return DriftResult(
                    feature=feature_name,
                    feature_type='continuous',
                    drift_detected=False,
                    p_value=1.0,
                    statistic=0.0,
                    threshold=threshold,
                    severity='none',
                    description="Insufficient data for drift detection"
                )
            
            statistic, p_value = stats.ks_2samp(train_clean, test_clean)
            drift_detected = statistic > threshold
            
            severity = self._assess_severity(statistic, self.thresholds['severity_low'],
                                            self.thresholds['severity_medium'],
                                            self.thresholds['severity_high'])
            
            description = f"KS statistic: {statistic:.4f}, p-value: {p_value:.4f}"
            if drift_detected:
                description += f" (drift detected, exceeds threshold {threshold:.4f})"
            
            return DriftResult(
                feature=feature_name,
                feature_type='continuous',
                drift_detected=drift_detected,
                p_value=p_value,
                statistic=statistic,
                threshold=threshold,
                severity=severity,
                description=description,
                train_dist={'mean': float(train_clean.mean()), 'std': float(train_clean.std())},
                test_dist={'mean': float(test_clean.mean()), 'std': float(test_clean.std())}
            )
            
        except Exception as e:
            logger.error(f"Error in KS test for feature {feature_name}: {e}")
            raise

    def detect_drift_chi2_test(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        feature_name: str = "feature",
        p_value_threshold: Optional[float] = None
    ) -> DriftResult:
        """Detect drift in categorical feature using Chi-square test."""
        p_value_threshold = p_value_threshold or self.thresholds['chi2_p_value']
        
        try:
            train_counts = pd.Series(train_data).value_counts()
            test_counts = pd.Series(test_data).value_counts()
            
            all_categories = set(train_counts.index) | set(test_counts.index)
            train_counts = train_counts.reindex(all_categories, fill_value=0)
            test_counts = test_counts.reindex(all_categories, fill_value=0)
            
            if (train_counts == 0).any() or (test_counts == 0).any():
                train_counts = train_counts + 1e-6
                test_counts = test_counts + 1e-6
            
            chi2_stat, p_value = stats.chisquare(test_counts, train_counts)
            drift_detected = p_value < p_value_threshold
            
            df = len(all_categories) - 1
            normalized_stat = chi2_stat / max(1, df)
            
            severity = self._assess_severity(normalized_stat, self.thresholds['severity_low'],
                                            self.thresholds['severity_medium'],
                                            self.thresholds['severity_high'])
            
            description = f"Chi-square: {chi2_stat:.4f}, p-value: {p_value:.4f}"
            if drift_detected:
                description += f" (drift detected, p-value < {p_value_threshold:.4f})"
            
            return DriftResult(
                feature=feature_name,
                feature_type='categorical',
                drift_detected=drift_detected,
                p_value=p_value,
                statistic=chi2_stat,
                threshold=p_value_threshold,
                severity=severity,
                description=description,
                train_dist=train_counts.to_dict(),
                test_dist=test_counts.to_dict()
            )
            
        except Exception as e:
            logger.error(f"Error in Chi-square test for feature {feature_name}: {e}")
            raise

    def detect_feature_drift(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> List[DriftResult]:
        """Detect drift for multiple features."""
        if feature_columns is None:
            feature_columns = train_df.columns.tolist()
        
        if categorical_features is None:
            categorical_features = []
        
        results = []
        
        for feature in feature_columns:
            if feature not in train_df.columns or feature not in test_df.columns:
                logger.warning(f"Feature {feature} not found in one of the datasets")
                continue
            
            try:
                if feature in categorical_features:
                    result = self.detect_drift_chi2_test(
                        train_df[feature].values,
                        test_df[feature].values,
                        feature_name=feature
                    )
                else:
                    result = self.detect_drift_ks_test(
                        train_df[feature].values,
                        test_df[feature].values,
                        feature_name=feature
                    )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error detecting drift for feature {feature}: {e}")
                continue
        
        return results

    def assess_overall_drift(
        self,
        feature_results: List[DriftResult],
        drift_ratio_threshold: Optional[float] = None
    ) -> OverallDriftResult:
        """Assess overall drift across all features."""
        drift_ratio_threshold = drift_ratio_threshold or self.thresholds['drift_ratio']
        
        if not feature_results:
            return OverallDriftResult(
                overall_drift_detected=False,
                num_features_drifted=0,
                total_features_checked=0,
                drift_percentage=0.0,
                severity='none',
                feature_results=[],
                timestamp=datetime.now().isoformat(),
                description="No features to check"
            )
        
        drifted_features = [r for r in feature_results if r.drift_detected]
        num_drifted = len(drifted_features)
        total_features = len(feature_results)
        drift_percentage = num_drifted / total_features if total_features > 0 else 0
        
        overall_drift_detected = drift_percentage > drift_ratio_threshold
        
        if num_drifted == 0:
            overall_severity = 'none'
        else:
            severity_scores = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
            avg_severity_score = np.mean([severity_scores.get(r.severity, 0) for r in feature_results])
            
            if avg_severity_score < 1:
                overall_severity = 'low'
            elif avg_severity_score < 2:
                overall_severity = 'medium'
            else:
                overall_severity = 'high'
        
        description = f"{num_drifted}/{total_features} features show drift ({drift_percentage*100:.1f}%)"
        if overall_drift_detected:
            description += f" - Overall drift detected (exceeds {drift_ratio_threshold*100:.1f}% threshold)"
        
        return OverallDriftResult(
            overall_drift_detected=overall_drift_detected,
            num_features_drifted=num_drifted,
            total_features_checked=total_features,
            drift_percentage=drift_percentage,
            severity=overall_severity,
            feature_results=feature_results,
            timestamp=datetime.now().isoformat(),
            description=description
        )

    def detect_model_performance_drift(
        self,
        train_scores: List[float],
        test_scores: List[float],
        metric_name: str = "accuracy",
        performance_threshold: float = 0.05
    ) -> Tuple[bool, float, str]:
        """Detect if model performance has degraded."""
        try:
            train_mean = np.mean(train_scores)
            test_mean = np.mean(test_scores)
            
            degradation = train_mean - test_mean
            degradation_ratio = degradation / train_mean if train_mean > 0 else 0
            
            drift_detected = degradation_ratio > performance_threshold
            
            description = f"Train {metric_name}: {train_mean:.4f}, Test {metric_name}: {test_mean:.4f}"
            description += f", Degradation: {degradation:.4f} ({degradation_ratio*100:.1f}%)"
            
            if drift_detected:
                description += f" (exceeds {performance_threshold*100:.1f}% threshold)"
            
            logger.info(f"Performance drift for {metric_name}: {description}")
            
            return drift_detected, degradation_ratio, description
            
        except Exception as e:
            logger.error(f"Error detecting performance drift: {e}")
            raise

    def _assess_severity(
        self,
        statistic: float,
        threshold_low: float,
        threshold_medium: float,
        threshold_high: float
    ) -> str:
        """Assess severity level based on statistic value."""
        if statistic < threshold_low:
            return 'none'
        elif statistic < threshold_medium:
            return 'low'
        elif statistic < threshold_high:
            return 'medium'
        else:
            return 'high'

    def get_drift_summary_dataframe(self, results: List[DriftResult]) -> pd.DataFrame:
        """Generate summary DataFrame from drift results."""
        data = []
        for result in results:
            data.append({
                'Feature': result.feature,
                'Type': result.feature_type,
                'Drift Detected': result.drift_detected,
                'Statistic': f"{result.statistic:.4f}",
                'P-Value': f"{result.p_value:.4f}",
                'Severity': result.severity.upper(),
                'Description': result.description[:50] + '...' if len(result.description) > 50 else result.description
            })
        
        return pd.DataFrame(data)

    def compare_distributions(
        self,
        feature_name: str,
        train_data: np.ndarray,
        test_data: np.ndarray,
        bins: int = 20
    ) -> Dict[str, Any]:
        """Get statistics for comparing distributions."""
        try:
            train_clean = train_data[~np.isnan(train_data)]
            test_clean = test_data[~np.isnan(test_data)]
            
            train_hist, bin_edges = np.histogram(train_clean, bins=bins)
            test_hist, _ = np.histogram(test_clean, bins=bin_edges)
            
            return {
                'feature': feature_name,
                'train_hist': train_hist.tolist(),
                'test_hist': test_hist.tolist(),
                'bins': bin_edges.tolist(),
                'train_mean': float(train_clean.mean()),
                'train_std': float(train_clean.std()),
                'test_mean': float(test_clean.mean()),
                'test_std': float(test_clean.std()),
                'train_count': len(train_clean),
                'test_count': len(test_clean)
            }
            
        except Exception as e:
            logger.error(f"Error comparing distributions for {feature_name}: {e}")
            raise
