"""Visualization utilities for the Streamlit frontend."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_bool_dtype,
    is_numeric_dtype,
    is_string_dtype,
    is_object_dtype,
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class Visualizer:
    """Create visualizations for model evaluation and exploration."""

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
        """Create confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        
        return fig

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str = "ROC Curve"):
        """Create ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        
        return fig

    @staticmethod
    def plot_feature_distribution(df: pd.DataFrame, column: str, target_col: str = None):
        """
        Backward compatible wrapper.
        Uses auto_feature_view under the hood.
        """
        fig, meta = Visualizer.auto_feature_view(df, column, target_col=target_col)
        # If None, return an empty figure with a readable title
        if fig is None:
            empty = go.Figure()
            empty.update_layout(
                title=f"No chart for {column}",
                xaxis_title="",
                yaxis_title="",
            )
            return empty
        return fig
    @staticmethod
    def _column_profile(df: pd.DataFrame, column: str) -> dict:
        s = df[column]
        n = len(s)
        nunique = int(s.nunique(dropna=True))
        missing = int(s.isna().sum())
        unique_ratio = (nunique / n) if n > 0 else 0.0

        # Basic dtype classification
        dtype_kind = "other"
        if is_datetime64_any_dtype(s):
            dtype_kind = "datetime"
        elif is_bool_dtype(s):
            dtype_kind = "boolean"
        elif is_numeric_dtype(s):
            dtype_kind = "numeric"
        elif is_string_dtype(s) or is_object_dtype(s):
            dtype_kind = "categorical"
        else:
            dtype_kind = "categorical"

        # Detect "ID-like" columns:
        # - high uniqueness ratio
        # - column name contains id-ish tokens
        name_lower = column.lower()
        looks_like_id_name = any(tok in name_lower for tok in ["id", "uuid", "guid", "key"])
        looks_like_id = (unique_ratio >= 0.90 and nunique >= 50) or (looks_like_id_name and unique_ratio >= 0.70)

        # Numeric discreteness: numeric but small number of unique values
        numeric_discrete = dtype_kind == "numeric" and nunique <= 15

        return {
            "n": n,
            "nunique": nunique,
            "missing": missing,
            "unique_ratio": unique_ratio,
            "dtype_kind": dtype_kind,
            "looks_like_id": looks_like_id,
            "numeric_discrete": numeric_discrete,
        }

    @staticmethod
    def auto_feature_view(
        df: pd.DataFrame,
        column: str,
        target_col: str = None,
        top_n: int = 20,
    ):
        """
        Automatically choose the best visualization for a selected column.

        Returns:
            fig (plotly.graph_objects.Figure | None),
            meta (dict): view_type, reason, notes
        """
        prof = Visualizer._column_profile(df, column)
        s = df[column]

        # 1) ID-like: no chart (best chart is no chart)
        if prof["looks_like_id"]:
            return None, {
                "view_type": "none",
                "reason": "This column looks like an identifier (almost all values are unique).",
                "notes": [
                    "Identifiers are useful for joining tables, not for statistical visualization.",
                    f"Unique values: {prof['nunique']} out of {prof['n']} rows ({prof['unique_ratio']:.0%} unique).",
                    "Suggestion: drop this column from training features, or keep only for reference.",
                ],
            }

        # 2) Datetime: time trend (counts over time)
        if prof["dtype_kind"] == "datetime":
            tmp = df[[column]].copy()
            tmp = tmp.dropna()
            tmp["count"] = 1
            # group by day (simple default)
            tmp = tmp.set_index(column).resample("D")["count"].sum().reset_index()

            fig = px.line(tmp, x=column, y="count", markers=True, title=f"Events over time: {column}")
            return fig, {
                "view_type": "time_series",
                "reason": "Datetime columns are best shown as trends over time.",
                "notes": ["You can later add a resample selector (day/week/month)."],
            }

        # 3) Numeric discrete: bar chart (0/1, small integer sets)
        if prof["numeric_discrete"]:
            counts = s.value_counts(dropna=False).reset_index()
            counts.columns = [column, "count"]
            fig = px.bar(counts, x=column, y="count", title=f"Value counts: {column}")
            return fig, {
                "view_type": "bar_counts",
                "reason": "This numeric column has few unique values, so a bar chart is clearer than a histogram.",
                "notes": [f"Unique values: {prof['nunique']}."],
            }

        # 4) Numeric continuous: histogram (optionally overlay by target)
        if prof["dtype_kind"] == "numeric":
            fig = go.Figure()
            if target_col and target_col in df.columns:
                # overlay hist by class (good for churn / yes-no)
                for label in sorted(df[target_col].dropna().unique()):
                    mask = df[target_col] == label
                    fig.add_trace(
                        go.Histogram(
                            x=df.loc[mask, column],
                            name=f"{target_col}={label}",
                            opacity=0.6,
                        )
                    )
                fig.update_layout(barmode="overlay")
                reason = "Numeric feature shown as overlapping histograms split by target."
            else:
                fig.add_trace(go.Histogram(x=df[column], name=column))
                reason = "Numeric feature shown as a histogram to reveal distribution shape."

            fig.update_layout(
                title=f"Distribution of {column}",
                xaxis_title=column,
                yaxis_title="Frequency",
                hovermode="x unified",
            )
            return fig, {"view_type": "histogram", "reason": reason, "notes": []}

        # 5) Categorical: choose bar vs top-N
        # If too many categories, show top-N + 'Other'
        if prof["dtype_kind"] == "categorical":
            nunique = prof["nunique"]

            if nunique > top_n:
                vc = s.value_counts(dropna=False)
                top = vc.head(top_n)
                other_count = int(vc.iloc[top_n:].sum())
                counts = top.reset_index()
                counts.columns = [column, "count"]
                if other_count > 0:
                    counts = pd.concat([counts, pd.DataFrame([{column: "Other", "count": other_count}])], ignore_index=True)

                fig = px.bar(counts, x=column, y="count", title=f"Top {top_n} categories: {column}")
                return fig, {
                    "view_type": "topn_bar",
                    "reason": f"Too many categories ({nunique}). Showing top {top_n} + Other.",
                    "notes": ["This prevents unreadable charts with hundreds of bars."],
                }

            # small categorical: grouped by target if available
            if target_col and target_col in df.columns and target_col != column:
                counts = df.groupby([column, target_col]).size().reset_index(name="count")
                fig = px.bar(counts, x=column, y="count", color=target_col, barmode="group",
                             title=f"{column} by {target_col}")
                return fig, {
                    "view_type": "grouped_bar",
                    "reason": "Categorical feature grouped by target to compare outcomes.",
                    "notes": [],
                }

            counts = s.value_counts(dropna=False).reset_index()
            counts.columns = [column, "count"]
            fig = px.bar(counts, x=column, y="count", title=f"Category counts: {column}")
            return fig, {
                "view_type": "bar",
                "reason": "Categorical feature with few categories shown as a bar chart.",
                "notes": [],
            }

        # Fallback
        return None, {
            "view_type": "none",
            "reason": "No suitable visualization rule matched this column type.",
            "notes": ["Try converting the column type or selecting a different feature."],
        }
    
    @staticmethod
    def plot_metrics_comparison(models_metrics: dict, metric_name: str = 'f1'):
        """Create bar chart comparing metrics across models."""
        models = list(models_metrics.keys())
        values = [models_metrics[m].get(metric_name, 0) for m in models]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=models, y=values, text=[f'{v:.3f}' for v in values], textposition='auto'))
        
        fig.update_layout(
            title=f'Model Comparison - {metric_name.upper()}',
            xaxis_title='Model',
            yaxis_title=metric_name,
            showlegend=False,
            hovermode='x'
        )
        
        return fig

    @staticmethod
    def plot_metrics_table(models_metrics: dict):
        """Create a table of all metrics."""
        data = []
        
        for model_name, metrics in models_metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[df[col] for col in df.columns], fill_color='lavender', align='left')
        )])
        
        fig.update_layout(title='Model Metrics Summary')
        
        return fig

    @staticmethod
    def plot_feature_importance_permutation(importance_dict: dict):
        """Create bar chart of permutation importance."""
        features = list(importance_dict.keys())
        values = list(importance_dict.values())
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=features,
            y=values,
            text=[f'{v:.4f}' for v in values],
            textposition='auto',
            marker=dict(color=values, colorscale='Viridis')
        ))
        
        fig.update_layout(
            title='Feature Importance (Permutation)',
            xaxis_title='Feature',
            yaxis_title='Importance Score',
            hovermode='x',
            showlegend=False
        )
        
        return fig

    @staticmethod
    def plot_sensitivity_analysis(sensitivity_df: pd.DataFrame, feature_name: str):
        """Create line plot for sensitivity analysis."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sensitivity_df['feature_value'],
            y=sensitivity_df['prediction'],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='#3498db', width=2),
            marker=dict(size=8)
        ))
        
        if 'confidence' in sensitivity_df.columns:
            fig.add_trace(go.Scatter(
                x=sensitivity_df['feature_value'],
                y=sensitivity_df['confidence'],
                mode='lines',
                name='Confidence',
                line=dict(color='#e74c3c', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title=f'Sensitivity Analysis: {feature_name}',
            xaxis_title=feature_name,
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        return fig
    @staticmethod
    def plot_precision_recall_curve(pr_data: dict, current_threshold: float = 0.5):
        """
        Create interactive precision-recall curve with threshold indicator.
        
        Args:
            pr_data: Dict with 'precision', 'recall', 'thresholds' lists
            current_threshold: Current decision threshold to highlight
        """
        fig = go.Figure()
        
        # Add precision-recall curve
        fig.add_trace(go.Scatter(
            x=pr_data['recall'],
            y=pr_data['precision'],
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(color='#3498db', width=3)
        ))
        
        # Add threshold markers
        fig.add_trace(go.Scatter(
            x=pr_data['recall'],
            y=pr_data['precision'],
            mode='markers',
            name='Thresholds',
            marker=dict(
                size=6,
                color=pr_data['thresholds'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Threshold")
            ),
            text=[f"Threshold: {t:.2f}" for t in pr_data['thresholds']],
            hovertemplate='<b>%{text}</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Precision-Recall Tradeoff (Current Threshold: {current_threshold:.2f})',
            xaxis_title='Recall',
            yaxis_title='Precision',
            hovermode='closest',
            height=500
        )
        
        return fig