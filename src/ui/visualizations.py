"""Visualization utilities for the Streamlit frontend."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List

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
        """Plot distribution of a feature."""
        fig = go.Figure()
        
        if df[column].dtype in ['int64', 'float64']:
            # Numeric feature
            if target_col and target_col in df.columns:
                for label in df[target_col].unique():
                    mask = df[target_col] == label
                    fig.add_trace(go.Histogram(
                        x=df[mask][column],
                        name=f'{target_col}={label}',
                        opacity=0.7
                    ))
            else:
                fig.add_trace(go.Histogram(x=df[column], name=column))
        else:
            # Categorical feature
            if target_col and target_col in df.columns:
                counts = df.groupby([column, target_col]).size().reset_index(name='count')
                fig = px.bar(counts, x=column, y='count', color=target_col, barmode='group')
            else:
                counts = df[column].value_counts().reset_index()
                counts.columns = [column, 'count']
                fig = px.bar(counts, x=column, y='count')
        
        fig.update_layout(
            title=f'Distribution of {column}',
            xaxis_title=column,
            yaxis_title='Frequency',
            hovermode='x unified'
        )
        
        return fig

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