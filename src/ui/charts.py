"""Chart Generation Utilities - Consolidated visualization functions"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any


def plot_data_distribution(df: pd.DataFrame, column: str, title: str = None) -> None:
    """Plot distribution of a column."""
    if column not in df.columns:
        st.error(f"Column '{column}' not found")
        return
    
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=title or f"Distribution of {column}",
        labels={column: column}
    )
    
    fig.update_layout(showlegend=False, height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(df: pd.DataFrame, max_features: int = 15, title: str = "Feature Correlation Matrix") -> None:
    """Plot correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        st.warning("No numeric columns found")
        return
    
    if len(numeric_df.columns) > max_features:
        numeric_df = numeric_df.loc[:, numeric_df.var().nlargest(max_features).index]
    
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        )
    )
    
    fig.update_layout(title=title, height=500, xaxis_title="Features", yaxis_title="Features")
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(importances: Dict[str, float], top_n: int = 15, title: str = "Top Features by Importance") -> None:
    """Plot feature importance bar chart."""
    if not importances:
        st.warning("No importance data available")
        return
    
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, scores = zip(*sorted_imp)
    
    fig = go.Figure(
        data=go.Bar(y=list(features), x=list(scores), orientation='h', marker_color='steelblue')
    )
    
    fig.update_layout(title=title, xaxis_title="Importance Score", height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_whatif_analysis(
    test_values: List[float],
    formatted_test_values: List[str],
    confidences: List[float],
    feature_name: str
) -> None:
    """Plot What-If analysis: how prediction changes with feature."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_values,
        y=confidences,
        mode='lines+markers',
        name='Confidence',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{customdata}</b><br>Confidence: %{y:.1f}%<extra></extra>',
        customdata=formatted_test_values
    ))
    
    fig.update_layout(
        title=f"Confidence vs {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Confidence (%)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_prediction_distribution(predictions: List[float], actual: List[int] = None, title: str = "Prediction Confidence Distribution") -> None:
    """Plot histogram of prediction confidences."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(x=predictions, nbinsx=20, name='Predictions', marker_color='steelblue'))
    
    fig.update_layout(
        title=title,
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
