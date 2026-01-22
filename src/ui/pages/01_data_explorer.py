"""Streamlit Data Explorer Page"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

from src.ui.session_manager import init_session_state
from src.ui.utils import load_demo_dataset, render_error_message
from src.ui.charts import plot_data_distribution, plot_correlation_heatmap

# Configure page FIRST (must be first Streamlit command)
st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state ONCE
init_session_state()

st.title("📊 Data Explorer")

# ===== DATA LOADING =====
st.markdown("## 📤 Load Your Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("📁 Use Demo Dataset", use_container_width=True):
        try:
            df = load_demo_dataset()
            st.session_state.df = df
            st.success("✅ Demo dataset loaded (10,000 rows)")
        except FileNotFoundError:
            st.error("⚠️ Demo dataset not found")
            st.stop()

with col2:
    uploaded_file = st.file_uploader("📥 Or upload your CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ Loaded {uploaded_file.name} ({len(df):,} rows)")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Load data from session or default to demo
if 'df' not in st.session_state or st.session_state.df is None:
    try:
        st.session_state.df = load_demo_dataset()
    except:
        st.warning("⚠️ Please load data using buttons above")
        st.stop()

df = st.session_state.df

# ===== DATA QUALITY CHECKS =====
from src.ui.utils import validate_dataset

quality_metrics = validate_dataset(df)

if quality_metrics['warnings']:
    st.warning("🔍 **Data Quality Insights:**")
    for warning in quality_metrics['warnings']:
        st.caption(warning)

# ===== OVERVIEW =====
st.markdown("## 📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Rows", f"{len(df):,}")

with col2:
    st.metric("📈 Columns", df.shape[1])

with col3:
    st.metric("🔢 Numeric", df.select_dtypes(include=[np.number]).shape[1])

with col4:
    st.metric("📝 Categorical", df.select_dtypes(include=['object']).shape[1])

# ===== DATA PREVIEW =====
st.markdown("## 🔍 Data Preview")

preview_rows = st.slider("Rows to display:", min_value=5, max_value=50, value=10, key="preview_rows_tab1")
st.dataframe(df.head(preview_rows), use_container_width=True)

# ===== DATA TYPES & INFO =====
st.markdown("## 📌 Column Information")

col_info = []
for col in df.columns:
    col_info.append({
        "Column": col,
        "Type": str(df[col].dtype),
        "Non-Null": f"{df[col].notna().sum():,}",
        "Unique": df[col].nunique(),
        "Missing": f"{(df[col].isnull().sum() / len(df) * 100):.1f}%"
    })

st.dataframe(pd.DataFrame(col_info), use_container_width=True)

# ===== STATISTICS =====
st.markdown("## 📊 Statistical Summary")

st.dataframe(df.describe(), use_container_width=True)

# ===== VISUALIZATIONS =====
st.markdown("## 📈 Visualizations")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📌 Targets"])

with tab1:
    if numeric_cols:
        col_dist = st.selectbox("Select column to visualize:", numeric_cols)
        plot_data_distribution(df, col_dist)
    else:
        st.info("No numeric columns to visualize")

with tab2:
    if len(numeric_cols) > 1:
        with st.spinner("📊 Calculating correlations..."):
            plot_correlation_heatmap(df)
    else:
        st.info("Not enough numeric columns for correlation analysis")

with tab3:
    target_candidates = [col for col in df.columns if 2 <= df[col].nunique() <= 10]
    
    if target_candidates:
        target = st.selectbox("Select target column:", target_candidates)
        
        target_counts = df[target].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(x=target_counts.index, y=target_counts.values, marker_color='#FF6B6B')
        ])
        fig.update_layout(
            title=f"Distribution of {target}",
            xaxis_title=target,
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**{target} Distribution:**")
        for idx, (val, count) in enumerate(target_counts.items()):
            pct = count / len(df) * 100
            st.write(f"  • {val}: {count:,} ({pct:.1f}%)")
    else:
        st.info("No suitable target columns found")

# ===== MISSING DATA =====
st.markdown("## ⚠️ Missing Data Analysis")

missing = df.isnull().sum()
if missing.sum() > 0:
    missing_df = pd.DataFrame({
        "Column": missing[missing > 0].index,
        "Missing Count": missing[missing > 0].values,
        "Percentage": (missing[missing > 0].values / len(df) * 100).round(2)
    })
    st.dataframe(missing_df, use_container_width=True)
else:
    st.success("✅ No missing values!")

# Footer
st.markdown("---")
st.markdown("""
### 💡 Tips
- Use this page to understand your data before training
- Look for patterns and relationships
- Check for missing values and outliers
- Ready to train? Go to "Train Model" page
""")
