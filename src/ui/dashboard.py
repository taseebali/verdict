"""
Verdict ML Platform - Home Page
Streamlined 5-step workflow for ML model building
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ui.session_manager import init_session_state

# Configure page
st.set_page_config(
    page_title="Verdict ML Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
init_session_state()

# Custom CSS for styling
st.markdown("""
<style>
    .workflow-step {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .workflow-step.inactive {
        background: #e0e0e0;
        color: #666;
    }
    .step-number {
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .step-title {
        font-size: 1.2em;
        margin-bottom: 5px;
    }
    .step-description {
        font-size: 0.9em;
        opacity: 0.95;
    }
    .workflow-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin: 20px 0;
    }
    .faq-section {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    .navigation-button {
        margin: 10px 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("# 🚀 Welcome to Verdict ML Platform")
st.markdown("""
### Build and Deploy Machine Learning Models in 5 Simple Steps

This platform guides you through the complete ML workflow—from understanding your data to making predictions. 
No experience necessary. **No mumbo jumbo. Just clear, simple steps.**
""")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Workflow Guide", "❓ FAQ", "⚡ Quick Start", "📚 Learn More"])

# ============================================================================
# TAB 1: WORKFLOW GUIDE
# ============================================================================
with tab1:
    st.markdown("## Your 5-Step Journey to ML Mastery")
    
    col1, col2 = st.columns(2)
    
    # Step 1: Load Data
    with col1:
        st.markdown("""
        ### 📁 Step 1: Load Your Data
        
        **What to do:**
        - Start with a CSV file (or use our demo dataset)
        - Upload your data to the platform
        
        **Why it matters:**
        - Your data is the foundation of everything
        - Good data = good predictions
        
        **Time:** 2 minutes
        """)
        if st.button("→ Go to Data Explorer", key="btn_explore"):
            st.switch_page("pages/01_data_explorer.py")
    
    # Step 2: Explore Data
    with col2:
        st.markdown("""
        ### 🔍 Step 2: Understand Your Data
        
        **What to do:**
        - See statistics and patterns in your data
        - Visualize relationships between columns
        - Identify what you want to predict
        
        **Why it matters:**
        - Understanding your data helps you train better models
        - You'll see what's useful and what's not
        
        **Time:** 5 minutes
        """)
    
    col1, col2 = st.columns(2)
    
    # Step 3: Train Model
    with col1:
        st.markdown("""
        ### 🤖 Step 3: Train Your Model
        
        **What to do:**
        - Select which column you want to predict
        - Exclude columns that won't help (we suggest which ones)
        - Click "Train Model" and wait
        
        **Why it matters:**
        - The model learns patterns from your data
        - Good column selection = more accurate predictions
        
        **Time:** 2-10 minutes (depending on data size)
        """)
        if st.button("→ Go to Model Training", key="btn_train"):
            st.switch_page("pages/02_model_training.py")
    
    # Step 4: Make Predictions
    with col2:
        st.markdown("""
        ### 🎯 Step 4: Make Predictions
        
        **What to do:**
        - Use sliders to set feature values
        - See what the model predicts
        - Try What-If scenarios
        
        **Why it matters:**
        - This is where you use your trained model
        - See real predictions on real data
        
        **Time:** 1 minute per prediction
        """)
        if st.button("→ Go to Predictions", key="btn_predict"):
            st.switch_page("pages/03_predictions.py")
    
    col1, col2 = st.columns(2)
    
    # Step 5: Review Results
    with col1:
        st.markdown("""
        ### 📋 Step 5: Review and Audit
        
        **What to do:**
        - See all your predictions in one place
        - Check accuracy and performance metrics
        - Export results
        
        **Why it matters:**
        - Verify your model is working correctly
        - Track what was predicted and when
        
        **Time:** 1 minute
        """)
        if st.button("→ Go to Audit Logs", key="btn_audit"):
            st.switch_page("pages/04_audit_logs.py")

# ============================================================================
# TAB 2: FAQ
# ============================================================================
with tab2:
    st.markdown("## Common Questions")
    
    with st.expander("❓ What are 'excluded columns'?"):
        st.markdown("""
        When training a model, some columns DON'T help predict your target. Examples:
        
        - **customer_id**: Just an ID number, not predictive
        - **signup_date**: The exact date doesn't matter, only how long they've been a customer
        - **timestamps**: When data was recorded, not useful for predictions
        
        **Why exclude them?**
        - More clarity: The model focuses on what actually matters
        - Better accuracy: Less noise = clearer patterns
        - Faster training: Fewer columns = faster computation
        
        **Our recommendation:**
        The Training page automatically detects and suggests which columns to exclude.
        We explain WHY each one should be excluded.
        """)
    
    with st.expander("❓ Why do the prediction sliders show 'real' values like '$89.50' and '35 years'?"):
        st.markdown("""
        Because that's how your data actually looks!
        
        **Examples:**
        - If your data contains ages like 25, 35, 45, the slider shows "35 years" (the average)
        - If monthly charges range from $20-$150, the slider shows "$89.50" (the average)
        - If tenure is 5-60 months, the slider shows "24 months" (the average)
        
        **Why?**
        - Realistic predictions: You're predicting on real-world data
        - No confusion: You see "$89.50" instead of "89.50" - much clearer
        - Good defaults: Slider starts at the average, not zero
        """)
    
    with st.expander("❓ What's the difference between Training Accuracy and Test Accuracy?"):
        st.markdown("""
        The model learns from **training data** and is tested on **test data** it's never seen.
        
        | Metric | What It Means |
        |--------|--------------|
        | **Training Accuracy** | How well it predicts data it already learned from |
        | **Test Accuracy** | How well it predicts NEW data it's never seen |
        
        **The catch:**
        - Training accuracy can be inflated (the model memorized patterns)
        - Test accuracy is more honest (shows real-world performance)
        - Look at TEST accuracy for the true story
        
        If they're very different, your model might be "overfitting" (memorizing instead of learning).
        """)
    
    with st.expander("❓ What is Feature Importance?"):
        st.markdown("""
        Shows which columns (features) the model uses MOST to make predictions.
        
        **Example:**
        - If predicting customer churn:
          - Feature importance might show: tenure (40%), monthly_charges (25%), complaints (20%)
          - Meaning: How long they've been a customer matters most
        
        **Why it matters:**
        - Understand what drives your predictions
        - Focus on what actually matters
        - Spot if the model is learning the right things
        """)
    
    with st.expander("❓ What if my predictions look wrong?"):
        st.markdown("""
        Three things to check:
        
        1. **Do you have enough data?**
           - More data = better learning = better predictions
           - Ideally 1000+ samples
        
        2. **Are you excluding the right columns?**
           - Too many exclusions: Model loses important info
           - Too few exclusions: Model gets confused by noise
           - Use our recommendations as a starting point
        
        3. **Is there a clear pattern to predict?**
           - Some targets are harder to predict than others
           - If target is random, even a perfect model can't help
        
        **Solution:** Train a new model, excluding different columns, and compare results.
        """)

# ============================================================================
# TAB 3: QUICK START
# ============================================================================
with tab3:
    st.markdown("## Quick Start Guides")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🏃 I have 5 minutes
        
        1. Click "Go to Data Explorer"
        2. Review the demo dataset overview
        3. Click "Go to Model Training"
        4. Click "Train Model" with defaults
        5. Click "Go to Predictions"
        6. Play with the sliders
        
        **Result:** You'll understand how the platform works!
        """)
    
    with col2:
        st.markdown("""
        ### 🚴 I have 15 minutes
        
        1. Go to **Data Explorer**
           - Review overview and statistics
           - Check the correlation heatmap
        2. Go to **Model Training**
           - Review recommended columns
           - Train the model
           - Check feature importance
        3. Go to **Predictions**
           - Try different What-If scenarios
           - See how changes affect predictions
        4. Go to **Audit Logs**
           - See prediction history
        
        **Result:** Full end-to-end workflow demo!
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎓 I want to understand ML
        
        1. **Data Explorer:** Understand what the numbers mean
        2. **Model Training:** Learn why we exclude columns
        3. **Read Feature Importance:** See what the model values
        4. **Predictions (What-If):** Experiment with scenarios
        5. **Explore the code:** It's all documented!
        
        **Bonus:** Read the FAQ section above
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 I have my own data
        
        1. Prepare a CSV file with:
           - Column headers on first row
           - Numbers or categories
           - NO weird formatting
        2. Use the Data Explorer to upload
        3. Follow the workflow steps
        4. Train on YOUR data
        5. Make real predictions
        
        **Tip:** The more data, the better!
        """)

# ============================================================================
# TAB 4: LEARN MORE
# ============================================================================
with tab4:
    st.markdown("## Key Concepts Explained")
    
    st.markdown("""
    ### What is Machine Learning?
    
    Machine learning trains a computer to find patterns in data, so it can:
    - **Predict** outcomes (e.g., will customer churn?)
    - **Classify** things (e.g., satisfied or unsatisfied?)
    - **Rank** importance (e.g., what matters most?)
    
    **The workflow:**
    1. Feed data to the model
    2. Model finds patterns
    3. Model makes predictions on new data
    
    ---
    
    ### Training vs. Testing
    
    To know if your model is any good, we need to test it on data it's never seen:
    
    - **Training (70% of data):** Model learns from this
    - **Testing (30% of data):** We check accuracy on this
    
    This tells us: "How well will it work on completely new data?"
    
    ---
    
    ### Why Features Matter
    
    A "feature" is just a column in your data. Some features are useful for predictions:
    - ✅ Age (older customers might behave differently)
    - ✅ Tenure (loyal customers are predictable)  
    - ❌ Customer ID (just an ID, not predictive)
    - ❌ Record timestamp (when data was saved, not predictive)
    
    **The art of ML:** Finding the RIGHT features
    
    ---
    
    ### Accuracy vs. Reality
    
    An 85% accurate model means:
    - ✅ Out of 100 predictions, ~85 will be correct
    - ❌ Out of 100 predictions, ~15 will be wrong
    
    **Is 85% good?** Depends on:
    - Your use case (predicting fraud? Higher bar!)
    - What it's replacing (better than human guesses? Great!)
    - Cost of mistakes (expensive? Demand higher accuracy!)
    
    ---
    
    ### Common Pitfalls
    
    🚫 **Overfitting:** Model memorizes training data but fails on new data
    - Solution: Check test accuracy, ensure it's close to training accuracy
    
    🚫 **Not enough data:** 10 samples can't teach a model anything
    - Solution: Collect more data
    
    🚫 **Bad features:** Including junk columns confuses the model
    - Solution: Exclude non-predictive columns (we help with this!)
    
    🚫 **Wrong target:** Can't predict something with no pattern
    - Solution: Pick a target with real variation
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; margin-top: 30px;'>
    <h3>Ready to get started?</h3>
    <p>Click "Data Explorer" in the sidebar → to begin your ML journey!</p>
    <p style='font-size: 0.9em; color: #999;'>
        Verdict ML Platform • Built for clarity, accuracy, and ease of use
    </p>
</div>
""", unsafe_allow_html=True)
