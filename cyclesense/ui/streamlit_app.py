"""
CycleSense Streamlit Dashboard.
Provides a user interface for cycle prediction and pattern insights.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import json

import os

# API configuration (reads environment variable API_URL, defaults to local API)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="CycleSense",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional health-tech aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0B3D3F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5B6B69;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #FFF3CD;
        border-left: 4px solid #FFC107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def call_api(endpoint, method="GET", data=None):
    """Call the CycleSense API."""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, json=data, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None


def render_disclaimer():
    """Render medical disclaimer."""
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> CycleSense provides data-driven cycle estimates 
        for educational demonstration and pattern exploration. Predictions are estimates and should 
        not be used for diagnosis, contraception, fertility planning, or medical decisions.
    </div>
    """, unsafe_allow_html=True)


def render_dashboard():
    """Render main dashboard."""
    st.markdown('<h1 class="main-header">🩺 CycleSense</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Menstrual Cycle Prediction & Pattern Intelligence</p>', unsafe_allow_html=True)
    
    render_disclaimer()
    
    # API Health Check
    with st.spinner("Connecting to CycleSense API..."):
        health = call_api("/health")
    
    if health and health.get("model_loaded"):
        st.success("✅ Connected to CycleSense API")
    else:
        st.error("❌ Unable to connect to CycleSense API")
        st.info(f"Please ensure the API is accessible at `{API_URL}`")
        return


def render_predict_page():
    """Render prediction page."""
    st.header("🔮 Predict Next Cycle")
    
    render_disclaimer()
    
    # Profile Section
    st.subheader("User Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 50, 28)
        bmi = st.slider("BMI", 15.0, 40.0, 22.8, 0.1)
        diet_quality = st.selectbox("Diet Quality", ["Poor", "Fair", "Good", "Excellent"])
        exercise_frequency = st.selectbox("Exercise Frequency", 
                                        ["1-2 days/week", "3-4 days/week", "5-6 days/week"])
        sleep_hours = st.slider("Average Sleep Hours", 4.0, 10.0, 7.0, 0.1)
    
    with col2:
        caffeine_intake = st.slider("Caffeine Intake (cups/day)", 0.0, 5.0, 1.8, 0.1)
        water_intake = st.slider("Water Intake (liters/day)", 0.5, 5.0, 2.3, 0.1)
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Weekly"])
        smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])
        birth_control = st.selectbox("Birth Control Use", [0, 1], format_func=lambda x: "Yes" if x else "No")
        pcos_diagnosed = st.selectbox("PCOS Diagnosed", [0, 1], format_func=lambda x: "Yes" if x else "No")
        stress_baseline = st.slider("Baseline Stress Score (1-10)", 1.0, 10.0, 5.6, 0.1)
    
    # Cycle History Section
    st.subheader("Current Cycle Information")
    col3, col4 = st.columns(2)
    
    with col3:
        cycle_length = st.slider("Current Cycle Length (days)", 20.0, 45.0, 28.0, 0.1)
        prev_cycle_length = st.slider("Previous Cycle Length (days)", 20.0, 45.0, 28.0, 0.1)
        cycle_phase = st.selectbox("Current Cycle Phase", ["Follicular", "Luteal", "Ovulation"])
        flow_level = st.selectbox("Flow Level", ["Light", "Medium", "Heavy"])
    
    with col4:
        pain_level = st.slider("Pain Level (1-10)", 1, 10, 5)
        pms_symptoms = st.selectbox("PMS Symptoms", ["No", "Yes"])
        mood_score = st.slider("Mood Score (1-10)", 1, 10, 7)
        stress_cycle = st.slider("Current Stress Score (1-10)", 1.0, 10.0, 5.8, 0.1)
        sleep_cycle = st.slider("Current Sleep Hours", 4.0, 10.0, 7.0, 0.1)
    
    # Historical Cycles
    st.subheader("Historical Cycle Lengths")
    historical_cycles = st.text_input(
        "Enter previous cycle lengths (comma-separated, most recent first)",
        "28, 29, 27, 28, 30"
    )
    
    try:
        historical_list = [float(x.strip()) for x in historical_cycles.split(",") if x.strip()]
    except ValueError:
        historical_list = []
    
    # Prediction Button
    if st.button("🔮 Predict Next Cycle", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            # Prepare request
            request_data = {
                "profile": {
                    "age": age,
                    "bmi": bmi,
                    "diet_quality": diet_quality,
                    "exercise_frequency": exercise_frequency,
                    "sleep_hours": sleep_hours,
                    "caffeine_intake": caffeine_intake,
                    "water_intake_liters": water_intake,
                    "alcohol_consumption": alcohol_consumption,
                    "smoking_status": smoking_status,
                    "birth_control_use": birth_control,
                    "pcos_diagnosed": pcos_diagnosed,
                    "stress_score_baseline": stress_baseline
                },
                "cycle_history": {
                    "cycle_length_days": cycle_length,
                    "prev_cycle_length": prev_cycle_length,
                    "cycle_phase": cycle_phase,
                    "flow_level": flow_level,
                    "pain_level": pain_level,
                    "pms_symptoms": pms_symptoms,
                    "mood_score": mood_score,
                    "stress_score_cycle": stress_cycle,
                    "sleep_hours_cycle": sleep_cycle,
                    "energy_level": 7,
                    "concentration_score": 7,
                    "work_hours_lost": 3.0,
                    "start_date": date.today().isoformat()
                },
                "historical_cycles": historical_list
            }
            
            # Call API
            response = call_api("/predict", method="POST", data=request_data)
            
            if response:
                # Display prediction
                st.success("✅ Prediction Complete!")
                
                col5, col6 = st.columns(2)
                with col5:
                    st.metric(
                        "Predicted Next Cycle Length",
                        f"{response['predicted_next_cycle_length_days']:.1f} days",
                        "ML Estimate"
                    )
                
                with col6:
                    st.metric(
                        "Model Version",
                        response['model_version'],
                        "Educational Estimate"
                    )
                
                st.info(f"Prediction Type: {response['prediction_type']}")


def render_pattern_insights():
    """Render pattern insights page."""
    st.header("📊 Pattern Insights")
    
    render_disclaimer()
    
    st.info("Pattern insights would be displayed here based on historical cycle data.")
    st.info("This feature requires historical cycle data analysis.")


def render_model_insights():
    """Render model insights page."""
    st.header("🤖 Model Insights")
    
    render_disclaimer()
    
    # Get model info
    model_info = call_api("/model-info")
    
    if model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            st.json({
                "Model Name": model_info.get("model_name"),
                "Model Type": model_info.get("model_type"),
                "Version": model_info.get("model_version"),
                "Feature Set": model_info.get("feature_set")
            })
        
        with col2:
            st.subheader("Performance Metrics")
            metrics = model_info.get("metrics", {})
            if metrics:
                st.metric("MAE", f"{metrics.get('mae', 'N/A'):.3f} days")
                st.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.3f} days")
                st.metric("R²", f"{metrics.get('r2', 'N/A'):.3f}")
        
        st.subheader("Medical Disclaimer")
        st.warning(model_info.get("medical_disclaimer", "See medical disclaimer"))


def render_about():
    """Render about page."""
    st.header("ℹ️ About CycleSense")
    
    render_disclaimer()
    
    st.markdown("""
    ## What is CycleSense?
    
    CycleSense is an explainable machine-learning system that learns from a user's profile 
    and previous menstrual-cycle history to estimate their next cycle length and analyze 
    patterns influencing that prediction.
    
    ## Dataset
    
    - **User Profiles**: 2,000 users with demographic and lifestyle information
    - **Cycle Logs**: 17,976 cycle records with detailed measurements
    - **Features**: Profile data, cycle history, hormonal measurements, symptom tracking
    
    ## Feature Engineering
    
    - **Historical Features**: Lag features, rolling statistics, cycle variability
    - **Profile Features**: Age, BMI, lifestyle factors, baseline stress
    - **Cross-Source Features**: Stress deltas, sleep deltas, interaction terms
    - **Date Features**: Seasonal patterns, cyclical representations
    
    ## Model
    
    - **Algorithm**: Histogram-based Gradient Boosting
    - **Performance**: MAE ≈ 1.7 days
    - **Explainability**: SHAP values, permutation importance
    
    ## MLOps Pipeline
    
    - **Data Versioning**: DVC for dataset tracking
    - **Experiment Tracking**: MLflow for model experiments
    - **Model Registry**: MLflow Model Registry for versioning
    - **API**: FastAPI for production inference
    - **Monitoring**: Prediction metrics and drift detection
    
    ## Limitations
    
    - Educational demonstration purposes only
    - Not a diagnostic or medical advice system
    - Based on synthetic educational dataset
    - Should not replace professional medical care
    
    ## Technology Stack
    
    - Python, scikit-learn, pandas
    - MLflow, DVC
    - FastAPI, Streamlit
    - Docker, GitHub Actions
    """)


def main():
    """Main application."""
    # Sidebar navigation
    st.sidebar.title("🩺 CycleSense")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Predict Next Cycle", "Pattern Insights", "Model Insights", "About"]
    )
    
    # Render selected page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Predict Next Cycle":
        render_predict_page()
    elif page == "Pattern Insights":
        render_pattern_insights()
    elif page == "Model Insights":
        render_model_insights()
    elif page == "About":
        render_about()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Built for educational demonstration**
    
    CycleSense provides data-driven cycle estimates for educational demonstration and pattern exploration.
    """)


if __name__ == "__main__":
    main()
