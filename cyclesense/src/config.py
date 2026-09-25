"""
CycleSense Configuration
Centralized configuration for reproducibility.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "CycleSense"
MODEL_REGISTRY_NAME = "CycleSenseCycleLengthModel"

# Data files
PERIOD_LOG_FILE = RAW_DATA_DIR / "Period_Log.csv"
USER_PROFILE_FILE = RAW_DATA_DIR / "User_Profile.csv"

# Random seed for reproducibility
RANDOM_STATE = 42

# Model parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering parameters
HISTORICAL_LAGS = [1, 2, 3]
ROLLING_WINDOWS = [3, 5]

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
STREAMLIT_PORT = 8501

# Medical disclaimer
MEDICAL_DISCLAIMER = """
CycleSense provides data-driven cycle estimates for educational demonstration 
and pattern exploration. Predictions are estimates and should not be used for 
diagnosis, contraception, fertility planning, or medical decisions.
"""
