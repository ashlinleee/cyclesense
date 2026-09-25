"""
Data loading and validation for CycleSense.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

from src.config import (
    PERIOD_LOG_FILE,
    USER_PROFILE_FILE,
    RANDOM_STATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Expected schema
EXPECTED_PERIOD_LOG_COLUMNS = [
    "user_id", "cycle_number", "start_date", "cycle_length_days", "prev_cycle_length",
    "cycle_phase", "flow_level", "pain_level", "pms_symptoms", "mood_score",
    "stress_score_cycle", "sleep_hours_cycle", "energy_level", "concentration_score",
    "work_hours_lost", "estrogen_pgml", "progesterone_ngml", "ovulation_result",
    "overall_health_score", "log_consistency_score", "prepared_before_period"
]

EXPECTED_USER_PROFILE_COLUMNS = [
    "user_id", "state", "age", "bmi", "diet_quality", "exercise_frequency",
    "sleep_hours", "caffeine_intake", "water_intake_liters", "alcohol_consumption",
    "smoking_status", "birth_control_use", "pcos_diagnosed", "stress_score_baseline"
]


def load_period_log() -> pd.DataFrame:
    """Load Period_Log.csv with validation."""
    logger.info(f"Loading period log from {PERIOD_LOG_FILE}")
    
    if not PERIOD_LOG_FILE.exists():
        raise FileNotFoundError(f"Period log file not found: {PERIOD_LOG_FILE}")
    
    df = pd.read_csv(PERIOD_LOG_FILE)
    
    # Validate schema
    actual_columns = list(df.columns)
    if actual_columns != EXPECTED_PERIOD_LOG_COLUMNS:
        logger.warning(f"Schema mismatch in Period_Log.csv")
        logger.warning(f"Expected: {EXPECTED_PERIOD_LOG_COLUMNS}")
        logger.warning(f"Actual: {actual_columns}")
    
    logger.info(f"Loaded {len(df)} period log records with {len(df.columns)} columns")
    return df


def load_user_profile() -> pd.DataFrame:
    """Load User_Profile.csv with validation."""
    logger.info(f"Loading user profile from {USER_PROFILE_FILE}")
    
    if not USER_PROFILE_FILE.exists():
        raise FileNotFoundError(f"User profile file not found: {USER_PROFILE_FILE}")
    
    df = pd.read_csv(USER_PROFILE_FILE)
    
    # Validate schema
    actual_columns = list(df.columns)
    if actual_columns != EXPECTED_USER_PROFILE_COLUMNS:
        logger.warning(f"Schema mismatch in User_Profile.csv")
        logger.warning(f"Expected: {EXPECTED_USER_PROFILE_COLUMNS}")
        logger.warning(f"Actual: {actual_columns}")
    
    logger.info(f"Loaded {len(df)} user profiles with {len(df.columns)} columns")
    return df


def validate_dataset_schema() -> dict:
    """
    Programmatically validate dataset schema and report discrepancies.
    Returns a dictionary with validation results.
    """
    logger.info("=== DATASET SCHEMA VALIDATION ===")
    
    period_df = load_period_log()
    profile_df = load_user_profile()
    
    validation_report = {
        "period_log": {
            "expected_rows": 17976,
            "actual_rows": len(period_df),
            "expected_columns": 21,
            "actual_columns": len(period_df.columns),
            "schema_match": list(period_df.columns) == EXPECTED_PERIOD_LOG_COLUMNS,
            "expected_columns_list": EXPECTED_PERIOD_LOG_COLUMNS,
            "actual_columns_list": list(period_df.columns)
        },
        "user_profile": {
            "expected_rows": 2000,
            "actual_rows": len(profile_df),
            "expected_columns": 14,
            "actual_columns": len(profile_df.columns),
            "schema_match": list(profile_df.columns) == EXPECTED_USER_PROFILE_COLUMNS,
            "expected_columns_list": EXPECTED_USER_PROFILE_COLUMNS,
            "actual_columns_list": list(profile_df.columns)
        }
    }
    
    # Log results
    for dataset, report in validation_report.items():
        logger.info(f"\n{dataset.upper()}:")
        logger.info(f"  Rows: Expected {report['expected_rows']}, Actual {report['actual_rows']}")
        logger.info(f"  Columns: Expected {report['expected_columns']}, Actual {report['actual_columns']}")
        logger.info(f"  Schema Match: {report['schema_match']}")
        
        if not report['schema_match']:
            logger.warning(f"  SCHEMA MISMATCH DETECTED!")
    
    return validation_report


def load_and_join_datasets() -> pd.DataFrame:
    """
    Load both datasets and join them on user_id.
    Validate the join for data quality.
    """
    logger.info("=== LOADING AND JOINING DATASETS ===")
    
    period_df = load_period_log()
    profile_df = load_user_profile()
    
    # Check for profile uniqueness
    profile_counts = profile_df['user_id'].value_counts()
    duplicate_profiles = profile_counts[profile_counts > 1]
    
    if len(duplicate_profiles) > 0:
        raise ValueError(f"Profile uniqueness violated: {len(duplicate_profiles)} users have duplicate profiles")
    
    # Perform left join (each period record gets profile info)
    merged_df = period_df.merge(profile_df, on='user_id', how='left')
    
    # Check for missing profiles
    missing_profiles = merged_df['age'].isna().sum()
    if missing_profiles > 0:
        logger.warning(f"{missing_profiles} period records have missing user profiles")
    
    logger.info(f"Joined datasets: {len(merged_df)} records")
    logger.info(f"Period records without profile: {missing_profiles}")
    
    return merged_df


if __name__ == "__main__":
    # Run validation
    report = validate_dataset_schema()
    
    # Test join
    merged = load_and_join_datasets()
    print(f"\nMerged dataset shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
