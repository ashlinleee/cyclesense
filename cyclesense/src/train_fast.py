"""
Fast training script for CycleSense models - optimized for cold starts.
Trains a single best model without full experiment suite.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import time
from pathlib import Path
import joblib
import json

from src.config import (
    RANDOM_STATE, 
    ARTIFACTS_DIR
)
from src.data import load_and_join_datasets
from src.target import construct_next_cycle_target, create_train_test_split
from src.features import engineer_features
from src.preprocessing import prepare_model_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure artifacts directory exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def train_fast_model():
    """
    Train a single best model quickly for cold starts.
    Uses gradient boosting with simplified configuration.
    """
    logger.info("=== FAST MODEL TRAINING FOR COLD STARTS ===")
    
    start_time = time.time()
    
    # Load and prepare data
    logger.info("Loading datasets...")
    df = load_and_join_datasets()
    
    # Check if we got synthetic data (empty original data)
    if len(df) < 100:
        logger.warning(f"Dataset too small for training: {len(df)} records")
        logger.warning("This is expected in Streamlit deployment - using pre-trained model fallback")
        raise ValueError("Dataset too small for training. In production, ensure data files are present or use a pre-trained model.")
    
    df_with_target = construct_next_cycle_target(df)
    df_engineered = engineer_features(df_with_target)
    
    # Split data
    train_df, test_df = create_train_test_split(df_engineered)
    
    # Prepare model data
    logger.info("Preparing training data...")
    preprocessor, X_train, y_train, feature_names = prepare_model_data(
        train_df, feature_set='strict', use_scaling=True
    )
    
    # Prepare test data
    from src.preprocessing import select_features_by_set
    test_feature_df = select_features_by_set(test_df, 'strict')
    
    engineered_feature_cols = [
        'cycle_length_lag_1', 'cycle_length_lag_2', 'cycle_length_lag_3',
        'cycle_length_mean_last_3', 'cycle_length_std_last_3',
        'cycle_length_mean_last_5', 'cycle_length_std_last_5',
        'cycle_length_change', 'cycle_length_abs_change',
        'stress_delta', 'sleep_delta', 'stress_sleep_interaction',
        'month', 'quarter', 'day_of_year', 'month_sin', 'month_cos'
    ]
    
    for col in engineered_feature_cols:
        if col in test_df.columns and col not in test_feature_df.columns:
            test_feature_df[col] = test_df[col]
    
    X_test = preprocessor.transform(test_feature_df)
    y_test = test_df['next_cycle_length']
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {X_train.shape[1]}")
    
    # Train gradient boosting with simplified config for speed
    logger.info("Training gradient boosting model...")
    model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        max_iter=50,  # Reduced from 100 for faster training
        max_depth=6,
        learning_rate=0.1
    )
    
    train_start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - train_start
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    total_time = time.time() - start_time
    
    logger.info(f"MAE: {metrics['mae']:.3f} days")
    logger.info(f"RMSE: {metrics['rmse']:.3f} days")
    logger.info(f"R²: {metrics['r2']:.3f}")
    logger.info(f"Training time: {training_time:.3f}s")
    logger.info(f"Total time: {total_time:.3f}s")
    
    # Save model and preprocessor
    model_path = ARTIFACTS_DIR / 'model.joblib'
    joblib.dump({
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }, model_path)
    
    # Save metadata
    metadata = {
        'model_name': 'gradient_boosting_fast',
        'model_type': 'HistGradientBoostingRegressor',
        'feature_set': 'strict',
        'metrics': metrics,
        'training_time': training_time,
        'total_time': total_time,
        'n_features': X_train.shape[1],
        'n_training_samples': len(X_train),
        'random_state': RANDOM_STATE,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    metadata_path = ARTIFACTS_DIR / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    
    return model, metrics


if __name__ == "__main__":
    print("=" * 80)
    print("CYCLESENSE FAST MODEL TRAINING")
    print("=" * 80)
    
    model, metrics = train_fast_model()
    
    print("\n" + "=" * 80)
    print("FAST TRAINING COMPLETE")
    print(f"MAE: {metrics['mae']:.3f} days")
    print("=" * 80)
