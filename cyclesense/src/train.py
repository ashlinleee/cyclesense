"""
Training script for CycleSense models.
Implements baseline and advanced regression models with MLflow tracking.
"""

import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import logging
import time
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
import json

from src.config import (
    RANDOM_STATE, 
    MLFLOW_TRACKING_URI, 
    MLFLOW_EXPERIMENT_NAME,
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

# Set MLflow tracking safely
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
except Exception as e:
    logger.warning(f"MLflow initialization skipped: {e}")


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


def train_baseline_mean(X_train, y_train, X_test, y_test, run_name=None):
    """Baseline A: Predict global training-set mean cycle length."""
    logger.info("=== BASELINE A: GLOBAL MEAN PREDICTOR ===")
    
    with mlflow.start_run(run_name=run_name or "EXP-001_Baseline_Mean"):
        mlflow.log_params({
            "model_type": "DummyRegressor",
            "strategy": "mean",
            "feature_set": "strict"
        })
        
        model = DummyRegressor(strategy='mean')
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        
        mlflow.log_metrics({
            "mae": metrics['mae'],
            "rmse": metrics['rmse'],
            "r2": metrics['r2'],
            "training_time": training_time
        })
        
        logger.info(f"MAE: {metrics['mae']:.3f} days")
        logger.info(f"RMSE: {metrics['rmse']:.3f} days")
        logger.info(f"R²: {metrics['r2']:.3f}")
        logger.info(f"Training time: {training_time:.3f}s")
        
        mlflow.sklearn.log_model(model, "model")
    
    return model, metrics, training_time


def train_baseline_previous_cycle(X_train, y_train, X_test, y_test, df_train, df_test, run_name=None):
    """Baseline B: Predict next cycle ≈ previous cycle length."""
    logger.info("=== BASELINE B: PREVIOUS CYCLE PREDICTOR ===")
    
    with mlflow.start_run(run_name=run_name or "EXP-002_Baseline_Previous_Cycle"):
        mlflow.log_params({
            "model_type": "PreviousCycleBaseline",
            "strategy": "previous_cycle",
            "feature_set": "strict"
        })
        
        # Use cycle_length_days as prediction for next_cycle_length
        # This simulates "next cycle will be like current cycle"
        y_pred_test = df_test['cycle_length_days'].values
        
        # Calculate metrics on test set
        metrics = calculate_metrics(y_test, y_pred_test)
        training_time = 0.0
        
        mlflow.log_metrics({
            "mae": metrics['mae'],
            "rmse": metrics['rmse'],
            "r2": metrics['r2'],
            "training_time": training_time
        })
        
        logger.info(f"MAE: {metrics['mae']:.3f} days")
        logger.info(f"RMSE: {metrics['rmse']:.3f} days")
        logger.info(f"R²: {metrics['r2']:.3f}")
        logger.info(f"Training time: 0.000s (no training)")
        
        # Return a simple model for consistency
        class PreviousCycleBaseline:
            def __init__(self, default_value=28.0):
                self.default_value = default_value
            
            def fit(self, X, y=None):
                # Find the index of cycle_length_days if it exists
                self.cycle_length_idx = 0  # Default to first column
                return self
            
            def predict(self, X):
                # Return the first column (assuming it's cycle_length_days)
                if len(X.shape) > 1:
                    return X[:, 0]
                return X
        
        model = PreviousCycleBaseline()
        model.fit(X_train)  # Fit for consistency
        
        mlflow.sklearn.log_model(model, "model")
    
    return model, metrics, training_time


def train_linear_regression(X_train, y_train, X_test, y_test, run_name=None):
    """Train Linear Regression model."""
    logger.info("=== LINEAR REGRESSION ===")
    
    with mlflow.start_run(run_name=run_name or "EXP-003_Linear_Regression"):
        mlflow.log_params({
            "model_type": "LinearRegression",
            "feature_set": "strict",
            "scaling": True
        })
        
        model = LinearRegression()
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        
        mlflow.log_metrics({
            "mae": metrics['mae'],
            "rmse": metrics['rmse'],
            "r2": metrics['r2'],
            "training_time": training_time
        })
        
        logger.info(f"MAE: {metrics['mae']:.3f} days")
        logger.info(f"RMSE: {metrics['rmse']:.3f} days")
        logger.info(f"R²: {metrics['r2']:.3f}")
        logger.info(f"Training time: {training_time:.3f}s")
        
        mlflow.sklearn.log_model(model, "model")
    
    return model, metrics, training_time


def train_ridge(X_train, y_train, X_test, y_test, alpha=1.0, run_name=None):
    """Train Ridge regression model."""
    logger.info(f"=== RIDGE REGRESSION (alpha={alpha}) ===")
    
    with mlflow.start_run(run_name=run_name or "EXP-004_Ridge_Regression"):
        mlflow.log_params({
            "model_type": "Ridge",
            "alpha": alpha,
            "feature_set": "strict",
            "scaling": True,
            "random_state": RANDOM_STATE
        })
        
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        
        mlflow.log_metrics({
            "mae": metrics['mae'],
            "rmse": metrics['rmse'],
            "r2": metrics['r2'],
            "training_time": training_time
        })
        
        logger.info(f"MAE: {metrics['mae']:.3f} days")
        logger.info(f"RMSE: {metrics['rmse']:.3f} days")
        logger.info(f"R²: {metrics['r2']:.3f}")
        logger.info(f"Training time: {training_time:.3f}s")
        
        mlflow.sklearn.log_model(model, "model")
    
    return model, metrics, training_time


def train_lasso(X_train, y_train, X_test, y_test, alpha=1.0, run_name=None):
    """Train Lasso regression model."""
    logger.info(f"=== LASSO REGRESSION (alpha={alpha}) ===")
    
    with mlflow.start_run(run_name=run_name or "EXP-005_Lasso_Regression"):
        mlflow.log_params({
            "model_type": "Lasso",
            "alpha": alpha,
            "feature_set": "strict",
            "scaling": True,
            "random_state": RANDOM_STATE
        })
        
        model = Lasso(alpha=alpha, random_state=RANDOM_STATE)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        features_used = np.sum(model.coef_ != 0)
        
        mlflow.log_metrics({
            "mae": metrics['mae'],
            "rmse": metrics['rmse'],
            "r2": metrics['r2'],
            "training_time": training_time,
            "features_used": features_used
        })
        
        logger.info(f"MAE: {metrics['mae']:.3f} days")
        logger.info(f"RMSE: {metrics['rmse']:.3f} days")
        logger.info(f"R²: {metrics['r2']:.3f}")
        logger.info(f"Training time: {training_time:.3f}s")
        logger.info(f"Features used: {features_used}")
        
        mlflow.sklearn.log_model(model, "model")
    
    return model, metrics, training_time


def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, run_name=None):
    """Train Random Forest regressor."""
    logger.info(f"=== RANDOM FOREST (n_estimators={n_estimators}) ===")
    
    with mlflow.start_run(run_name=run_name or "EXP-006_Random_Forest"):
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": n_estimators,
            "feature_set": "strict",
            "scaling": False,  # RF doesn't need scaling
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        })
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        
        mlflow.log_metrics({
            "mae": metrics['mae'],
            "rmse": metrics['rmse'],
            "r2": metrics['r2'],
            "training_time": training_time
        })
        
        logger.info(f"MAE: {metrics['mae']:.3f} days")
        logger.info(f"RMSE: {metrics['rmse']:.3f} days")
        logger.info(f"R²: {metrics['r2']:.3f}")
        logger.info(f"Training time: {training_time:.3f}s")
        
        mlflow.sklearn.log_model(model, "model")
    
    return model, metrics, training_time


def train_gradient_boosting(X_train, y_train, X_test, y_test, run_name=None):
    """Train Histogram-based Gradient Boosting regressor."""
    logger.info("=== GRADIENT BOOSTING ===")
    
    with mlflow.start_run(run_name=run_name or "EXP-007_Gradient_Boosting"):
        mlflow.log_params({
            "model_type": "HistGradientBoostingRegressor",
            "max_iter": 100,
            "feature_set": "strict",
            "scaling": False,  # GB doesn't need scaling
            "random_state": RANDOM_STATE
        })
        
        model = HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            max_iter=100
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        
        mlflow.log_metrics({
            "mae": metrics['mae'],
            "rmse": metrics['rmse'],
            "r2": metrics['r2'],
            "training_time": training_time
        })
        
        logger.info(f"MAE: {metrics['mae']:.3f} days")
        logger.info(f"RMSE: {metrics['rmse']:.3f} days")
        logger.info(f"R²: {metrics['r2']:.3f}")
        logger.info(f"Training time: {training_time:.3f}s")
        
        mlflow.sklearn.log_model(model, "model")
    
    return model, metrics, training_time


def run_experiments(feature_set='strict'):
    """
    Run baseline and advanced model experiments with MLflow tracking.
    
    Args:
        feature_set: 'strict' or 'extended' feature set
    """
    logger.info(f"=== RUNNING EXPERIMENTS ({feature_set.upper()} FEATURE SET) ===")
    
    # Load and prepare data
    df = load_and_join_datasets()
    df_with_target = construct_next_cycle_target(df)
    df_engineered = engineer_features(df_with_target)
    
    # Split data
    train_df, test_df = create_train_test_split(df_engineered)
    
    # Prepare model data
    preprocessor, X_train, y_train, feature_names = prepare_model_data(
        train_df, feature_set=feature_set, use_scaling=True
    )
    
    # Prepare test data using the same preprocessor
    # First select features for test set
    from src.preprocessing import select_features_by_set
    test_feature_df = select_features_by_set(test_df, feature_set)
    
    # Add engineered features
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
    
    # Run experiments
    results = {}
    
    # Baseline A: Global mean
    model_a, metrics_a, time_a = train_baseline_mean(X_train, y_train, X_test, y_test)
    results['baseline_mean'] = {
        'model': model_a,
        'metrics': metrics_a,
        'training_time': time_a,
        'feature_set': feature_set
    }
    
    # Baseline B: Previous cycle
    model_b, metrics_b, time_b = train_baseline_previous_cycle(
        X_train, y_train, X_test, y_test, train_df, test_df
    )
    results['baseline_previous'] = {
        'model': model_b,
        'metrics': metrics_b,
        'training_time': time_b,
        'feature_set': feature_set
    }
    
    # Linear models
    model_lr, metrics_lr, time_lr = train_linear_regression(X_train, y_train, X_test, y_test)
    results['linear_regression'] = {
        'model': model_lr,
        'metrics': metrics_lr,
        'training_time': time_lr,
        'feature_set': feature_set
    }
    
    model_ridge, metrics_ridge, time_ridge = train_ridge(X_train, y_train, X_test, y_test)
    results['ridge'] = {
        'model': model_ridge,
        'metrics': metrics_ridge,
        'training_time': time_ridge,
        'feature_set': feature_set
    }
    
    model_lasso, metrics_lasso, time_lasso = train_lasso(X_train, y_train, X_test, y_test)
    results['lasso'] = {
        'model': model_lasso,
        'metrics': metrics_lasso,
        'training_time': time_lasso,
        'feature_set': feature_set
    }
    
    # Tree-based models
    model_rf, metrics_rf, time_rf = train_random_forest(X_train, y_train, X_test, y_test)
    results['random_forest'] = {
        'model': model_rf,
        'metrics': metrics_rf,
        'training_time': time_rf,
        'feature_set': feature_set
    }
    
    model_gb, metrics_gb, time_gb = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results['gradient_boosting'] = {
        'model': model_gb,
        'metrics': metrics_gb,
        'training_time': time_gb,
        'feature_set': feature_set
    }
    
    # Print summary
    logger.info("\n=== EXPERIMENT SUMMARY ===")
    logger.info(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'Time':<10}")
    logger.info("-" * 65)
    
    for name, result in results.items():
        m = result['metrics']
        t = result['training_time']
        logger.info(f"{name:<25} {m['mae']:<10.3f} {m['rmse']:<10.3f} {m['r2']:<10.3f} {t:<10.3f}")
    
    # Calculate improvement over baselines
    baseline_mean_mae = results['baseline_mean']['metrics']['mae']
    baseline_prev_mae = results['baseline_previous']['metrics']['mae']
    
    logger.info(f"\nImprovement over baseline mean:")
    for name, result in results.items():
        if 'baseline' not in name:
            improvement = (baseline_mean_mae - result['metrics']['mae']) / baseline_mean_mae * 100
            logger.info(f"  {name}: {improvement:+.1f}%")
    
    logger.info(f"\nImprovement over baseline previous cycle:")
    for name, result in results.items():
        if 'baseline' not in name:
            improvement = (baseline_prev_mae - result['metrics']['mae']) / baseline_prev_mae * 100
            logger.info(f"  {name}: {improvement:+.1f}%")
    
    # Find best model based on MAE
    best_model_name = min(
        [name for name in results.keys() if 'baseline' not in name],
        key=lambda x: results[x]['metrics']['mae']
    )
    
    logger.info(f"\n=== BEST MODEL: {best_model_name.upper()} ===")
    logger.info(f"MAE: {results[best_model_name]['metrics']['mae']:.3f} days")
    logger.info(f"RMSE: {results[best_model_name]['metrics']['rmse']:.3f} days")
    logger.info(f"R²: {results[best_model_name]['metrics']['r2']:.3f}")
    
    # Save best model with preprocessing pipeline
    best_model = results[best_model_name]['model']
    
    # Save model and preprocessor separately
    model_path = ARTIFACTS_DIR / 'model.joblib'
    joblib.dump({
        'model': best_model,
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }, model_path)
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'model_type': type(best_model).__name__,
        'feature_set': feature_set,
        'metrics': results[best_model_name]['metrics'],
        'training_time': results[best_model_name]['training_time'],
        'n_features': X_train.shape[1],
        'n_training_samples': len(X_train),
        'random_state': RANDOM_STATE
    }
    
    metadata_path = ARTIFACTS_DIR / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Best model saved to: {model_path}")
    logger.info(f"Model metadata saved to: {metadata_path}")
    
    return results, best_model_name


if __name__ == "__main__":
    # Run experiments with strict feature set
    print("=" * 80)
    print("CYCLESENSE MODEL EXPERIMENTS WITH MLFLOW TRACKING")
    print("=" * 80)
    
    results_strict, best_model_name = run_experiments(feature_set='strict')
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print(f"Best model: {best_model_name}")
    print("To view MLflow UI, run: mlflow ui")
    print("=" * 80)
