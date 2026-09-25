"""
Feature selection and explainability for CycleSense models.
Implements feature importance analysis, permutation importance, and SHAP values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import logging
from pathlib import Path

from src.config import ARTIFACTS_DIR, FIGURES_DIR, RANDOM_STATE
from src.data import load_and_join_datasets
from src.target import construct_next_cycle_target, create_train_test_split
from src.features import engineer_features
from src.preprocessing import prepare_model_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure figures directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_trained_model():
    """Load the trained model and preprocessor."""
    model_path = ARTIFACTS_DIR / 'model.joblib'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    
    model_data = joblib.load(model_path)
    return model_data['model'], model_data['preprocessor'], model_data['feature_names']


def calculate_feature_importance(model, feature_names):
    """Calculate feature importance from tree-based models."""
    logger.info("=== CALCULATING FEATURE IMPORTANCE ===")
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 features by importance:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    else:
        logger.warning("Model does not support feature_importances_")
        return None


def calculate_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10):
    """Calculate permutation importance."""
    logger.info("=== CALCULATING PERMUTATION IMPORTANCE ===")
    
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    logger.info(f"Top 10 features by permutation importance:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance_mean']:.4f} (+/- {row['importance_std']:.4f})")
    
    return importance_df


def plot_feature_importance(importance_df, title="Feature Importance", save_path=None, importance_col='importance'):
    """Plot feature importance."""
    plt.figure(figsize=(12, 8))
    
    top_n = min(20, len(importance_df))
    plot_data = importance_df.head(top_n).copy()
    
    # Handle different column names for importance
    if importance_col not in plot_data.columns:
        if 'importance_mean' in plot_data.columns:
            importance_col = 'importance_mean'
        elif 'shap_importance' in plot_data.columns:
            importance_col = 'shap_importance'
    
    sns.barplot(data=plot_data, x=importance_col, y='feature', palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved feature importance plot to: {save_path}")
    
    plt.close()


def select_features_using_statistical_methods(X_train, y_train, X_test, feature_names, k=20):
    """Select features using statistical methods."""
    logger.info("=== STATISTICAL FEATURE SELECTION ===")
    
    # Method 1: SelectKBest with f_regression
    selector_f = SelectKBest(f_regression, k=k)
    X_train_selected_f = selector_f.fit_transform(X_train, y_train)
    X_test_selected_f = selector_f.transform(X_test)
    
    selected_features_f = [feature_names[i] for i in selector_f.get_support(indices=True)]
    
    logger.info(f"SelectKBest (f_regression) selected {len(selected_features_f)} features:")
    for feat in selected_features_f[:10]:
        logger.info(f"  - {feat}")
    
    # Method 2: Mutual Information
    selector_mi = SelectKBest(mutual_info_regression, k=k)
    X_train_selected_mi = selector_mi.fit_transform(X_train, y_train)
    X_test_selected_mi = selector_mi.transform(X_test)
    
    selected_features_mi = [feature_names[i] for i in selector_mi.get_support(indices=True)]
    
    logger.info(f"SelectKBest (mutual_info) selected {len(selected_features_mi)} features:")
    for feat in selected_features_mi[:10]:
        logger.info(f"  - {feat}")
    
    return {
        'f_regression': {
            'selector': selector_f,
            'selected_features': selected_features_f,
            'X_train': X_train_selected_f,
            'X_test': X_test_selected_f
        },
        'mutual_info': {
            'selector': selector_mi,
            'selected_features': selected_features_mi,
            'X_train': X_train_selected_mi,
            'X_test': X_test_selected_mi
        }
    }


def calculate_shap_values(model, X_sample, feature_names, max_samples=100):
    """Calculate SHAP values for model explainability."""
    logger.info("=== CALCULATING SHAP VALUES ===")
    
    try:
        import shap
        
        # Sample data for SHAP calculation (computationally expensive)
        if len(X_sample) > max_samples:
            indices = np.random.choice(len(X_sample), max_samples, replace=False)
            X_sample = X_sample[indices]
        
        # Use TreeExplainer for tree-based models
        if hasattr(model, 'estimators_') or hasattr(model, '_predictors'):
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other models
            explainer = shap.KernelExplainer(model.predict, X_sample[:50])
        
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        shap_importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_shap
        }).sort_values('shap_importance', ascending=False)
        
        logger.info(f"Top 10 features by SHAP importance:")
        for idx, row in shap_importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['shap_importance']:.4f}")
        
        return shap_values, shap_importance_df
        
    except ImportError:
        logger.warning("SHAP not installed. Skipping SHAP analysis.")
        return None, None
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {e}")
        return None, None


def generate_explainability_report():
    """Generate comprehensive explainability report."""
    logger.info("=== GENERATING EXPLAINABILITY REPORT ===")
    
    # Load data
    df = load_and_join_datasets()
    df_with_target = construct_next_cycle_target(df)
    df_engineered = engineer_features(df_with_target)
    
    # Split data
    train_df, test_df = create_train_test_split(df_engineered)
    
    # Prepare model data
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
    
    # Load trained model
    model, _, _ = load_trained_model()
    
    # Calculate feature importance
    importance_df = calculate_feature_importance(model, feature_names)
    
    if importance_df is not None and len(importance_df) > 0:
        # Plot feature importance
        plot_path = FIGURES_DIR / 'feature_importance.png'
        plot_feature_importance(importance_df, "Feature Importance", plot_path)
    
    # Calculate permutation importance
    perm_importance_df = calculate_permutation_importance(
        model, X_test, y_test, feature_names, n_repeats=5
    )
    
    # Plot permutation importance
    if perm_importance_df is not None:
        plot_path = FIGURES_DIR / 'permutation_importance.png'
        plot_feature_importance(
            perm_importance_df, 
            "Permutation Importance",
            plot_path,
            importance_col='importance_mean'
        )
    
    # Statistical feature selection
    selection_results = select_features_using_statistical_methods(
        X_train, y_train, X_test, feature_names, k=20
    )
    
    # SHAP analysis (optional, computationally expensive)
    shap_values, shap_importance_df = calculate_shap_values(
        model, X_test, feature_names, max_samples=50
    )
    
    if shap_importance_df is not None:
        # Plot SHAP importance
        plot_path = FIGURES_DIR / 'shap_importance.png'
        plot_feature_importance(
            shap_importance_df,
            "SHAP Feature Importance",
            plot_path,
            importance_col='shap_importance'
        )
    
    # Save feature importance results
    results = {
        'feature_importance': importance_df.to_dict('records') if importance_df is not None else None,
        'permutation_importance': perm_importance_df.to_dict('records') if perm_importance_df is not None else None,
        'selected_features_f_regression': selection_results['f_regression']['selected_features'],
        'selected_features_mutual_info': selection_results['mutual_info']['selected_features'],
        'shap_importance': shap_importance_df.to_dict('records') if shap_importance_df is not None else None
    }
    
    results_path = ARTIFACTS_DIR / 'feature_importance.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Feature importance results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("CYCLESENSE FEATURE SELECTION AND EXPLAINABILITY")
    print("=" * 80)
    
    results = generate_explainability_report()
    
    print("\n" + "=" * 80)
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 80)
