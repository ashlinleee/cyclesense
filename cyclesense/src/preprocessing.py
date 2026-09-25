"""
Preprocessing pipeline for CycleSense.
Creates sklearn-compatible preprocessing with categorical encoding and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

from src.features import get_feature_categories
from src.leakage_audit import FEATURE_CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_strict_feature_set() -> list:
    """Get strict feature set (safe at prediction time)."""
    return FEATURE_CATEGORIES["SAFE_AT_PREDICTION_TIME"].copy()


def get_extended_feature_set() -> list:
    """Get extended feature set (includes conditional features)."""
    return (FEATURE_CATEGORIES["SAFE_AT_PREDICTION_TIME"].copy() + 
            FEATURE_CATEGORIES["CONDITIONAL"].copy())


def select_features_by_set(df: pd.DataFrame, feature_set: str = 'strict') -> pd.DataFrame:
    """
    Select features based on feature set.
    
    Args:
        df: DataFrame with all features
        feature_set: 'strict', 'extended', or 'full'
        
    Returns:
        DataFrame with selected features
    """
    if feature_set == 'strict':
        features = get_strict_feature_set()
    elif feature_set == 'extended':
        features = get_extended_feature_set()
    else:
        # Use all safe + conditional features
        features = get_extended_feature_set()
    
    # Filter to only features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    
    logger.info(f"Feature set: {feature_set}")
    logger.info(f"Requested features: {len(features)}")
    logger.info(f"Available features: {len(available_features)}")
    
    return df[available_features].copy()


def create_preprocessing_pipeline(
    numerical_features: list,
    categorical_features: list,
    use_scaling: bool = True
) -> ColumnTransformer:
    """
    Create sklearn preprocessing pipeline.
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        use_scaling: Whether to apply scaling to numerical features
        
    Returns:
        ColumnTransformer with preprocessing steps
    """
    logger.info("=== CREATING PREPROCESSING PIPELINE ===")
    
    # Numerical preprocessing
    numerical_steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ]
    
    if use_scaling:
        numerical_steps.append(('scaler', StandardScaler()))
    
    numerical_transformer = Pipeline(steps=numerical_steps)
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns not specified
    )
    
    logger.info(f"Numerical features: {len(numerical_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    logger.info(f"Scaling: {use_scaling}")
    
    return preprocessor


def prepare_model_data(
    df: pd.DataFrame,
    feature_set: str = 'strict',
    target_col: str = 'next_cycle_length',
    use_scaling: bool = True
) -> tuple:
    """
    Prepare data for modeling with preprocessing pipeline.
    
    Args:
        df: DataFrame with engineered features and target
        feature_set: 'strict', 'extended', or 'full'
        target_col: Name of target column
        use_scaling: Whether to apply scaling
        
    Returns:
        Tuple of (preprocessor, X, y, feature_names)
    """
    logger.info(f"=== PREPARING MODEL DATA ({feature_set.upper()} FEATURE SET) ===")
    
    # Select features based on feature set
    feature_df = select_features_by_set(df, feature_set)
    
    # Add engineered features if they exist
    engineered_feature_cols = [
        'cycle_length_lag_1', 'cycle_length_lag_2', 'cycle_length_lag_3',
        'cycle_length_mean_last_3', 'cycle_length_std_last_3',
        'cycle_length_mean_last_5', 'cycle_length_std_last_5',
        'cycle_length_change', 'cycle_length_abs_change',
        'stress_delta', 'sleep_delta', 'stress_sleep_interaction',
        'month', 'quarter', 'day_of_year', 'month_sin', 'month_cos'
    ]
    
    for col in engineered_feature_cols:
        if col in df.columns and col not in feature_df.columns:
            feature_df[col] = df[col]
    
    # Get target
    y = df[target_col].copy()
    
    # Identify feature types
    feature_categories = get_feature_categories(feature_df)
    numerical_features = feature_categories['numerical']
    categorical_features = feature_categories['categorical']
    
    logger.info(f"Total features: {len(feature_df.columns)}")
    logger.info(f"Numerical features: {len(numerical_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        numerical_features,
        categorical_features,
        use_scaling
    )
    
    # Fit preprocessor and transform data
    X_processed = preprocessor.fit_transform(feature_df)
    
    # Get feature names after preprocessing
    feature_names = get_feature_names_after_preprocessing(
        preprocessor, numerical_features, categorical_features
    )
    
    logger.info(f"Processed features: {X_processed.shape[1]}")
    
    return preprocessor, X_processed, y, feature_names


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
    numerical_features: list,
    categorical_features: list
) -> list:
    """
    Get feature names after preprocessing (including one-hot encoded features).
    
    Args:
        preprocessor: Fitted ColumnTransformer
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        
    Returns:
        List of feature names after preprocessing
    """
    feature_names = []
    
    # Numerical feature names (unchanged)
    feature_names.extend(numerical_features)
    
    # Categorical feature names (one-hot encoded)
    cat_transformer = preprocessor.named_transformers_['cat']
    if hasattr(cat_transformer, 'named_steps'):
        onehot = cat_transformer.named_steps['onehot']
    else:
        onehot = cat_transformer
    
    if hasattr(onehot, 'get_feature_names_out'):
        cat_feature_names = onehot.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    else:
        # Fallback for older sklearn versions
        for cat_feature in categorical_features:
            unique_values = onehot.categories_[categorical_features.index(cat_feature)]
            feature_names.extend([f"{cat_feature}_{val}" for val in unique_values])
    
    return feature_names


if __name__ == "__main__":
    # Test preprocessing pipeline
    from src.data import load_and_join_datasets
    from src.target import construct_next_cycle_target
    from src.features import engineer_features
    
    # Load and prepare data
    df = load_and_join_datasets()
    df_with_target = construct_next_cycle_target(df)
    df_engineered = engineer_features(df_with_target)
    
    # Test different feature sets
    for feature_set in ['strict', 'extended']:
        print(f"\n=== Testing {feature_set.upper()} feature set ===")
        preprocessor, X, y, feature_names = prepare_model_data(
            df_engineered, 
            feature_set=feature_set,
            use_scaling=True
        )
        
        print(f"Processed shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature names: {len(feature_names)}")
