"""
Feature engineering for CycleSense.
Implements sklearn-compatible transformations for historical cycle features.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

from src.config import HISTORICAL_LAGS, ROLLING_WINDOWS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalCycleFeatures(BaseEstimator, TransformerMixin):
    """
    Create leakage-safe historical cycle features.
    
    Features created:
    - Lag features: cycle_length_lag_1, cycle_length_lag_2, cycle_length_lag_3
    - Rolling statistics: mean, std, min, max over last 3 and 5 cycles
    - Cycle change: cycle_length_change, cycle_length_abs_change
    """
    
    def __init__(self, lags=None, windows=None):
        self.lags = lags or HISTORICAL_LAGS
        self.windows = windows or ROLLING_WINDOWS
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """Fit transformer - stores feature names."""
        # This transformer doesn't need to learn anything
        # But we need to handle the case where X might not have user_id
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = self._get_feature_names(X)
        return self
    
    def _get_feature_names(self, X):
        """Generate feature names for the engineered features."""
        names = []
        
        # Lag features
        for lag in self.lags:
            names.append(f'cycle_length_lag_{lag}')
        
        # Rolling statistics
        for window in self.windows:
            names.extend([
                f'cycle_length_mean_last_{window}',
                f'cycle_length_std_last_{window}',
                f'cycle_length_min_last_{window}',
                f'cycle_length_max_last_{window}'
            ])
        
        # Change features
        names.extend([
            'cycle_length_change',
            'cycle_length_abs_change'
        ])
        
        return names
    
    def transform(self, X):
        """
        Transform data to create historical features.
        
        Args:
            X: DataFrame with columns ['user_id', 'cycle_number', 'cycle_length_days']
               Must be sorted by user_id and cycle_number before transformation
        
        Returns:
            DataFrame with engineered historical features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Make a copy to avoid modifying original
        df = X.copy()
        
        # Ensure proper sorting
        df = df.sort_values(['user_id', 'cycle_number'])
        
        # Create lag features
        for lag in self.lags:
            df[f'cycle_length_lag_{lag}'] = df.groupby('user_id')['cycle_length_days'].shift(lag)
        
        # Create rolling statistics (lagged by 1 to prevent current target leakage)
        for window in self.windows:
            rolling = df.groupby('user_id')['cycle_length_days'].shift(1).groupby(df['user_id']).rolling(
                window=window, min_periods=1
            )
            
            df[f'cycle_length_mean_last_{window}'] = rolling.mean().reset_index(level=0, drop=True)
            df[f'cycle_length_std_last_{window}'] = rolling.std().reset_index(level=0, drop=True)
            df[f'cycle_length_min_last_{window}'] = rolling.min().reset_index(level=0, drop=True)
            df[f'cycle_length_max_last_{window}'] = rolling.max().reset_index(level=0, drop=True)
        
        # Create change features
        df['cycle_length_change'] = df.groupby('user_id')['cycle_length_days'].diff()
        df['cycle_length_abs_change'] = df['cycle_length_change'].abs()
        
        # Return only the engineered features
        feature_cols = self._get_feature_names(X)
        return df[feature_cols]


class CrossSourceFeatures(BaseEstimator, TransformerMixin):
    """
    Create cross-source interaction features.
    
    Features created:
    - stress_delta = stress_score_cycle - stress_score_baseline
    - sleep_delta = sleep_hours_cycle - sleep_hours
    - stress_sleep_interaction
    """
    
    def __init__(self):
        self.feature_names_ = [
            'stress_delta',
            'sleep_delta',
            'stress_sleep_interaction'
        ]
    
    def fit(self, X, y=None):
        """Fit transformer - stores feature names."""
        return self
    
    def transform(self, X):
        """
        Transform data to create cross-source features.
        
        Args:
            X: DataFrame with relevant columns
        
        Returns:
            DataFrame with engineered cross-source features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        df = X.copy()
        
        # Create delta features if columns exist
        if 'stress_score_cycle' in df.columns and 'stress_score_baseline' in df.columns:
            df['stress_delta'] = df['stress_score_cycle'] - df['stress_score_baseline']
        else:
            df['stress_delta'] = 0
        
        if 'sleep_hours_cycle' in df.columns and 'sleep_hours' in df.columns:
            df['sleep_delta'] = df['sleep_hours_cycle'] - df['sleep_hours']
        else:
            df['sleep_delta'] = 0
        
        # Create interaction feature
        if 'stress_delta' in df.columns and 'sleep_delta' in df.columns:
            df['stress_sleep_interaction'] = df['stress_delta'] * df['sleep_delta']
        else:
            df['stress_sleep_interaction'] = 0
        
        # Return only the engineered features
        return df[self.feature_names_]


class DateFeatures(BaseEstimator, TransformerMixin):
    """
    Create date-based features from start_date.
    
    Features created:
    - month, quarter, day_of_year
    - month_sin, month_cos (cyclical representation)
    """
    
    def __init__(self, date_col='start_date'):
        self.date_col = date_col
        self.feature_names_ = [
            'month', 'quarter', 'day_of_year',
            'month_sin', 'month_cos'
        ]
    
    def fit(self, X, y=None):
        """Fit transformer - stores feature names."""
        return self
    
    def transform(self, X):
        """
        Transform data to create date features.
        
        Args:
            X: DataFrame with date column
        
        Returns:
            DataFrame with engineered date features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        df = X.copy()
        
        # Convert to datetime if not already
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            
            # Extract date features
            df['month'] = df[self.date_col].dt.month
            df['quarter'] = df[self.date_col].dt.quarter
            df['day_of_year'] = df[self.date_col].dt.dayofyear
            
            # Cyclical representation
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        else:
            # If date column doesn't exist, create zeros
            for feat in self.feature_names_:
                df[feat] = 0
        
        # Return only the engineered features
        return df[self.feature_names_]


def create_preprocessing_pipeline(
    numerical_features: list,
    categorical_features: list,
    use_scaling: bool = True
) -> Pipeline:
    """
    Create sklearn preprocessing pipeline.
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        use_scaling: Whether to apply scaling to numerical features
        
    Returns:
        sklearn Pipeline with preprocessing steps
    """
    logger.info("=== CREATING PREPROCESSING PIPELINE ===")
    
    # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler() if use_scaling else 'passthrough')
    ])
    
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


def get_feature_categories(df: pd.DataFrame) -> dict:
    """
    Automatically identify numerical and categorical features.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with 'numerical' and 'categorical' feature lists
    """
    # Exclude identifier columns
    exclude_cols = ['user_id', 'cycle_number', 'start_date', 'next_cycle_length']
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Remove excluded columns
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    return {
        'numerical': list(numerical_cols),
        'categorical': list(categorical_cols)
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: DataFrame with target variable constructed
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("=== COMPLETE FEATURE ENGINEERING ===")
    
    df_engineered = df.copy()
    
    # Ensure proper sorting
    df_engineered = df_engineered.sort_values(['user_id', 'cycle_number'])
    
    # Create historical cycle features
    logger.info("Creating historical cycle features...")
    historical_transformer = HistoricalCycleFeatures()
    historical_features = historical_transformer.fit_transform(
        df_engineered[['user_id', 'cycle_number', 'cycle_length_days']]
    )
    
    # Add historical features to main dataframe
    for col in historical_features.columns:
        df_engineered[col] = historical_features[col]
    
    # Create cross-source features
    logger.info("Creating cross-source features...")
    cross_source_transformer = CrossSourceFeatures()
    cross_source_features = cross_source_transformer.fit_transform(df_engineered)
    
    # Add cross-source features to main dataframe
    for col in cross_source_features.columns:
        df_engineered[col] = cross_source_features[col]
    
    # Create date features
    logger.info("Creating date features...")
    date_transformer = DateFeatures()
    date_features = date_transformer.fit_transform(df_engineered)
    
    # Add date features to main dataframe
    for col in date_features.columns:
        df_engineered[col] = date_features[col]
    
    logger.info(f"Original features: {len(df.columns)}")
    logger.info(f"Engineered features: {len(df_engineered.columns)}")
    logger.info(f"New features added: {len(df_engineered.columns) - len(df.columns)}")
    
    return df_engineered


if __name__ == "__main__":
    # Test feature engineering
    from src.data import load_and_join_datasets
    from src.target import construct_next_cycle_target
    
    # Load and prepare data
    df = load_and_join_datasets()
    df_with_target = construct_next_cycle_target(df)
    
    # Engineer features
    df_engineered = engineer_features(df_with_target)
    
    print(f"\n=== FEATURE ENGINEERING COMPLETE ===")
    print(f"Original shape: {df_with_target.shape}")
    print(f"Engineered shape: {df_engineered.shape}")
    print(f"New features: {len(df_engineered.columns) - len(df_with_target.columns)}")
    print(f"\nNew feature columns:")
    new_cols = set(df_engineered.columns) - set(df_with_target.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")
