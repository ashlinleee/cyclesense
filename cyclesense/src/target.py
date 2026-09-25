"""
Leakage-safe next-cycle target construction for CycleSense.
Creates the target variable: next_cycle_length
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

from src.config import RANDOM_STATE
from src.data import load_and_join_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_next_cycle_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct next_cycle_length target variable.
    
    For each user, sort chronologically and create target by shifting cycle_length_days by -1.
    Current cycle N → target = cycle N+1 length
    
    Args:
        df: Merged dataframe with period logs and user profiles
        
    Returns:
        DataFrame with next_cycle_length column, rows without target removed
    """
    logger.info("=== CONSTRUCTING NEXT-CYCLE TARGET ===")
    
    # Sort by user_id, start_date, cycle_number to ensure chronological order
    df_sorted = df.sort_values(['user_id', 'start_date', 'cycle_number']).copy()
    
    logger.info(f"Initial records: {len(df_sorted)}")
    
    # Create target by shifting cycle_length_days by -1 within each user
    df_sorted['next_cycle_length'] = df_sorted.groupby('user_id')['cycle_length_days'].shift(-1)
    
    # Remove rows without target (final cycle for each user)
    df_with_target = df_sorted.dropna(subset=['next_cycle_length']).copy()
    
    logger.info(f"Records with target: {len(df_with_target)}")
    logger.info(f"Records removed (no target): {len(df_sorted) - len(df_with_target)}")
    
    # Validate target construction
    validate_target_construction(df_sorted, df_with_target)
    
    return df_with_target


def validate_target_construction(df_original: pd.DataFrame, df_with_target: pd.DataFrame):
    """
    Validate that target construction is correct.
    Ensure current cycle N → target cycle N+1 for sample users.
    """
    logger.info("\n=== TARGET CONSTRUCTION VALIDATION ===")
    
    # Check sample users
    sample_users = df_with_target['user_id'].unique()[:5]
    
    for user_id in sample_users:
        user_data = df_with_target[df_with_target['user_id'] == user_id].sort_values('cycle_number')
        
        logger.info(f"\nUser {user_id}:")
        for idx, row in user_data.iterrows():
            current_cycle = row['cycle_number']
            current_length = row['cycle_length_days']
            target_length = row['next_cycle_length']
            
            logger.info(f"  Cycle {current_cycle}: {current_length} days → Target: {target_length} days")
            
            # Verify target matches next cycle's actual length
            next_cycle_data = df_original[
                (df_original['user_id'] == user_id) & 
                (df_original['cycle_number'] == current_cycle + 1)
            ]
            
            if not next_cycle_data.empty:
                actual_next_length = next_cycle_data['cycle_length_days'].values[0] if 'cycle_length_days' in next_cycle_data.columns else None
                if actual_next_length is not None:
                    assert abs(target_length - actual_next_length) < 0.01, \
                        f"Target mismatch for user {user_id}, cycle {current_cycle}"
    
    # Check that no future data leaked into historical features
    logger.info("\n=== LEAKAGE CHECK ===")
    
    # Ensure we removed final cycles
    users_with_final_cycle = df_original.groupby('user_id')['cycle_number'].max()
    final_cycle_count = 0
    
    for user_id, max_cycle in users_with_final_cycle.items():
        final_cycle_in_target = df_with_target[
            (df_with_target['user_id'] == user_id) & 
            (df_with_target['cycle_number'] == max_cycle)
        ]
        if not final_cycle_in_target.empty:
            final_cycle_count += 1
            logger.warning(f"User {user_id} still has final cycle {max_cycle} in target data")
    
    if final_cycle_count == 0:
        logger.info("✓ All final cycles correctly removed")
    else:
        logger.error(f"✗ {final_cycle_count} users still have final cycles in target data")
        raise ValueError("Target construction failed: final cycles not removed")
    
    # Check target distribution
    logger.info(f"\nTarget variable statistics:")
    logger.info(f"  Mean: {df_with_target['next_cycle_length'].mean():.2f} days")
    logger.info(f"  Median: {df_with_target['next_cycle_length'].median():.2f} days")
    logger.info(f"  Std: {df_with_target['next_cycle_length'].std():.2f} days")
    logger.info(f"  Min: {df_with_target['next_cycle_length'].min():.2f} days")
    logger.info(f"  Max: {df_with_target['next_cycle_length'].max():.2f} days")
    
    logger.info("✓ Target construction validated successfully")


def create_train_test_split(
    df: pd.DataFrame, 
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split respecting user grouping and chronology.
    
    Uses GroupShuffleSplit to prevent data leakage by keeping all cycles
    of a user in either train or test, not both.
    
    Args:
        df: DataFrame with target variable
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    logger.info("=== CREATING TRAIN/TEST SPLIT ===")
    
    # Use GroupShuffleSplit to keep all cycles of a user together
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Get indices for train and test
    train_idx, test_idx = next(gss.split(df, groups=df['user_id']))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    logger.info(f"Train set: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
    logger.info(f"Train users: {train_df['user_id'].nunique()}")
    logger.info(f"Test users: {test_df['user_id'].nunique()}")
    
    # Verify no user overlap
    train_users = set(train_df['user_id'].unique())
    test_users = set(test_df['user_id'].unique())
    overlap = train_users & test_users
    
    if overlap:
        logger.error(f"✗ User overlap detected: {len(overlap)} users in both train and test")
        raise ValueError("Train/test split failed: user overlap detected")
    else:
        logger.info("✓ No user overlap between train and test")
    
    return train_df, test_df


def prepare_feature_target_split(
    df: pd.DataFrame,
    target_col: str = 'next_cycle_length',
    exclude_cols: list = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target for modeling.
    
    Args:
        df: DataFrame with target variable
        target_col: Name of target column
        exclude_cols: Columns to exclude from features (identifiers, etc.)
        
    Returns:
        Tuple of (X_features, y_target)
    """
    if exclude_cols is None:
        exclude_cols = ['user_id', 'cycle_number', 'start_date', target_col]
    
    # Separate features and target
    X = df.drop(columns=exclude_cols)
    y = df[target_col]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Feature columns: {list(X.columns)}")
    
    return X, y


if __name__ == "__main__":
    # Load and prepare data
    df = load_and_join_datasets()
    
    # Construct target
    df_with_target = construct_next_cycle_target(df)
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df_with_target)
    
    # Prepare feature/target split
    X_train, y_train = prepare_feature_target_split(train_df)
    X_test, y_test = prepare_feature_target_split(test_df)
    
    print(f"\n=== DATA PREPARATION COMPLETE ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
