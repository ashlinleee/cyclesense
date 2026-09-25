"""
Tests for feature engineering pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import HistoricalCycleFeatures, CrossSourceFeatures, DateFeatures


def test_historical_cycle_features():
    """Test historical cycle feature engineering."""
    # Create sample data
    df = pd.DataFrame({
        'user_id': ['U001', 'U001', 'U001', 'U002', 'U002'],
        'cycle_number': [1, 2, 3, 1, 2],
        'cycle_length_days': [28, 29, 27, 30, 28]
    })
    
    transformer = HistoricalCycleFeatures()
    result = transformer.fit_transform(df)
    
    # Check that features were created
    assert 'cycle_length_lag_1' in result.columns
    assert 'cycle_length_mean_last_3' in result.columns
    assert 'cycle_length_change' in result.columns
    
    # Check that NaN values are handled correctly
    assert result['cycle_length_lag_1'].isna().sum() > 0  # First cycle should have NaN lag


def test_cross_source_features():
    """Test cross-source feature engineering."""
    df = pd.DataFrame({
        'stress_score_cycle': [6.0, 7.0, 5.0],
        'stress_score_baseline': [5.0, 5.0, 5.0],
        'sleep_hours_cycle': [7.0, 6.0, 8.0],
        'sleep_hours': [7.0, 7.0, 7.0]
    })
    
    transformer = CrossSourceFeatures()
    result = transformer.fit_transform(df)
    
    assert 'stress_delta' in result.columns
    assert 'sleep_delta' in result.columns
    assert 'stress_sleep_interaction' in result.columns
    
    # Check delta calculation
    assert result['stress_delta'].iloc[0] == 1.0  # 6.0 - 5.0


def test_date_features():
    """Test date feature engineering."""
    df = pd.DataFrame({
        'start_date': pd.to_datetime(['2024-01-15', '2024-06-15', '2024-12-15'])
    })
    
    transformer = DateFeatures()
    result = transformer.fit_transform(df)
    
    assert 'month' in result.columns
    assert 'quarter' in result.columns
    assert 'month_sin' in result.columns
    assert 'month_cos' in result.columns
    
    # Check month extraction
    assert result['month'].iloc[0] == 1
    assert result['month'].iloc[1] == 6


def test_no_future_leakage():
    """Test that future cycles don't leak into historical features."""
    df = pd.DataFrame({
        'user_id': ['U001'] * 5,
        'cycle_number': [1, 2, 3, 4, 5],
        'cycle_length_days': [28, 29, 27, 30, 28]
    })
    
    transformer = HistoricalCycleFeatures()
    result = transformer.fit_transform(df)
    
    # For cycle 3, rolling mean should only include cycles 1-2
    # For cycle 5, rolling mean should include cycles 2-4 (not cycle 5 itself)
    
    # Check that rolling mean for cycle 5 doesn't include cycle 5's value
    cycle_5_idx = 4
    cycle_5_rolling_mean = result['cycle_length_mean_last_3'].iloc[cycle_5_idx]
    
    # Manual calculation of cycles 2-4: (29 + 27 + 30) / 3 = 28.67
    expected_mean = (29 + 27 + 30) / 3
    
    assert abs(cycle_5_rolling_mean - expected_mean) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
