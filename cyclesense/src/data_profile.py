"""
Data profiling and EDA for CycleSense datasets.
Generates comprehensive data quality reports and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

from src.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    RANDOM_STATE
)
from src.data import load_period_log, load_user_profile, load_and_join_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def create_directory_structure():
    """Ensure directories exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_dataset_dimensions(df: pd.DataFrame, name: str) -> dict:
    """Analyze basic dataset dimensions."""
    logger.info(f"\n=== {name} DIMENSIONS ===")
    info = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2
    }
    logger.info(f"Rows: {info['rows']:,}")
    logger.info(f"Columns: {info['columns']}")
    logger.info(f"Memory: {info['memory_mb']:.2f} MB")
    return info


def analyze_data_types(df: pd.DataFrame, name: str) -> dict:
    """Analyze data types."""
    logger.info(f"\n=== {name} DATA TYPES ===")
    dtype_counts = df.dtypes.value_counts()
    logger.info(f"Data type distribution:\n{dtype_counts}")
    return dtype_counts.to_dict()


def analyze_missing_values(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Analyze missing values."""
    logger.info(f"\n=== {name} MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'count': missing,
        'percentage': missing_pct
    }).sort_values('count', ascending=False)
    
    missing_df = missing_df[missing_df['count'] > 0]
    
    if len(missing_df) > 0:
        logger.info(f"Columns with missing values:\n{missing_df}")
    else:
        logger.info("No missing values found")
    
    return missing_df


def analyze_duplicates(df: pd.DataFrame, name: str) -> dict:
    """Analyze duplicates."""
    logger.info(f"\n=== {name} DUPLICATES ===")
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / len(df) * 100).round(2)
    logger.info(f"Duplicate rows: {duplicate_count} ({duplicate_pct}%)")
    return {"count": duplicate_count, "percentage": duplicate_pct}


def analyze_unique_users(df: pd.DataFrame) -> dict:
    """Analyze unique users and cycles per user."""
    logger.info(f"\n=== USER ANALYSIS ===")
    unique_users = df['user_id'].nunique()
    cycles_per_user = df.groupby('user_id').size()
    
    logger.info(f"Unique users: {unique_users:,}")
    logger.info(f"Cycles per user statistics:")
    logger.info(f"  Mean: {cycles_per_user.mean():.2f}")
    logger.info(f"  Median: {cycles_per_user.median():.2f}")
    logger.info(f"  Min: {cycles_per_user.min()}")
    logger.info(f"  Max: {cycles_per_user.max()}")
    logger.info(f"  Std: {cycles_per_user.std():.2f}")
    
    return {
        "unique_users": unique_users,
        "mean_cycles_per_user": cycles_per_user.mean(),
        "median_cycles_per_user": cycles_per_user.median(),
        "min_cycles": cycles_per_user.min(),
        "max_cycles": cycles_per_user.max(),
        "std_cycles": cycles_per_user.std()
    }


def analyze_categorical_cardinality(df: pd.DataFrame) -> dict:
    """Analyze categorical variable cardinality."""
    logger.info(f"\n=== CATEGORICAL CARDINALITY ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    cardinality = {}
    for col in categorical_cols:
        unique_count = df[col].nunique()
        cardinality[col] = unique_count
        logger.info(f"{col}: {unique_count} unique values")
    
    return cardinality


def analyze_numerical_distributions(df: pd.DataFrame) -> dict:
    """Analyze numerical distributions."""
    logger.info(f"\n=== NUMERICAL DISTRIBUTIONS ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = {}
    for col in numerical_cols:
        col_stats = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skew': df[col].skew()
        }
        stats[col] = col_stats
        logger.info(f"{col}: mean={col_stats['mean']:.2f}, median={col_stats['median']:.2f}, std={col_stats['std']:.2f}")
    
    return stats


def analyze_outliers(df: pd.DataFrame, column: str) -> dict:
    """Analyze outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        'count': len(outliers),
        'percentage': len(outliers) / len(df) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlations between numerical variables."""
    logger.info(f"\n=== CORRELATION ANALYSIS ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    # Find high correlations
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr:
        logger.info(f"High correlations (>0.7):")
        for corr in high_corr:
            logger.info(f"  {corr['var1']} - {corr['var2']}: {corr['correlation']:.3f}")
    
    return corr_matrix


def analyze_date_range(df: pd.DataFrame, date_col: str = 'start_date') -> dict:
    """Analyze date range."""
    logger.info(f"\n=== DATE RANGE ANALYSIS ===")
    df[date_col] = pd.to_datetime(df[date_col])
    
    date_info = {
        'min_date': df[date_col].min(),
        'max_date': df[date_col].max(),
        'date_range_days': (df[date_col].max() - df[date_col].min()).days
    }
    
    logger.info(f"Date range: {date_info['min_date']} to {date_info['max_date']}")
    logger.info(f"Span: {date_info['date_range_days']} days")
    
    return date_info


def plot_cycle_length_distribution(df: pd.DataFrame):
    """Plot cycle length distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['cycle_length_days'], bins=30, kde=True, color='teal')
    plt.title('Cycle Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Cycle Length (Days)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(df['cycle_length_days'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["cycle_length_days"].mean():.1f} days')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cycle_length_distribution.png', dpi=300)
    plt.close()
    logger.info("Saved: cycle_length_distribution.png")


def plot_missing_values(df: pd.DataFrame):
    """Plot missing values heatmap."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'missing_values.png', dpi=300)
    plt.close()
    logger.info("Saved: missing_values.png")


def plot_correlation_matrix(corr_matrix: pd.DataFrame):
    """Plot correlation matrix."""
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_matrix.png', dpi=300)
    plt.close()
    logger.info("Saved: correlation_matrix.png")


def plot_cycles_per_user(df: pd.DataFrame):
    """Plot cycles per user distribution."""
    cycles_per_user = df.groupby('user_id').size()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(cycles_per_user, bins=30, kde=True, color='coral')
    plt.title('Cycles Per User Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Cycles', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.axvline(cycles_per_user.mean(), color='red', linestyle='--',
                label=f'Mean: {cycles_per_user.mean():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cycles_per_user.png', dpi=300)
    plt.close()
    logger.info("Saved: cycles_per_user.png")


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'cycle_length_days'):
    """Plot target variable distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col], bins=30, kde=True, color='purple')
    plt.title(f'{target_col} Distribution', fontsize=14, fontweight='bold')
    plt.xlabel(target_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(df[target_col].mean(), color='red', linestyle='--',
                label=f'Mean: {df[target_col].mean():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'target_distribution.png', dpi=300)
    plt.close()
    logger.info("Saved: target_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame):
    """Plot distributions of key numeric variables."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    key_cols = ['cycle_length_days', 'age', 'bmi', 'stress_score_cycle', 
                'sleep_hours_cycle', 'mood_score']
    
    # Filter to only columns that exist
    available_cols = [col for col in key_cols if col in numerical_cols]
    
    if not available_cols:
        logger.warning("No key numeric columns found for distribution plots")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(available_cols):
        if idx < len(axes):
            sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[idx], color='steelblue')
            axes[idx].set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('')
    
    # Hide unused subplots
    for idx in range(len(available_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'numeric_distributions.png', dpi=300)
    plt.close()
    logger.info("Saved: numeric_distributions.png")


def generate_data_profile_report():
    """Generate comprehensive data profile report."""
    logger.info("=== GENERATING DATA PROFILE REPORT ===")
    
    create_directory_structure()
    
    # Load data
    period_df = load_period_log()
    profile_df = load_user_profile()
    merged_df = load_and_join_datasets()
    
    # Analyses
    report_lines = []
    report_lines.append("# CycleSense Data Profile Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n---\n")
    
    # Period Log Analysis
    report_lines.append("## Period_Log.csv Analysis")
    report_lines.append("\n### Dimensions")
    period_dims = analyze_dataset_dimensions(period_df, "Period_Log")
    report_lines.append(f"- Rows: {period_dims['rows']:,}")
    report_lines.append(f"- Columns: {period_dims['columns']}")
    report_lines.append(f"- Memory: {period_dims['memory_mb']:.2f} MB")
    
    report_lines.append("\n### Data Types")
    period_dtypes = analyze_data_types(period_df, "Period_Log")
    for dtype, count in period_dtypes.items():
        report_lines.append(f"- {dtype}: {count}")
    
    report_lines.append("\n### Missing Values")
    period_missing = analyze_missing_values(period_df, "Period_Log")
    if not period_missing.empty:
        for col in period_missing.index:
            count = period_missing.loc[col, 'count']
            percentage = period_missing.loc[col, 'percentage']
            report_lines.append(f"- {col}: {count} ({percentage}%)")
    
    report_lines.append("\n### Duplicates")
    period_dupes = analyze_duplicates(period_df, "Period_Log")
    report_lines.append(f"- Duplicate rows: {period_dupes['count']} ({period_dupes['percentage']}%)")
    
    # User Profile Analysis
    report_lines.append("\n---\n")
    report_lines.append("## User_Profile.csv Analysis")
    report_lines.append("\n### Dimensions")
    profile_dims = analyze_dataset_dimensions(profile_df, "User_Profile")
    report_lines.append(f"- Rows: {profile_dims['rows']:,}")
    report_lines.append(f"- Columns: {profile_dims['columns']}")
    report_lines.append(f"- Memory: {profile_dims['memory_mb']:.2f} MB")
    
    report_lines.append("\n### Data Types")
    profile_dtypes = analyze_data_types(profile_df, "User_Profile")
    for dtype, count in profile_dtypes.items():
        report_lines.append(f"- {dtype}: {count}")
    
    report_lines.append("\n### Missing Values")
    profile_missing = analyze_missing_values(profile_df, "User_Profile")
    if not profile_missing.empty:
        for col in profile_missing.index:
            count = profile_missing.loc[col, 'count']
            percentage = profile_missing.loc[col, 'percentage']
            report_lines.append(f"- {col}: {count} ({percentage}%)")
    
    # Merged Dataset Analysis
    report_lines.append("\n---\n")
    report_lines.append("## Merged Dataset Analysis")
    report_lines.append("\n### Dimensions")
    merged_dims = analyze_dataset_dimensions(merged_df, "Merged")
    report_lines.append(f"- Rows: {merged_dims['rows']:,}")
    report_lines.append(f"- Columns: {merged_dims['columns']}")
    
    report_lines.append("\n### User Analysis")
    user_stats = analyze_unique_users(merged_df)
    report_lines.append(f"- Unique users: {user_stats['unique_users']:,}")
    report_lines.append(f"- Mean cycles per user: {user_stats['mean_cycles_per_user']:.2f}")
    report_lines.append(f"- Median cycles per user: {user_stats['median_cycles_per_user']:.2f}")
    report_lines.append(f"- Min cycles: {user_stats['min_cycles']}")
    report_lines.append(f"- Max cycles: {user_stats['max_cycles']}")
    
    report_lines.append("\n### Categorical Cardinality")
    cardinality = analyze_categorical_cardinality(merged_df)
    for col, count in cardinality.items():
        report_lines.append(f"- {col}: {count} unique values")
    
    report_lines.append("\n### Numerical Distributions")
    numerical_stats = analyze_numerical_distributions(merged_df)
    key_vars = ['cycle_length_days', 'age', 'bmi', 'stress_score_cycle', 'sleep_hours_cycle']
    for var in key_vars:
        if var in numerical_stats:
            stats = numerical_stats[var]
            report_lines.append(f"\n{var}:")
            report_lines.append(f"  - Mean: {stats['mean']:.2f}")
            report_lines.append(f"  - Median: {stats['median']:.2f}")
            report_lines.append(f"  - Std: {stats['std']:.2f}")
            report_lines.append(f"  - Min: {stats['min']:.2f}")
            report_lines.append(f"  - Max: {stats['max']:.2f}")
            report_lines.append(f"  - Skew: {stats['skew']:.2f}")
    
    report_lines.append("\n### Date Range")
    date_info = analyze_date_range(merged_df)
    report_lines.append(f"- Date range: {date_info['min_date']} to {date_info['max_date']}")
    report_lines.append(f"- Span: {date_info['date_range_days']} days")
    
    report_lines.append("\n### Correlation Analysis")
    corr_matrix = analyze_correlations(merged_df)
    
    # Outlier Analysis
    report_lines.append("\n### Outlier Analysis (IQR Method)")
    outlier_cols = ['cycle_length_days', 'age', 'bmi', 'stress_score_cycle']
    for col in outlier_cols:
        if col in merged_df.columns:
            outliers = analyze_outliers(merged_df, col)
            report_lines.append(f"\n{col}:")
            report_lines.append(f"  - Outliers: {outliers['count']} ({outliers['percentage']:.2f}%)")
            report_lines.append(f"  - Bounds: [{outliers['lower_bound']:.2f}, {outliers['upper_bound']:.2f}]")
    
    # Generate plots
    logger.info("\n=== GENERATING VISUALIZATIONS ===")
    plot_cycle_length_distribution(period_df)
    plot_missing_values(merged_df)
    plot_correlation_matrix(corr_matrix)
    plot_cycles_per_user(period_df)
    plot_target_distribution(period_df)
    plot_numeric_distributions(merged_df)
    
    # Write report
    report_path = REPORTS_DIR / 'data_profile.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"\nData profile report saved to: {report_path}")
    logger.info("All visualizations saved to: {FIGURES_DIR}")
    
    return report_path


if __name__ == "__main__":
    generate_data_profile_report()
