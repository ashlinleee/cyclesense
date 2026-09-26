# CycleSense Feature Engineering & Model Analysis Report

## Overview

This report provides a comprehensive analysis of the feature engineering techniques implemented in the CycleSense project, along with model performance outcomes and comparison data.

**Project**: CycleSense - Menstrual Cycle Prediction & Pattern Intelligence  
**Objective**: Predict next menstrual cycle length using user profiles and historical cycle data  
**Dataset**: 2,000 users, 17,976 cycle records, 34 features

---

## Feature Engineering Techniques

### 1. Historical Cycle Features (Leakage-Safe)

**Purpose**: Capture temporal patterns in cycle history without data leakage

#### Lag Features
- `cycle_length_lag_1`: Previous cycle length (1 cycle back)
- `cycle_length_lag_2`: Cycle length 2 cycles back
- `cycle_length_lag_3`: Cycle length 3 cycles back

**Implementation**: Uses `groupby('user_id')['cycle_length_days'].shift(lag)` to create lagged values while respecting user boundaries.

#### Rolling Statistics
- `cycle_length_mean_last_3`: Mean cycle length over last 3 cycles
- `cycle_length_std_last_3`: Standard deviation over last 3 cycles
- `cycle_length_min_last_3`: Minimum cycle length over last 3 cycles
- `cycle_length_max_last_3`: Maximum cycle length over last 3 cycles
- `cycle_length_mean_last_5`: Mean cycle length over last 5 cycles
- `cycle_length_std_last_5`: Standard deviation over last 5 cycles

**Implementation**: Rolling windows with `shift(1)` to prevent current target leakage:
```python
rolling = df.groupby('user_id')['cycle_length_days'].shift(1).groupby(df['user_id']).rolling(
    window=window, min_periods=1
)
```

#### Change Features
- `cycle_length_change`: Difference between current and previous cycle length
- `cycle_length_abs_change`: Absolute value of cycle length change

**Implementation**: 
```python
df['cycle_length_change'] = df.groupby('user_id')['cycle_length_days'].diff()
df['cycle_length_abs_change'] = df['cycle_length_change'].abs()
```

**Outcome**: These features capture cycle variability and trends, which are strong predictors of future cycle length.

---

### 2. Cross-Source Features

**Purpose**: Create interaction features between baseline profile and current cycle measurements

#### Delta Features
- `stress_delta`: `stress_score_cycle - stress_score_baseline`
- `sleep_delta`: `sleep_hours_cycle - sleep_hours`

**Rationale**: Captures deviation from baseline behavior, which may indicate cycle changes.

#### Interaction Features
- `stress_sleep_interaction`: Product of stress_delta and sleep_delta

**Implementation**:
```python
df['stress_delta'] = df['stress_score_cycle'] - df['stress_score_baseline']
df['sleep_delta'] = df['sleep_hours_cycle'] - df['sleep_hours']
df['stress_sleep_interaction'] = df['stress_delta'] * df['sleep_delta']
```

**Outcome**: These features capture how combined stress and sleep changes affect cycle patterns.

---

### 3. Date Features

**Purpose**: Capture seasonal and temporal patterns in cycle data

#### Basic Date Features
- `month`: Month of the year (1-12)
- `quarter`: Quarter of the year (1-4)
- `day_of_year`: Day number in the year (1-365/366)

#### Cyclical Representations
- `month_sin`: Sine transformation of month (preserves cyclical nature)
- `month_cos`: Cosine transformation of month

**Implementation**:
```python
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

**Rationale**: Cyclical representations ensure that December (12) and January (1) are recognized as adjacent in the cycle, unlike standard encoding.

**Outcome**: Captures potential seasonal variations in cycle patterns.

---

### 4. Preprocessing Pipeline

**Purpose**: Prepare features for modeling with consistent transformations

#### Numerical Features
- **Imputation**: Median imputation for missing values
- **Scaling**: StandardScaler (z-score normalization) - applied only for linear models

#### Categorical Features
- **Imputation**: Most frequent value for missing categories
- **Encoding**: OneHotEncoder with `handle_unknown='ignore'`

**Implementation**:
```python
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler() if use_scaling else 'passthrough')
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
```

**Rationale for Conditional Scaling**: 
- Linear models (Ridge, Lasso) are sensitive to feature scale
- Tree-based models (Random Forest, Gradient Boosting) are scale-invariant
- Applied scaling only where needed for optimization

---

## Leakage Prevention Strategy

### Feature Categorization

The project implements a rigorous leakage audit to categorize features based on prediction-time availability:

#### SAFE_AT_PREDICTION_TIME (14 features)
Features realistically available when making a prediction:
- User profile: age, bmi, diet_quality, exercise_frequency, sleep_hours, caffeine_intake, water_intake_liters, alcohol_consumption, smoking_status
- Medical factors: birth_control_use, pcos_diagnosed, stress_score_baseline
- Current cycle: cycle_length_days, prev_cycle_length

#### CONDITIONAL (10 features)
Features that might be available depending on prediction timing:
- cycle_phase, flow_level, pain_level, pms_symptoms, mood_score
- stress_score_cycle, sleep_hours_cycle, energy_level, concentration_score, work_hours_lost

#### EXCLUDED_LEAKAGE (6 features)
Features that would NOT be available at prediction time:
- estrogen_pgml, progesterone_ngml, ovulation_result
- overall_health_score, log_consistency_score, prepared_before_period

#### IDENTIFIER_ONLY (4 features)
Features used only for grouping, not prediction:
- user_id, cycle_number, start_date, state

### Feature Sets for Experimentation

1. **Strict Feature Set** (14 features): Only information realistically available at prediction time
2. **Extended Feature Set** (24 features): Includes conditional features, still excludes direct leakage
3. **Full Feature Set** (25 features): Extended + state for evaluation

**Production Decision**: Strict feature set used for all production models to ensure defensible predictions.

---

## Target Construction

### Method
- **Target Variable**: `next_cycle_length`
- **Construction**: For each user, sort chronologically and shift `cycle_length_days` by -1
- **Logic**: Current cycle N → target = cycle N+1 length
- **Removal**: Final cycle for each user removed (no next cycle to predict)

### Validation
- Verified target matches next cycle's actual length for sample users
- Ensured no future data leaked into historical features
- Confirmed all final cycles correctly removed

### Train/Test Split Strategy
- **Method**: GroupShuffleSplit (keeps all cycles of a user together)
- **Rationale**: Prevents data leakage by ensuring no user appears in both train and test
- **Split**: 80% train, 20% test
- **Verification**: Confirmed zero user overlap between train and test sets

---

## Model Comparison Data

### Baseline Models

#### Baseline A: Global Mean Predictor
- **Strategy**: Predict global training-set mean cycle length
- **MAE**: 1.94 days
- **RMSE**: 2.49 days
- **R²**: 0.000
- **Training Time**: 0.01s

#### Baseline B: Previous Cycle Predictor
- **Strategy**: Predict next cycle ≈ current cycle length
- **MAE**: 2.30 days
- **RMSE**: 2.87 days
- **R²**: -0.132
- **Training Time**: 0.000s (no training)

### Linear Models

#### Linear Regression
- **MAE**: 1.74 days
- **RMSE**: 2.24 days
- **R²**: 0.266
- **Training Time**: 0.16s
- **Scaling**: Yes
- **Improvement over baseline mean**: +10.3%

#### Ridge Regression (α=1.0)
- **MAE**: 1.74 days
- **RMSE**: 2.24 days
- **R²**: 0.266
- **Training Time**: 0.03s
- **Scaling**: Yes
- **Features Used**: All (no feature selection)
- **Improvement over baseline mean**: +10.3%

#### Lasso Regression (α=1.0)
- **MAE**: 1.94 days
- **RMSE**: 2.62 days
- **R²**: -0.006
- **Training Time**: 0.03s
- **Scaling**: Yes
- **Features Used**: 0 (aggressive regularization)
- **Improvement over baseline mean**: 0.0%

### Tree-Based Models

#### Random Forest (n_estimators=100)
- **MAE**: 1.73 days
- **RMSE**: 2.21 days
- **R²**: 0.284
- **Training Time**: 3.08s
- **Scaling**: No (not needed)
- **Improvement over baseline mean**: +10.8%

#### Histogram-based Gradient Boosting
- **MAE**: 1.70 days ⭐ **BEST**
- **RMSE**: 2.18 days
- **R²**: 0.306
- **Training Time**: 0.45s
- **Scaling**: No (not needed)
- **Max Iterations**: 100
- **Improvement over baseline mean**: +12.4%
- **Improvement over previous cycle baseline**: +26.1%

---

## Feature Importance Analysis

### Top Features by Permutation Importance

1. **pcos_diagnosed** (0.250): PCOS diagnosis is the strongest predictor
2. **cycle_length_mean_last_5** (0.126): Rolling mean over last 5 cycles
3. **cycle_length_mean_last_3** (0.022): Rolling mean over last 3 cycles
4. **bmi** (0.013): Body Mass Index
5. **birth_control_use** (0.008): Birth control usage

### Insights
- **Medical factors dominate**: PCOS diagnosis and birth control use are top predictors
- **Historical patterns matter**: Rolling statistics of past cycle lengths are highly predictive
- **Profile factors contribute**: BMI and other static profile features provide meaningful signal
- **Lifestyle factors**: Stress, sleep, and exercise have moderate importance

---

## Data Profile Summary

### Dataset Characteristics
- **Users**: 2,000 unique users
- **Cycle Records**: 17,976 total records
- **Features**: 34 columns after joining datasets
- **Date Range**: 460 days (2024-01-01 to 2025-04-05)
- **Cycles per User**: Mean 8.99, Median 9.00, Range 6-12

### Key Distributions

#### Cycle Length (Target)
- **Mean**: 27.82 days
- **Median**: 28.00 days
- **Std**: 2.40 days
- **Range**: 22-44 days
- **Skew**: 0.71 (slightly right-skewed)
- **Outliers**: 374 (2.08%) using IQR method

#### User Profile
- **Age**: Mean 25.79, Range 16-43, Skew 0.10
- **BMI**: Mean 22.99, Range 16-41.2, Skew 0.33
- **Stress Score**: Mean 5.75, Range 1-10, Skew -0.07
- **Sleep Hours**: Mean 6.96, Range 4.5-10, Skew 0.10

### Missing Values
- **prev_cycle_length**: 2,000 (11.13%) - First cycle for each user
- **exercise_frequency**: 385 (19.25%) in user profiles

### Correlation Analysis
- No correlations > 0.7 detected between numerical features
- Features are relatively independent, good for modeling

---

## Model Selection Rationale

### Why Gradient Boosting?
1. **Best Performance**: Lowest MAE (1.70 days) and highest R² (0.306)
2. **Training Efficiency**: 0.45s training time (faster than Random Forest)
3. **No Scaling Required**: Scale-invariant, simplifies pipeline
4. **Handles Non-linearity**: Captures complex feature interactions
5. **Robust to Outliers**: Less sensitive to extreme values than linear models

### Why Not Other Models?
- **Linear Regression**: Good baseline, but limited by linear assumptions
- **Ridge**: Similar to Linear Regression, no improvement
- **Lasso**: Over-regularized, zero features used
- **Random Forest**: Good performance, but slower training (3.08s)

---

## Outcomes & Impact

### Performance Improvements
- **12.4% improvement** over baseline mean predictor
- **26.1% improvement** over previous cycle baseline
- **MAE of 1.70 days**: Predictions differ from actual next cycle by ~1.7 days on average

### Business Impact
- **More accurate predictions**: Enables better cycle planning
- **Personalized insights**: Historical patterns captured per user
- **Explainable**: Feature importance provides actionable insights
- **Production-ready**: Leakage-safe features ensure defensible predictions

### Limitations
- **Educational dataset**: Based on synthetic data, not real patient data
- **Simplified features**: Real-world applications would need more sophisticated features
- **No uncertainty estimates**: Current model provides point predictions only
- **Static model**: Requires retraining for new patterns

---

## Technical Implementation Highlights

### MLOps Pipeline
- **Data Versioning**: DVC tracks raw datasets and transformations
- **Experiment Tracking**: MLflow logs all experiments, metrics, and artifacts
- **Model Registry**: Registered model with version management
- **Quality Gates**: MAE threshold (2.5 days), R² threshold (0.2)

### Production Architecture
- **API**: FastAPI for model serving
- **UI**: Streamlit for interactive dashboard
- **Deployment**: Docker containerization
- **CI/CD**: GitHub Actions for automated testing and deployment

### Code Quality
- **Modular design**: Separate modules for data, features, target, preprocessing, training, evaluation
- **sklearn-compatible**: Custom transformers follow sklearn API
- **Logging**: Comprehensive logging throughout pipeline
- **Testing**: Unit tests for feature engineering

---

## Recommendations

### Immediate Improvements
1. **Uncertainty Estimation**: Add prediction intervals using quantile regression or bootstrapping
2. **Feature Selection**: Implement recursive feature elimination to reduce dimensionality
3. **Hyperparameter Tuning**: Optimize Gradient Boosting hyperparameters (learning rate, max_depth, etc.)
4. **Cross-validation**: Use k-fold cross-validation for more robust performance estimates

### Long-term Enhancements
1. **Real-world Data**: Integrate with actual health data sources
2. **Advanced Features**: More sophisticated temporal features, external factors (medications, health conditions)
3. **Multi-task Learning**: Predict additional outcomes (symptoms, mood, ovulation timing)
4. **User Feedback Loop**: Continuous learning from user corrections
5. **Real-time Monitoring**: Drift detection and performance monitoring in production

---

## Conclusion

The CycleSense project demonstrates a comprehensive approach to feature engineering for time-series prediction with rigorous leakage prevention. The implemented techniques—including lag features, rolling statistics, cross-source interactions, and cyclical date features—combined with careful preprocessing and model selection, resulted in a Gradient Boosting model that achieves a MAE of 1.70 days, representing a 12.4% improvement over baseline predictions.

The project showcases best practices in ML engineering, including data versioning, experiment tracking, model explainability, and production deployment, making it a solid foundation for real-world health-tech applications.

---

**Report Generated**: 2026-09-26  
**Project Repository**: CycleSense  
**Model Version**: 1.0  
**Best Model**: Histogram-based Gradient Boosting Regressor