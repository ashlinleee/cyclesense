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



# CycleSense Feature Engineering Complete Chart

## Overview

This chart visualizes all feature engineering techniques implemented in the CycleSense project, showing inputs, outputs, and relationships.

---

## Complete Feature Engineering Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RAW DATA INPUT                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  User Profile (14 features)              Period Log (21 features)          │
│  ├── age (16-43)                          ├── cycle_length_days (22-44)        │
│  ├── bmi (16-41.2)                        ├── prev_cycle_length              │
│  ├── diet_quality (4 levels)              ├── cycle_phase (3 levels)         │
│  ├── exercise_frequency (3 levels)        ├── flow_level (3 levels)          │
│  ├── sleep_hours (4.5-10)                 ├── pain_level                     │
│  ├── caffeine_intake                       ├── pms_symptoms                   │
│  ├── water_intake_liters                  ├── mood_score                     │
│  ├── alcohol_consumption (3 levels)       ├── stress_score_cycle (1-10)     │
│  ├── smoking_status (2 levels)             ├── sleep_hours_cycle (4.5-10)     │
│  ├── birth_control_use (2 levels)         ├── energy_level                   │
│  ├── pcos_diagnosed (2 levels)            ├── concentration_score             │
│  └── stress_score_baseline (1-10)          ├── work_hours_lost                │
│                                         └── start_date (datetime)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DATA JOINING & TARGET CONSTRUCTION                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MERGED DATASET (34 features)                                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  User Profile (14) + Period Log (20) = 34 total features           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TARGET CONSTRUCTION                                                │   │
│  │  Formula: next_cycle_length = shift(cycle_length_days, -1)          │   │
│  │  Result: Current cycle N → target = cycle N+1 length                 │   │
│  │  Validation: Remove final cycles (no next cycle to predict)        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. HISTORICAL CYCLE FEATURES (Leakage-Safe)                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  INPUT: user_id, cycle_number, cycle_length_days                      │   │
│  │  OUTPUT: 8 features                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │ LAG FEATURES (3)                                               │   │   │
│  │  │   ├── cycle_length_lag_1 = shift(cycle_length_days, 1)       │   │   │
│  │  │   ├── cycle_length_lag_2 = shift(cycle_length_days, 2)       │   │   │
│  │  │   └── cycle_length_lag_3 = shift(cycle_length_days, 3)       │   │   │
│  │  │   Purpose: Capture recent historical cycle lengths            │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │ ROLLING STATISTICS (8)                                          │   │   │
│  │  │   Window 3:                                                   │   │   │
│  │  │   ├── cycle_length_mean_last_3 = rolling(shift(1), mean)      │   │   │
│  │  │   ├── cycle_length_std_last_3 = rolling(shift(1), std)       │   │   │
│  │  │   ├── cycle_length_min_last_3 = rolling(shift(1), min)       │   │   │
│  │  │   └── cycle_length_max_last_3 = rolling(shift(1), max)       │   │   │
│  │  │   Window 5:                                                   │   │   │
│  │  │   ├── cycle_length_mean_last_5 = rolling(shift(1), mean)      │   │   │
│  │  │   ├── cycle_length_std_last_5 = rolling(shift(1), std)       │   │   │
│  │  │   ├── cycle_length_min_last_5 = rolling(shift(1), min)       │   │   │
│  │  │   └── cycle_length_max_last_5 = rolling(shift(1), max)       │   │   │
│  │  │   Purpose: Capture trends and consistency over time             │   │   │
│  │  │   Leakage Prevention: shift(1) excludes current cycle         │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │ CHANGE FEATURES (2)                                             │   │   │
│  │  │   ├── cycle_length_change = diff(cycle_length_days)           │   │   │
│  │  │   └── cycle_length_abs_change = abs(cycle_length_change)      │   │   │
│  │  │   Purpose: Detect lengthening/shortening trends                │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  2. CROSS-SOURCE FEATURES (Delta & Interaction)                      │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  INPUT: stress_score_cycle, stress_score_baseline,                 │   │
│  │         sleep_hours_cycle, sleep_hours                               │   │
│  │  OUTPUT: 3 features                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │ DELTA FEATURES (2)                                             │   │   │
│  │  │   ├── stress_delta = stress_score_cycle - stress_score_baseline │   │   │
│  │  │   └── sleep_delta = sleep_hours_cycle - sleep_hours           │   │   │
│  │  │   Purpose: Capture deviations from baseline behavior            │   │   │
│  │  │   Example: +2.0 = 2 points above baseline stress             │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │ INTERACTION FEATURE (1)                                        │   │   │
│  │  │   └── stress_sleep_interaction = stress_delta × sleep_delta   │   │   │
│  │  │   Purpose: Capture combined effect of stress and sleep          │   │   │
│  │  │   Example: -2.0 = high stress + poor sleep (negative impact)   │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3. DATE FEATURES (Temporal)                                         │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  INPUT: start_date (datetime)                                        │   │
│  │  OUTPUT: 5 features                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │ BASIC DATE FEATURES (3)                                         │   │   │
│  │  │   ├── month = start_date.dt.month (1-12)                        │   │   │
│  │  │   ├── quarter = start_date.dt.quarter (1-4)                     │   │   │
│  │  │   └── day_of_year = start_date.dt.dayofyear (1-365)            │   │   │
│  │  │   Purpose: Capture seasonal patterns                            │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │ CYCLICAL FEATURES (2)                                           │   │   │
│  │  │   ├── month_sin = sin(2π × month / 12)                         │   │   │
│  │  │   └── month_cos = cos(2π × month / 12)                         │   │   │
│  │  │   Purpose: Preserve cyclical nature (Dec ≈ Jan)                  │   │   │
│  │  │   Visualization: Position on yearly circle                       │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LEAKAGE AUDIT & FEATURE SELECTION                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FEATURE CATEGORIZATION                                               │   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  SAFE_AT_PREDICTION_TIME (14) ✅                                     │   │
│  │  ├── User profile: age, bmi, diet_quality, exercise_frequency,      │   │
│  │  │                 sleep_hours, caffeine_intake, water_intake,       │   │
│  │  │                 alcohol_consumption, smoking_status,              │   │
│  │  │                 birth_control_use, pcos_diagnosed,               │   │
│  │  │                 stress_score_baseline                            │   │
│  │  └── Current cycle: cycle_length_days, prev_cycle_length            │   │
│  │                                                                       │   │
│  │  CONDITIONAL (10) ⚠️                                                  │   │
│  │  └── cycle_phase, flow_level, pain_level, pms_symptoms,            │   │
│  │      mood_score, stress_score_cycle, sleep_hours_cycle,              │   │
│  │      energy_level, concentration_score, work_hours_lost             │   │
│  │                                                                       │   │
│  │  EXCLUDED_LEAKAGE (6) ❌                                              │   │
│  │  └── prepared_before_period, overall_health_score,                    │   │
│  │      log_consistency_score, estrogen_pgml, progesterone_ngml,       │   │
│  │      ovulation_result                                                │   │
│  │                                                                       │   │
│  │  IDENTIFIER_ONLY (4) ⊘                                                │   │
│  │  └── user_id, cycle_number, start_date, state                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FEATURE SET SELECTION                                                │   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  STRICT FEATURE SET (14 base + 18 engineered = 32) ✅ PRODUCTION  │   │
│  │  └── Only SAFE features + leakage-safe engineered features          │   │
│  │                                                                       │   │
│  │  EXTENDED FEATURE SET (24 base + 18 engineered = 42) 📊 RESEARCH  │   │
│  │  └── SAFE + CONDITIONAL features + engineered features              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MISSING VALUE IMPUTATION                                            │   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  NUMERICAL (29 features)                                              │   │
│  │  └── SimpleImputer(strategy='median') ✅ Robust to outliers          │   │
│  │                                                                       │   │
│  │  CATEGORICAL (6 features)                                            │   │
│  │  └── SimpleImputer(strategy='most_frequent') ✅ Preserves distribution │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FEATURE ENCODING                                                     │   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  ONE-HOT ENCODING (6 categorical → 16 binary features)                 │   │
│  │  ├── diet_quality (4 values) → 4 binary features                     │   │
│  │  ├── exercise_frequency (3 values) → 3 binary features               │   │
│  │  ├── alcohol_consumption (3 values) → 3 binary features              │   │
│  │  ├── smoking_status (2 values) → 2 binary features                    │   │
│  │  ├── birth_control_use (2 values) → 2 binary features                 │   │
│  │  └── pcos_diagnosed (2 values) → 2 binary features                    │   │
│  │  Parameters: handle_unknown='ignore', sparse_output=False            │   │
│  │  Purpose: No ordinal assumption, works with all models              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FEATURE SCALING (Conditional)                                       │   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  LINEAR MODELS (Linear, Ridge, Lasso) ✅ WITH SCALING                │   │
│  │  └── StandardScaler() (mean=0, std=1)                              │   │
│  │      Purpose: Optimal for regularization, handles outliers           │   │
│  │                                                                       │   │
│  │  TREE-BASED MODELS (Random Forest, Gradient Boosting) ❌ NO SCALING │   │
│  │  └── No scaling (scale-invariant, split-based)                       │   │
│  │      Purpose: Faster training, tree models don't need scaling        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FINAL FEATURE COUNT                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BEFORE PREPROCESSING:                                                    │
│  ├── Base features: 14                                                    │
│  ├── Engineered features: 21                                              │
│  └── Total: 35 features                                                   │
│                                                                             │
│  AFTER PREPROCESSING:                                                     │
│  ├── Numerical features: 29 (base + engineered numerical)                 │
│  ├── One-hot encoded features: 16 (categorical expanded)                   │
│  └── TOTAL: 45 features for model training ✅                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODEL: Histogram-based Gradient Boosting Regressor                       │
│  INPUT: 45 features (29 numerical + 16 one-hot encoded)                    │
│  OUTPUT: Predicted next cycle length (days)                                │
│  PERFORMANCE: MAE = 1.70 days, R² = 0.306                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Engineering Techniques Summary Table

| # | Technique | Input Features | Output Features | Count | Purpose | Leakage Safe |
|---|-----------|---------------|----------------|-------|---------|--------------|
| 1 | **Lag Features** | cycle_length_days | cycle_length_lag_1, lag_2, lag_3 | 3 | Historical context | ✅ Yes (shift) |
| 2 | **Rolling Statistics** | cycle_length_days | mean/std/min/max for windows 3,5 | 8 | Trends & consistency | ✅ Yes (shift+rolling) |
| 3 | **Change Features** | cycle_length_days | cycle_length_change, abs_change | 2 | Rate of change | ✅ Yes (diff) |
| 4 | **Delta Features** | stress/sleep baseline & current | stress_delta, sleep_delta | 2 | Deviation from baseline | ✅ Yes (baseline available) |
| 5 | **Interaction Features** | stress_delta, sleep_delta | stress_sleep_interaction | 1 | Combined effects | ✅ Yes (from deltas) |
| 6 | **Date Features** | start_date | month, quarter, day_of_year | 3 | Seasonal patterns | ✅ Yes (date available) |
| 7 | **Cyclical Features** | month | month_sin, month_cos | 2 | Preserve cyclical nature | ✅ Yes (from date) |
| **TOTAL** | **7 Techniques** | **Multiple sources** | **21 engineered** | **Comprehensive** | **All Safe** |

---

## Feature Flow Diagram

```
RAW DATA (34 features)
    │
    ├─→ User Profile (14) ─────────────────────────┐
    │                                             │
    └─→ Period Log (20) ─────────────────────────┤
                                                  │
                                         DATA JOINING
                                                  ↓
                                    MERGED DATASET (34)
                                                  │
                                    TARGET CONSTRUCTION
                                                  ↓
                                  DATASET WITH TARGET (33)
                                                  │
    ┌─────────────────────────────────────────────┼─────────────────────────┐
    │                                             │                         │
    ↓                                             ↓                         ↓
HISTORICAL FEATURES                        CROSS-SOURCE              DATE FEATURES
    │                                             │                         │
    ├─→ Lag (3)                                   ├─→ Delta (2)             ├─→ Basic (3)
    ├─→ Rolling (8)                              ├─→ Interaction (1)        ├─→ Cyclical (2)
    │                                             │                         │
    └─────────────────────────────────────────────┴─────────────────────────┘
                                                  │
                                        ENGINEERED FEATURES (21)
                                                  │
                                    LEAKAGE AUDIT & SELECTION
                                                  │
                                        STRICT FEATURE SET (32)
                                                  │
                                    PREPROCESSING PIPELINE
                                                  │
    ├─→ Imputation (median/mode) ─────────────────┤
    ├─→ One-Hot Encoding (6→16) ──────────────────┤
    └─→ Standard Scaling (conditional) ────────────┘
                                                  │
                                    FINAL FEATURES (45)
                                                  │
                                    MODEL TRAINING
                                                  ↓
                                  PREDICTIONS (MAE: 1.70 days)
```

---

## Feature Engineering Impact

### **Model Performance Comparison**

| Model | Features Used | MAE (days) | RMSE (days) | R² | Improvement over Baseline |
|-------|---------------|------------|-------------|----|-------------------------|
| **Baseline Mean** | None | 1.94 | 2.49 | 0.000 | - |
| **Previous Cycle** | cycle_length_days | 2.30 | 2.87 | -0.132 | - |
| **Linear Regression** | 45 features | 1.74 | 2.24 | 0.266 | +10.3% |
| **Ridge** | 45 features | 1.74 | 2.24 | 0.266 | +10.3% |
| **Lasso** | 45 features | 1.94 | 2.62 | -0.006 | 0.0% |
| **Random Forest** | 45 features | 1.73 | 2.21 | 0.284 | +10.8% |
| **Gradient Boosting** ⭐ | 45 features | **1.70** | **2.18** | **0.306** | **+12.4%** |

### **Feature Importance (Top 5)**

| Rank | Feature | Type | Importance | Interpretation |
|------|---------|------|------------|----------------|
| 1 | pcos_diagnosed | Base | 0.250 | Strongest predictor |
| 2 | cycle_length_mean_last_5 | Engineered (Rolling) | 0.126 | Historical patterns |
| 3 | cycle_length_mean_last_3 | Engineered (Rolling) | 0.022 | Recent patterns |
| 4 | bmi | Base | 0.013 | Body composition |
| 5 | birth_control_use | Base | 0.008 | Medication effects |

---

## Key Design Decisions

### **Why These 7 Techniques?**

1. **Time-Series Appropriate**: Lag, rolling, and change features perfect for temporal data
2. **Leakage-Safe**: All techniques respect prediction-time availability
3. **Domain-Relevant**: Delta features capture medical/biological insights
4. **Seasonality**: Date/cyclical features capture temporal patterns
5. **Model-Optimized**: No unnecessary complexity for Gradient Boosting
6. **Interpretable**: All features can be explained to users
7. **Production-Ready**: Consistent preprocessing from training to deployment

### **Why NOT Other Techniques?**

| Technique | Not Used | Reason |
|-----------|----------|--------|
| IQR Winsorization | ❌ | Tree models robust, outliers might be genuine |
| Box-Cox Transformation | ❌ | No severe skewness, tree models don't need normality |
| MinMaxScaler | ❌ | StandardScaler better for regularization |
| Polynomial Features | ❌ | Gradient Boosting captures interactions automatically |
| Binning | ❌ | Loses information, trees handle continuous well |
| Target Encoding | ❌ | Risk of overfitting, one-hot sufficient |
| Log Transformation | ❌ | No severe skewness requiring it |

---

## Leakage Prevention in Feature Engineering

### **Leakage-Safe Implementation**

| Technique | Leakage Prevention Mechanism | Validation |
|-----------|----------------------------|------------|
| **Lag Features** | `shift(lag)` uses only past cycles | ✅ Verified |
| **Rolling Statistics** | `shift(1)` before rolling excludes current cycle | ✅ Verified |
| **Change Features** | `diff()` uses only previous cycle | ✅ Verified |
| **Delta Features** | Baseline values always available | ✅ Verified |
| **Interaction Features** | Derived from leakage-safe deltas | ✅ Verified |
| **Date Features** | Date always available at prediction time | ✅ Verified |
| **Cyclical Features** | Derived from date features | ✅ Verified |

### **Feature Set Safety**

- **Strict Feature Set**: 14 base features + 18 engineered = 32 features
- **All features**: Validated for prediction-time availability
- **Excluded features**: 6 features removed (direct leakage risk)
- **Production model**: Uses only strict feature set

---

## Summary

**Total Feature Engineering Techniques**: 7  
**Total Engineered Features Created**: 21  
**Total Features After Preprocessing**: 45  
**Leakage Safety**: 100% (all techniques leakage-safe)  
**Model Performance**: MAE = 1.70 days (12.4% improvement over baseline)

The feature engineering pipeline is **comprehensive, leakage-safe, and production-ready**, designed specifically for time-series health data prediction using tree-based models.

**Report Generated**: 2026-09-26  
**Project Repository**: CycleSense  
**Model Version**: 1.0  
**Best Model**: Histogram-based Gradient Boosting Regressor