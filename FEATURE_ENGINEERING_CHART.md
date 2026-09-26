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