# CycleSense Leakage Audit Report

## Overview

This audit categorizes each feature based on prediction-time availability.
The key question for each feature: **Would this information actually be available
when predicting the next cycle?**

## Feature Categories

### SAFE_AT_PREDICTION_TIME

Features that are realistically available when making a prediction:
- User profile information (static)
- Current cycle length
- Previous cycle length (if recorded)
- Historical cycle patterns (engineered from past cycles)

### CONDITIONAL

Features that might be available depending on when prediction is made:
- Current cycle phase (only if predicting mid-cycle)
- Current symptoms (pain, mood, stress, etc.)
- Current measurements (sleep, energy, concentration)

### EXCLUDED_LEAKAGE

Features that would NOT be available at prediction time (direct leakage):
- Response to current cycle (prepared_before_period)
- Assessment of current cycle (overall_health_score)
- Current cycle logging quality (log_consistency_score)
- Hormonal measurements during current cycle
- Ovulation result (determined during current cycle)

### IDENTIFIER_ONLY

Features used only for grouping, not prediction:
- user_id (for grouping and splitting)
- cycle_number (for ordering)
- start_date (for temporal features)
- state (may be high-cardinality noise - evaluate)

## Audit Results

- Total features analyzed: 34
- Safe at prediction time: 14
- Conditional (timing-dependent): 10
- Excluded (leakage risk): 6
- Identifiers only: 4
- Uncategorized: 0

## Safe Features

- cycle_length_days
- prev_cycle_length
- age
- bmi
- diet_quality
- exercise_frequency
- sleep_hours
- caffeine_intake
- water_intake_liters
- alcohol_consumption
- smoking_status
- birth_control_use
- pcos_diagnosed
- stress_score_baseline

## Conditional Features

- cycle_phase
- flow_level
- pain_level
- pms_symptoms
- mood_score
- stress_score_cycle
- sleep_hours_cycle
- energy_level
- concentration_score
- work_hours_lost

## Excluded Features (Leakage Risk)

- estrogen_pgml
- progesterone_ngml
- ovulation_result
- overall_health_score
- log_consistency_score
- prepared_before_period

## Identifier Features

- user_id
- cycle_number
- start_date
- state

## Feature Sets for Experimentation

### Strict Feature Set

**14 features**
Only information realistically available before/at prediction time.
This is the defensible production feature set.

- age
- bmi
- diet_quality
- exercise_frequency
- sleep_hours
- caffeine_intake
- water_intake_liters
- alcohol_consumption
- smoking_status
- birth_control_use
- pcos_diagnosed
- stress_score_baseline
- cycle_length_days
- prev_cycle_length

### Extended Research Feature Set

**24 features**
Includes conditional features that may be available depending on prediction timing.
Still excludes direct target leakage.

- age
- bmi
- diet_quality
- exercise_frequency
- sleep_hours
- caffeine_intake
- water_intake_liters
- alcohol_consumption
- smoking_status
- birth_control_use
- pcos_diagnosed
- stress_score_baseline
- cycle_length_days
- prev_cycle_length
- cycle_phase
- flow_level
- pain_level
- pms_symptoms
- mood_score
- stress_score_cycle
- sleep_hours_cycle
- energy_level
- concentration_score
- work_hours_lost

### Full Feature Set

**25 features**
Includes state for evaluation (may be high-cardinality noise).

- age
- bmi
- diet_quality
- exercise_frequency
- sleep_hours
- caffeine_intake
- water_intake_liters
- alcohol_consumption
- smoking_status
- birth_control_use
- pcos_diagnosed
- stress_score_baseline
- cycle_length_days
- prev_cycle_length
- cycle_phase
- flow_level
- pain_level
- pms_symptoms
- mood_score
- stress_score_cycle
- sleep_hours_cycle
- energy_level
- concentration_score
- work_hours_lost
- state

## Recommendations

1. **Primary model**: Use the **strict feature set** for production.
   This ensures predictions are defensible and based on information
   that would realistically be available at prediction time.

2. **Research experiments**: Compare strict vs extended feature sets
   through MLflow to quantify the value of conditional features.

3. **State evaluation**: Test whether `state` provides useful
   generalizable signal or merely creates high-cardinality noise.

4. **Leakage prevention**: Never use excluded features in model training.
   These would create artificially good performance that doesn't
   generalize to real-world prediction scenarios.

5. **user_id handling**: Use `user_id` only for grouping, splitting,
   and temporal feature creation. Never as a direct predictive feature.

## Feature Engineering Plan

Based on this audit, the feature engineering will focus on:

1. **Historical cycle features** (leakage-safe):
   - Lag features: cycle_length_lag_1, cycle_length_lag_2, cycle_length_lag_3
   - Rolling statistics: mean, std, min, max over last 3 and 5 cycles
   - Cycle change: cycle_length_change, cycle_length_abs_change

2. **Profile features** (static, always available):
   - Age, BMI, lifestyle factors
   - Baseline stress score

3. **Cross-source features** (interactions):
   - stress_delta = stress_score_cycle - stress_score_baseline
   - sleep_delta = sleep_hours_cycle - sleep_hours
   - Interaction terms where statistically defensible

4. **Date features** (from start_date):
   - Month, quarter, day_of_year
   - Cyclical representations (sin/cos) if useful

All engineered features will be validated for leakage safety before
inclusion in the production model.