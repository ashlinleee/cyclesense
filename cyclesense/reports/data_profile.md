# CycleSense Data Profile Report

Generated: 2026-09-26 00:03:17

---

## Period_Log.csv Analysis

### Dimensions
- Rows: 17,976
- Columns: 21
- Memory: 8.62 MB

### Data Types
- float64: 8
- int64: 7
- object: 6

### Missing Values
- prev_cycle_length: 2000 (11.13%)

### Duplicates
- Duplicate rows: 0 (0.0%)

---

## User_Profile.csv Analysis

### Dimensions
- Rows: 2,000
- Columns: 14
- Memory: 0.90 MB

### Data Types
- object: 6
- float64: 5
- int64: 3

### Missing Values
- exercise_frequency: 385 (19.25%)

---

## Merged Dataset Analysis

### Dimensions
- Rows: 17,976
- Columns: 34

### User Analysis
- Unique users: 2,000
- Mean cycles per user: 8.99
- Median cycles per user: 9.00
- Min cycles: 6
- Max cycles: 12

### Categorical Cardinality
- user_id: 2000 unique values
- start_date: 407 unique values
- cycle_phase: 3 unique values
- flow_level: 3 unique values
- pms_symptoms: 2 unique values
- ovulation_result: 2 unique values
- state: 50 unique values
- diet_quality: 4 unique values
- exercise_frequency: 3 unique values
- alcohol_consumption: 3 unique values
- smoking_status: 2 unique values

### Numerical Distributions

cycle_length_days:
  - Mean: 27.82
  - Median: 28.00
  - Std: 2.40
  - Min: 22.00
  - Max: 44.00
  - Skew: 0.71

age:
  - Mean: 25.79
  - Median: 26.00
  - Std: 4.92
  - Min: 16.00
  - Max: 43.00
  - Skew: 0.10

bmi:
  - Mean: 22.99
  - Median: 22.80
  - Std: 4.22
  - Min: 16.00
  - Max: 41.20
  - Skew: 0.33

stress_score_cycle:
  - Mean: 5.75
  - Median: 5.80
  - Std: 1.77
  - Min: 1.00
  - Max: 10.00
  - Skew: -0.07

sleep_hours_cycle:
  - Mean: 6.96
  - Median: 7.00
  - Std: 1.32
  - Min: 4.50
  - Max: 10.00
  - Skew: 0.10

### Date Range
- Date range: 2024-01-01 00:00:00 to 2025-04-05 00:00:00
- Span: 460 days

### Correlation Analysis

### Outlier Analysis (IQR Method)

cycle_length_days:
  - Outliers: 374 (2.08%)
  - Bounds: [21.50, 33.50]

age:
  - Outliers: 18 (0.10%)
  - Bounds: [11.50, 39.50]

bmi:
  - Outliers: 73 (0.41%)
  - Bounds: [10.50, 35.30]

stress_score_cycle:
  - Outliers: 0 (0.00%)
  - Bounds: [0.75, 10.75]