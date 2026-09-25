"""
Leakage audit for CycleSense features.
Categorizes features based on prediction-time availability.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List

from src.config import REPORTS_DIR
from src.data import load_and_join_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Feature categorization based on prediction-time availability
FEATURE_CATEGORIES = {
    "SAFE_AT_PREDICTION_TIME": [
        # User profile (static)
        "age",
        "bmi",
        "diet_quality",
        "exercise_frequency",
        "sleep_hours",
        "caffeine_intake",
        "water_intake_liters",
        "alcohol_consumption",
        "smoking_status",
        "birth_control_use",
        "pcos_diagnosed",
        "stress_score_baseline",
        
        # Current cycle info (available at prediction time)
        "cycle_length_days",  # Current cycle length
        "prev_cycle_length",  # Previous cycle length (if recorded)
        
        # Historical cycle measurements (would be engineered from past cycles)
        # These will be created in feature engineering:
        # "cycle_length_lag_1",
        # "cycle_length_lag_2", 
        # "cycle_length_lag_3",
        # "cycle_length_mean_last_3",
        # "cycle_length_std_last_3",
        # "cycle_length_mean_last_5",
        # "cycle_length_std_last_5",
    ],
    
    "CONDITIONAL": [
        # These might be available depending on when prediction is made
        "cycle_phase",  # Only if predicting mid-cycle
        "flow_level",   # Only if current cycle has started
        "pain_level",   # Only if symptoms have started
        "pms_symptoms", # Only if PMS phase has started
        "mood_score",   # Only if tracking current symptoms
        "stress_score_cycle",  # Only if tracking current stress
        "sleep_hours_cycle",   # Only if tracking current sleep
        "energy_level",        # Only if tracking current energy
        "concentration_score", # Only if tracking current concentration
        "work_hours_lost",     # Only if work impact has occurred
    ],
    
    "EXCLUDED_LEAKAGE": [
        # These would NOT be available at prediction time
        "prepared_before_period",  # Response to current cycle
        "overall_health_score",   # Assessment of current cycle
        "log_consistency_score",  # Based on current cycle logging
        "estrogen_pgml",          # Hormonal measurement during current cycle
        "progesterone_ngml",      # Hormonal measurement during current cycle
        "ovulation_result",       # Determined during current cycle
    ],
    
    "IDENTIFIER_ONLY": [
        "user_id",       # For grouping only, not prediction
        "cycle_number",  # For ordering only, not prediction
        "start_date",    # For temporal features only, not direct prediction
        "state",         # May be high-cardinality noise - evaluate
    ]
}


def perform_leakage_audit(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive leakage audit on all features.
    
    For each feature, answer: "Would this information actually be available 
    when predicting the next cycle?"
    """
    logger.info("=== LEAKAGE AUDIT ===")
    
    audit_report = {
        "total_features": len(df.columns),
        "safe_features": [],
        "conditional_features": [],
        "excluded_features": [],
        "identifier_features": [],
        "uncategorized": []
    }
    
    all_columns = set(df.columns)
    
    # Categorize each column
    for col in df.columns:
        if col in FEATURE_CATEGORIES["SAFE_AT_PREDICTION_TIME"]:
            audit_report["safe_features"].append(col)
            logger.info(f"✓ SAFE: {col} - Available at prediction time")
        elif col in FEATURE_CATEGORIES["CONDITIONAL"]:
            audit_report["conditional_features"].append(col)
            logger.info(f"⚠ CONDITIONAL: {col} - Depends on prediction timing")
        elif col in FEATURE_CATEGORIES["EXCLUDED_LEAKAGE"]:
            audit_report["excluded_features"].append(col)
            logger.info(f"✗ EXCLUDED: {col} - Leakage risk (not available at prediction time)")
        elif col in FEATURE_CATEGORIES["IDENTIFIER_ONLY"]:
            audit_report["identifier_features"].append(col)
            logger.info(f"⊘ IDENTIFIER: {col} - For grouping/identification only")
        else:
            audit_report["uncategorized"].append(col)
            logger.warning(f"? UNCATEGORIZED: {col} - Needs manual review")
    
    # Generate summary
    logger.info(f"\n=== LEAKAGE AUDIT SUMMARY ===")
    logger.info(f"Total features: {audit_report['total_features']}")
    logger.info(f"Safe at prediction time: {len(audit_report['safe_features'])}")
    logger.info(f"Conditional (timing-dependent): {len(audit_report['conditional_features'])}")
    logger.info(f"Excluded (leakage risk): {len(audit_report['excluded_features'])}")
    logger.info(f"Identifiers only: {len(audit_report['identifier_features'])}")
    logger.info(f"Uncategorized: {len(audit_report['uncategorized'])}")
    
    return audit_report


def create_feature_sets() -> Dict[str, List[str]]:
    """
    Create different feature sets for experimentation.
    
    Returns:
        Dictionary with feature set names and their feature lists
    """
    logger.info("=== CREATING FEATURE SETS ===")
    
    # Strict feature set - only information realistically available before/at prediction time
    strict_features = FEATURE_CATEGORIES["SAFE_AT_PREDICTION_TIME"].copy()
    
    # Extended research feature set - includes conditional features
    # May include additional current-cycle measurements, but still excludes direct target leakage
    extended_features = strict_features + FEATURE_CATEGORIES["CONDITIONAL"].copy()
    
    # Full feature set - includes everything except identifiers and excluded leakage
    full_features = (strict_features + 
                    FEATURE_CATEGORIES["CONDITIONAL"].copy() +
                    ["state"])  # Evaluate state separately
    
    feature_sets = {
        "strict": strict_features,
        "extended": extended_features,
        "full": full_features
    }
    
    for set_name, features in feature_sets.items():
        logger.info(f"{set_name.upper()} feature set: {len(features)} features")
    
    return feature_sets


def write_leakage_audit_report(audit_report: Dict, feature_sets: Dict[str, List[str]]):
    """Write comprehensive leakage audit report to markdown."""
    report_path = REPORTS_DIR / 'leakage_audit.md'
    
    report_lines = [
        "# CycleSense Leakage Audit Report",
        "",
        "## Overview",
        "",
        "This audit categorizes each feature based on prediction-time availability.",
        "The key question for each feature: **Would this information actually be available",
        "when predicting the next cycle?**",
        "",
        "## Feature Categories",
        "",
        "### SAFE_AT_PREDICTION_TIME",
        "",
        "Features that are realistically available when making a prediction:",
        "- User profile information (static)",
        "- Current cycle length",
        "- Previous cycle length (if recorded)",
        "- Historical cycle patterns (engineered from past cycles)",
        "",
        "### CONDITIONAL",
        "",
        "Features that might be available depending on when prediction is made:",
        "- Current cycle phase (only if predicting mid-cycle)",
        "- Current symptoms (pain, mood, stress, etc.)",
        "- Current measurements (sleep, energy, concentration)",
        "",
        "### EXCLUDED_LEAKAGE",
        "",
        "Features that would NOT be available at prediction time (direct leakage):",
        "- Response to current cycle (prepared_before_period)",
        "- Assessment of current cycle (overall_health_score)",
        "- Current cycle logging quality (log_consistency_score)",
        "- Hormonal measurements during current cycle",
        "- Ovulation result (determined during current cycle)",
        "",
        "### IDENTIFIER_ONLY",
        "",
        "Features used only for grouping, not prediction:",
        "- user_id (for grouping and splitting)",
        "- cycle_number (for ordering)",
        "- start_date (for temporal features)",
        "- state (may be high-cardinality noise - evaluate)",
        "",
        "## Audit Results",
        "",
        f"- Total features analyzed: {audit_report['total_features']}",
        f"- Safe at prediction time: {len(audit_report['safe_features'])}",
        f"- Conditional (timing-dependent): {len(audit_report['conditional_features'])}",
        f"- Excluded (leakage risk): {len(audit_report['excluded_features'])}",
        f"- Identifiers only: {len(audit_report['identifier_features'])}",
        f"- Uncategorized: {len(audit_report['uncategorized'])}",
        "",
        "## Safe Features",
        "",
    ]
    
    for feature in audit_report['safe_features']:
        report_lines.append(f"- {feature}")
    
    report_lines.extend([
        "",
        "## Conditional Features",
        ""
    ])
    
    for feature in audit_report['conditional_features']:
        report_lines.append(f"- {feature}")
    
    report_lines.extend([
        "",
        "## Excluded Features (Leakage Risk)",
        ""
    ])
    
    for feature in audit_report['excluded_features']:
        report_lines.append(f"- {feature}")
    
    report_lines.extend([
        "",
        "## Identifier Features",
        ""
    ])
    
    for feature in audit_report['identifier_features']:
        report_lines.append(f"- {feature}")
    
    if audit_report['uncategorized']:
        report_lines.extend([
            "",
            "## Uncategorized Features",
            ""
        ])
        for feature in audit_report['uncategorized']:
            report_lines.append(f"- {feature}")
    
    report_lines.extend([
        "",
        "## Feature Sets for Experimentation",
        "",
        "### Strict Feature Set",
        "",
        f"**{len(feature_sets['strict'])} features**",
        "Only information realistically available before/at prediction time.",
        "This is the defensible production feature set.",
        "",
    ])
    
    for feature in feature_sets['strict']:
        report_lines.append(f"- {feature}")
    
    report_lines.extend([
        "",
        "### Extended Research Feature Set",
        "",
        f"**{len(feature_sets['extended'])} features**",
        "Includes conditional features that may be available depending on prediction timing.",
        "Still excludes direct target leakage.",
        "",
    ])
    
    for feature in feature_sets['extended']:
        report_lines.append(f"- {feature}")
    
    report_lines.extend([
        "",
        "### Full Feature Set",
        "",
        f"**{len(feature_sets['full'])} features**",
        "Includes state for evaluation (may be high-cardinality noise).",
        "",
    ])
    
    for feature in feature_sets['full']:
        report_lines.append(f"- {feature}")
    
    report_lines.extend([
        "",
        "## Recommendations",
        "",
        "1. **Primary model**: Use the **strict feature set** for production.",
        "   This ensures predictions are defensible and based on information",
        "   that would realistically be available at prediction time.",
        "",
        "2. **Research experiments**: Compare strict vs extended feature sets",
        "   through MLflow to quantify the value of conditional features.",
        "",
        "3. **State evaluation**: Test whether `state` provides useful",
        "   generalizable signal or merely creates high-cardinality noise.",
        "",
        "4. **Leakage prevention**: Never use excluded features in model training.",
        "   These would create artificially good performance that doesn't",
        "   generalize to real-world prediction scenarios.",
        "",
        "5. **user_id handling**: Use `user_id` only for grouping, splitting,",
        "   and temporal feature creation. Never as a direct predictive feature.",
        "",
        "## Feature Engineering Plan",
        "",
        "Based on this audit, the feature engineering will focus on:",
        "",
        "1. **Historical cycle features** (leakage-safe):",
        "   - Lag features: cycle_length_lag_1, cycle_length_lag_2, cycle_length_lag_3",
        "   - Rolling statistics: mean, std, min, max over last 3 and 5 cycles",
        "   - Cycle change: cycle_length_change, cycle_length_abs_change",
        "",
        "2. **Profile features** (static, always available):",
        "   - Age, BMI, lifestyle factors",
        "   - Baseline stress score",
        "",
        "3. **Cross-source features** (interactions):",
        "   - stress_delta = stress_score_cycle - stress_score_baseline",
        "   - sleep_delta = sleep_hours_cycle - sleep_hours",
        "   - Interaction terms where statistically defensible",
        "",
        "4. **Date features** (from start_date):",
        "   - Month, quarter, day_of_year",
        "   - Cyclical representations (sin/cos) if useful",
        "",
        "All engineered features will be validated for leakage safety before",
        "inclusion in the production model.",
    ])
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Leakage audit report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    # Load data
    df = load_and_join_datasets()
    
    # Perform leakage audit
    audit_report = perform_leakage_audit(df)
    
    # Create feature sets
    feature_sets = create_feature_sets()
    
    # Write report
    write_leakage_audit_report(audit_report, feature_sets)
    
    print("\n=== LEAKAGE AUDIT COMPLETE ===")
    print(f"Safe features: {len(audit_report['safe_features'])}")
    print(f"Excluded features: {len(audit_report['excluded_features'])}")
    print(f"Report saved to: {REPORTS_DIR / 'leakage_audit.md'}")
