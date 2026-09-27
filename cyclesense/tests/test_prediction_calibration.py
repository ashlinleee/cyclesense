"""Tests for person-specific prediction calibration."""

from datetime import date

from api.main import personalize_cycle_estimate
from api.schemas import CycleHistory, PredictionRequest, UserProfile


def build_request(history, current_cycle):
    """Create the smallest complete request needed for calibration tests."""
    return PredictionRequest(
        profile=UserProfile(
            age=28,
            bmi=22.8,
            diet_quality="Good",
            exercise_frequency="3-4 days/week",
            sleep_hours=7.0,
            caffeine_intake=1.5,
            water_intake_liters=2.3,
            alcohol_consumption="Occasionally",
            smoking_status="No",
            birth_control_use=0,
            pcos_diagnosed=0,
            stress_score_baseline=5.5,
        ),
        cycle_history=CycleHistory(
            cycle_length_days=current_cycle,
            prev_cycle_length=current_cycle,
            cycle_phase="Follicular",
            flow_level="Medium",
            pain_level=5,
            pms_symptoms="No",
            mood_score=7,
            stress_score_cycle=5.5,
            sleep_hours_cycle=7.0,
            energy_level=7,
            concentration_score=7,
            work_hours_lost=0.0,
            start_date=date(2026, 9, 1),
        ),
        historical_cycles=history,
    )


def test_recent_history_changes_a_mean_model_estimate():
    """A 28-day population estimate must not flatten distinct user histories."""
    short_cycle_user = personalize_cycle_estimate(
        28.0, build_request([21, 22, 20], 21.0)
    )
    long_cycle_user = personalize_cycle_estimate(
        28.0, build_request([40, 39, 41], 40.0)
    )

    assert short_cycle_user < 26.0
    assert long_cycle_user > 32.0
    assert long_cycle_user - short_cycle_user > 6.0


def test_estimate_stays_within_supported_cycle_range():
    """Calibration honours the API's validated cycle bounds."""
    estimate = personalize_cycle_estimate(
        50.0, build_request([45, 45, 45], 45.0)
    )

    assert 20.0 <= estimate <= 45.0
