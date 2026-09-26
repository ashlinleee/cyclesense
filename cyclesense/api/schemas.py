"""
Pydantic schemas for CycleSense API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class UserProfile(BaseModel):
    """User profile information."""
    age: int = Field(..., ge=18, le=50, description="User age (18-50)")
    bmi: float = Field(..., ge=15.0, le=40.0, description="Body Mass Index")
    diet_quality: str = Field(..., description="Diet quality: Poor, Fair, Good, Excellent")
    exercise_frequency: str = Field(..., description="Exercise frequency: 1-2 days/week, 3-4 days/week, 5-6 days/week")
    sleep_hours: float = Field(..., ge=4.0, le=10.0, description="Average daily sleep hours")
    caffeine_intake: float = Field(..., ge=0.0, le=5.0, description="Daily caffeine intake (cups)")
    water_intake_liters: float = Field(..., ge=0.5, le=5.0, description="Daily water intake (liters)")
    alcohol_consumption: str = Field(..., description="Alcohol consumption: Never, Occasionally, Weekly")
    smoking_status: str = Field(..., description="Smoking status: No, Yes")
    birth_control_use: int = Field(..., ge=0, le=1, description="Birth control use: 0 or 1")
    pcos_diagnosed: int = Field(..., ge=0, le=1, description="PCOS diagnosis: 0 or 1")
    stress_score_baseline: float = Field(..., ge=1.0, le=10.0, description="Baseline stress score (1-10)")


class CycleHistory(BaseModel):
    """Historical cycle information."""
    cycle_length_days: float = Field(..., ge=20.0, le=45.0, description="Current cycle length in days")
    prev_cycle_length: Optional[float] = Field(None, description="Previous cycle length in days")
    cycle_phase: Optional[str] = Field(None, description="Current cycle phase")
    flow_level: Optional[str] = Field(None, description="Flow level: Light, Medium, Heavy")
    pain_level: Optional[int] = Field(None, ge=1, le=10, description="Pain level (1-10)")
    pms_symptoms: Optional[str] = Field(None, description="PMS symptoms: Yes, No")
    mood_score: Optional[int] = Field(None, ge=1, le=10, description="Mood score (1-10)")
    stress_score_cycle: Optional[float] = Field(None, ge=1.0, le=10.0, description="Current cycle stress score")
    sleep_hours_cycle: Optional[float] = Field(None, ge=4.0, le=10.0, description="Current cycle sleep hours")
    energy_level: Optional[int] = Field(None, ge=1, le=10, description="Energy level (1-10)")
    concentration_score: Optional[int] = Field(None, ge=1, le=10, description="Concentration score (1-10)")
    work_hours_lost: Optional[float] = Field(None, ge=0.0, le=10.0, description="Work hours lost")
    start_date: Optional[date] = Field(None, description="Cycle start date")


class PredictionRequest(BaseModel):
    """Prediction request with profile and cycle history."""
    profile: UserProfile
    cycle_history: CycleHistory
    historical_cycles: Optional[List[float]] = Field(
        None, 
        description="List of previous cycle lengths for lag features (most recent first)"
    )


class PredictionResponse(BaseModel):
    """Prediction response."""
    predicted_next_cycle_length_days: float = Field(..., description="Predicted next cycle length in days")
    predicted_next_period_date: Optional[str] = Field(None, description="Predicted next period date (YYYY-MM-DD)")
    model_version: str = Field(..., description="Model version")
    prediction_type: str = Field(..., description="Type of prediction: educational_estimate")
    confidence_interval: Optional[tuple] = Field(None, description="Confidence interval (if available)")


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    model_version: str
    feature_set: str
    metrics: dict
    training_date: str
    medical_disclaimer: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    """API metrics response."""
    prediction_count: int
    error_count: int
    average_latency_ms: float
    model_version: str
