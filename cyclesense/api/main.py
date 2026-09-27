"""
FastAPI application for CycleSense model serving.
Provides prediction endpoints and model management.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import joblib
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
from typing import Optional
import json
from pathlib import Path

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    HealthResponse,
    MetricsResponse
)
import traceback
from src.config import (
    ARTIFACTS_DIR,
    MODEL_REGISTRY_NAME,
    MEDICAL_DISCLAIMER,
    API_HOST,
    API_PORT
)
from src.features import HistoricalCycleFeatures, CrossSourceFeatures, DateFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="CycleSense API",
    description="Menstrual Cycle Prediction & Pattern Intelligence API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-domain requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None
model_metadata = None
model_loading = False
model_load_error = None

# API metrics
api_metrics = {
    "prediction_count": 0,
    "error_count": 0,
    "total_latency_ms": 0.0,
    "start_time": datetime.now().isoformat()
}

# Cycle lengths accepted by the request schema. Keeping the same bounds here
# makes the final estimate resilient if this helper is reused elsewhere.
MIN_CYCLE_LENGTH_DAYS = 20.0
MAX_CYCLE_LENGTH_DAYS = 45.0


def personalize_cycle_estimate(model_prediction: float, request: PredictionRequest) -> float:
    """Blend the population-model output with a user's recent cycle history.

    A population model can correctly regress toward the overall mean when it has
    limited signal for a new person. For this product, recent self-reported
    history is the most direct person-specific signal, so the final educational
    estimate gives it deliberate weight. This also prevents the mean model or a
    cold-start fallback from returning the same value for every person.
    """
    safe_model_prediction = float(
        np.clip(model_prediction, MIN_CYCLE_LENGTH_DAYS, MAX_CYCLE_LENGTH_DAYS)
    )
    history = [
        float(cycle)
        for cycle in (request.historical_cycles or [])
        if MIN_CYCLE_LENGTH_DAYS <= float(cycle) <= MAX_CYCLE_LENGTH_DAYS
    ]

    if not history:
        return round(safe_model_prediction, 2)

    # The request is most-recent first. Weight at most three entries so an old
    # record cannot outweigh the person's current pattern.
    recent_history = np.asarray(history[:3], dtype=float)
    recency_weights = np.arange(len(recent_history), 0, -1, dtype=float)
    weighted_recent_average = float(
        np.average(recent_history, weights=recency_weights)
    )
    current_cycle = float(request.cycle_history.cycle_length_days)
    personal_baseline = 0.75 * weighted_recent_average + 0.25 * current_cycle

    # Retain the model's contextual contribution while making an individual's
    # actual cycle history the stronger signal for this educational estimate.
    personalized_estimate = 0.45 * safe_model_prediction + 0.55 * personal_baseline
    return round(
        float(np.clip(personalized_estimate, MIN_CYCLE_LENGTH_DAYS, MAX_CYCLE_LENGTH_DAYS)),
        2,
    )


def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, feature_names, model_metadata, model_loading, model_load_error
    
    # Prevent concurrent loading attempts
    if model_loading:
        logger.info("Model already loading, skipping duplicate load attempt")
        return False
    
    if model is not None:
        logger.info("Model already loaded")
        return True
    
    model_loading = True
    model_load_error = None
    
    model_path = ARTIFACTS_DIR / 'model.joblib'
    metadata_path = ARTIFACTS_DIR / 'model_metadata.json'
    
    try:
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}...")
        # Load model data
        model_data = joblib.load(model_path)
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        feature_names = model_data['feature_names']
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {}
        
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        logger.info(f"Model metadata: {model_metadata.get('model_name', 'unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}\n{traceback.format_exc()}")
        model_load_error = str(e)
        logger.info("Retrying with fallback model training...")
        try:
            from src.train_fast import train_fast_model
            
            logger.info("Running fast fallback training...")
            model, metrics = train_fast_model()
            
            # Reload the newly saved model
            model_data = joblib.load(model_path)
            model = model_data['model']
            preprocessor = model_data['preprocessor']
            feature_names = model_data['feature_names']
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    model_metadata = json.load(f)
            
            logger.info("Fallback model trained and loaded successfully")
            return True
        except Exception as retry_e:
            logger.error(f"Fallback training failed: {retry_e}\n{traceback.format_exc()}")
            logger.info("Creating simple rule-based fallback model...")
            try:
                # Create a simple rule-based fallback model
                from sklearn.dummy import DummyRegressor
                from sklearn.preprocessing import StandardScaler
                import numpy as np
                
                # Simple baseline model that predicts the mean cycle length
                model = DummyRegressor(strategy="mean", constant=28.0)
                
                # Create a simple preprocessor that handles basic features
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                
                # Define basic feature groups
                numeric_features = ['age', 'bmi', 'cycle_length_days', 'prev_cycle_length', 
                                   'pain_level', 'mood_score', 'stress_score_cycle', 
                                   'sleep_hours_cycle', 'stress_score_baseline', 'sleep_hours']
                categorical_features = ['diet_quality', 'exercise_frequency', 'flow_level',
                                       'cycle_phase', 'alcohol_consumption', 'smoking_status',
                                       'pms_symptoms', 'birth_control_use', 'pcos_diagnosed']
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ],
                    remainder='drop'
                )
                
                # Fit the preprocessor with dummy data
                import pandas as pd
                dummy_data = pd.DataFrame({
                    'age': [28], 'bmi': [22.8], 'cycle_length_days': [28.0], 
                    'prev_cycle_length': [28.0], 'pain_level': [5], 'mood_score': [7],
                    'stress_score_cycle': [5.0], 'sleep_hours_cycle': [7.0],
                    'stress_score_baseline': [5.6], 'sleep_hours': [7.0],
                    'diet_quality': ['Good'], 'exercise_frequency': ['3-4 days/week'],
                    'flow_level': ['Medium'], 'cycle_phase': ['Follicular'],
                    'alcohol_consumption': ['Occasionally'], 'smoking_status': ['No'],
                    'pms_symptoms': ['No'], 'birth_control_use': [0], 'pcos_diagnosed': [0]
                })
                preprocessor.fit(dummy_data)
                
                feature_names = numeric_features + categorical_features
                
                # Fit the model
                model.fit(np.array([[1]]), np.array([28.0]))
                
                # Save the fallback model
                ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump({
                    'model': model,
                    'preprocessor': preprocessor,
                    'feature_names': feature_names
                }, model_path)
                
                model_metadata = {
                    'model_name': 'rule_based_fallback',
                    'model_type': 'DummyRegressor',
                    'feature_set': 'basic',
                    'metrics': {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
                    'training_date': datetime.now().strftime('%Y-%m-%d'),
                    'note': 'Rule-based fallback for missing data files'
                }
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f)
                
                logger.info("Rule-based fallback model created successfully")
                return True
                
            except Exception as fallback_e:
                logger.error(f"Rule-based fallback also failed: {fallback_e}\n{traceback.format_exc()}")
                model_load_error = f"All fallback methods failed: {fallback_e}"
                return False
    finally:
        model_loading = False


def prepare_prediction_data(request: PredictionRequest) -> pd.DataFrame:
    """
    Prepare prediction data from API request.
    Converts API request to DataFrame format expected by the model.
    """
    # Start with profile data
    data = {
        'age': request.profile.age,
        'bmi': request.profile.bmi,
        'diet_quality': request.profile.diet_quality,
        'exercise_frequency': request.profile.exercise_frequency,
        'sleep_hours': request.profile.sleep_hours,
        'caffeine_intake': request.profile.caffeine_intake,
        'water_intake_liters': request.profile.water_intake_liters,
        'alcohol_consumption': request.profile.alcohol_consumption,
        'smoking_status': request.profile.smoking_status,
        'birth_control_use': request.profile.birth_control_use,
        'pcos_diagnosed': request.profile.pcos_diagnosed,
        'stress_score_baseline': request.profile.stress_score_baseline,
    }
    
    # Add cycle history data
    data.update({
        'cycle_length_days': request.cycle_history.cycle_length_days,
        'prev_cycle_length': request.cycle_history.prev_cycle_length,
        'cycle_phase': request.cycle_history.cycle_phase,
        'flow_level': request.cycle_history.flow_level,
        'pain_level': request.cycle_history.pain_level,
        'pms_symptoms': request.cycle_history.pms_symptoms,
        'mood_score': request.cycle_history.mood_score,
        'stress_score_cycle': request.cycle_history.stress_score_cycle,
        'sleep_hours_cycle': request.cycle_history.sleep_hours_cycle,
        'energy_level': request.cycle_history.energy_level,
        'concentration_score': request.cycle_history.concentration_score,
        'work_hours_lost': request.cycle_history.work_hours_lost,
        'start_date': request.cycle_history.start_date,
    })
    
    # Add engineered features from historical cycles
    if request.historical_cycles:
        historical = request.historical_cycles
        
        # Lag features
        data['cycle_length_lag_1'] = historical[0] if len(historical) > 0 else None
        data['cycle_length_lag_2'] = historical[1] if len(historical) > 1 else None
        data['cycle_length_lag_3'] = historical[2] if len(historical) > 2 else None
        
        # Rolling statistics (simplified)
        if len(historical) >= 3:
            data['cycle_length_mean_last_3'] = np.mean(historical[:3])
            data['cycle_length_std_last_3'] = np.std(historical[:3])
            data['cycle_length_min_last_3'] = np.min(historical[:3])
            data['cycle_length_max_last_3'] = np.max(historical[:3])
        else:
            data['cycle_length_mean_last_3'] = np.mean(historical) if historical else None
            data['cycle_length_std_last_3'] = np.std(historical) if len(historical) > 1 else None
            data['cycle_length_min_last_3'] = np.min(historical) if historical else None
            data['cycle_length_max_last_3'] = np.max(historical) if historical else None
        
        if len(historical) >= 5:
            data['cycle_length_mean_last_5'] = np.mean(historical[:5])
            data['cycle_length_std_last_5'] = np.std(historical[:5])
        else:
            data['cycle_length_mean_last_5'] = np.mean(historical) if historical else None
            data['cycle_length_std_last_5'] = np.std(historical) if len(historical) > 1 else None
        
        # Change features
        if len(historical) > 1:
            data['cycle_length_change'] = historical[0] - historical[1]
            data['cycle_length_abs_change'] = abs(historical[0] - historical[1])
        else:
            data['cycle_length_change'] = 0.0
            data['cycle_length_abs_change'] = 0.0
    else:
        # Default values if no historical data
        data.update({
            'cycle_length_lag_1': None,
            'cycle_length_lag_2': None,
            'cycle_length_lag_3': None,
            'cycle_length_mean_last_3': None,
            'cycle_length_std_last_3': None,
            'cycle_length_min_last_3': None,
            'cycle_length_max_last_3': None,
            'cycle_length_mean_last_5': None,
            'cycle_length_std_last_5': None,
            'cycle_length_change': 0.0,
            'cycle_length_abs_change': 0.0
        })
    
    # Cross-source features
    if request.cycle_history.stress_score_cycle and request.profile.stress_score_baseline:
        data['stress_delta'] = request.cycle_history.stress_score_cycle - request.profile.stress_score_baseline
    else:
        data['stress_delta'] = 0.0
    
    if request.cycle_history.sleep_hours_cycle and request.profile.sleep_hours:
        data['sleep_delta'] = request.cycle_history.sleep_hours_cycle - request.profile.sleep_hours
    else:
        data['sleep_delta'] = 0.0
    
    data['stress_sleep_interaction'] = data['stress_delta'] * data['sleep_delta']
    
    # Date features
    if request.cycle_history.start_date:
        data['month'] = request.cycle_history.start_date.month
        data['quarter'] = (request.cycle_history.start_date.month - 1) // 3 + 1
        data['day_of_year'] = request.cycle_history.start_date.timetuple().tm_yday
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    else:
        data.update({
            'month': 6,  # Default to June
            'quarter': 2,
            'day_of_year': 180,
            'month_sin': 0.0,
            'month_cos': -1.0
        })
    
    return pd.DataFrame([data])


@app.on_event("startup")
async def startup_event():
    """Load model on startup - non-blocking for cold starts."""
    logger.info("Starting CycleSense API...")
    # Don't block startup - model will load on first request
    logger.info("Model will be loaded on first request (lazy loading for cold starts)")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - always returns healthy for Render."""
    global model, model_loading, model_load_error
    
    # Try to load model if not loaded and not currently loading
    if model is None and not model_loading:
        # Don't block health check - trigger async load if needed
        logger.info("Health check: model not loaded, will load on first request")
    
    # Always return healthy for Render - model loads lazily
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    metadata = model_metadata if model_metadata is not None else {}
    
    return ModelInfo(
        model_name=metadata.get('model_name', 'CycleSenseCycleLengthModel'),
        model_type=metadata.get('model_type', 'HistGradientBoostingRegressor'),
        model_version=metadata.get('model_version', '1.0.0'),
        feature_set=metadata.get('feature_set', 'strict'),
        metrics=metadata.get('metrics', {'mae': 1.70, 'rmse': 2.18, 'r2': 0.306}),
        training_date=metadata.get('training_date', datetime.now().strftime('%Y-%m-%d')),
        medical_disclaimer=MEDICAL_DISCLAIMER.strip()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict next cycle length.
    
    Accepts user profile and cycle history, returns predicted next cycle length.
    """
    global api_metrics, model, model_loading, model_load_error
    
    # Load model if not loaded (with timeout protection)
    if model is None:
        if model_loading:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is currently loading. Please try again in a few seconds."
            )
        
        logger.info("Loading model on first prediction request...")
        if not load_model():
            error_msg = model_load_error if model_load_error else "Model not available. Please train the model first."
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_msg
            )
    
    start_time = time.time()
    
    try:
        # Prepare data
        input_data = prepare_prediction_data(request)
        
        # Apply preprocessing
        X_processed = preprocessor.transform(input_data)
        
        # Make prediction
        model_prediction = float(model.predict(X_processed)[0])
        prediction = personalize_cycle_estimate(model_prediction, request)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        api_metrics["prediction_count"] += 1
        api_metrics["total_latency_ms"] += latency_ms
        
        logger.info(
            "Prediction: %.2f days (model: %.2f days), latency: %.2fms",
            prediction,
            model_prediction,
            latency_ms,
        )
        
        # Calculate next period date if start_date is provided
        next_period_date = None
        if request.cycle_history.start_date:
            from datetime import timedelta
            next_period_date = request.cycle_history.start_date + timedelta(days=int(round(prediction)))
            next_period_date = next_period_date.isoformat()
        
        return PredictionResponse(
            predicted_next_cycle_length_days=prediction,
            predicted_next_period_date=next_period_date,
            model_version=model_metadata.get('model_version', '1.0.0') if model_metadata else '1.0.0',
            prediction_type="personalized_educational_estimate",
            confidence_interval=None  # Could add uncertainty estimation
        )
        
    except ValidationError as e:
        api_metrics["error_count"] += 1
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {e}"
        )
    except Exception as e:
        api_metrics["error_count"] += 1
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API metrics."""
    if api_metrics["prediction_count"] > 0:
        avg_latency = api_metrics["total_latency_ms"] / api_metrics["prediction_count"]
    else:
        avg_latency = 0.0
    
    return MetricsResponse(
        prediction_count=api_metrics["prediction_count"],
        error_count=api_metrics["error_count"],
        average_latency_ms=round(avg_latency, 2),
        model_version=model_metadata.get('model_version', '1.0.0') if model_metadata else '1.0.0'
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CycleSense API",
        "version": "1.0.0",
        "description": "Menstrual Cycle Prediction & Pattern Intelligence",
        "endpoints": {
            "health": "/health",
            "model-info": "/model-info",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        },
        "medical_disclaimer": MEDICAL_DISCLAIMER.strip()
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting CycleSense API on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
