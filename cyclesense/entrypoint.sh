#!/bin/bash

# CycleSense Docker Entrypoint
# Starts FastAPI backend with optimized cold start handling

set -e

echo "Starting CycleSense Application..."

PORT=${PORT:-8000}

# Set environment variables for production
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Create necessary directories
mkdir -p data/raw data/processed data/reference artifacts reports/figures

# Train model if not present (with timeout protection)
if [ ! -f artifacts/model.joblib ]; then
    echo "Model artifact not found. Training model..."
    # Use fast training script for quicker cold starts
    python src/train_fast.py || {
        echo "Fast training failed, will use fallback on first request"
    }
fi

# Start FastAPI backend server on Render PORT with timeout configurations
echo "Starting FastAPI on port ${PORT}..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --timeout-keep-alive 60 \
    --timeout-graceful-shutdown 30 \
    --workers 1 \
    --log-level info
