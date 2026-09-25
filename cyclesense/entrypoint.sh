#!/bin/bash

# CycleSense Docker Entrypoint
# Starts FastAPI backend and Streamlit frontend

set -e

echo "Starting CycleSense Application..."

PORT=${PORT:-8000}

# Train model if not present
if [ ! -f artifacts/model.joblib ]; then
    echo "Model artifact not found. Training model..."
    python src/train.py
fi

# Start FastAPI backend server on Render PORT
echo "Starting FastAPI on port ${PORT}..."
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
