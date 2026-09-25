#!/bin/bash

# CycleSense Docker Entrypoint
# Starts FastAPI backend and Streamlit frontend

set -e

echo "Starting CycleSense Application..."

# Train model if not present
if [ ! -f artifacts/model.joblib ]; then
    echo "Model artifact not found. Training model..."
    python src/train.py
fi

# Start FastAPI backend in background
echo "Starting FastAPI on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be healthy
echo "Waiting for API to be ready..."
timeout=60
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is ready!"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

if [ $elapsed -ge $timeout ]; then
    echo "Warning: API health check timed out"
fi

# Start Streamlit frontend
echo "Starting Streamlit on port 8501..."
streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0

# Cleanup
wait $API_PID
