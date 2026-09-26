FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set PYTHONPATH so python can locate modules in src/ and api/
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY ui/ ./ui/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY reports/ ./reports/

# Create necessary directories
RUN mkdir -p data/raw data/processed data/reference artifacts reports/figures

# Expose port
EXPOSE 8000

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Health check
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]
