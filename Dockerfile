# Customer Intelligence & Campaign Orchestration Platform (CICOP)
# Multi-stage Docker build for production deployment

# Stage 1: Base Python environment
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    pkg-config \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data logs models config

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Generate sample data if not exists
RUN python -c "import os; \
    import runpy; \
    os.makedirs('data', exist_ok=True); \
    0 if os.path.exists('data/customer_insights.db') else runpy.run_path('scripts/generate_standalone_data.py')"

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["full"]

# Stage 2: Development environment
FROM python:3.11-slim as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter

# Copy source code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose ports
EXPOSE 8501 8000

# Default command
CMD ["python", "main.py"]

# Stage 3: Production environment
FROM python:3.11-slim as production

# Install production dependencies only
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn

# Copy source code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Default command for production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "main:app"]

# Stage 4: ML Model Serving
FROM base as ml-serving

# Install ML-specific dependencies
RUN pip install --no-cache-dir \
    tensorflow \
    torch \
    scikit-learn \
    mlflow \
    prometheus-client

# Copy ML models and source code
COPY src/ml/ ./src/ml/
COPY config/ ./config/
COPY models/ ./models/

# Create model serving directory
RUN mkdir -p /app/models

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose ML model serving port
EXPOSE 8001

# Default command for ML serving
CMD ["python", "-m", "src.ml.model_server"]

# Stage 5: API Gateway
FROM base as api-gateway

# Install API gateway dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    redis \
    prometheus-client

# Copy API source code
COPY src/api/ ./src/api/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose API gateway port
EXPOSE 8002

# Default command for API gateway
CMD ["uvicorn", "src.api.gateway:app", "--host", "0.0.0.0", "--port", "8002"]

# Stage 6: Worker Processes
FROM base as worker

# Install worker dependencies
RUN pip install --no-cache-dir \
    celery \
    redis \
    prometheus-client

# Copy worker source code
COPY src/workers/ ./src/workers/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Default command for worker processes
CMD ["celery", "-A", "src.workers.celery_app", "worker", "--loglevel=info"]
