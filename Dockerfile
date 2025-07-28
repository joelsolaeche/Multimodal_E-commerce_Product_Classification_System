# Multi-stage Railway-optimized Dockerfile with minimal footprint
FROM python:3.9-slim-bullseye as base

# Set working directory
WORKDIR /app

# Install minimal system dependencies (only what's needed for pandas/numpy/scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: Copy minimal requirements FIRST for layer caching optimization
COPY requirements-railway.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application files
COPY api_server.py /app/
COPY src/ /app/src/

# CRITICAL: Create necessary directories for data files
RUN mkdir -p /app/data/Raw /app/data/images /app/Embeddings

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# MANDATORY: Use Railway's dynamic PORT
EXPOSE $PORT

# Set environment variable to prevent Python from buffering output
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV USE_POSTGRES=true

# REQUIRED: Health check for monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import requests; requests.get('http://localhost:${PORT:-8000}/health', timeout=5)" || exit 1

# Use startup script
CMD ["/app/start.sh"]
