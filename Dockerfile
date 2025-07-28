# Multi-stage Railway-optimized Dockerfile
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    zlib1g-dev \
    libjpeg-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: Copy requirements FIRST for layer caching optimization
COPY requirements-api.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application files
COPY api_server.py /app/
COPY src/ /app/src/

# CRITICAL: Bundle all necessary data files for production
# Create data structure and copy essential files
RUN mkdir -p /app/data/Raw /app/Embeddings

# CRITICAL: Bundle essential data files for production
# Copy only small essential files (categories.json)
COPY data/Raw/ /app/data/Raw/
# Large embedding files are handled with mock data in production via load_data() function

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# MANDATORY: Use Railway's dynamic PORT
EXPOSE $PORT

# Set environment variable to prevent Python from buffering output
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# REQUIRED: Health check for monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import requests; requests.get('http://localhost:${PORT:-8000}/health', timeout=5)" || exit 1

# Use startup script
CMD ["/app/start.sh"]
