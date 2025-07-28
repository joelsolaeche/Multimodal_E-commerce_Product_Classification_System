#!/bin/bash

# CRITICAL: Railway dynamic port handling - never hardcode ports
if [ -z "$PORT" ]; then
    PORT=8000
fi

echo "🚀 Starting Multimodal E-commerce API on port $PORT"
echo "Environment: ${RAILWAY_ENVIRONMENT:-local}"

# Start uvicorn with Railway's dynamic port
uvicorn api_server:app --host 0.0.0.0 --port $PORT 