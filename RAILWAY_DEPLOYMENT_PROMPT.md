# 🚀 Railway Deployment Success Blueprint - AI Agent Instructions

## **MISSION**: Apply proven Railway deployment practices for guaranteed success

This prompt contains battle-tested practices from a successful Railway deployment. Follow these instructions to ensure your deployments work flawlessly on Railway.

---

## 🎯 **CORE PRINCIPLE: Dual Architecture Strategy**

**CRITICAL**: Implement TWO different deployment architectures:

### **Strategy 1: Local Development (Multi-Service)**
```yaml
# docker-compose.yml for local development
version: "3.8"
services:
  app:
    build: ./src/app
    ports: ["8000:80"]
    depends_on: [redis, model]
    environment: [REDIS_HOST=redis, REDIS_PORT=6379]
  
  model:
    build: ./src/models
    depends_on: [redis]
    environment: [REDIS_HOST=redis, REDIS_PORT=6379]
  
  redis:
    image: "redis:alpine"
```

### **Strategy 2: Railway Production (Single Container)**
```dockerfile
# Root Dockerfile for Railway (MANDATORY)
FROM python:3.9-slim as base
WORKDIR /app

# Copy requirements FIRST (caching optimization)
COPY src/app/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/app/ /app/
# CRITICAL: Bundle ML model in production
COPY src/models/predict_model.pkl /app/predict_model.pkl

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# MANDATORY: Use Railway's dynamic PORT
EXPOSE $PORT

# REQUIRED: Health check for monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import requests; requests.get('http://localhost:${PORT:-8000}/health', timeout=5)" || exit 1

CMD ["/app/start.sh"]
```

---

## 🔧 **MANDATORY Railway Configurations**

### **1. railway.json (REQUIRED)**
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "dockerfile",
    "dockerfilePath": "Dockerfile"
  }
}
```

### **2. Dynamic Port Handling (CRITICAL)**
```bash
# start.sh - NEVER hardcode ports
#!/bin/bash
if [ -z "$PORT" ]; then
    PORT=8000
fi
echo "Starting uvicorn on port $PORT"
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### **3. Environment-Aware Configuration (MANDATORY)**
```python
# Smart Redis connection - detects Railway vs local
redis_url = os.getenv('REDIS_URL') or os.getenv('REDIS_IP')
if redis_url and redis_url.startswith('redis://'):
    # Railway managed Redis
    db = redis.from_url(redis_url)
else:
    # Local development
    db = redis.Redis(
        host=os.getenv('REDIS_IP', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB_ID', 0))
    )
```

---

## 🏥 **CRITICAL: Health Check Implementation**

**MANDATORY**: Implement comprehensive health checks

```python
@app.get("/health", include_in_schema=False)
def health_check():
    """Railway deployment health monitoring"""
    try:
        # Test database/Redis connection
        db.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    # Check critical services
    model_status = "loaded" if ml_model is not None else "not loaded"
    
    return {
        "status": "healthy",
        "service": "your-service-name",
        "redis": redis_status,
        "ml_model": model_status,  # Replace with your services
        "version": "1.0.0"
    }
```

---

## 🐳 **Docker Optimization Rules**

### **Rule 1: Multi-Stage Builds**
```dockerfile
FROM python:3.9-slim as base
# Base setup...

FROM base as build
# Production optimizations...
```

### **Rule 2: Layer Caching (CRITICAL)**
```dockerfile
# ALWAYS copy requirements.txt FIRST
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# THEN copy application code
COPY ./ /app/
```

### **Rule 3: Use Official Base Images**
```dockerfile
# For FastAPI applications
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# For general Python apps
FROM python:3.9-slim
```

---

## 📁 **Required File Structure**

```
project/
├── railway.json              # Railway configuration
├── Dockerfile               # Production container
├── start.sh                 # Startup script
├── requirements.txt         # Dependencies
├── env.example             # Environment template
├── src/
│   ├── app/                # Web application
│   │   ├── Dockerfile      # Local dev container
│   │   ├── requirements.txt
│   │   └── app.py
│   ├── models/             # ML services (if applicable)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── model.pkl
│   └── docker-compose.yml  # Local development
```

---

## ⚙️ **Environment Variables Template**

```bash
# env.example
# === Railway Auto-Configured ===
# PORT=8000                    # Railway sets this
# REDIS_URL=redis://...        # Railway sets this

# === Your Configuration ===
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=60
API_SLEEP=0.5

# === Local Development ===
REDIS_IP=redis
REDIS_PORT=6379
REDIS_DB_ID=0
```

---

## 🎯 **Implementation Checklist**

### **Phase 1: Setup (MANDATORY)**
- [ ] Create `railway.json` with dockerfile builder
- [ ] Create root `Dockerfile` with single-container strategy
- [ ] Create `start.sh` with dynamic port handling
- [ ] Setup environment-aware configurations

### **Phase 2: Health & Monitoring (CRITICAL)**
- [ ] Implement `/health` endpoint
- [ ] Add Docker health checks
- [ ] Test all service dependencies in health check

### **Phase 3: Optimization (REQUIRED)**
- [ ] Multi-stage Docker builds
- [ ] Requirements.txt layer caching
- [ ] Use official base images
- [ ] Bundle all assets in production container

### **Phase 4: Local Development (RECOMMENDED)**
- [ ] Create `docker-compose.yml` for local testing
- [ ] Separate service containers for development
- [ ] Environment-specific configurations

---

## 🚨 **Common Failure Points to AVOID**

### **❌ NEVER DO:**
- Hardcode ports (use `$PORT`)
- Use localhost in production configs
- Skip health checks
- Forget to bundle ML models/assets
- Use development configurations in production

### **✅ ALWAYS DO:**
- Use Railway's environment variables
- Implement comprehensive health checks
- Bundle everything in production container
- Test both local and production configurations
- Use layer caching in Docker builds

---

## 🧪 **Testing Your Implementation**

### **Local Testing:**
```bash
docker-compose up --build
curl http://localhost:8000/health
```

### **Production Simulation:**
```bash
docker build -t test-app .
docker run -p 8000:8000 -e PORT=8000 test-app
curl http://localhost:8000/health
```

---

## 📋 **Success Criteria**

Your deployment is ready when:
- [ ] Health check returns 200 OK
- [ ] Application starts without hardcoded ports
- [ ] All dependencies are bundled in container
- [ ] Environment variables are properly handled
- [ ] Docker build completes without errors

---

**REMEMBER**: Railway works best with single, self-contained containers that adapt to their environment. Follow this blueprint exactly for guaranteed deployment success! 🚀 