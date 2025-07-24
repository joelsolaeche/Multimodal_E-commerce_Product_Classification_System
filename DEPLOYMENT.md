# 🚀 Deployment Guide

## Quick Start (Development)

### 1. **Backend (FastAPI)**

```bash
# Install dependencies
pip install -r requirements-api.txt

# Start the FastAPI server
python api_server.py
# Server will run on http://localhost:8000
```

### 2. **Frontend (Next.js)**

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
# Frontend will run on http://localhost:3000
```

## 🌐 Production Deployment

### **Backend Options:**

#### **Option 1: Railway** (Recommended)
1. Connect your GitHub repo to Railway
2. Set environment variables:
   ```
   PORT=8000
   PYTHONPATH=/app
   ```
3. Railway will automatically deploy from `api_server.py`

#### **Option 2: Render**
1. Create new Web Service
2. Build Command: `pip install -r requirements-api.txt`
3. Start Command: `python api_server.py`

#### **Option 3: Heroku**
```bash
# Create Procfile
echo "web: python api_server.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### **Frontend: Vercel** (Recommended)

1. **Connect to Vercel:**
   ```bash
   # Install Vercel CLI
   npm i -g vercel
   
   # Deploy from frontend directory
   cd frontend
   vercel
   ```

2. **Environment Variables:**
   ```
   NEXT_PUBLIC_API_URL=https://your-api-url.railway.app
   ```

3. **Build Settings:**
   - Framework: Next.js
   - Root Directory: `frontend`
   - Build Command: `npm run build`

## 🔧 Configuration

### **CORS Setup**
Update `api_server.py` with your frontend URL:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend-url.vercel.app"
    ],
    # ...
)
```

### **Environment Variables**

**Backend (.env):**
```
PORT=8000
ENVIRONMENT=production
```

**Frontend (.env.local):**
```
NEXT_PUBLIC_API_URL=https://your-api-url.railway.app
```

## 📊 Monitoring & Analytics

### **Backend Monitoring**
```python
# Add to api_server.py
import logging
logging.basicConfig(level=logging.INFO)

# Health check endpoint available at /health
```

### **Frontend Analytics**
```bash
# Add Vercel Analytics
npm install @vercel/analytics

# Add to layout.tsx
import { Analytics } from '@vercel/analytics/react'
```

## 🚨 Troubleshooting

### **Common Issues:**

1. **CORS Errors:**
   - Ensure API URL is correct in frontend
   - Check CORS settings in FastAPI

2. **Model Loading Errors:**
   - Increase memory limits on hosting platform
   - Consider model optimization

3. **Build Failures:**
   - Check Node.js version (use 18+)
   - Verify all dependencies in package.json

### **Performance Optimization:**

1. **Backend:**
   - Use model caching
   - Implement connection pooling
   - Add request rate limiting

2. **Frontend:**
   - Enable Next.js Image Optimization
   - Use dynamic imports for heavy components
   - Implement service worker caching

## 🎯 Live Demo URLs

Once deployed, update these URLs:
- **Frontend:** https://your-app.vercel.app
- **API:** https://your-api.railway.app
- **API Docs:** https://your-api.railway.app/docs

## 📝 Notes

- **Free Tier Limits:** Both Railway and Vercel offer generous free tiers
- **Custom Domains:** Can be added to both services
- **SSL:** Automatically provided by both platforms
- **Scaling:** Both platforms support automatic scaling

---

**Total Setup Time:** ~15 minutes for basic deployment 🚀 