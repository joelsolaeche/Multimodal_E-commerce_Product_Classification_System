# 🚀 Deployment Guide - Multimodal E-commerce AI Demo

This guide will help you deploy both the **Next.js frontend** and **FastAPI backend** to showcase your multimodal AI system.

## 🎯 Architecture Overview

```
Frontend (Next.js)     Backend (FastAPI)
     ↓                      ↓
   Vercel              Railway/Render
     ↓                      ↓
User Interface  ←→  AI Classification API
```

## 📋 Prerequisites

- GitHub account
- Vercel account (free)
- Railway account (free) OR Render account (free)

## 🔧 Option 1: Vercel + Railway (Recommended)

### **Step 1: Deploy Backend to Railway**

1. **Create Railway Account**: Go to [railway.app](https://railway.app)

2. **Deploy from GitHub**:
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `multimodal-ecommerce-ai-demo` repository
   - Railway will auto-detect Python/FastAPI

3. **Configure Environment**:
   - Railway will automatically set `RAILWAY_ENVIRONMENT=true`
   - Your backend will use mock data (perfect for demo)

4. **Get Your API URL**:
   - After deployment, Railway will provide a URL like: `https://your-app.railway.app`
   - Copy this URL - you'll need it for frontend

### **Step 2: Deploy Frontend to Vercel**

1. **Create Vercel Account**: Go to [vercel.com](https://vercel.com)

2. **Deploy from GitHub**:
   - Click "New Project" → Import your GitHub repo
   - Select the `multimodal-ecommerce-demo` folder as root directory

3. **Configure Environment Variables**:
   - In Vercel dashboard → Settings → Environment Variables
   - Add: `NEXT_PUBLIC_API_URL` = `https://your-railway-backend-url.railway.app`

4. **Deploy**:
   - Vercel will automatically build and deploy your Next.js app
   - You'll get a URL like: `https://your-frontend.vercel.app`

## 🔧 Option 2: All-in-One Railway Deployment

1. **Deploy to Railway**:
   - Create new project from your GitHub repo
   - Railway will detect both frontend and backend

2. **Configure Build**:
   - Set build command: `cd multimodal-ecommerce-demo && npm install && npm run build`
   - Set start command: `python api_server.py & cd multimodal-ecommerce-demo && npm start`

## 🌐 Option 3: Render Deployment

### **Backend on Render**:
1. Go to [render.com](https://render.com)
2. New Web Service → Connect GitHub repo
3. Runtime: Python 3.11
4. Build Command: `pip install -r requirements-api.txt`
5. Start Command: `python api_server.py`

### **Frontend on Render**:
1. New Static Site → Connect GitHub repo
2. Build Command: `cd multimodal-ecommerce-demo && npm install && npm run build`
3. Publish Directory: `multimodal-ecommerce-demo/.next`

## ⚙️ Environment Variables

### **Backend** (Railway/Render):
```
RAILWAY_ENVIRONMENT=true  # Auto-set by Railway
RENDER=true              # Auto-set by Render
PORT=8000               # Auto-set by platform
```

### **Frontend** (Vercel):
```
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

## 🎉 Demo Features in Production

Your deployed demo will showcase:

- ✅ **Interactive Classification**: Upload images and enter text
- ✅ **Real-time Predictions**: Using TF-IDF and similarity search
- ✅ **Performance Analytics**: Charts showing model comparisons
- ✅ **Model Explorer**: 12+ ML architectures with metrics
- ✅ **Professional UI**: Responsive design with animations
- ✅ **Mock Data**: 1000 demo products across 10 categories

## 🔗 Post-Deployment

1. **Test Your Demo**:
   - Visit your frontend URL
   - Try image classification
   - Test text classification
   - Check performance charts

2. **Update README**:
   - Add live demo links
   - Update with deployment URLs

3. **Share Your Work**:
   - Portfolio websites
   - LinkedIn posts
   - GitHub repository description

## 🛠️ Troubleshooting

### **Common Issues**:

1. **CORS Errors**: Backend automatically allows all origins for demo
2. **API Not Found**: Check `NEXT_PUBLIC_API_URL` environment variable
3. **Build Failures**: Ensure all dependencies are in requirements files
4. **Memory Issues**: Platforms have memory limits, mock data keeps it light

### **Logs**:
- **Railway**: View logs in Railway dashboard
- **Vercel**: View function logs in Vercel dashboard
- **Render**: View logs in Render dashboard

## 💡 Pro Tips

1. **Custom Domain**: Both Vercel and Railway support custom domains
2. **Monitoring**: Use built-in platform monitoring
3. **Scaling**: Upgrade to paid plans for better performance
4. **Analytics**: Add Google Analytics to track demo usage

## 🎯 Expected Performance

- **Frontend**: Lightning fast on Vercel's CDN
- **Backend**: ~500ms response time for classifications
- **Uptime**: 99.9% on both platforms
- **Cost**: $0 on free tiers (perfect for portfolio demos)

---

🚀 **Ready to deploy?** Follow the steps above and your multimodal AI demo will be live within 10 minutes! 