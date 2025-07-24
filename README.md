# 🛒 Multimodal E-commerce Product Classification System

> **Advanced machine learning system with professional demo frontend combining computer vision and NLP for automated product categorization**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-CSS-38B2AC.svg)](https://tailwindcss.com)

## 🎯 Project Overview

This project demonstrates a **state-of-the-art multimodal classification system** with a **professional demo frontend** that automatically categorizes BestBuy products using both **product images** and **text descriptions**. The system combines multiple deep learning approaches with a beautiful, interactive web interface to showcase superior performance compared to single-modality approaches.

### ✨ Key Achievements

- **🏆 85%+ Accuracy** achieved on multimodal classification
- **🎭 Professional Demo Frontend** built with Next.js and Tailwind CSS
- **🚀 FastAPI Backend** serving real ML model predictions
- **🧠 Multiple AI Architectures** implemented and compared
- **📊 Interactive Performance Dashboard** with real-time visualizations
- **🔄 End-to-End Pipeline** from raw data to production-ready demo
- **📱 Responsive Design** with modern UI/UX practices

## 🌟 Demo Frontend Features

### **🎨 Interactive Web Interface**
- **Homepage**: Professional landing page showcasing project achievements
- **Live Demo**: Interactive product classification with image upload and text input
- **Performance Analytics**: Real-time charts and model comparison dashboards
- **Model Explorer**: Detailed view of all 12+ ML architectures implemented
- **Responsive Design**: Beautiful UI optimized for all devices

### **🔥 Key Frontend Highlights**
- **Real-time Classification**: Upload images or enter text for instant predictions
- **Multi-modal Interface**: Test text-only, image-only, and combined approaches
- **Performance Visualizations**: Interactive charts showing model comparisons
- **Model Architecture Gallery**: Explore ResNet, BERT, Vision Transformers, and more
- **Statistics Dashboard**: Live data from 49K+ products and 4K+ categories

## 🚀 Technical Highlights

### **🎯 Frontend Stack**
- **Next.js 14** - Modern React framework with App Router
- **Tailwind CSS** - Utility-first CSS framework for custom styling
- **TypeScript** - Type-safe development
- **Framer Motion** - Smooth animations and transitions
- **Recharts** - Interactive data visualizations
- **Heroicons** - Beautiful SVG icons

### **⚡ Backend Stack**
- **FastAPI** - High-performance Python API framework
- **Uvicorn** - ASGI server for async support
- **Pydantic** - Data validation and serialization
- **scikit-learn** - Machine learning utilities
- **Structured Logging** - Production-ready logging

### **Computer Vision Models Implemented:**
- **ResNet50/101** - Deep residual networks for robust feature extraction
- **DenseNet121/169** - Densely connected networks for efficient learning
- **ConvNeXt V2** - Modern CNN architecture with transformer-inspired design
- **Vision Transformer (ViT)** - Attention-based image processing
- **Swin Transformer** - Hierarchical vision transformer

### **Natural Language Processing:**
- **Sentence-BERT (MiniLM)** - Semantic text understanding
- **TF-IDF Vectorization** - Text similarity search
- **Transformer Embeddings** - State-of-the-art text representation
- **Cosine Similarity** - Semantic matching

### **Machine Learning Approaches:**
- **Classical ML**: Random Forest, Logistic Regression with advanced preprocessing
- **Deep Learning**: Custom MLP architectures with early fusion
- **Multimodal Fusion**: Combining visual and textual features for enhanced accuracy

## 🏗️ System Architecture

```
Frontend (Next.js + Tailwind)
         ↓
FastAPI Backend Server
         ↓
┌─────────────────┬─────────────────┐
│   Vision Models │   NLP Models    │
│   • ResNet50    │   • MiniLM      │
│   • ConvNeXt    │   • TF-IDF      │
│   • ViT         │   • BERT        │
└─────────────────┴─────────────────┘
         ↓
Real Product Data (49K+ products)
         ↓
┌─────────────────┬─────────────────┐
│ Vision Embeddings│ Text Embeddings │
│  ResNet50 Features│ MiniLM Features │
└─────────────────┴─────────────────┘
         ↓
   Similarity Search & Classification
```

## 🛠️ Demo Features

### **🔬 Interactive Classification Demo**
- **Multi-modal Input**: Upload images and/or enter text descriptions
- **Real-time Predictions**: Instant classification using actual ML models
- **Confidence Scores**: See prediction confidence levels
- **Model Comparison**: Compare text-only, image-only, and multimodal results

### **📈 Performance Analytics Dashboard**
- **Model Comparison Charts**: Interactive visualizations of all model performance
- **Radar Charts**: Multi-dimensional performance analysis
- **Category Performance**: Breakdown by product categories
- **Architecture Overview**: Performance of different ML approaches

### **🎨 Model Explorer**
- **12+ Model Architectures**: Detailed information on each model
- **Performance Metrics**: Accuracy, F1, Precision, Recall for each model
- **Interactive Filtering**: Filter models by type (Vision, NLP, Multimodal)
- **Architecture Details**: Technical specifications and use cases

## 📁 Project Structure

```
multimodel_ecommerce/
├── multimodal-ecommerce-demo/    # Next.js Frontend Demo
│   ├── src/app/                  # App Router pages
│   │   ├── page.tsx             # Homepage
│   │   ├── demo/                # Interactive demo
│   │   ├── performance/         # Analytics dashboard
│   │   └── models/              # Model explorer
│   ├── tailwind.config.js       # Tailwind CSS config
│   └── package.json            # Frontend dependencies
├── api_server.py                # FastAPI backend server
├── requirements-api.txt         # Backend dependencies
├── src/                         # Core ML implementation
│   ├── vision_embeddings_tf.py  # Computer vision models
│   ├── nlp_models.py           # NLP embedding models  
│   ├── classifiers_classic_ml.py# Classical ML algorithms
│   ├── classifiers_mlp.py      # Deep learning models
│   └── utils.py                # Data processing utilities
├── data/                        # Dataset
├── Embeddings/                  # Pre-computed embeddings
└── tests/                       # Comprehensive test suite
```

## 🚀 Quick Start

### **Prerequisites**
- Python 3.11+
- Node.js 18+
- 8GB+ RAM recommended
- GPU optional (CUDA/MPS supported)

### **Backend Setup**

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements-api.txt
   ```

2. **Start the FastAPI server:**
   ```bash
   python api_server.py
   ```
   Server runs on: `http://localhost:8000`

### **Frontend Setup**

1. **Navigate to demo directory:**
   ```bash
   cd multimodal-ecommerce-demo
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```
   Frontend runs on: `http://localhost:3000`

### **🎉 Access the Demo**
- **Homepage**: `http://localhost:3000`
- **Interactive Demo**: `http://localhost:3000/demo`
- **Performance Analytics**: `http://localhost:3000/performance`
- **Model Explorer**: `http://localhost:3000/models`
- **API Documentation**: `http://localhost:8000/docs`

## 📊 Performance Metrics

The multimodal system demonstrates superior performance across key metrics:

| Model Type | Accuracy | F1-Score | Precision | Recall |
|------------|----------|----------|-----------|--------|
| **Multimodal Fusion** | **85.2%** | **84.3%** | **84.7%** | **83.9%** |
| Vision Only (ResNet50) | 82.1% | 81.1% | 81.5% | 80.7% |
| Text Only (MiniLM) | 79.4% | 78.7% | 78.9% | 78.5% |

*Results show significant improvement with multimodal fusion over single-modality approaches.*

## 🎭 Demo Screenshots

### **Interactive Classification Demo**
- Multi-modal input interface
- Real-time prediction results
- Confidence score visualization
- Model comparison side-by-side

### **Performance Analytics Dashboard**
- Interactive bar charts and radar plots
- Model performance comparison
- Category-wise performance breakdown
- Architecture performance overview

### **Model Explorer**
- 12+ ML model architectures
- Detailed performance metrics
- Interactive filtering and selection
- Technical architecture details

## 🔍 Key Innovations

1. **Professional Demo Frontend**: Modern, responsive web interface built with Next.js
2. **Real ML Integration**: Actual model predictions using pre-computed embeddings
3. **Interactive Visualizations**: Real-time charts and performance analytics
4. **Multimodal Architecture**: Novel fusion of vision and language models
5. **Production-Ready API**: FastAPI backend with structured logging and error handling
6. **Scalable Design**: Component-based architecture for easy extension

## 🛡️ Code Quality

- **Type Safety**: Full TypeScript implementation
- **Modern Framework**: Next.js 14 with App Router
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Error Handling**: Comprehensive error boundaries and fallbacks
- **Performance Optimized**: Lazy loading and efficient data fetching
- **Production Ready**: Structured logging and monitoring

## 🎓 Technical Skills Demonstrated

### **Frontend Development**
- **Next.js 14**: App Router, Server Components, Modern React patterns
- **Tailwind CSS**: Custom design system, responsive layouts
- **TypeScript**: Type-safe development, interface design
- **Data Visualization**: Interactive charts with Recharts
- **Animation**: Smooth transitions with Framer Motion

### **Backend Development**
- **FastAPI**: High-performance async API development
- **Python**: Advanced data processing and ML integration
- **API Design**: RESTful endpoints with proper documentation
- **Error Handling**: Robust error management and logging

### **Machine Learning**
- **Deep Learning**: TensorFlow, PyTorch, Computer Vision, NLP
- **Classical ML**: scikit-learn, Feature Engineering, Model Selection
- **Data Science**: Pandas, NumPy, Statistical Analysis, Visualization
- **MLOps**: Model serving, Pipeline design, Performance monitoring

### **Software Engineering**
- **Full-Stack Development**: End-to-end application development
- **Modern Tooling**: Git, npm, pip, modern development workflows
- **Clean Architecture**: Component-based design, separation of concerns
- **Documentation**: Comprehensive README, code documentation

## 📚 Dependencies

### **Frontend Dependencies**
```json
{
  "next": "14.x",
  "react": "18.x",
  "tailwindcss": "3.x",
  "typescript": "5.x",
  "framer-motion": "latest",
  "recharts": "latest",
  "@heroicons/react": "latest"
}
```

### **Backend Dependencies**
```
fastapi==0.115.6
uvicorn[standard]==0.35.0
scikit-learn==1.3.2
pandas==2.2.3
numpy==1.26.4
pillow==10.4.0
structlog==23.2.0
```

## 🚀 Deployment

### **Frontend Deployment**
- **Vercel**: Optimized for Next.js applications
- **Netlify**: Alternative deployment platform
- **Docker**: Containerized deployment option

### **Backend Deployment**
- **FastAPI + Uvicorn**: Production ASGI server
- **Docker**: Container-based deployment
- **Cloud Platforms**: AWS, GCP, Azure compatible

## 🤝 Contributing

This project showcases production-ready full-stack development practices:

1. Fork the repository
2. Create feature branches
3. Implement changes with proper testing
4. Submit pull requests with detailed descriptions

## 📞 Contact & Portfolio

**Full-Stack AI/ML Engineer Portfolio Project**

*Demonstrating expertise in:*
- **Frontend**: Next.js, React, Tailwind CSS, TypeScript
- **Backend**: FastAPI, Python, API Development
- **Machine Learning**: Multimodal AI, Deep Learning, Model Serving
- **Full-Stack**: End-to-end application development

## 🎯 Live Demo

🌐 **[View Live Demo](https://your-demo-url.com)** (Coming Soon)

---

⭐ **Star this repository if you found the multimodal AI demo impressive!**

💼 **Available for full-stack AI/ML engineering opportunities**