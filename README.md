# Multimodal E-commerce Product Classification

This project implements a multimodal (text + image) product classification system for e-commerce applications.

## 🆕 Real Data Implementation

The project now supports real data using PostgreSQL on Railway. See the [Real Data Deployment Guide](DEPLOYMENT_GUIDE.md) for detailed instructions.

### Key Features:

- PostgreSQL database integration for product data
- Support for large embedding files (text and vision)
- Automatic fallback to mock data when database is unavailable
- Scalable architecture for Railway deployment

## Project Structure

- `api_server.py`: FastAPI server for predictions
- `src/`: Core ML model implementations
- `data/`: Product data and images
- `Embeddings/`: Pre-computed embeddings for text and images
- `multimodal-ecommerce-demo/`: Next.js frontend
- `import_data_to_postgres.py`: Script to import data to PostgreSQL

## Setup Instructions

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements-api.txt

# Start the API server
python api_server.py
```

### 2. Database Setup

```bash
# Set PostgreSQL connection string
export DATABASE_URL="postgresql://postgres:password@localhost:5432/postgres"

# Import data to PostgreSQL
python import_data_to_postgres.py
```

### 3. Railway Deployment

See the [Deployment Guide](DEPLOYMENT_GUIDE.md) for detailed instructions on deploying to Railway with PostgreSQL.

## API Endpoints

- `/api/classify/text`: Text-based product classification
- `/api/classify/image`: Image-based product classification
- `/api/classify/multimodal`: Combined text + image classification
- `/api/products`: Get product listings
- `/health`: API health check

## Frontend Demo

The Next.js frontend demo showcases the multimodal classification capabilities:

```bash
# Start the frontend
cd multimodal-ecommerce-demo
npm install
npm run dev
```

## Data Management Options

1. **PostgreSQL on Railway (Recommended)**:
   - Store product data (18MB) and sample embeddings
   - Free tier includes 1GB storage
   - Automatic backups and scaling

2. **Google Cloud Storage**:
   - Store large embedding files (800MB+)
   - Download on-demand or at startup
   - Free tier includes 5GB storage

3. **Hybrid Approach**:
   - Store product data in PostgreSQL
   - Store embeddings in cloud storage
   - Cache embeddings for performance

## Performance Considerations

- Railway free tier: 1GB PostgreSQL storage
- Recommended approach: Store product data + sample embeddings
- For full embeddings: Use cloud storage or upgrade to hobby tier

## License

MIT