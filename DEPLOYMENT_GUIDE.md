# Real Data Deployment Guide

This guide explains how to deploy your multimodal e-commerce application with real data using Railway's PostgreSQL and optional cloud storage.

## Option 1: PostgreSQL on Railway (Recommended)

### Step 1: Create PostgreSQL Database on Railway

1. Log into [Railway](https://railway.app)
2. Click "New Project" → "Database" → "PostgreSQL"
3. Wait for provisioning to complete
4. Click on your database to view connection details

### Step 2: Convert CSV to PostgreSQL Tables

Create a script to import your product data:

```python
# import_data_to_postgres.py
import os
import pandas as pd
from sqlalchemy import create_engine

# Railway PostgreSQL connection string (from environment variables)
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Load product data
print("Loading product data...")
products_df = pd.read_csv('data/processed_products_with_images.csv')

# Write to PostgreSQL
print("Writing to PostgreSQL...")
products_df.to_sql('products', engine, if_exists='replace', index=False)

# Optional: Load smaller embedding files
# For example, if you have a subset of embeddings that fit in Railway's free tier
try:
    print("Loading text embeddings sample...")
    # Load a sample/subset of embeddings that fits within Railway limits
    text_embeddings_sample = pd.read_csv('Embeddings/text_embeddings_minilm.csv', nrows=5000)
    text_embeddings_sample.to_sql('text_embeddings', engine, if_exists='replace', index=False)
    print(f"Uploaded {len(text_embeddings_sample)} text embeddings")
except Exception as e:
    print(f"Error loading text embeddings: {str(e)}")

print("Data import completed!")
```

### Step 3: Update API Server to Use PostgreSQL

Modify `api_server.py` to connect to PostgreSQL instead of loading CSV files:

```python
# Add to imports
from sqlalchemy import create_engine
from sqlalchemy.sql import text

# In load_data function:
def load_data():
    """Load all necessary data files"""
    global product_data, text_embeddings_data, vision_embeddings_data, categories_data, model_performance, tfidf_vectorizer, tfidf_matrix
    
    try:
        logger.info("Loading product data...")
        
        # CRITICAL: Environment-aware detection for Railway
        is_production = (
            os.environ.get('RAILWAY_ENVIRONMENT') or 
            os.environ.get('RENDER') or 
            os.environ.get('ENVIRONMENT') == 'production'
        )
        
        if is_production:
            # Connect to PostgreSQL in production
            database_url = os.environ.get("DATABASE_URL")
            if database_url:
                logger.info("Connecting to PostgreSQL database")
                engine = create_engine(database_url)
                
                # Load products from database
                product_data = pd.read_sql("SELECT * FROM products LIMIT 5000", engine)
                logger.info(f"Loaded {len(product_data)} products from PostgreSQL")
                
                # Load embeddings from database (if available)
                try:
                    text_embeddings_data = pd.read_sql("SELECT * FROM text_embeddings LIMIT 5000", engine)
                    logger.info(f"Loaded {len(text_embeddings_data)} text embeddings from PostgreSQL")
                except:
                    logger.warning("Text embeddings table not found in database")
                    text_embeddings_data = None
                
                # Load categories from products
                if product_data is not None:
                    categories_data = {}
                    # Try to load categories from database or extract from products
                    try:
                        categories_df = pd.read_sql("SELECT * FROM categories", engine)
                        categories_data = dict(zip(categories_df['id'], categories_df['name']))
                    except:
                        # Extract unique categories from products
                        unique_categories = product_data['class_id'].unique()
                        categories_data = {cat: f"Category {cat}" for cat in unique_categories}
                    
                    logger.info(f"Loaded {len(categories_data)} categories")
            else:
                logger.warning("DATABASE_URL not found, using mock data")
                # Fall back to mock data (existing code)
                # ...
```

### Step 4: Update Railway Environment Variables

In your Railway project:
1. Go to your API service → Variables
2. Add the PostgreSQL connection string (Railway auto-generates this)
3. Add any other required environment variables

## Option 2: Cloud Storage for Large Files

For embedding files too large for PostgreSQL free tier:

### Step 1: Set Up Google Cloud Storage

1. Create a [Google Cloud account](https://cloud.google.com)
2. Create a new bucket in Google Cloud Storage
3. Upload your large embedding files
4. Set up authentication (service account key)

### Step 2: Add Code to Download Embeddings On-Demand

```python
# Add to imports
from google.cloud import storage

# Function to download embeddings from GCS
def download_embeddings_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads embeddings file from GCS to local file system"""
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        # Download file
        blob.download_to_filename(destination_file_name)
        logger.info(f"Downloaded {source_blob_name} to {destination_file_name}")
        return True
    except Exception as e:
        logger.error(f"Error downloading from GCS: {str(e)}")
        return False

# In load_data function, add:
if is_production and not os.path.exists('Embeddings/text_embeddings_minilm.csv'):
    # Download embeddings from GCS if needed
    gcs_bucket = os.environ.get("GCS_BUCKET_NAME")
    if gcs_bucket:
        logger.info("Downloading embeddings from Google Cloud Storage")
        os.makedirs('Embeddings', exist_ok=True)
        download_embeddings_from_gcs(
            gcs_bucket, 
            "text_embeddings_minilm.csv", 
            "Embeddings/text_embeddings_minilm.csv"
        )
```

## Recommended Approach for Railway Free Tier

Given Railway's free tier limitations:

1. **Store product data (18MB) in PostgreSQL**
2. **Store a subset of embeddings in PostgreSQL**
3. **Implement on-demand loading of full embeddings**
4. **Use caching to improve performance**

## Deployment Steps

1. **Prepare your data:**
   ```bash
   # Run locally to upload data to PostgreSQL
   export DATABASE_URL="postgresql://postgres:password@localhost:5432/postgres"
   python import_data_to_postgres.py
   ```

2. **Deploy to Railway:**
   ```bash
   # Push your code to GitHub
   git add .
   git commit -m "Update for PostgreSQL integration"
   git push
   
   # Railway will automatically deploy
   ```

3. **Verify deployment:**
   ```bash
   # Test your API endpoints
   curl https://your-app-name.railway.app/health
   ```

## Scaling Beyond Free Tier

When you're ready to scale:

1. Upgrade to Railway's hobby tier for more PostgreSQL storage
2. Consider using pgvector extension for vector similarity search
3. Implement proper caching and pagination for large datasets 