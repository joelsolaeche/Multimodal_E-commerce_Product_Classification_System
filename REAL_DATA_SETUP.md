# Step-by-Step Guide: Real Data Setup with PostgreSQL on Railway

This guide will walk you through the process of setting up your multimodal e-commerce application with real data using PostgreSQL on Railway.

## Overview

We've implemented a hybrid approach that:
1. Stores product data (18MB) in PostgreSQL
2. Stores embeddings in Google Cloud Storage
3. Provides fallback to mock data if needed

## Step 1: Set Up PostgreSQL on Railway

1. **Create a Railway Account**
   - Sign up at [Railway.app](https://railway.app)
   - You can start with the free tier (1GB storage)

2. **Create a PostgreSQL Database**
   - Click "New Project" → "Database" → "PostgreSQL"
   - Wait for provisioning to complete
   - Railway will automatically generate connection details

3. **Get Your Database Connection String**
   - In your Railway project, click on the PostgreSQL service
   - Go to "Connect" tab
   - Copy the "Postgres Connection URL"
   - It will look like: `postgresql://postgres:password@containers-us-west-XXX.railway.app:XXXX/railway`

## Step 2: Import Product Data to PostgreSQL

1. **Install Required Dependencies**
   ```bash
   pip install -r requirements-api.txt
   ```

2. **Run the Import Script Locally**
   ```bash
   # Set your Railway PostgreSQL connection string
   export DATABASE_URL="your_railway_postgres_connection_string"
   
   # Run the import script for products and categories only
   python import_data_to_postgres.py --skip-embeddings
   ```

3. **Verify Data Import**
   ```bash
   # Check that products and categories were imported
   python import_data_to_postgres.py --check-only
   ```

## Step 3: Set Up Google Cloud Storage for Embeddings

1. **Create a Google Cloud Account**
   - Sign up for [Google Cloud](https://cloud.google.com)
   - Create a new project

2. **Create a Storage Bucket**
   - Go to Cloud Storage → Create Bucket
   - Choose a unique name (e.g., `your-project-embeddings`)
   - Set access to "Fine-grained"
   - Choose a region close to your users
   - Click "Create"

3. **Upload Embedding Files**
   - In your bucket, click "Upload Files"
   - Upload the following files:
     - `text_embeddings_minilm.csv` (from your local `Embeddings/` folder)
     - `Embeddings_resnet50.csv` (from your local `Embeddings/` folder)

4. **Create a Service Account**
   - Go to IAM & Admin → Service Accounts
   - Click "Create Service Account"
   - Name: `embeddings-reader`
   - Description: `Service account for reading embeddings`
   - Click "Create and Continue"
   - Add Role: "Storage Object Viewer"
   - Click "Continue" and "Done"

5. **Create and Download a Key**
   - Find your new service account in the list
   - Click on the three dots (actions) → "Manage keys"
   - Click "Add Key" → "Create new key"
   - Select "JSON" format
   - Click "Create" to download the key file

## Step 4: Deploy Your API to Railway

1. **Connect Your GitHub Repository**
   - Push your code changes to GitHub
   - In Railway, create a new project
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Set Environment Variables**
   - In your Railway project, go to the "Variables" tab
   - Add the following variables:
     - `DATABASE_URL`: This will be automatically linked by Railway
     - `PORT`: `8000`
     - `ENVIRONMENT`: `production`
     - `USE_POSTGRES`: `true`
     - `USE_GCS`: `true`
     - `GCS_BUCKET_NAME`: Your bucket name (e.g., `your-project-embeddings`)
     - `GOOGLE_APPLICATION_CREDENTIALS_JSON`: The entire contents of your downloaded JSON key file

3. **Deploy Your API**
   - Railway will automatically detect your `railway.json` and `Dockerfile`
   - Wait for the deployment to complete
   - Your API will be available at a URL like `https://your-app-name.railway.app`

## Step 5: Test Your API

1. **Check Health Endpoint**
   ```bash
   curl https://your-app-name.railway.app/health
   ```

2. **Check API Stats**
   ```bash
   curl https://your-app-name.railway.app/api/stats
   ```
   Verify that:
   - `database_connected` is `true`
   - `gcs_connected` is `true`
   - `has_embeddings` is `true`

3. **Test Text Classification**
   ```bash
   curl -X POST https://your-app-name.railway.app/api/classify/text \
     -H "Content-Type: application/json" \
     -d '{"text": "duracell batteries"}'
   ```
   You should get relevant categories like "Batteries" or "Accessories" instead of random categories.

## Step 6: Connect Your Frontend

1. **Update API URL in Frontend**
   - In your Next.js project, update the API URL to point to your Railway deployment
   - Edit `multimodal-ecommerce-demo/.env.local`:
     ```
     NEXT_PUBLIC_API_URL=https://your-app-name.railway.app
     ```

2. **Deploy Frontend to Vercel**
   - Connect your GitHub repository to Vercel
   - Set the root directory to `multimodal-ecommerce-demo`
   - Add the environment variable above
   - Deploy the frontend

## Troubleshooting

1. **Database Connection Issues**
   - Check your `DATABASE_URL` environment variable
   - Verify network access to Railway
   - Check PostgreSQL logs in Railway dashboard

2. **Google Cloud Storage Issues**
   - Verify your service account has the "Storage Object Viewer" role
   - Check that your JSON key is correctly formatted in the environment variable
   - Verify that the bucket name is correct
   - Check that the embedding files are uploaded to the bucket

3. **API Errors**
   - Check Railway logs for errors
   - Verify that tables exist in PostgreSQL
   - Test locally before deploying

4. **Performance Issues**
   - Consider adding indexes to frequently queried columns
   - Limit the number of embeddings loaded
   - Implement caching for frequent queries

## Next Steps

1. **Implement pgvector Extension**
   - For more efficient vector similarity search
   - Requires Railway hobby tier

2. **Add Caching Layer**
   - Implement Redis for caching frequent queries
   - Improve response times for common searches

3. **Optimize Embeddings**
   - Consider dimensionality reduction techniques
   - Implement quantization for smaller storage

4. **Implement Pagination**
   - For large result sets
   - Improve frontend performance 