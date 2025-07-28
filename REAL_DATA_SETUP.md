# Step-by-Step Guide: Real Data Setup with PostgreSQL on Railway

This guide will walk you through the process of setting up your multimodal e-commerce application with real data using PostgreSQL on Railway.

## Overview

We've implemented a hybrid approach that:
1. Stores product data (18MB) in PostgreSQL
2. Stores a sample of embeddings in PostgreSQL
3. Provides options for larger embedding files

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

## Step 2: Import Your Data to PostgreSQL

1. **Install Required Dependencies**
   ```bash
   pip install -r requirements-api.txt
   ```

2. **Run the Import Script Locally**
   ```bash
   # Set your Railway PostgreSQL connection string
   export DATABASE_URL="your_railway_postgres_connection_string"
   
   # Run the import script
   python import_data_to_postgres.py
   ```

3. **Options for Large Files**
   - For the full dataset:
     ```bash
     python import_data_to_postgres.py
     ```
   
   - For just product data (no embeddings):
     ```bash
     python import_data_to_postgres.py --skip-embeddings
     ```
   
   - For a smaller sample of embeddings:
     ```bash
     python import_data_to_postgres.py --sample-size 1000
     ```

4. **Monitor the Import Process**
   - The script will log its progress
   - For large files, this may take some time
   - Check Railway dashboard to monitor database storage usage

## Step 3: Deploy Your API to Railway

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

3. **Deploy Your API**
   - Railway will automatically detect your `railway.json` and `Dockerfile`
   - Wait for the deployment to complete
   - Your API will be available at a URL like `https://your-app-name.railway.app`

## Step 4: Test Your API

1. **Check Health Endpoint**
   ```bash
   curl https://your-app-name.railway.app/health
   ```

2. **Test Text Classification**
   ```bash
   curl -X POST https://your-app-name.railway.app/api/classify/text \
     -H "Content-Type: application/json" \
     -d '{"text": "wireless bluetooth headphones"}'
   ```

## Step 5: Connect Your Frontend

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

## Advanced: Google Cloud Storage for Large Files

If your embedding files are too large for Railway's free tier:

1. **Create a Google Cloud Account**
   - Sign up for [Google Cloud](https://cloud.google.com)
   - Create a new project

2. **Create a Storage Bucket**
   - Go to Cloud Storage → Create Bucket
   - Choose a unique name
   - Set access to "Fine-grained"
   - Create the bucket

3. **Upload Your Large Files**
   - Upload your embedding files to the bucket
   - Set permissions to allow your service to access them

4. **Create a Service Account**
   - Go to IAM & Admin → Service Accounts
   - Create a new service account
   - Grant "Storage Object Viewer" role
   - Create and download a JSON key file

5. **Add to Railway**
   - In Railway, add these environment variables:
     - `GCS_BUCKET_NAME`: Your bucket name
     - `GOOGLE_APPLICATION_CREDENTIALS`: Contents of your JSON key file

## Scaling Considerations

1. **Railway Free Tier (1GB)**
   - Store product data (~18MB)
   - Store a subset of embeddings (~5000 rows)
   - Use mock data as fallback

2. **Railway Hobby Tier ($5/month, 5GB)**
   - Store all product data
   - Store more embeddings
   - Better performance and reliability

3. **Hybrid Approach**
   - Store product data in PostgreSQL
   - Store full embeddings in Google Cloud Storage
   - Cache frequently used embeddings

## Troubleshooting

1. **Database Connection Issues**
   - Check your `DATABASE_URL` environment variable
   - Verify network access to Railway
   - Check PostgreSQL logs in Railway dashboard

2. **Import Script Errors**
   - Ensure CSV files are properly formatted
   - Try importing smaller batches
   - Check for memory limitations

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