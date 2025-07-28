#!/usr/bin/env python3
"""
Google Cloud Storage utilities for loading embeddings
"""

import os
import io
import json
import logging
import pandas as pd
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCSStorage:
    """Google Cloud Storage handler for embeddings"""
    
    def __init__(self):
        """Initialize GCS client"""
        self.client = None
        self.bucket = None
        self.bucket_name = os.environ.get('GCS_BUCKET_NAME')
        self.initialized = False
        
        # Try to initialize the client
        try:
            # Check if credentials are provided as environment variable content
            creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
            if creds_json:
                # Parse JSON credentials from environment variable
                try:
                    creds_info = json.loads(creds_json)
                    credentials = service_account.Credentials.from_service_account_info(creds_info)
                    self.client = storage.Client(credentials=credentials)
                except json.JSONDecodeError:
                    logger.error("Failed to parse GOOGLE_APPLICATION_CREDENTIALS_JSON")
            else:
                # Use default credentials file path
                creds_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                if creds_file:
                    self.client = storage.Client.from_service_account_json(creds_file)
                else:
                    # Try default authentication
                    self.client = storage.Client()
            
            # Get bucket
            if self.client and self.bucket_name:
                self.bucket = self.client.bucket(self.bucket_name)
                self.initialized = True
                logger.info(f"GCS initialized with bucket: {self.bucket_name}")
            else:
                logger.warning("GCS not fully initialized. Missing client or bucket name.")
        except Exception as e:
            logger.error(f"Error initializing GCS: {str(e)}")
    
    def is_ready(self):
        """Check if GCS is initialized and ready"""
        return self.initialized
    
    def list_files(self, prefix=None):
        """List files in the bucket with optional prefix"""
        if not self.is_ready():
            logger.error("GCS not initialized")
            return []
        
        try:
            blobs = list(self.client.list_blobs(self.bucket_name, prefix=prefix))
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []
    
    def load_csv_as_dataframe(self, blob_name, nrows=None):
        """Load a CSV file from GCS as a pandas DataFrame"""
        if not self.is_ready():
            logger.error("GCS not initialized")
            return None
        
        try:
            # Get the blob
            blob = self.bucket.blob(blob_name)
            
            # Download as bytes
            content = blob.download_as_bytes()
            
            # Convert to DataFrame
            df = pd.read_csv(io.BytesIO(content), nrows=nrows)
            logger.info(f"Loaded {blob_name} with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading {blob_name}: {str(e)}")
            return None
    
    def load_text_embeddings(self, file_name="text_embeddings_minilm.csv", sample_size=None):
        """Load text embeddings from GCS"""
        df = self.load_csv_as_dataframe(file_name, nrows=sample_size)
        if df is None:
            return None
        
        # Process embeddings if needed
        if 'embeddings' in df.columns:
            logger.info("Processing 'embeddings' column")
            try:
                # Function to convert string representation to list
                def parse_embedding(embedding_str):
                    try:
                        # Remove brackets and split by comma
                        values = embedding_str.strip('[]').split(',')
                        return [float(v.strip()) for v in values]
                    except:
                        return []
                
                # Apply parsing to the embeddings column
                embeddings_lists = df['embeddings'].apply(parse_embedding)
                
                # Get the max length of embeddings
                max_len = max(len(e) for e in embeddings_lists if e)
                logger.info(f"Max embedding length: {max_len}")
                
                # Create separate columns for each embedding dimension
                for i in range(max_len):
                    df[f'embedding_{i}'] = embeddings_lists.apply(
                        lambda x: x[i] if i < len(x) else 0.0
                    )
                
                # Drop the original embeddings column
                df = df.drop(columns=['embeddings'])
            except Exception as e:
                logger.error(f"Error processing embeddings column: {str(e)}")
        
        return df
    
    def load_vision_embeddings(self, file_name="Embeddings_resnet50.csv", sample_size=None):
        """Load vision embeddings from GCS"""
        df = self.load_csv_as_dataframe(file_name, nrows=sample_size)
        if df is None:
            return None
        
        # Rename numeric columns to feature_X format if needed
        numeric_cols = [col for col in df.columns if str(col).isdigit()]
        if numeric_cols:
            logger.info(f"Found {len(numeric_cols)} numeric columns, renaming to feature_X format")
            rename_dict = {col: f'feature_{col}' for col in numeric_cols}
            df = df.rename(columns=rename_dict)
        
        # Rename ImageName column to sku if it exists
        if 'ImageName' in df.columns:
            logger.info("Renaming 'ImageName' column to 'sku'")
            df = df.rename(columns={'ImageName': 'sku'})
        
        return df

# Singleton instance
gcs_storage = GCSStorage() 