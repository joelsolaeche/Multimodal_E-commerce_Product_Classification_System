#!/usr/bin/env python3
"""
Script to upload embedding files to Google Cloud Storage
"""

import os
import argparse
import logging
from google.cloud import storage
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Upload embeddings to Google Cloud Storage')
    parser.add_argument('--credentials-file', type=str, required=True,
                        help='Path to Google Cloud service account credentials JSON file')
    parser.add_argument('--bucket-name', type=str, required=True,
                        help='Google Cloud Storage bucket name')
    parser.add_argument('--text-embeddings-file', type=str, default='Embeddings/text_embeddings_minilm.csv',
                        help='Path to text embeddings CSV file')
    parser.add_argument('--vision-embeddings-file', type=str, default='Embeddings/Embeddings_resnet50.csv',
                        help='Path to vision embeddings CSV file')
    parser.add_argument('--skip-text', action='store_true',
                        help='Skip uploading text embeddings')
    parser.add_argument('--skip-vision', action='store_true',
                        help='Skip uploading vision embeddings')
    return parser.parse_args()

def upload_file_to_gcs(credentials_file, bucket_name, source_file_path, destination_blob_name):
    """Upload a file to Google Cloud Storage"""
    try:
        # Initialize storage client
        credentials = service_account.Credentials.from_service_account_file(credentials_file)
        client = storage.Client(credentials=credentials)
        
        # Get bucket
        bucket = client.bucket(bucket_name)
        
        # Create blob and upload
        blob = bucket.blob(destination_blob_name)
        
        # Check if file exists
        if not os.path.exists(source_file_path):
            logger.error(f"File not found: {source_file_path}")
            return False
        
        # Get file size
        file_size_mb = os.path.getsize(source_file_path) / (1024 * 1024)
        logger.info(f"Uploading {source_file_path} ({file_size_mb:.2f} MB) to {destination_blob_name}")
        
        # Upload file
        blob.upload_from_filename(source_file_path)
        
        logger.info(f"File {source_file_path} uploaded to {destination_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return False

def main():
    """Main function"""
    args = parse_args()
    
    # Upload text embeddings
    if not args.skip_text:
        text_embeddings_file = args.text_embeddings_file
        text_embeddings_blob = os.path.basename(text_embeddings_file)
        logger.info(f"Uploading text embeddings from {text_embeddings_file} to {text_embeddings_blob}")
        upload_file_to_gcs(
            args.credentials_file,
            args.bucket_name,
            text_embeddings_file,
            text_embeddings_blob
        )
    
    # Upload vision embeddings
    if not args.skip_vision:
        vision_embeddings_file = args.vision_embeddings_file
        vision_embeddings_blob = os.path.basename(vision_embeddings_file)
        logger.info(f"Uploading vision embeddings from {vision_embeddings_file} to {vision_embeddings_blob}")
        upload_file_to_gcs(
            args.credentials_file,
            args.bucket_name,
            vision_embeddings_file,
            vision_embeddings_blob
        )
    
    logger.info("Upload completed")

if __name__ == "__main__":
    main() 