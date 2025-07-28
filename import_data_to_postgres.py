#!/usr/bin/env python3
"""
Script to import product data and embeddings to PostgreSQL
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Import data to PostgreSQL')
    parser.add_argument('--database-url', type=str, help='PostgreSQL connection string')
    parser.add_argument('--products-file', type=str, default='data/processed_products_with_images.csv', 
                        help='Path to products CSV file')
    parser.add_argument('--text-embeddings-file', type=str, default='Embeddings/text_embeddings_minilm.csv',
                        help='Path to text embeddings CSV file')
    parser.add_argument('--vision-embeddings-file', type=str, default='Embeddings/Embeddings_resnet50.csv',
                        help='Path to vision embeddings CSV file')
    parser.add_argument('--categories-file', type=str, default='data/Raw/categories.json',
                        help='Path to categories JSON file')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Number of embeddings to import (for large files)')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip importing embeddings (useful for limited storage)')
    return parser.parse_args()

def import_products(engine, products_file):
    """Import products data to PostgreSQL"""
    try:
        logger.info(f"Loading products from {products_file}")
        products_df = pd.read_csv(products_file)
        
        # Clean up data for SQL import
        for col in products_df.columns:
            if products_df[col].dtype == object:
                products_df[col] = products_df[col].fillna('')
        
        logger.info(f"Importing {len(products_df)} products to PostgreSQL")
        products_df.to_sql('products', engine, if_exists='replace', index=False)
        logger.info("Products import completed")
        return True
    except Exception as e:
        logger.error(f"Error importing products: {str(e)}")
        return False

def import_categories(engine, categories_file):
    """Import categories to PostgreSQL"""
    try:
        if os.path.exists(categories_file):
            logger.info(f"Loading categories from {categories_file}")
            with open(categories_file, 'r') as f:
                categories_list = json.load(f)
            
            # Convert to DataFrame
            categories_df = pd.DataFrame([
                {'id': cat['id'], 'name': cat['name']} 
                for cat in categories_list
            ])
            
            logger.info(f"Importing {len(categories_df)} categories to PostgreSQL")
            categories_df.to_sql('categories', engine, if_exists='replace', index=False)
            logger.info("Categories import completed")
            return True
        else:
            logger.warning(f"Categories file not found: {categories_file}")
            return False
    except Exception as e:
        logger.error(f"Error importing categories: {str(e)}")
        return False

def import_text_embeddings(engine, text_embeddings_file, sample_size):
    """Import text embeddings to PostgreSQL"""
    try:
        if os.path.exists(text_embeddings_file):
            logger.info(f"Loading text embeddings from {text_embeddings_file}")
            
            # Check file size
            file_size_gb = os.path.getsize(text_embeddings_file) / (1024**3)
            logger.info(f"Text embeddings file size: {file_size_gb:.2f} GB")
            
            if file_size_gb > 0.8:  # Close to Railway's 1GB free tier limit
                logger.warning("Large file detected, sampling data")
                # Read only a sample
                text_embeddings_df = pd.read_csv(text_embeddings_file, nrows=sample_size)
            else:
                text_embeddings_df = pd.read_csv(text_embeddings_file)
            
            # Clean up data for SQL import
            for col in text_embeddings_df.columns:
                if text_embeddings_df[col].dtype == object:
                    text_embeddings_df[col] = text_embeddings_df[col].fillna('')
            
            logger.info(f"Importing {len(text_embeddings_df)} text embeddings to PostgreSQL")
            text_embeddings_df.to_sql('text_embeddings', engine, if_exists='replace', index=False)
            logger.info("Text embeddings import completed")
            return True
        else:
            logger.warning(f"Text embeddings file not found: {text_embeddings_file}")
            return False
    except Exception as e:
        logger.error(f"Error importing text embeddings: {str(e)}")
        return False

def import_vision_embeddings(engine, vision_embeddings_file, sample_size):
    """Import vision embeddings to PostgreSQL"""
    try:
        if os.path.exists(vision_embeddings_file):
            logger.info(f"Loading vision embeddings from {vision_embeddings_file}")
            
            # Check file size
            file_size_gb = os.path.getsize(vision_embeddings_file) / (1024**3)
            logger.info(f"Vision embeddings file size: {file_size_gb:.2f} GB")
            
            if file_size_gb > 0.8:  # Close to Railway's 1GB free tier limit
                logger.warning("Large file detected, sampling data")
                # Read only a sample
                vision_embeddings_df = pd.read_csv(vision_embeddings_file, nrows=sample_size)
            else:
                vision_embeddings_df = pd.read_csv(vision_embeddings_file)
            
            # Clean up data for SQL import
            for col in vision_embeddings_df.columns:
                if vision_embeddings_df[col].dtype == object:
                    vision_embeddings_df[col] = vision_embeddings_df[col].fillna('')
            
            logger.info(f"Importing {len(vision_embeddings_df)} vision embeddings to PostgreSQL")
            vision_embeddings_df.to_sql('vision_embeddings', engine, if_exists='replace', index=False)
            logger.info("Vision embeddings import completed")
            return True
        else:
            logger.warning(f"Vision embeddings file not found: {vision_embeddings_file}")
            return False
    except Exception as e:
        logger.error(f"Error importing vision embeddings: {str(e)}")
        return False

def create_indexes(engine):
    """Create indexes for better query performance"""
    try:
        logger.info("Creating indexes")
        with engine.connect() as conn:
            # Index on products table
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_products_sku ON products (sku)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_products_class_id ON products (class_id)"))
            
            # Index on text_embeddings table
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_text_embeddings_sku ON text_embeddings (sku)"))
            
            # Index on vision_embeddings table
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vision_embeddings_sku ON vision_embeddings (sku)"))
            
            conn.commit()
        logger.info("Indexes created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating indexes: {str(e)}")
        return False

def main():
    """Main function"""
    args = parse_args()
    
    # Get database URL from args or environment
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    
    if not database_url:
        logger.error("DATABASE_URL not provided. Use --database-url or set DATABASE_URL environment variable")
        sys.exit(1)
    
    try:
        # Create SQLAlchemy engine
        logger.info("Connecting to PostgreSQL database")
        engine = create_engine(database_url)
        
        # Import products
        import_products(engine, args.products_file)
        
        # Import categories
        import_categories(engine, args.categories_file)
        
        # Import embeddings if not skipped
        if not args.skip_embeddings:
            import_text_embeddings(engine, args.text_embeddings_file, args.sample_size)
            import_vision_embeddings(engine, args.vision_embeddings_file, args.sample_size)
        else:
            logger.info("Skipping embeddings import as requested")
        
        # Create indexes
        create_indexes(engine)
        
        logger.info("Data import completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 