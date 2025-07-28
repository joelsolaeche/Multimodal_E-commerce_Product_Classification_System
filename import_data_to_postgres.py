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
    parser.add_argument('--check-only', action='store_true',
                        help='Only check database tables without importing')
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
            
            # Read the first row to get column names
            text_embeddings_sample = pd.read_csv(text_embeddings_file, nrows=1)
            
            # Check if the file has proper embedding columns
            embedding_cols = [col for col in text_embeddings_sample.columns if 'embedding' in col.lower()]
            if not embedding_cols:
                logger.warning("No embedding columns found in text embeddings file")
                # Try to detect numeric columns that might be embeddings
                numeric_cols = text_embeddings_sample.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude common non-embedding numeric columns
                numeric_cols = [col for col in numeric_cols if col not in ['sku', 'class_id', 'price']]
                if len(numeric_cols) > 100:  # Likely embedding columns
                    logger.info(f"Found {len(numeric_cols)} potential embedding columns")
                    # Rename columns to have 'embedding_' prefix
                    column_mapping = {col: f'embedding_{i}' for i, col in enumerate(numeric_cols)}
                else:
                    logger.error("Could not identify embedding columns in the file")
                    return False
            
            if file_size_gb > 0.8:  # Close to Railway's 1GB free tier limit
                logger.warning("Large file detected, sampling data")
                # Read only a sample
                text_embeddings_df = pd.read_csv(text_embeddings_file, nrows=sample_size)
            else:
                text_embeddings_df = pd.read_csv(text_embeddings_file)
            
            # Rename columns if needed
            if 'column_mapping' in locals():
                text_embeddings_df = text_embeddings_df.rename(columns=column_mapping)
            
            # Ensure 'sku' column exists
            if 'sku' not in text_embeddings_df.columns:
                logger.warning("No 'sku' column found in text embeddings, adding index as sku")
                text_embeddings_df['sku'] = [f'EMBEDDING{i:06d}' for i in range(len(text_embeddings_df))]
            
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
            
            # Read the first row to get column names
            vision_embeddings_sample = pd.read_csv(vision_embeddings_file, nrows=1)
            
            # Check if the file has proper feature columns
            feature_cols = [col for col in vision_embeddings_sample.columns if 'feature' in col.lower()]
            if not feature_cols:
                logger.warning("No feature columns found in vision embeddings file")
                # Try to detect numeric columns that might be embeddings
                numeric_cols = vision_embeddings_sample.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude common non-embedding numeric columns
                numeric_cols = [col for col in numeric_cols if col not in ['sku', 'class_id', 'price']]
                if len(numeric_cols) > 100:  # Likely embedding columns
                    logger.info(f"Found {len(numeric_cols)} potential feature columns")
                    # Rename columns to have 'feature_' prefix
                    column_mapping = {col: f'feature_{i}' for i, col in enumerate(numeric_cols)}
                else:
                    logger.error("Could not identify feature columns in the file")
                    return False
            
            if file_size_gb > 0.8:  # Close to Railway's 1GB free tier limit
                logger.warning("Large file detected, sampling data")
                # Read only a sample
                vision_embeddings_df = pd.read_csv(vision_embeddings_file, nrows=sample_size)
            else:
                vision_embeddings_df = pd.read_csv(vision_embeddings_file)
            
            # Rename columns if needed
            if 'column_mapping' in locals():
                vision_embeddings_df = vision_embeddings_df.rename(columns=column_mapping)
            
            # Ensure 'sku' column exists
            if 'sku' not in vision_embeddings_df.columns:
                logger.warning("No 'sku' column found in vision embeddings, adding index as sku")
                vision_embeddings_df['sku'] = [f'EMBEDDING{i:06d}' for i in range(len(vision_embeddings_df))]
            
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

def check_database_tables(engine):
    """Check database tables and their contents"""
    try:
        logger.info("Checking database tables")
        
        # Get list of tables
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        tables = pd.read_sql(tables_query, engine)
        logger.info(f"Found tables: {tables['table_name'].tolist()}")
        
        # Check each table
        for table in tables['table_name']:
            try:
                # Get row count
                count_query = f"SELECT COUNT(*) FROM {table}"
                count = pd.read_sql(count_query, engine).iloc[0, 0]
                
                # Get column info
                cols_query = f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}'"
                cols = pd.read_sql(cols_query, engine)
                
                # Get sample row
                sample_query = f"SELECT * FROM {table} LIMIT 1"
                sample = pd.read_sql(sample_query, engine)
                
                logger.info(f"Table '{table}': {count} rows, {len(cols)} columns")
                logger.info(f"Sample columns: {cols['column_name'].tolist()[:10]}")
                
                # For embeddings tables, check for embedding columns
                if table == 'text_embeddings':
                    embedding_cols = [col for col in sample.columns if 'embedding' in col.lower()]
                    logger.info(f"Found {len(embedding_cols)} embedding columns in text_embeddings")
                elif table == 'vision_embeddings':
                    feature_cols = [col for col in sample.columns if 'feature' in col.lower()]
                    logger.info(f"Found {len(feature_cols)} feature columns in vision_embeddings")
            except Exception as e:
                logger.error(f"Error checking table {table}: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking database tables: {str(e)}")
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
        
        # If check-only flag is set, just check the database
        if args.check_only:
            check_database_tables(engine)
            return
        
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
        
        # Check database tables after import
        check_database_tables(engine)
        
        logger.info("Data import completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 