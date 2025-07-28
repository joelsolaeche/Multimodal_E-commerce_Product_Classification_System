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
            
            # For text_embeddings_minilm.csv, the file has a specific format
            # First, try to read with pandas directly
            try:
                # Read only a sample if file is large
                if file_size_gb > 0.8:
                    logger.warning("Large file detected, sampling data")
                    text_embeddings_df = pd.read_csv(text_embeddings_file, nrows=sample_size)
                else:
                    text_embeddings_df = pd.read_csv(text_embeddings_file)
                
                logger.info(f"Successfully read text embeddings file with shape: {text_embeddings_df.shape}")
                
                # Check if we have the expected columns
                if 'embeddings' in text_embeddings_df.columns:
                    logger.info("Found 'embeddings' column, need to parse it")
                    
                    # The embeddings column might be a string representation of a list
                    # We need to parse it into separate columns
                    try:
                        # Sample the first row to see the format
                        sample_embedding = text_embeddings_df['embeddings'].iloc[0]
                        logger.info(f"Sample embedding format: {type(sample_embedding)}")
                        
                        # If it's a string representation of a list, parse it
                        if isinstance(sample_embedding, str):
                            logger.info("Parsing embeddings from string format")
                            
                            # Function to convert string representation to list
                            def parse_embedding(embedding_str):
                                try:
                                    # Remove brackets and split by comma
                                    values = embedding_str.strip('[]').split(',')
                                    return [float(v.strip()) for v in values]
                                except:
                                    return []
                            
                            # Apply parsing to the embeddings column
                            embeddings_lists = text_embeddings_df['embeddings'].apply(parse_embedding)
                            
                            # Get the max length of embeddings
                            max_len = max(len(e) for e in embeddings_lists if e)
                            logger.info(f"Max embedding length: {max_len}")
                            
                            # Create separate columns for each embedding dimension
                            for i in range(max_len):
                                text_embeddings_df[f'embedding_{i}'] = embeddings_lists.apply(
                                    lambda x: x[i] if i < len(x) else 0.0
                                )
                            
                            # Drop the original embeddings column
                            text_embeddings_df = text_embeddings_df.drop(columns=['embeddings'])
                    except Exception as e:
                        logger.error(f"Error parsing embeddings column: {str(e)}")
                
                # Ensure 'sku' column exists
                if 'sku' not in text_embeddings_df.columns:
                    logger.warning("No 'sku' column found in text embeddings, adding index as sku")
                    text_embeddings_df['sku'] = [f'EMBEDDING{i:06d}' for i in range(len(text_embeddings_df))]
                
                # Clean up data for SQL import
                for col in text_embeddings_df.columns:
                    if text_embeddings_df[col].dtype == object:
                        text_embeddings_df[col] = text_embeddings_df[col].fillna('')
                
                logger.info(f"Importing {len(text_embeddings_df)} text embeddings to PostgreSQL")
                
                # Create a subset with just sku and embedding columns to reduce size
                embedding_cols = [col for col in text_embeddings_df.columns if 'embedding_' in col or col == 'sku']
                if len(embedding_cols) > 1:  # At least sku and one embedding column
                    text_embeddings_subset = text_embeddings_df[embedding_cols]
                    logger.info(f"Created subset with {len(embedding_cols)} columns for import")
                    text_embeddings_subset.to_sql('text_embeddings', engine, if_exists='replace', index=False)
                else:
                    logger.warning("No embedding columns found, attempting direct import")
                    text_embeddings_df.to_sql('text_embeddings', engine, if_exists='replace', index=False)
                
                logger.info("Text embeddings import completed")
                return True
                
            except Exception as e:
                logger.error(f"Error reading text embeddings file with pandas: {str(e)}")
                return False
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
            
            # For Embeddings_resnet50.csv, the file has numeric columns with names 0, 1, 2, etc.
            try:
                # Read only a sample if file is large
                if file_size_gb > 0.8:
                    logger.warning("Large file detected, sampling data")
                    vision_embeddings_df = pd.read_csv(vision_embeddings_file, nrows=sample_size)
                else:
                    vision_embeddings_df = pd.read_csv(vision_embeddings_file)
                
                logger.info(f"Successfully read vision embeddings file with shape: {vision_embeddings_df.shape}")
                
                # Rename numeric columns to feature_X format
                numeric_cols = [col for col in vision_embeddings_df.columns if str(col).isdigit()]
                if numeric_cols:
                    logger.info(f"Found {len(numeric_cols)} numeric columns, renaming to feature_X format")
                    rename_dict = {col: f'feature_{col}' for col in numeric_cols}
                    vision_embeddings_df = vision_embeddings_df.rename(columns=rename_dict)
                
                # Check if we have an image name column that can be used as SKU
                if 'ImageName' in vision_embeddings_df.columns:
                    logger.info("Found 'ImageName' column, using as SKU")
                    vision_embeddings_df = vision_embeddings_df.rename(columns={'ImageName': 'sku'})
                
                # Ensure 'sku' column exists
                if 'sku' not in vision_embeddings_df.columns:
                    logger.warning("No 'sku' column found in vision embeddings, adding index as sku")
                    vision_embeddings_df['sku'] = [f'VISION{i:06d}' for i in range(len(vision_embeddings_df))]
                
                # Clean up data for SQL import
                for col in vision_embeddings_df.columns:
                    if vision_embeddings_df[col].dtype == object:
                        vision_embeddings_df[col] = vision_embeddings_df[col].fillna('')
                
                logger.info(f"Importing {len(vision_embeddings_df)} vision embeddings to PostgreSQL")
                vision_embeddings_df.to_sql('vision_embeddings', engine, if_exists='replace', index=False)
                logger.info("Vision embeddings import completed")
                return True
                
            except Exception as e:
                logger.error(f"Error reading vision embeddings file with pandas: {str(e)}")
                return False
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
            
            # Check if text_embeddings table exists before creating index
            try:
                conn.execute(text("SELECT 1 FROM text_embeddings LIMIT 1"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_text_embeddings_sku ON text_embeddings (sku)"))
                logger.info("Created index on text_embeddings.sku")
            except Exception as e:
                logger.warning(f"Could not create index on text_embeddings: {str(e)}")
            
            # Check if vision_embeddings table exists before creating index
            try:
                conn.execute(text("SELECT 1 FROM vision_embeddings LIMIT 1"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vision_embeddings_sku ON vision_embeddings (sku)"))
                logger.info("Created index on vision_embeddings.sku")
            except Exception as e:
                logger.warning(f"Could not create index on vision_embeddings: {str(e)}")
            
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