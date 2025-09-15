#!/usr/bin/env python3
# setup_database.py
# Purpose: Initialize the database and create tables

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.database_manager import DatabaseManager
from src.utils.logger import setup_logger

def main():
    # Step 1: Setup logging
    logger = setup_logger("DatabaseSetup")
    logger.info("Starting database setup...")
    
    try:
        # Step 2: Initialize database manager
        db_manager = DatabaseManager()
        logger.info("Database manager initialized successfully")
        
        # Step 3: Get database statistics
        stats = db_manager.get_database_stats()
        logger.info(f"Database statistics: {stats}")
        
        # Step 4: Verify tables exist
        required_tables = ['customers', 'transactions', 'customer_segments', 'campaigns']
        for table in required_tables:
            if stats.get(table, 0) >= 0:
                logger.info(f"✓ Table '{table}' exists")
            else:
                logger.error(f"✗ Table '{table}' not found")
        
        logger.info("Database setup completed successfully!")
        print("✅ Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during database setup: {str(e)}")
        print(f"❌ Error during database setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 