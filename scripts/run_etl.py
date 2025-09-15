#!/usr/bin/env python3
# run_etl.py
# Purpose: Run the complete ETL pipeline

import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.database_manager import DatabaseManager
from src.ml.customer_segmentation import CustomerSegmentation
from src.ml.recommendation_engine import RecommendationEngine
from src.utils.logger import setup_logger

class ETLPipeline:
    def __init__(self):
        # Step 1: Initialize ETL pipeline components
        self.logger = setup_logger("ETLPipeline")
        self.data_loader = DataLoader()
        self.data_preprocessor = DataPreprocessor()
        self.db_manager = DatabaseManager()
        self.segmentation = CustomerSegmentation()
        self.recommendation_engine = RecommendationEngine()
    
    def extract_data(self):
        # Step 1: Extract data from various sources
        self.logger.info("Starting data extraction...")
        
        try:
            # Step 2: Load sample data from data directory
            data_files = self.data_loader.load_sample_data()
            
            if not data_files:
                self.logger.warning("No data files found. Please run generate_sample_data.py first.")
                return None, None
            
            # Step 3: Extract transactions and customers
            transactions_df = data_files.get('transactions')
            customers_df = data_files.get('customers')
            
            if transactions_df is not None:
                self.logger.info(f"Extracted {len(transactions_df)} transactions")
            if customers_df is not None:
                self.logger.info(f"Extracted {len(customers_df)} customers")
            
            return transactions_df, customers_df
            
        except Exception as e:
            self.logger.error(f"Error during data extraction: {str(e)}")
            raise
    
    def transform_data(self, transactions_df, customers_df):
        # Step 1: Transform and clean the data
        self.logger.info("Starting data transformation...")
        
        try:
            # Step 2: Clean transaction data
            if transactions_df is not None:
                cleaned_transactions = self.data_preprocessor.clean_transaction_data(transactions_df)
                self.logger.info(f"Cleaned {len(cleaned_transactions)} transactions")
            else:
                cleaned_transactions = None
            
            # Step 3: Clean customer data
            if customers_df is not None:
                cleaned_customers = self.data_preprocessor.clean_customer_data(customers_df)
                self.logger.info(f"Cleaned {len(cleaned_customers)} customers")
            else:
                cleaned_customers = None
            
            # Step 4: Create customer features
            if cleaned_transactions is not None:
                customer_features = self.data_preprocessor.create_customer_features(cleaned_transactions)
                self.logger.info(f"Created features for {len(customer_features)} customers")
            else:
                customer_features = None
            
            return cleaned_transactions, cleaned_customers, customer_features
            
        except Exception as e:
            self.logger.error(f"Error during data transformation: {str(e)}")
            raise
    
    def load_data(self, transactions_df, customers_df):
        # Step 1: Load data into database
        self.logger.info("Starting data loading...")
        
        try:
            # Step 2: Load customers
            if customers_df is not None:
                self.db_manager.insert_dataframe(customers_df, 'customers', if_exists='replace')
                self.logger.info(f"Loaded {len(customers_df)} customers")
            
            # Step 3: Load transactions
            if transactions_df is not None:
                self.db_manager.insert_dataframe(transactions_df, 'transactions', if_exists='replace')
                self.logger.info(f"Loaded {len(transactions_df)} transactions")
            
            self.logger.info("Data loading completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during data loading: {str(e)}")
            raise
    
    def perform_segmentation(self, customer_features):
        # Step 1: Perform customer segmentation
        self.logger.info("Starting customer segmentation...")
        
        try:
            if customer_features is not None and len(customer_features) > 0:
                # Step 2: Perform segmentation
                segmentation_results = self.segmentation.perform_segmentation(customer_features)
                
                # Step 3: Save segmentation results
                segments_df = segmentation_results[['customer_id', 'segment_id', 'segment_name']].copy()
                segments_df['last_updated'] = datetime.now().strftime('%Y-%m-%d')
                self.db_manager.update_customer_segments(segments_df)
                
                self.logger.info(f"Segmentation completed for {len(segments_df)} customers")
                return segmentation_results
            else:
                self.logger.warning("No customer features available for segmentation")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during segmentation: {str(e)}")
            raise
    
    def setup_recommendation_engine(self, transactions_df):
        # Step 1: Setup recommendation engine
        self.logger.info("Setting up recommendation engine...")
        
        try:
            if transactions_df is not None and len(transactions_df) > 0:
                # Step 2: Create customer-item matrix
                customer_item_matrix = self.recommendation_engine.create_customer_item_matrix(transactions_df)
                
                # Step 3: Calculate customer similarity
                similarity_matrix = self.recommendation_engine.calculate_customer_similarity()
                
                self.logger.info("Recommendation engine setup completed")
                return True
            else:
                self.logger.warning("No transaction data available for recommendation engine")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting up recommendation engine: {str(e)}")
            raise
    
    def run_pipeline(self):
        # Step 1: Run complete ETL pipeline
        self.logger.info("Starting ETL pipeline...")
        
        try:
            # Step 2: Extract
            transactions_df, customers_df = self.extract_data()
            
            # Step 3: Transform
            cleaned_transactions, cleaned_customers, customer_features = self.transform_data(
                transactions_df, customers_df
            )
            
            # Step 4: Load
            self.load_data(cleaned_transactions, cleaned_customers)
            
            # Step 5: Perform segmentation
            segmentation_results = self.perform_segmentation(customer_features)
            
            # Step 6: Setup recommendation engine
            recommendation_ready = self.setup_recommendation_engine(cleaned_transactions)
            
            # Step 7: Print summary
            self._print_pipeline_summary(
                cleaned_transactions, cleaned_customers, 
                segmentation_results, recommendation_ready
            )
            
            self.logger.info("ETL pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in ETL pipeline: {str(e)}")
            return False
    
    def _print_pipeline_summary(self, transactions_df, customers_df, 
                              segmentation_results, recommendation_ready):
        # Step 1: Print pipeline summary
        print("\n" + "="*60)
        print("🔄 ETL PIPELINE SUMMARY")
        print("="*60)
        
        if transactions_df is not None:
            print(f"🛒 Transactions Processed: {len(transactions_df):,}")
            print(f"💰 Total Revenue: ${transactions_df['amount'].sum():,.2f}")
            print(f"📈 Average Transaction: ${transactions_df['amount'].mean():.2f}")
        
        if customers_df is not None:
            print(f"👥 Customers Processed: {len(customers_df):,}")
        
        if segmentation_results is not None:
            print(f"🏷️ Segments Created: {segmentation_results['segment_name'].nunique()}")
            print(f"📊 Segment Distribution:")
            segment_counts = segmentation_results['segment_name'].value_counts()
            for segment, count in segment_counts.items():
                print(f"   • {segment}: {count:,} customers")
        
        print(f"🎯 Recommendation Engine: {'✅ Ready' if recommendation_ready else '❌ Not Ready'}")
        
        print("="*60)

def main():
    # Step 1: Main execution
    print("🚀 Starting ETL Pipeline...")
    
    try:
        # Step 2: Initialize and run pipeline
        pipeline = ETLPipeline()
        success = pipeline.run_pipeline()
        
        if success:
            print("✅ ETL pipeline completed successfully!")
        else:
            print("❌ ETL pipeline failed!")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during ETL pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 