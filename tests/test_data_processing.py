# test_data_processing.py
# Purpose: Unit tests for data processing functionality

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from data.database_manager import DatabaseManager

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Step 1: Setup test data
        self.data_loader = DataLoader()
        
        # Step 2: Create sample test data
        self.sample_transactions = pd.DataFrame({
            'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003'],
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_001'],
            'transaction_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'amount': [100.0, 200.0, 150.0],
            'category': ['Shopping', 'Food', 'Shopping'],
            'merchant': ['Amazon', 'McDonald\'s', 'Target']
        })
        
        self.sample_customers = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'],
            'first_name': ['John', 'Jane'],
            'last_name': ['Doe', 'Smith'],
            'email': ['john@example.com', 'jane@example.com']
        })
    
    def test_validate_data_quality(self):
        # Step 1: Test data quality validation
        quality_report = self.data_loader.validate_data_quality(
            self.sample_transactions, 'transactions'
        )
        
        # Step 2: Assert expected results
        self.assertEqual(quality_report['total_records'], 3)
        self.assertIn('null_counts', quality_report)
        self.assertIn('duplicate_records', quality_report)
        self.assertIn('data_types', quality_report)
    
    def test_supported_formats(self):
        # Step 1: Test supported file formats
        self.assertIn('.csv', self.data_loader.supported_formats)
        self.assertIn('.xlsx', self.data_loader.supported_formats)
        self.assertIn('.json', self.data_loader.supported_formats)

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        # Step 1: Setup test data
        self.preprocessor = DataPreprocessor()
        
        # Step 2: Create sample data with issues
        self.dirty_transactions = pd.DataFrame({
            'transaction_id': ['TXN_001', 'TXN_002', 'TXN_001', 'TXN_003'],  # Duplicate
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_001', 'CUST_003'],
            'transaction_date': ['2024-01-01', '2024-01-02', '2024-01-01', '2025-01-01'],  # Future date
            'amount': [100.0, -50.0, 100.0, 0.0],  # Negative and zero amounts
            'category': ['shopping', 'FOOD', 'Shopping', 'entertainment'],  # Inconsistent case
            'merchant': ['Amazon', 'McDonald\'s', 'Amazon', 'Netflix'],
            'subcategory': ['Electronics', None, 'Electronics', 'Streaming'],  # Missing values
            'payment_method': ['Credit Card', 'Debit Card', None, 'PayPal']  # Missing values
        })
    
    def test_clean_transaction_data(self):
        # Step 1: Test transaction data cleaning
        cleaned_data = self.preprocessor.clean_transaction_data(self.dirty_transactions)
        
        # Step 2: Assert cleaning results
        self.assertLess(len(cleaned_data), len(self.dirty_transactions))  # Duplicates removed
        self.assertTrue(all(cleaned_data['amount'] > 0))  # No negative amounts
        self.assertTrue(all(cleaned_data['category'].str.istitle()))  # Title case
        self.assertFalse(cleaned_data['subcategory'].isnull().any())  # No null subcategories
    
    def test_create_customer_features(self):
        # Step 1: Create clean transaction data
        clean_transactions = self.preprocessor.clean_transaction_data(self.dirty_transactions)
        
        # Step 2: Test feature creation
        customer_features = self.preprocessor.create_customer_features(clean_transactions)
        
        # Step 3: Assert expected features
        expected_features = [
            'total_spent', 'avg_transaction_amount', 'transaction_count',
            'spending_std', 'first_transaction', 'last_transaction',
            'primary_category', 'unique_merchants'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, customer_features.columns)
    
    def test_detect_outliers(self):
        # Step 1: Create test data with outliers
        test_data = pd.DataFrame({
            'amount': [10, 20, 30, 40, 50, 1000, 2000]  # Last two are outliers
        })
        
        # Step 2: Test outlier detection
        outliers = self.preprocessor.detect_outliers(test_data, 'amount', method='iqr')
        
        # Step 3: Assert outlier detection
        self.assertTrue(outliers.iloc[-2])  # 1000 should be outlier
        self.assertTrue(outliers.iloc[-1])  # 2000 should be outlier
        self.assertFalse(outliers.iloc[0])  # 10 should not be outlier

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Step 1: Setup test database
        self.db_manager = DatabaseManager(":memory:")  # Use in-memory database for testing
        
        # Step 2: Create test data
        self.test_customers = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'],
            'first_name': ['John', 'Jane'],
            'last_name': ['Doe', 'Smith'],
            'email': ['john@example.com', 'jane@example.com'],
            'registration_date': ['2024-01-01', '2024-01-02'],
            'age': [30, 25],
            'gender': ['Male', 'Female'],
            'location': ['New York, NY', 'Los Angeles, CA']
        })
        
        self.test_transactions = pd.DataFrame({
            'transaction_id': ['TXN_001', 'TXN_002'],
            'customer_id': ['CUST_001', 'CUST_002'],
            'transaction_date': ['2024-01-01', '2024-01-02'],
            'amount': [100.0, 200.0],
            'category': ['Shopping', 'Food'],
            'subcategory': ['Electronics', 'Restaurant'],
            'merchant': ['Amazon', 'McDonald\'s'],
            'payment_method': ['Credit Card', 'Debit Card']
        })
    
    def test_insert_and_query_dataframe(self):
        # Step 1: Test inserting data
        self.db_manager.insert_dataframe(self.test_customers, 'customers')
        
        # Step 2: Test querying data
        result = self.db_manager.query_to_dataframe("SELECT * FROM customers")
        
        # Step 3: Assert results
        self.assertEqual(len(result), 2)
        self.assertIn('CUST_001', result['customer_id'].values)
        self.assertIn('CUST_002', result['customer_id'].values)
    
    def test_get_customer_data(self):
        # Step 1: Insert test data
        self.db_manager.insert_dataframe(self.test_customers, 'customers')
        
        # Step 2: Test getting customer data
        customers = self.db_manager.get_customer_data()
        
        # Step 3: Assert results
        self.assertEqual(len(customers), 2)
        self.assertIn('first_name', customers.columns)
        self.assertIn('last_name', customers.columns)
    
    def test_get_transaction_data(self):
        # Step 1: Insert test data
        self.db_manager.insert_dataframe(self.test_customers, 'customers')
        self.db_manager.insert_dataframe(self.test_transactions, 'transactions')
        
        # Step 2: Test getting transaction data
        transactions = self.db_manager.get_transaction_data()
        
        # Step 3: Assert results
        self.assertEqual(len(transactions), 2)
        self.assertIn('amount', transactions.columns)
        self.assertIn('category', transactions.columns)
    
    def test_get_database_stats(self):
        # Step 1: Insert test data
        self.db_manager.insert_dataframe(self.test_customers, 'customers')
        self.db_manager.insert_dataframe(self.test_transactions, 'transactions')
        
        # Step 2: Test getting database stats
        stats = self.db_manager.get_database_stats()
        
        # Step 3: Assert results
        self.assertEqual(stats['customers'], 2)
        self.assertEqual(stats['transactions'], 2)
        self.assertIn('customer_segments', stats)
        self.assertIn('campaigns', stats)

def run_tests():
    # Step 1: Create test suite
    test_suite = unittest.TestSuite()
    
    # Step 2: Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestDataPreprocessor))
    test_suite.addTest(unittest.makeSuite(TestDatabaseManager))
    
    # Step 3: Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Step 4: Return results
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 