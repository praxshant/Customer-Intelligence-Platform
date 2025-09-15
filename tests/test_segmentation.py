# test_segmentation.py
# Purpose: Unit tests for customer segmentation functionality

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from sklearn.cluster import KMeans
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.customer_segmentation import CustomerSegmentation
from utils.logger import setup_logger

class TestCustomerSegmentation(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and segmentation instance"""
        self.segmentation = CustomerSegmentation()
        self.logger = setup_logger("TestSegmentation")
        
        # Create sample customer features for testing
        np.random.seed(42)
        n_customers = 100
        
        self.sample_features = pd.DataFrame({
            'customer_id': [f'CUST_{i:03d}' for i in range(n_customers)],
            'total_spent': np.random.uniform(100, 5000, n_customers),
            'avg_transaction_amount': np.random.uniform(20, 200, n_customers),
            'transaction_count': np.random.randint(5, 50, n_customers),
            'avg_monthly_spend': np.random.uniform(50, 500, n_customers),
            'transaction_frequency': np.random.uniform(0.5, 5.0, n_customers),
            'recency_days': np.random.randint(1, 365, n_customers),
            'unique_merchants': np.random.randint(3, 20, n_customers)
        })
    
    def test_feature_preparation(self):
        """Test feature preparation and scaling"""
        # Test feature preparation
        prepared_features = self.segmentation._prepare_features(self.sample_features)
        
        # Check that features are scaled
        self.assertIsInstance(prepared_features, np.ndarray)
        self.assertEqual(prepared_features.shape[0], len(self.sample_features))
        self.assertEqual(prepared_features.shape[1], 7)  # 7 feature columns
        
        # Check that features are properly scaled (mean close to 0, std close to 1)
        self.assertAlmostEqual(prepared_features.mean(), 0, places=1)
        self.assertAlmostEqual(prepared_features.std(), 1, places=1)
    
    def test_optimal_cluster_finding(self):
        """Test finding optimal number of clusters"""
        prepared_features = self.segmentation._prepare_features(self.sample_features)
        
        # Test elbow method
        cluster_range = range(2, 11)
        inertias = []
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(prepared_features)
            inertias.append(kmeans.inertia_)
        
        elbow_scores = self.segmentation._find_elbow_point(inertias, cluster_range)
        self.assertIsInstance(elbow_scores, int)
        self.assertGreater(elbow_scores, 1)
        self.assertLessEqual(elbow_scores, 10)
        
        # Test silhouette method
        silhouette_scores = self.segmentation._find_silhouette_optimal(prepared_features)
        self.assertIsInstance(silhouette_scores, dict)
        self.assertIn('optimal_clusters', silhouette_scores)
        self.assertIn('scores', silhouette_scores)
        
        # Test combined method
        optimal_clusters = self.segmentation._find_optimal_clusters(prepared_features)
        self.assertIsInstance(optimal_clusters, int)
        self.assertGreater(optimal_clusters, 1)
        self.assertLessEqual(optimal_clusters, 10)
    
    def test_clustering(self):
        """Test K-means clustering"""
        # Test clustering with sample data
        segments_df = self.segmentation.perform_segmentation(self.sample_features)
        
        # Check output format
        self.assertIsInstance(segments_df, pd.DataFrame)
        self.assertIn('customer_id', segments_df.columns)
        self.assertIn('segment_id', segments_df.columns)
        self.assertIn('segment_name', segments_df.columns)
        self.assertIn('last_updated', segments_df.columns)
        
        # Check that all customers have segments
        self.assertEqual(len(segments_df), len(self.sample_features))
        self.assertTrue(segments_df['segment_id'].notna().all())
        self.assertTrue(segments_df['segment_name'].notna().all())
        
        # Check segment distribution
        segment_counts = segments_df['segment_id'].value_counts()
        self.assertGreater(len(segment_counts), 1)  # Should have multiple segments
    
    def test_segment_naming(self):
        """Test automatic segment naming"""
        # Create segments with known characteristics
        segments_df = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'segment_id': [0, 1, 2],
            'segment_name': ['', '', ''],
            'last_updated': [datetime.now()] * 3
        })
        
        # Mock segment centers for testing
        segment_centers = np.array([
            [1000, 100, 20, 200, 2.0, 30, 10],   # High value
            [500, 50, 10, 100, 1.0, 90, 5],      # Medium value
            [200, 20, 5, 50, 0.5, 180, 3]        # Low value
        ])
        
        named_segments = self.segmentation._assign_segment_names(
            segments_df, segment_centers
        )
        
        # Check that segment names are assigned
        self.assertTrue(named_segments['segment_name'].notna().all())
        self.assertFalse((named_segments['segment_name'] == '').any())
    
    def test_model_persistence(self):
        """Test saving and loading segmentation model"""
        # Perform segmentation to create model
        self.segmentation.perform_segmentation(self.sample_features)
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pkl')
            
            # Test saving model
            self.segmentation.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Test loading model
            loaded_segmentation = CustomerSegmentation()
            loaded_segmentation.load_model(model_path)
            
            # Test prediction with loaded model
            test_customer = self.sample_features.iloc[0:1]
            prediction = loaded_segmentation.predict_segment(test_customer)
            
            self.assertIsInstance(prediction, pd.DataFrame)
            self.assertEqual(len(prediction), 1)
            self.assertIn('segment_id', prediction.columns)
    
    def test_segment_profiling(self):
        """Test segment profiling functionality"""
        # Perform segmentation
        segments_df = self.segmentation.perform_segmentation(self.sample_features)
        
        # Test segment profiling
        profiles = self.segmentation.generate_segment_profiles(
            self.sample_features, segments_df
        )
        
        # Check profile structure
        self.assertIsInstance(profiles, dict)
        self.assertGreater(len(profiles), 0)
        
        # Check each segment profile
        for segment_id, profile in profiles.items():
            self.assertIn('size', profile)
            self.assertIn('avg_total_spent', profile)
            self.assertIn('avg_transaction_count', profile)
            self.assertIn('characteristics', profile)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.segmentation.perform_segmentation(empty_df)
        
        # Test with missing required columns
        invalid_df = pd.DataFrame({
            'customer_id': ['CUST_001'],
            'total_spent': [1000]
        })
        with self.assertRaises(ValueError):
            self.segmentation.perform_segmentation(invalid_df)
    
    def test_feature_engineering(self):
        """Test feature engineering from transaction data"""
        # Create sample transaction data
        transactions = pd.DataFrame({
            'customer_id': ['CUST_001'] * 10 + ['CUST_002'] * 8,
            'transaction_date': pd.date_range('2023-01-01', periods=18),
            'amount': np.random.uniform(10, 200, 18),
            'category': ['Food'] * 5 + ['Shopping'] * 5 + ['Food'] * 4 + ['Shopping'] * 4
        })
        
        # Test feature engineering
        features = self.segmentation._engineer_features(transactions)
        
        # Check output
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIn('customer_id', features.columns)
        self.assertIn('total_spent', features.columns)
        self.assertIn('transaction_count', features.columns)
        self.assertIn('avg_transaction_amount', features.columns)

if __name__ == '__main__':
    unittest.main() 