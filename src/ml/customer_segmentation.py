# customer_segmentation.py
# Purpose: Perform customer segmentation using clustering algorithms

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from src.utils.logger import setup_logger
from src.utils.config_loader import load_ml_config

class CustomerSegmentation:
    def __init__(self):
        # Step 1: Initialize segmentation model with configuration
        self.logger = setup_logger("CustomerSegmentation")
        self.config = load_ml_config()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        self.segment_profiles = {}
    
    def prepare_features(self, customer_features_df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Select relevant features for clustering
        feature_columns = [
            'total_spent', 'avg_transaction_amount', 'transaction_count',
            'avg_monthly_spend', 'transaction_frequency', 'recency_days',
            'unique_merchants'
        ]
        
        # Step 2: Filter to only include available columns
        available_features = [col for col in feature_columns if col in customer_features_df.columns]
        self.feature_columns = available_features
        
        # Step 3: Create feature matrix
        feature_matrix = customer_features_df[available_features].copy()
        
        # Step 4: Handle missing values
        feature_matrix = feature_matrix.fillna(feature_matrix.median())
        
        # Step 5: Log transformation for skewed features
        skewed_features = ['total_spent', 'avg_transaction_amount', 'avg_monthly_spend']
        for feature in skewed_features:
            if feature in feature_matrix.columns:
                feature_matrix[feature] = np.log1p(feature_matrix[feature])
        
        self.logger.info(f"Prepared {len(available_features)} features for clustering")
        return feature_matrix
    
    def find_optimal_clusters(self, X: pd.DataFrame, max_clusters: int = 10) -> Dict[str, Any]:
        # Step 1: Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 2: Test different numbers of clusters
        cluster_range = range(2, max_clusters + 1)
        inertias = []
        silhouette_scores = []
        
        for n_clusters in cluster_range:
            # Fit KMeans model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(sil_score)
        
        # Step 3: Find optimal number using elbow method and silhouette score
        optimal_clusters = self._find_elbow_point(inertias, cluster_range)
        best_silhouette_idx = np.argmax(silhouette_scores)
        best_silhouette_clusters = cluster_range[best_silhouette_idx]
        
        # Step 4: Choose the best number of clusters
        optimal_n_clusters = min(optimal_clusters, best_silhouette_clusters)
        
        results = {
            'optimal_clusters': optimal_n_clusters,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'cluster_range': list(cluster_range)
        }
        
        self.logger.info(f"Optimal number of clusters: {optimal_n_clusters}")
        return results
    
    def perform_segmentation(self, customer_features_df: pd.DataFrame, n_clusters: Optional[int] = None) -> pd.DataFrame:
        # Step 1: Prepare features for clustering
        feature_matrix = self.prepare_features(customer_features_df)
        
        # Step 2: Find optimal number of clusters if not specified
        if n_clusters is None:
            optimal_results = self.find_optimal_clusters(feature_matrix)
            n_clusters = optimal_results['optimal_clusters']
        
        # Step 3: Scale features
        X_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Step 4: Perform clustering
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.model.fit_predict(X_scaled)
        
        # Step 5: Create segmentation results
        segmentation_results = customer_features_df.copy()
        segmentation_results['segment_id'] = cluster_labels
        
        # Step 6: Generate segment names and profiles
        segment_names = self._generate_segment_names(segmentation_results, n_clusters)
        segmentation_results['segment_name'] = segmentation_results['segment_id'].map(segment_names)
        
        # Step 7: Create segment profiles
        self.segment_profiles = self._create_segment_profiles(segmentation_results)
        
        # Add last_updated for API compatibility
        segmentation_results['last_updated'] = pd.Timestamp.now()
        self.logger.info(f"Completed segmentation with {n_clusters} clusters")
        return segmentation_results

    # Backward-compatibility helpers expected by tests
    def _prepare_features(self, customer_features_df: pd.DataFrame) -> pd.DataFrame:
        # Return scaled numpy array to align with tests
        features_df = self.prepare_features(customer_features_df)
        return self.scaler.fit_transform(features_df)

    def _engineer_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        # Basic feature engineering mirroring DataPreprocessor.create_customer_features
        df = transactions_df.copy()
        if 'merchant' not in df.columns:
            df['merchant'] = ''
        grouped = df.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count'],
            'transaction_date': ['min', 'max'],
            'merchant': 'nunique'
        }).reset_index()
        grouped.columns = [
            'customer_id', 'total_spent', 'avg_transaction_amount', 'transaction_count',
            'first_transaction', 'last_transaction', 'unique_merchants'
        ]
        grouped['avg_monthly_spend'] = grouped['total_spent'] / ((pd.to_datetime(grouped['last_transaction']) - pd.to_datetime(grouped['first_transaction'])).dt.days.clip(lower=1) / 30)
        grouped['transaction_frequency'] = grouped['transaction_count'] / ((pd.to_datetime(grouped['last_transaction']) - pd.to_datetime(grouped['first_transaction'])).dt.days.clip(lower=1))
        grouped['recency_days'] = (pd.Timestamp.now() - pd.to_datetime(grouped['last_transaction'])).dt.days
        return grouped

    def _assign_segment_names(self, segments_df: pd.DataFrame, centers: np.ndarray) -> pd.DataFrame:
        # Assign names based on centers heuristics using total_spent, transaction_frequency, recency_days
        names = {}
        for idx, center in enumerate(centers):
            total_spent, avg_tx, tx_count, avg_monthly, tx_freq, recency, unique_merchants = center
            spend_level = 'High Value' if total_spent >= np.max(centers[:, 0]) else ('Medium Value' if total_spent >= np.median(centers[:, 0]) else 'Low Value')
            freq_level = 'Frequent' if tx_freq >= np.median(centers[:, 4]) else 'Occasional'
            rec_level = 'Recent' if recency <= np.median(centers[:, 5]) else 'Inactive'
            names[idx] = f"{spend_level} - {freq_level} - {rec_level}"
        segments_df['segment_name'] = segments_df['segment_id'].map(names)
        return segments_df
    
    def _find_elbow_point(self, inertias: List[float], cluster_range: range) -> int:
        # Step 1: Calculate the rate of change in inertia
        inertia_changes = np.diff(inertias)
        inertia_change_rates = np.diff(inertia_changes)
        
        # Step 2: Find the elbow point (maximum rate of change)
        elbow_idx = np.argmax(inertia_change_rates) + 2  # +2 because of double diff
        return cluster_range[elbow_idx]
    
    def generate_segment_profiles(self, customer_features_df: pd.DataFrame, segments_df: pd.DataFrame) -> Dict[str, Any]:
        # Generate segment profiles for test compatibility
        profiles = {}
        for segment_id in segments_df['segment_id'].unique():
            segment_data = segments_df[segments_df['segment_id'] == segment_id]
            segment_name = segment_data['segment_name'].iloc[0]
            
            profile = {
                'segment_id': segment_id,
                'segment_name': segment_name,
                'size': len(segment_data),
                'avg_total_spent': segment_data['total_spent'].mean(),
                'avg_transaction_count': segment_data['transaction_count'].mean(),
                'characteristics': {
                    'spending_pattern': 'high' if segment_data['total_spent'].mean() > 3000 else 'medium' if segment_data['total_spent'].mean() > 1500 else 'low',
                    'frequency': 'high' if segment_data['transaction_count'].mean() > 50 else 'medium' if segment_data['transaction_count'].mean() > 25 else 'low',
                    'recency': 'recent' if segment_data['recency_days'].mean() < 30 else 'moderate' if segment_data['recency_days'].mean() < 90 else 'inactive'
                }
            }
            profiles[segment_name] = profile
        
        return profiles
    
    def _find_silhouette_optimal(self, prepared_features: pd.DataFrame) -> Dict[str, Any]:
        # Find optimal number of clusters using silhouette score
        from sklearn.metrics import silhouette_score
        
        best_score = -1
        best_k = 2
        scores = {}
        
        for k in range(2, min(11, len(prepared_features) // 2 + 1)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(prepared_features)
                score = silhouette_score(prepared_features, cluster_labels)
                scores[k] = score
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        return {
            'optimal_clusters': best_k,
            'scores': scores
        }
    
    def _find_optimal_clusters(self, prepared_features: pd.DataFrame) -> int:
        # Use elbow method as default
        cluster_range = range(2, min(11, len(prepared_features) // 2 + 1))
        inertias = []
        
        for k in cluster_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(prepared_features)
                inertias.append(kmeans.inertia_)
            except:
                inertias.append(0)
        
        return self._find_elbow_point(inertias, cluster_range)
    
    def _generate_segment_names(self, segmentation_results: pd.DataFrame, n_clusters: int) -> Dict[int, str]:
        # Step 1: Analyze each segment to generate descriptive names
        segment_names = {}
        
        for segment_id in range(n_clusters):
            segment_data = segmentation_results[segmentation_results['segment_id'] == segment_id]
            
            # Step 2: Calculate segment characteristics
            avg_spend = segment_data['total_spent'].mean()
            avg_frequency = segment_data['transaction_frequency'].mean()
            avg_recency = segment_data['recency_days'].mean()
            
            # Step 3: Generate descriptive name based on characteristics
            if avg_spend > segmentation_results['total_spent'].quantile(0.75):
                spend_level = "High-Value"
            elif avg_spend > segmentation_results['total_spent'].quantile(0.5):
                spend_level = "Medium-Value"
            else:
                spend_level = "Low-Value"
            
            if avg_frequency > segmentation_results['transaction_frequency'].quantile(0.75):
                frequency_level = "Frequent"
            elif avg_frequency > segmentation_results['transaction_frequency'].quantile(0.5):
                frequency_level = "Regular"
            else:
                frequency_level = "Occasional"
            
            if avg_recency < segmentation_results['recency_days'].quantile(0.25):
                recency_level = "Recent"
            elif avg_recency < segmentation_results['recency_days'].quantile(0.5):
                recency_level = "Active"
            else:
                recency_level = "Inactive"
            
            segment_names[segment_id] = f"{spend_level}-{frequency_level}-{recency_level}"
        
        return segment_names
    
    def _create_segment_profiles(self, segmentation_results: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        # Step 1: Create detailed profiles for each segment
        profiles = {}
        
        for segment_id in segmentation_results['segment_id'].unique():
            segment_data = segmentation_results[segmentation_results['segment_id'] == segment_id]
            segment_name = segment_data['segment_name'].iloc[0]
            
            # Step 2: Calculate segment statistics
            profile = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(segmentation_results) * 100,
                'avg_total_spent': segment_data['total_spent'].mean(),
                'avg_transaction_amount': segment_data['avg_transaction_amount'].mean(),
                'avg_transaction_count': segment_data['transaction_count'].mean(),
                'avg_recency_days': segment_data['recency_days'].mean(),
                'avg_monthly_spend': segment_data['avg_monthly_spend'].mean(),
                # Some tests' sample data may not include primary_category
                'primary_categories': segment_data['primary_category'].value_counts().head(3).to_dict() if 'primary_category' in segment_data.columns else {}
            }
            
            profiles[segment_name] = profile
        
        return profiles
    
    def save_model(self, model_path: str = "data/models/segmentation_model.pkl"):
        # Step 1: Save the trained model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'segment_profiles': self.segment_profiles
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "data/models/segmentation_model.pkl"):
        # Step 1: Load the trained model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.segment_profiles = model_data['segment_profiles']
            
            self.logger.info(f"Model loaded from {model_path}")
        else:
            self.logger.warning(f"Model file not found: {model_path}")
    
    def predict_segment(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Prepare features for prediction
        feature_matrix = customer_features[self.feature_columns].copy()
        feature_matrix = feature_matrix.fillna(feature_matrix.median())
        
        # Step 2: Apply log transformation if needed
        skewed_features = ['total_spent', 'avg_transaction_amount', 'avg_monthly_spend']
        for feature in skewed_features:
            if feature in feature_matrix.columns:
                feature_matrix[feature] = np.log1p(feature_matrix[feature])
        
        # Step 3: Scale features and predict
        X_scaled = self.scaler.transform(feature_matrix)
        cluster_labels = self.model.predict(X_scaled)
        
        # Step 4: Add predictions to dataframe
        result_df = customer_features.copy()
        result_df['segment_id'] = cluster_labels
        
        # Step 5: Map segment names
        segment_names = {i: name for i, name in enumerate(self.segment_profiles.keys())}
        result_df['segment_name'] = result_df['segment_id'].map(segment_names)
        
        return result_df 