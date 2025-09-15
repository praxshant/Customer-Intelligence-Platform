# recommendation_engine.py
# Purpose: Generate personalized recommendations for customers

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Optional, Any
from src.utils.logger import setup_logger
from src.utils.config_loader import load_ml_config

class RecommendationEngine:
    def __init__(self):
        # Step 1: Initialize recommendation engine
        self.logger = setup_logger("RecommendationEngine")
        self.config = load_ml_config()
        self.customer_item_matrix = None
        self.similarity_matrix = None
        self.item_features = None
    
    def create_customer_item_matrix(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Create customer-item interaction matrix
        customer_item_matrix = transactions_df.pivot_table(
            index='customer_id',
            columns='category',
            values='amount',
            aggfunc='sum',
            fill_value=0
        )
        
        # Step 2: Normalize the matrix
        customer_item_matrix = customer_item_matrix.div(customer_item_matrix.sum(axis=1), axis=0)
        
        self.customer_item_matrix = customer_item_matrix
        self.logger.info(f"Created customer-item matrix with shape {customer_item_matrix.shape}")
        
        return customer_item_matrix
    
    def calculate_customer_similarity(self, method: str = 'cosine') -> pd.DataFrame:
        # Step 1: Calculate customer similarity matrix
        if self.customer_item_matrix is None:
            raise ValueError("Customer-item matrix not created. Call create_customer_item_matrix first.")
        
        if method == 'cosine':
            similarity_matrix = cosine_similarity(self.customer_item_matrix)
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
        
        # Step 2: Convert to DataFrame with customer IDs as index and columns
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.customer_item_matrix.index,
            columns=self.customer_item_matrix.index
        )
        
        self.similarity_matrix = similarity_df
        self.logger.info(f"Calculated customer similarity matrix using {method} method")
        
        return similarity_df
    
    def get_similar_customers(self, customer_id: str, n_similar: int = 5) -> List[Tuple[str, float]]:
        # Step 1: Get similarity scores for the customer
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not calculated. Call calculate_customer_similarity first.")
        
        if customer_id not in self.similarity_matrix.index:
            raise ValueError(f"Customer {customer_id} not found in similarity matrix")
        
        # Step 2: Get similarity scores and sort
        customer_similarities = self.similarity_matrix.loc[customer_id].sort_values(ascending=False)
        
        # Step 3: Remove self-similarity and get top similar customers
        similar_customers = customer_similarities.drop(customer_id).head(n_similar)
        
        return list(zip(similar_customers.index, similar_customers.values))
    
    def generate_category_recommendations(self, customer_id: str, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        # Step 1: Get similar customers
        similar_customers = self.get_similar_customers(customer_id, n_similar=10)
        
        # Step 2: Get categories that similar customers prefer
        similar_customer_ids = [customer for customer, _ in similar_customers]
        
        # Step 3: Calculate weighted category preferences
        category_scores = {}
        total_weight = 0
        
        for similar_customer_id, similarity_score in similar_customers:
            customer_preferences = self.customer_item_matrix.loc[similar_customer_id]
            weight = similarity_score
            
            for category, preference_score in customer_preferences.items():
                if category not in category_scores:
                    category_scores[category] = 0
                category_scores[category] += preference_score * weight
                total_weight += weight
        
        # Step 4: Normalize scores and sort
        if total_weight > 0:
            category_scores = {k: v / total_weight for k, v in category_scores.items()}
        
        # Step 5: Sort by score and return top recommendations
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for category, score in sorted_categories[:n_recommendations]:
            recommendations.append({
                'category': category,
                'score': score,
                'recommendation_type': 'collaborative_filtering'
            })
        
        return recommendations
    
    def generate_merchant_recommendations(self, transactions_df: pd.DataFrame, customer_id: str, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        # Step 1: Get customer's preferred categories
        customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id]
        preferred_categories = customer_transactions['category'].value_counts().head(3).index.tolist()
        
        # Step 2: Find merchants in preferred categories that customer hasn't visited
        customer_merchants = set(customer_transactions['merchant'].unique())
        
        # Step 3: Get all merchants in preferred categories
        category_merchants = transactions_df[
            transactions_df['category'].isin(preferred_categories)
        ]['merchant'].unique()
        
        # Step 4: Filter out already visited merchants
        new_merchants = [m for m in category_merchants if m not in customer_merchants]
        
        # Step 5: Calculate merchant popularity scores
        merchant_scores = {}
        for merchant in new_merchants:
            merchant_transactions = transactions_df[transactions_df['merchant'] == merchant]
            popularity_score = len(merchant_transactions) / len(transactions_df)
            avg_amount = merchant_transactions['amount'].mean()
            
            merchant_scores[merchant] = {
                'popularity': popularity_score,
                'avg_amount': avg_amount,
                'total_score': popularity_score * avg_amount
            }
        
        # Step 6: Sort by total score and return recommendations
        sorted_merchants = sorted(merchant_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        
        recommendations = []
        for merchant, scores in sorted_merchants[:n_recommendations]:
            recommendations.append({
                'merchant': merchant,
                'category': transactions_df[transactions_df['merchant'] == merchant]['category'].iloc[0],
                'popularity_score': scores['popularity'],
                'avg_amount': scores['avg_amount'],
                'recommendation_type': 'merchant_based'
            })
        
        return recommendations
    
    def generate_personalized_offers(self, customer_id: str, transactions_df: pd.DataFrame, 
                                   customer_features: pd.DataFrame) -> List[Dict[str, Any]]:
        # Step 1: Get customer characteristics
        customer_data = customer_features[customer_features['customer_id'] == customer_id]
        if customer_data.empty:
            return []
        
        customer_row = customer_data.iloc[0]
        
        # Step 2: Generate offers based on customer characteristics
        offers = []
        
        # High-value customer offer
        if customer_row['total_spent'] > customer_features['total_spent'].quantile(0.8):
            offers.append({
                'offer_type': 'loyalty_reward',
                'description': 'Exclusive VIP discount for high-value customers',
                'discount_percentage': 15,
                'min_purchase': 100,
                'validity_days': 30
            })
        
        # Inactive customer win-back offer
        if customer_row['recency_days'] > customer_features['recency_days'].quantile(0.8):
            offers.append({
                'offer_type': 'win_back',
                'description': 'Special comeback offer for returning customers',
                'discount_percentage': 25,
                'min_purchase': 50,
                'validity_days': 60
            })
        
        # Frequent customer offer
        if customer_row['transaction_frequency'] > customer_features['transaction_frequency'].quantile(0.8):
            offers.append({
                'offer_type': 'frequency_reward',
                'description': 'Reward for your frequent shopping',
                'discount_percentage': 10,
                'min_purchase': 75,
                'validity_days': 45
            })
        
        # Category-specific offer
        if 'primary_category' in customer_row:
            primary_category = customer_row['primary_category']
            offers.append({
                'offer_type': 'category_specific',
                'description': f'Special offer on {primary_category}',
                'discount_percentage': 20,
                'min_purchase': 60,
                'validity_days': 30,
                'category': primary_category
            })
        
        return offers
    
    def get_recommendation_summary(self, customer_id: str, transactions_df: pd.DataFrame, 
                                 customer_features: pd.DataFrame) -> Dict[str, Any]:
        # Step 1: Generate all types of recommendations
        category_recs = self.generate_category_recommendations(customer_id)
        merchant_recs = self.generate_merchant_recommendations(transactions_df, customer_id)
        personalized_offers = self.generate_personalized_offers(customer_id, transactions_df, customer_features)
        
        # Step 2: Create summary
        summary = {
            'customer_id': customer_id,
            'category_recommendations': category_recs,
            'merchant_recommendations': merchant_recs,
            'personalized_offers': personalized_offers,
            'total_recommendations': len(category_recs) + len(merchant_recs) + len(personalized_offers)
        }
        
        return summary 