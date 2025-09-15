# data_preprocessor.py
# Purpose: Clean, transform, and prepare data for analysis

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from src.utils.logger import setup_logger
from src.utils.helpers import safe_divide

class DataPreprocessor:
    def __init__(self):
        # Step 1: Initialize preprocessor with logger
        self.logger = setup_logger("DataPreprocessor")
    
    def clean_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Create a copy of the dataframe
        cleaned_df = df.copy()
        
        # Step 2: Remove duplicate transactions
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['transaction_id'])
        duplicates_removed = initial_count - len(cleaned_df)
        
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate transactions")
        
        # Step 3: Convert date column to datetime
        cleaned_df['transaction_date'] = pd.to_datetime(cleaned_df['transaction_date'])
        
        # Step 4: Remove transactions with invalid amounts
        cleaned_df = cleaned_df[cleaned_df['amount'] > 0]
        
        # Step 5: Remove future transactions
        cleaned_df = cleaned_df[cleaned_df['transaction_date'] <= datetime.now()]
        
        # Step 6: Clean text fields
        text_columns = ['category', 'subcategory', 'merchant']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].str.strip().str.title()
        
        # Step 7: Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        self.logger.info(f"Cleaned transaction data: {len(cleaned_df)} records remaining")
        return cleaned_df
    
    def clean_customer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Create a copy and remove duplicates
        cleaned_df = df.copy()
        cleaned_df = cleaned_df.drop_duplicates(subset=['customer_id'])
        
        # Step 2: Clean text fields
        text_columns = ['first_name', 'last_name', 'email', 'location']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].str.strip()
        
        # Step 3: Validate email addresses
        if 'email' in cleaned_df.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            valid_emails = cleaned_df['email'].str.match(email_pattern, na=False)
            cleaned_df.loc[~valid_emails, 'email'] = None
        
        # Step 4: Validate age values
        if 'age' in cleaned_df.columns:
            cleaned_df['age'] = pd.to_numeric(cleaned_df['age'], errors='coerce')
            cleaned_df.loc[(cleaned_df['age'] < 18) | (cleaned_df['age'] > 100), 'age'] = None
        
        return cleaned_df
    
    def create_customer_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Calculate customer-level aggregated features
        customer_features = transactions_df.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'transaction_date': ['min', 'max'],
            'category': lambda x: x.value_counts().index[0],  # Most frequent category
            'merchant': 'nunique'
        }).round(2)
        
        # Step 2: Flatten multi-level column names
        customer_features.columns = [
            'total_spent', 'avg_transaction_amount', 'transaction_count',
            'spending_std', 'first_transaction', 'last_transaction',
            'primary_category', 'unique_merchants'
        ]
        
        # Step 3: Calculate derived features
        customer_features['customer_lifetime_days'] = (
            customer_features['last_transaction'] - customer_features['first_transaction']
        ).dt.days
        
        customer_features['avg_monthly_spend'] = customer_features.apply(
            lambda row: safe_divide(
                row['total_spent'],
                max(1, row['customer_lifetime_days'] / 30)
            ), axis=1
        )
        
        customer_features['transaction_frequency'] = customer_features.apply(
            lambda row: safe_divide(
                row['transaction_count'],
                max(1, row['customer_lifetime_days'])
            ), axis=1
        )
        
        # Step 4: Calculate recency (days since last transaction)
        customer_features['recency_days'] = (
            datetime.now() - customer_features['last_transaction']
        ).dt.days
        
        # Step 5: Reset index to make customer_id a column
        customer_features = customer_features.reset_index()
        
        self.logger.info(f"Created features for {len(customer_features)} customers")
        return customer_features
    
    def create_category_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Create customer-category spending matrix
        category_spending = transactions_df.groupby(['customer_id', 'category'])['amount'].sum().reset_index()
        category_pivot = category_spending.pivot(
            index='customer_id', 
            columns='category', 
            values='amount'
        ).fillna(0)
        
        # Step 2: Calculate category spending percentages
        category_percentages = category_pivot.div(category_pivot.sum(axis=1), axis=0).fillna(0)
        
        # Step 3: Add column prefix for clarity
        category_percentages.columns = [f'category_pct_{col.lower().replace(" ", "_")}' 
                                      for col in category_percentages.columns]
        
        return category_percentages.reset_index()
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Handle missing values based on column type and business logic
        if 'subcategory' in df.columns:
            df['subcategory'] = df['subcategory'].fillna('Other')
        
        if 'payment_method' in df.columns:
            df['payment_method'] = df['payment_method'].fillna('Unknown')
        
        # Step 2: Remove rows with missing critical information
        critical_columns = ['customer_id', 'transaction_date', 'amount', 'category']
        df = df.dropna(subset=critical_columns)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
        # Step 1: Detect outliers using specified method
        if method == 'iqr':
            # Use a stricter quantile method for small samples so extreme values are flagged
            values = df[column].dropna().values
            if len(values) == 0:
                return pd.Series([False] * len(df), index=df.index)
            # Compute quartiles using the 'lower' interpolation for robustness on small samples
            Q1 = np.percentile(values, 25, method='lower') if hasattr(np, 'percentile') else df[column].quantile(0.25)
            Q3 = np.percentile(values, 75, method='lower') if hasattr(np, 'percentile') else df[column].quantile(0.75)
            IQR = Q3 - Q1
            # If IQR becomes zero due to ties, fallback to z-score thresholding
            if IQR == 0:
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std(ddof=0))
                return z_scores > 3
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std(ddof=0))
            return z_scores > 3
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")