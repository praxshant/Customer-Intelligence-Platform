# data_utils.py
# Purpose: Data utilities for dashboard data processing

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from src.utils.helpers import calculate_date_range, format_currency

def calculate_dashboard_metrics(transactions_df: pd.DataFrame, customers_df: pd.DataFrame) -> Dict[str, Any]:
    # Step 1: Calculate comprehensive dashboard metrics
    metrics = {}
    
    # Step 2: Basic transaction metrics
    if not transactions_df.empty:
        metrics['total_revenue'] = transactions_df['amount'].sum()
        metrics['total_transactions'] = len(transactions_df)
        metrics['avg_transaction_value'] = transactions_df['amount'].mean()
        metrics['unique_customers'] = transactions_df['customer_id'].nunique()
        metrics['total_merchants'] = transactions_df['merchant'].nunique()
        metrics['total_categories'] = transactions_df['category'].nunique()
    else:
        metrics.update({
            'total_revenue': 0,
            'total_transactions': 0,
            'avg_transaction_value': 0,
            'unique_customers': 0,
            'total_merchants': 0,
            'total_categories': 0
        })
    
    # Step 3: Customer metrics
    if not customers_df.empty:
        metrics['total_customers'] = len(customers_df)
        if 'segment_name' in customers_df.columns:
            metrics['total_segments'] = customers_df['segment_name'].nunique()
        else:
            metrics['total_segments'] = 0
    else:
        metrics.update({
            'total_customers': 0,
            'total_segments': 0
        })
    
    # Step 4: Time-based metrics
    if not transactions_df.empty:
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
        metrics['date_range'] = {
            'start_date': transactions_df['transaction_date'].min(),
            'end_date': transactions_df['transaction_date'].max(),
            'total_days': (transactions_df['transaction_date'].max() - transactions_df['transaction_date'].min()).days + 1
        }
    
    return metrics

def calculate_period_comparison(transactions_df: pd.DataFrame, period: str = "30d") -> Dict[str, Any]:
    # Step 1: Calculate period comparison metrics
    if transactions_df.empty:
        return {}
    
    # Step 2: Get current and previous period data
    end_date = datetime.now()
    start_date, _ = calculate_date_range(period)
    
    current_period = transactions_df[
        (transactions_df['transaction_date'] >= start_date) &
        (transactions_df['transaction_date'] <= end_date)
    ]
    
    period_days = (end_date - start_date).days
    prev_start = start_date - timedelta(days=period_days)
    prev_end = start_date
    
    previous_period = transactions_df[
        (transactions_df['transaction_date'] >= prev_start) &
        (transactions_df['transaction_date'] < prev_end)
    ]
    
    # Step 3: Calculate comparison metrics
    comparison = {
        'current_period': {
            'revenue': current_period['amount'].sum(),
            'transactions': len(current_period),
            'customers': current_period['customer_id'].nunique(),
            'avg_value': current_period['amount'].mean()
        },
        'previous_period': {
            'revenue': previous_period['amount'].sum(),
            'transactions': len(previous_period),
            'customers': previous_period['customer_id'].nunique(),
            'avg_value': previous_period['amount'].mean()
        }
    }
    
    # Step 4: Calculate percentage changes
    for metric in ['revenue', 'transactions', 'customers', 'avg_value']:
        current_val = comparison['current_period'][metric]
        previous_val = comparison['previous_period'][metric]
        
        if previous_val > 0:
            change_pct = ((current_val - previous_val) / previous_val) * 100
        else:
            change_pct = 0
        
        comparison[f'{metric}_change_pct'] = change_pct
    
    return comparison

def get_top_performers(transactions_df: pd.DataFrame, customers_df: pd.DataFrame, 
                      top_n: int = 10) -> Dict[str, pd.DataFrame]:
    # Step 1: Get top performers in different categories
    top_performers = {}
    
    if not transactions_df.empty:
        # Step 2: Top customers by revenue
        top_customers = transactions_df.groupby('customer_id').agg({
            'amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        top_customers.columns = ['customer_id', 'total_revenue', 'transaction_count']
        top_customers = top_customers.nlargest(top_n, 'total_revenue')
        
        # Step 3: Merge with customer data
        if not customers_df.empty:
            top_customers = top_customers.merge(
                customers_df[['customer_id', 'first_name', 'last_name', 'email']], 
                on='customer_id', how='left'
            )
        
        top_performers['top_customers'] = top_customers
        
        # Step 4: Top merchants by revenue
        top_merchants = transactions_df.groupby('merchant').agg({
            'amount': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()
        top_merchants.columns = ['merchant', 'total_revenue', 'transaction_count', 'unique_customers']
        top_merchants = top_merchants.nlargest(top_n, 'total_revenue')
        top_performers['top_merchants'] = top_merchants
        
        # Step 5: Top categories by revenue
        top_categories = transactions_df.groupby('category').agg({
            'amount': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()
        top_categories.columns = ['category', 'total_revenue', 'transaction_count', 'unique_customers']
        top_categories = top_categories.nlargest(top_n, 'total_revenue')
        top_performers['top_categories'] = top_categories
    
    return top_performers

def calculate_segment_metrics(customers_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Calculate segment-level metrics
    if customers_df.empty or transactions_df.empty or 'segment_name' not in customers_df.columns:
        return pd.DataFrame()
    
    # Step 2: Merge customer and transaction data
    merged_data = transactions_df.merge(
        customers_df[['customer_id', 'segment_name']], 
        on='customer_id', how='left'
    )
    
    # Step 3: Calculate segment metrics
    segment_metrics = merged_data.groupby('segment_name').agg({
        'amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique',
        'merchant': 'nunique',
        'category': 'nunique'
    }).reset_index()
    
    # Step 4: Flatten column names
    segment_metrics.columns = [
        'segment_name', 'total_revenue', 'avg_transaction_value', 'transaction_count',
        'unique_customers', 'unique_merchants', 'unique_categories'
    ]
    
    # Step 5: Calculate additional metrics
    segment_metrics['avg_customer_revenue'] = segment_metrics['total_revenue'] / segment_metrics['unique_customers']
    segment_metrics['avg_customer_transactions'] = segment_metrics['transaction_count'] / segment_metrics['unique_customers']
    segment_metrics['revenue_percentage'] = (segment_metrics['total_revenue'] / segment_metrics['total_revenue'].sum()) * 100
    
    return segment_metrics

def get_trend_data(transactions_df: pd.DataFrame, group_by: str = 'date') -> pd.DataFrame:
    # Step 1: Get trend data for different time periods
    if transactions_df.empty:
        return pd.DataFrame()
    
    # Step 2: Ensure transaction_date is datetime
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    
    # Step 3: Group by specified time period
    if group_by == 'date':
        transactions_df['period'] = transactions_df['transaction_date'].dt.date
    elif group_by == 'week':
        transactions_df['period'] = transactions_df['transaction_date'].dt.to_period('W')
    elif group_by == 'month':
        transactions_df['period'] = transactions_df['transaction_date'].dt.to_period('M')
    elif group_by == 'quarter':
        transactions_df['period'] = transactions_df['transaction_date'].dt.to_period('Q')
    else:
        transactions_df['period'] = transactions_df['transaction_date'].dt.date
    
    # Step 4: Calculate trend metrics
    trend_data = transactions_df.groupby('period').agg({
        'amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique',
        'merchant': 'nunique'
    }).reset_index()
    
    # Step 5: Flatten column names
    trend_data.columns = [
        'period', 'total_revenue', 'avg_transaction_value', 'transaction_count',
        'unique_customers', 'unique_merchants'
    ]
    
    # Step 6: Sort by period
    trend_data = trend_data.sort_values('period')
    
    return trend_data

def calculate_customer_lifetime_value(transactions_df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Calculate customer lifetime value metrics
    if transactions_df.empty:
        return pd.DataFrame()
    
    # Step 2: Calculate CLV metrics
    clv_data = transactions_df.groupby('customer_id').agg({
        'amount': ['sum', 'mean', 'count'],
        'transaction_date': ['min', 'max']
    }).reset_index()
    
    # Step 3: Flatten column names
    clv_data.columns = [
        'customer_id', 'total_spent', 'avg_transaction_value', 'transaction_count',
        'first_transaction', 'last_transaction'
    ]
    
    # Step 4: Calculate additional CLV metrics
    clv_data['customer_lifetime_days'] = (
        pd.to_datetime(clv_data['last_transaction']) - 
        pd.to_datetime(clv_data['first_transaction'])
    ).dt.days
    
    clv_data['avg_monthly_spend'] = clv_data['total_spent'] / (clv_data['customer_lifetime_days'] / 30).clip(lower=1)
    clv_data['transaction_frequency'] = clv_data['transaction_count'] / clv_data['customer_lifetime_days'].clip(lower=1)
    
    # Step 5: Calculate recency
    clv_data['recency_days'] = (
        datetime.now() - pd.to_datetime(clv_data['last_transaction'])
    ).dt.days
    
    return clv_data

def get_geographic_insights(customers_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Dict[str, Any]:
    # Step 1: Get geographic insights if location data is available
    insights = {}
    
    if 'location' in customers_df.columns and not customers_df.empty and not transactions_df.empty:
        # Step 2: Merge customer and transaction data
        merged_data = transactions_df.merge(
            customers_df[['customer_id', 'location']], 
            on='customer_id', how='left'
        )
        
        # Step 3: Calculate location-based metrics
        location_metrics = merged_data.groupby('location').agg({
            'amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique',
            'merchant': 'nunique'
        }).reset_index()
        
        # Step 4: Flatten column names
        location_metrics.columns = [
            'location', 'total_revenue', 'avg_transaction_value', 'transaction_count',
            'unique_customers', 'unique_merchants'
        ]
        
        insights['location_metrics'] = location_metrics
        insights['top_locations'] = location_metrics.nlargest(10, 'total_revenue')
    
    return insights

def calculate_seasonal_patterns(transactions_df: pd.DataFrame) -> Dict[str, Any]:
    # Step 1: Calculate seasonal patterns
    patterns = {}
    
    if not transactions_df.empty:
        # Step 2: Add time-based columns
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
        transactions_df['month'] = transactions_df['transaction_date'].dt.month
        transactions_df['day_of_week'] = transactions_df['transaction_date'].dt.dayofweek
        transactions_df['hour'] = transactions_df['transaction_date'].dt.hour
        transactions_df['season'] = transactions_df['transaction_date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Step 3: Monthly patterns
        monthly_patterns = transactions_df.groupby('month').agg({
            'amount': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()
        patterns['monthly'] = monthly_patterns
        
        # Step 4: Day of week patterns
        dow_patterns = transactions_df.groupby('day_of_week').agg({
            'amount': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()
        patterns['day_of_week'] = dow_patterns
        
        # Step 5: Seasonal patterns
        seasonal_patterns = transactions_df.groupby('season').agg({
            'amount': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()
        patterns['seasonal'] = seasonal_patterns
    
    return patterns 