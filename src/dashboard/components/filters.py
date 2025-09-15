# filters.py
# Purpose: Filters component for data filtering

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from src.utils.helpers import calculate_date_range

def render_filters(transactions_df: pd.DataFrame, customers_df: pd.DataFrame) -> Tuple[Optional[Tuple[datetime, datetime]], List[str], List[str]]:
    # Step 1: Create filters section
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = render_date_filter()
    
    with col2:
        selected_categories = render_category_filter(transactions_df)
    
    with col3:
        selected_segments = render_segment_filter(customers_df)
    
    return date_range, selected_categories, selected_segments

def render_date_filter() -> Optional[Tuple[datetime, datetime]]:
    # Step 1: Date range filter
    st.write("**Date Range**")
    
    # Step 2: Quick date options
    date_option = st.selectbox(
        "Select Date Range",
        ["Custom", "Last 7 days", "Last 30 days", "Last 90 days", "Last 1 year", "All time"],
        index=1
    )
    
    # Step 3: Handle date selection
    if date_option == "Custom":
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        end_date = st.date_input("End Date", value=datetime.now())
        return (datetime.combine(start_date, datetime.min.time()), 
                datetime.combine(end_date, datetime.max.time()))
    
    elif date_option == "Last 7 days":
        start_date, end_date = calculate_date_range("7d")
        return (start_date, end_date)
    
    elif date_option == "Last 30 days":
        start_date, end_date = calculate_date_range("30d")
        return (start_date, end_date)
    
    elif date_option == "Last 90 days":
        start_date, end_date = calculate_date_range("90d")
        return (start_date, end_date)
    
    elif date_option == "Last 1 year":
        start_date, end_date = calculate_date_range("1y")
        return (start_date, end_date)
    
    else:  # All time
        return None

def render_category_filter(transactions_df: pd.DataFrame) -> List[str]:
    # Step 1: Category filter
    st.write("**Categories**")
    
    if transactions_df.empty:
        return []
    
    # Step 2: Get unique categories
    categories = sorted(transactions_df['category'].unique())
    
    # Step 3: Multi-select categories
    selected_categories = st.multiselect(
        "Select Categories",
        options=categories,
        default=categories[:5] if len(categories) > 5 else categories,
        help="Select categories to filter transactions"
    )
    
    return selected_categories

def render_segment_filter(customers_df: pd.DataFrame) -> List[str]:
    # Step 1: Segment filter
    st.write("**Customer Segments**")
    
    if customers_df.empty or 'segment_name' not in customers_df.columns:
        return []
    
    # Step 2: Get unique segments
    segments = sorted(customers_df['segment_name'].unique())
    
    # Step 3: Multi-select segments
    selected_segments = st.multiselect(
        "Select Segments",
        options=segments,
        default=segments,
        help="Select customer segments to filter data"
    )
    
    return selected_segments

def render_merchant_filter(transactions_df: pd.DataFrame) -> List[str]:
    # Step 1: Merchant filter
    st.write("**Merchants**")
    
    if transactions_df.empty:
        return []
    
    # Step 2: Get unique merchants
    merchants = sorted(transactions_df['merchant'].unique())
    
    # Step 3: Multi-select merchants
    selected_merchants = st.multiselect(
        "Select Merchants",
        options=merchants,
        default=[],
        help="Select merchants to filter transactions"
    )
    
    return selected_merchants

def render_amount_filter() -> Tuple[Optional[float], Optional[float]]:
    # Step 1: Amount range filter
    st.write("**Amount Range**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_amount = st.number_input(
            "Min Amount ($)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            help="Minimum transaction amount"
        )
    
    with col2:
        max_amount = st.number_input(
            "Max Amount ($)",
            min_value=0.0,
            value=1000.0,
            step=10.0,
            help="Maximum transaction amount"
        )
    
    return min_amount if min_amount > 0 else None, max_amount if max_amount > 0 else None

def render_customer_filter(customers_df: pd.DataFrame) -> List[str]:
    # Step 1: Customer filter
    st.write("**Customers**")
    
    if customers_df.empty:
        return []
    
    # Step 2: Search customers
    customer_search = st.text_input(
        "Search Customers",
        placeholder="Enter customer name or ID",
        help="Search customers by name or ID"
    )
    
    # Step 3: Filter customers based on search
    if customer_search:
        filtered_customers = customers_df[
            customers_df['customer_id'].str.contains(customer_search, case=False) |
            customers_df['first_name'].str.contains(customer_search, case=False) |
            customers_df['last_name'].str.contains(customer_search, case=False)
        ]
    else:
        filtered_customers = customers_df
    
    # Step 4: Multi-select customers
    customer_options = filtered_customers['customer_id'].tolist()
    selected_customers = st.multiselect(
        "Select Customers",
        options=customer_options,
        default=[],
        help="Select specific customers to filter data"
    )
    
    return selected_customers

def render_advanced_filters(transactions_df: pd.DataFrame, customers_df: pd.DataFrame):
    # Step 1: Advanced filters section
    with st.expander("🔧 Advanced Filters"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount filter
            min_amount, max_amount = render_amount_filter()
            
            # Merchant filter
            selected_merchants = render_merchant_filter(transactions_df)
        
        with col2:
            # Customer filter
            selected_customers = render_customer_filter(customers_df)
            
            # Payment method filter
            if 'payment_method' in transactions_df.columns:
                payment_methods = sorted(transactions_df['payment_method'].unique())
                selected_payment_methods = st.multiselect(
                    "Payment Methods",
                    options=payment_methods,
                    default=[],
                    help="Select payment methods to filter transactions"
                )
            else:
                selected_payment_methods = []
        
        return {
            'min_amount': min_amount,
            'max_amount': max_amount,
            'selected_merchants': selected_merchants,
            'selected_customers': selected_customers,
            'selected_payment_methods': selected_payment_methods
        }

def apply_advanced_filters(transactions_df: pd.DataFrame, filter_params: dict) -> pd.DataFrame:
    # Step 1: Apply advanced filters to transactions
    filtered_df = transactions_df.copy()
    
    # Step 2: Amount filter
    if filter_params.get('min_amount') is not None:
        filtered_df = filtered_df[filtered_df['amount'] >= filter_params['min_amount']]
    
    if filter_params.get('max_amount') is not None:
        filtered_df = filtered_df[filtered_df['amount'] <= filter_params['max_amount']]
    
    # Step 3: Merchant filter
    if filter_params.get('selected_merchants'):
        filtered_df = filtered_df[filtered_df['merchant'].isin(filter_params['selected_merchants'])]
    
    # Step 4: Customer filter
    if filter_params.get('selected_customers'):
        filtered_df = filtered_df[filtered_df['customer_id'].isin(filter_params['selected_customers'])]
    
    # Step 5: Payment method filter
    if filter_params.get('selected_payment_methods') and 'payment_method' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['payment_method'].isin(filter_params['selected_payment_methods'])]
    
    return filtered_df

def render_filter_summary(active_filters: dict):
    # Step 1: Display active filters summary
    if any(active_filters.values()):
        st.write("**Active Filters:**")
        
        filter_texts = []
        
        if active_filters.get('date_range'):
            start_date, end_date = active_filters['date_range']
            filter_texts.append(f"📅 Date: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        if active_filters.get('categories'):
            filter_texts.append(f"🛍️ Categories: {', '.join(active_filters['categories'][:3])}{'...' if len(active_filters['categories']) > 3 else ''}")
        
        if active_filters.get('segments'):
            filter_texts.append(f"👥 Segments: {', '.join(active_filters['segments'][:3])}{'...' if len(active_filters['segments']) > 3 else ''}")
        
        if active_filters.get('min_amount') or active_filters.get('max_amount'):
            amount_text = "💰 Amount: "
            if active_filters.get('min_amount'):
                amount_text += f"${active_filters['min_amount']}+"
            if active_filters.get('max_amount'):
                amount_text += f"${active_filters['max_amount']}-"
            filter_texts.append(amount_text)
        
        for text in filter_texts:
            st.write(f"• {text}")
        
        # Step 2: Clear filters button
        if st.button("🗑️ Clear All Filters"):
            st.rerun() 