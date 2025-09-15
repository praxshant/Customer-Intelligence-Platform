# charts.py
# Purpose: Charts component for data visualizations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
from src.utils.helpers import format_currency, generate_color_palette

def render_spending_chart(transactions_df: pd.DataFrame):
    # Step 1: Spending trend chart
    st.subheader("📈 Spending Trends")
    
    if transactions_df.empty:
        st.info("No transaction data available")
        return
    
    # Step 2: Prepare data for time series
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    daily_spending = transactions_df.groupby('transaction_date')['amount'].sum().reset_index()
    
    # Step 3: Create time series chart
    fig = px.line(
        daily_spending,
        x='transaction_date',
        y='amount',
        title="Daily Spending Trends",
        labels={'amount': 'Total Spending ($)', 'transaction_date': 'Date'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Total Spending ($)",
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')

def render_segment_chart(customers_df: pd.DataFrame):
    # Step 1: Customer segmentation chart
    st.subheader("👥 Customer Segments")
    
    if customers_df.empty or 'segment_name' not in customers_df.columns:
        st.info("No segment data available")
        return
    
    # Step 2: Prepare segment data
    segment_counts = customers_df['segment_name'].value_counts()
    
    # Step 3: Create pie chart
    colors = generate_color_palette(len(segment_counts))
    
    fig = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Distribution by Segment",
        color_discrete_sequence=colors
    )
    
    fig.update_layout(height=300)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, width='stretch')

def render_category_chart(transactions_df: pd.DataFrame):
    # Step 1: Category spending chart
    st.subheader("🛍️ Category Spending")
    
    if transactions_df.empty:
        st.info("No transaction data available")
        return
    
    # Step 2: Prepare category data
    category_spending = transactions_df.groupby('category')['amount'].sum().sort_values(ascending=False)
    
    # Step 3: Create bar chart
    colors = generate_color_palette(len(category_spending))
    
    fig = px.bar(
        x=category_spending.index,
        y=category_spending.values,
        title="Spending by Category",
        color=category_spending.values,
        color_continuous_scale='viridis',
        labels={'x': 'Category', 'y': 'Total Spending ($)'}
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Total Spending ($)",
        height=300,
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, width='stretch')

def render_merchant_chart(transactions_df: pd.DataFrame, top_n: int = 10):
    # Step 1: Top merchants chart
    st.subheader("🏪 Top Merchants")
    
    if transactions_df.empty:
        st.info("No transaction data available")
        return
    
    # Step 2: Prepare merchant data
    merchant_spending = transactions_df.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(top_n)
    
    # Step 3: Create horizontal bar chart
    fig = px.bar(
        x=merchant_spending.values,
        y=merchant_spending.index,
        orientation='h',
        title=f"Top {top_n} Merchants by Revenue",
        color=merchant_spending.values,
        color_continuous_scale='plasma',
        labels={'x': 'Total Revenue ($)', 'y': 'Merchant'}
    )
    
    fig.update_layout(
        xaxis_title="Total Revenue ($)",
        yaxis_title="Merchant",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')

def render_customer_value_chart(customer_features_df: pd.DataFrame):
    # Step 1: Customer value distribution chart
    st.subheader("💰 Customer Value Distribution")
    
    if customer_features_df.empty:
        st.info("No customer data available")
        return
    
    # Step 2: Create histogram
    fig = px.histogram(
        customer_features_df,
        x='total_spent',
        nbins=20,
        title="Distribution of Customer Lifetime Value",
        labels={'total_spent': 'Total Spent ($)', 'count': 'Number of Customers'}
    )
    
    fig.update_layout(
        xaxis_title="Total Spent ($)",
        yaxis_title="Number of Customers",
        height=300
    )
    
    st.plotly_chart(fig, width='stretch')

def render_recency_frequency_chart(customer_features_df: pd.DataFrame):
    # Step 1: Recency-Frequency chart
    st.subheader("📊 Customer Recency vs Frequency")
    
    if customer_features_df.empty:
        st.info("No customer data available")
        return
    
    # Step 2: Create scatter plot
    fig = px.scatter(
        customer_features_df,
        x='recency_days',
        y='transaction_frequency',
        size='total_spent',
        color='total_spent',
        title="Customer Recency vs Frequency Analysis",
        labels={
            'recency_days': 'Days Since Last Transaction',
            'transaction_frequency': 'Transaction Frequency (per day)',
            'total_spent': 'Total Spent ($)'
        },
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Days Since Last Transaction",
        yaxis_title="Transaction Frequency (per day)",
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')

def render_campaign_performance_chart(campaigns_df: pd.DataFrame):
    # Step 1: Campaign performance chart
    st.subheader("🎯 Campaign Performance")
    
    if campaigns_df.empty:
        st.info("No campaign data available")
        return
    
    # Step 2: Prepare campaign data
    campaign_performance = campaigns_df.groupby('campaign_type').agg({
        'discount_percentage': 'mean',
        'priority_score': 'mean',
        'campaign_id': 'count'
    }).reset_index()
    
    campaign_performance.columns = ['campaign_type', 'avg_discount', 'avg_priority', 'campaign_count']
    
    # Step 3: Create bubble chart
    fig = px.scatter(
        campaign_performance,
        x='avg_discount',
        y='avg_priority',
        size='campaign_count',
        color='campaign_type',
        title="Campaign Performance Analysis",
        labels={
            'avg_discount': 'Average Discount (%)',
            'avg_priority': 'Average Priority Score',
            'campaign_count': 'Number of Campaigns'
        }
    )
    
    fig.update_layout(
        xaxis_title="Average Discount (%)",
        yaxis_title="Average Priority Score",
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')

def render_monthly_trends_chart(transactions_df: pd.DataFrame):
    # Step 1: Monthly trends chart
    st.subheader("📅 Monthly Trends")
    
    if transactions_df.empty:
        st.info("No transaction data available")
        return
    
    # Step 2: Prepare monthly data
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    transactions_df['month'] = transactions_df['transaction_date'].dt.to_period('M')
    
    monthly_data = transactions_df.groupby('month').agg({
        'amount': 'sum',
        'transaction_id': 'count',
        'customer_id': 'nunique'
    }).reset_index()
    
    monthly_data['month'] = monthly_data['month'].astype(str)
    
    # Step 3: Create subplot with multiple metrics
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Revenue', 'Monthly Transactions & Customers'),
        vertical_spacing=0.1
    )
    
    # Revenue line
    fig.add_trace(
        go.Scatter(x=monthly_data['month'], y=monthly_data['amount'], 
                  name='Revenue', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Transactions bar
    fig.add_trace(
        go.Bar(x=monthly_data['month'], y=monthly_data['transaction_id'], 
               name='Transactions', marker_color='green'),
        row=2, col=1
    )
    
    # Customers line
    fig.add_trace(
        go.Scatter(x=monthly_data['month'], y=monthly_data['customer_id'], 
                  name='Customers', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=True)
    
    st.plotly_chart(fig, width='stretch')