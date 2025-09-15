# kpi_cards.py
# Purpose: KPI cards component for displaying key metrics

import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.utils.helpers import format_currency, calculate_percentage_change

def render_kpi_cards(kpi_data: Dict[str, Any]):
    # Step 1: Create KPI cards layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_revenue_card(kpi_data)
    
    with col2:
        render_transactions_card(kpi_data)
    
    with col3:
        render_avg_value_card(kpi_data)
    
    with col4:
        render_customers_card(kpi_data)

def render_revenue_card(kpi_data: Dict[str, Any]):
    # Step 1: Revenue KPI card
    total_revenue = kpi_data.get('total_revenue', 0)
    revenue_change = kpi_data.get('revenue_change', 0)
    
    # Step 2: Determine trend indicator and valid delta_color
    if revenue_change > 0:
        trend_icon = "📈"
        delta_color = "normal"   # green up
    elif revenue_change < 0:
        trend_icon = "📉"
        delta_color = "inverse"  # red down
    else:
        trend_icon = "➡️"
        delta_color = "off"
    
    # Step 3: Create card
    st.metric(
        label="💰 Total Revenue",
        value=format_currency(total_revenue),
        delta=f"{trend_icon} {revenue_change:.1f}%",
        delta_color=delta_color
    )

def render_transactions_card(kpi_data: Dict[str, Any]):
    # Step 1: Transactions KPI card
    total_transactions = kpi_data.get('total_transactions', 0)
    transaction_change = kpi_data.get('transaction_change', 0)
    
    # Step 2: Determine trend indicator and valid delta_color
    if transaction_change > 0:
        trend_icon = "📈"
        delta_color = "normal"
    elif transaction_change < 0:
        trend_icon = "📉"
        delta_color = "inverse"
    else:
        trend_icon = "➡️"
        delta_color = "off"
    
    # Step 3: Create card
    st.metric(
        label="🛒 Total Transactions",
        value=f"{total_transactions:,}",
        delta=f"{trend_icon} {transaction_change:.1f}%",
        delta_color=delta_color
    )

def render_avg_value_card(kpi_data: Dict[str, Any]):
    # Step 1: Average transaction value KPI card
    avg_value = kpi_data.get('avg_transaction_value', 0)
    
    # Step 2: Create card
    st.metric(
        label="💵 Avg Transaction Value",
        value=format_currency(avg_value),
        delta=None
    )

def render_customers_card(kpi_data: Dict[str, Any]):
    # Step 1: Unique customers KPI card
    unique_customers = kpi_data.get('unique_customers', 0)
    
    # Step 2: Create card
    st.metric(
        label="👥 Unique Customers",
        value=f"{unique_customers:,}",
        delta=None
    )

def render_segment_kpi_cards(segment_data: pd.DataFrame):
    # Step 1: Segment-specific KPI cards
    if segment_data.empty:
        st.info("No segment data available")
        return
    
    # Step 2: Calculate segment metrics
    segment_metrics = segment_data.groupby('segment_name').agg({
        'total_spent': 'sum',
        'transaction_count': 'sum',
        'customer_id': 'count'
    }).reset_index()
    
    # Step 3: Create segment cards
    st.subheader("📊 Segment Performance")
    
    cols = st.columns(len(segment_metrics))
    
    for i, (_, segment) in enumerate(segment_metrics.iterrows()):
        with cols[i]:
            st.metric(
                label=f"🏷️ {segment['segment_name']}",
                value=format_currency(segment['total_spent']),
                delta=f"{segment['customer_id']} customers"
            )

def render_campaign_kpi_cards(campaign_data: pd.DataFrame):
    # Step 1: Campaign-specific KPI cards
    if campaign_data.empty:
        st.info("No campaign data available")
        return
    
    # Step 2: Calculate campaign metrics
    total_campaigns = len(campaign_data)
    active_campaigns = len(campaign_data[campaign_data['status'] == 'active'])
    avg_discount = campaign_data['discount_percentage'].mean()
    total_potential_revenue = campaign_data['min_purchase'].sum()
    
    # Step 3: Create campaign cards
    st.subheader("🎯 Campaign Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📢 Total Campaigns",
            value=f"{total_campaigns:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="✅ Active Campaigns",
            value=f"{active_campaigns:,}",
            delta=f"{active_campaigns/total_campaigns*100:.1f}%" if total_campaigns > 0 else "0%"
        )
    
    with col3:
        st.metric(
            label="💰 Avg Discount",
            value=f"{avg_discount:.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="💵 Potential Revenue",
            value=format_currency(total_potential_revenue),
            delta=None
        ) 