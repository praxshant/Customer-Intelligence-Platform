# styling.py
# Purpose: Styling utilities for the dashboard

import streamlit as st
from typing import Dict, Any

def apply_custom_css():
    # Step 1: Apply custom CSS styling
    custom_css = """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    h3 {
        color: #34495e;
        font-weight: 500;
        margin-bottom: 0.6rem;
    }
    
    /* Metric cards styling */
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    /* Chart container styling */
    .chart-container {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Error message styling */
    .stError {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div {
        border-radius: 8px;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div > div {
        border-radius: 8px;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Date input styling */
    .stDateInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background-color: #f8f9fa;
        border: none;
        color: #6c757d;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Custom card styling */
    .custom-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    /* KPI card styling */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Status indicator styling */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #28a745;
    }
    
    .status-inactive {
        background-color: #dc3545;
    }
    
    .status-pending {
        background-color: #ffc107;
    }
    
    /* Loading spinner styling */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .custom-card {
            padding: 1rem;
        }
        
        .kpi-card {
            padding: 1rem;
        }
    }
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    # Step 1: Create a custom metric card
    card_html = f"""
    <div class="kpi-card">
        <h3 style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{title}</h3>
        <h2 style="margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">{value}</h2>
    """
    
    if delta:
        card_html += f'<p style="margin: 0; font-size: 0.8rem; opacity: 0.8;">{delta}</p>'
    
    card_html += "</div>"
    
    return card_html

def create_status_badge(status: str, text: str = None):
    # Step 1: Create a status badge
    status_colors = {
        'active': '#28a745',
        'inactive': '#dc3545',
        'pending': '#ffc107',
        'success': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545',
        'info': '#17a2b8'
    }
    
    color = status_colors.get(status.lower(), '#6c757d')
    display_text = text or status.title()
    
    badge_html = f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    ">
        {display_text}
    </span>
    """
    
    return badge_html

def create_progress_bar(value: float, max_value: float, label: str = None):
    # Step 1: Create a custom progress bar
    percentage = (value / max_value) * 100 if max_value > 0 else 0
    
    progress_html = f"""
    <div style="margin: 1rem 0;">
        {f'<p style="margin-bottom: 0.5rem; font-weight: 500;">{label}</p>' if label else ''}
        <div style="
            background-color: #e9ecef;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        ">
            <div style="
                background: linear-gradient(90deg, #1f77b4, #667eea);
                height: 100%;
                width: {percentage}%;
                transition: width 0.3s ease;
                border-radius: 10px;
            "></div>
        </div>
        <p style="margin-top: 0.25rem; font-size: 0.8rem; color: #6c757d;">
            {value:,.0f} / {max_value:,.0f} ({percentage:.1f}%)
        </p>
    </div>
    """
    
    return progress_html

def create_info_card(title: str, content: str, icon: str = "ℹ️"):
    # Step 1: Create an info card
    card_html = f"""
    <div class="custom-card">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            <h3 style="margin: 0;">{title}</h3>
        </div>
        <p style="margin: 0; color: #6c757d; line-height: 1.6;">{content}</p>
    </div>
    """
    
    return card_html

def create_alert_box(message: str, alert_type: str = "info"):
    # Step 1: Create an alert box
    alert_styles = {
        'info': {
            'background': '#d1ecf1',
            'border': '#bee5eb',
            'color': '#0c5460',
            'icon': 'ℹ️'
        },
        'success': {
            'background': '#d4edda',
            'border': '#c3e6cb',
            'color': '#155724',
            'icon': '✅'
        },
        'warning': {
            'background': '#fff3cd',
            'border': '#ffeaa7',
            'color': '#856404',
            'icon': '⚠️'
        },
        'error': {
            'background': '#f8d7da',
            'border': '#f5c6cb',
            'color': '#721c24',
            'icon': '❌'
        }
    }
    
    style = alert_styles.get(alert_type, alert_styles['info'])
    
    alert_html = f"""
    <div style="
        background-color: {style['background']};
        border: 1px solid {style['border']};
        color: {style['color']};
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
    ">
        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{style['icon']}</span>
        <span>{message}</span>
    </div>
    """
    
    return alert_html

def create_data_table(data: Dict[str, Any], title: str = None):
    # Step 1: Create a data table
    table_html = f"""
    <div class="custom-card">
        {f'<h3 style="margin-bottom: 1rem;">{title}</h3>' if title else ''}
        <table style="
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        ">
            <thead>
                <tr style="background-color: #f8f9fa;">
                    <th style="padding: 0.75rem; text-align: left; border-bottom: 2px solid #dee2e6;">Metric</th>
                    <th style="padding: 0.75rem; text-align: right; border-bottom: 2px solid #dee2e6;">Value</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for key, value in data.items():
        table_html += f"""
                <tr>
                    <td style="padding: 0.75rem; border-bottom: 1px solid #dee2e6; font-weight: 500;">{key}</td>
                    <td style="padding: 0.75rem; border-bottom: 1px solid #dee2e6; text-align: right;">{value}</td>
                </tr>
        """
    
    table_html += """
            </tbody>
        </table>
    </div>
    """
    
    return table_html 