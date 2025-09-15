# helpers.py
# Purpose: Common utility functions used across the application

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

def calculate_date_range(period: str) -> Tuple[datetime, datetime]:
    # Step 1: Get current date
    end_date = datetime.now()
    
    # Step 2: Calculate start date based on period
    if period == "7d":
        start_date = end_date - timedelta(days=7)
    elif period == "30d":
        start_date = end_date - timedelta(days=30)
    elif period == "90d":
        start_date = end_date - timedelta(days=90)
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    else:
        start_date = end_date - timedelta(days=30)  # Default to 30 days
    
    return start_date, end_date

def format_currency(amount: float) -> str:
    # Format amount as currency string
    return f"${amount:,.2f}"

def calculate_percentage_change(current: float, previous: float) -> float:
    # Calculate percentage change between two values
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100

def safe_divide(numerator: float, denominator: float) -> float:
    # Safely divide two numbers, avoiding division by zero
    return numerator / denominator if denominator != 0 else 0.0

def generate_color_palette(n_colors: int) -> List[str]:
    # Generate a color palette for visualizations
    base_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ]
    
    # Repeat colors if needed
    colors = (base_colors * ((n_colors // len(base_colors)) + 1))[:n_colors]
    return colors

def validate_email(email: str) -> bool:
    # Validate email format
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def calculate_age(birth_date: datetime) -> int:
    # Calculate age from birth date
    today = datetime.now()
    age = today.year - birth_date.year
    if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
        age -= 1
    return age

def get_season(date: datetime) -> str:
    # Get season from date
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def categorize_amount(amount: float) -> str:
    # Categorize transaction amount
    if amount < 10:
        return "Small"
    elif amount < 50:
        return "Medium"
    elif amount < 200:
        return "Large"
    else:
        return "Very Large" 