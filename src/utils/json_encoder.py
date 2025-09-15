# src/utils/json_encoder.py
import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from decimal import Decimal
from typing import Any

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy and pandas types"""
    
    def default(self, obj: Any) -> Any:
        # Handle NumPy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        
        # Handle pandas types
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        
        # Handle datetime types
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        
        # Let the base class handle other types
        return super().default(obj)

def convert_numpy_types(data: Any) -> Any:
    """Recursively convert NumPy and pandas types to native Python types"""
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_numpy_types(item) for item in data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, pd.Series):
        return convert_numpy_types(data.to_dict())
    elif isinstance(data, pd.DataFrame):
        return convert_numpy_types(data.to_dict('records'))
    else:
        return data
