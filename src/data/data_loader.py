# data_loader.py
# Purpose: Load and validate raw data from various sources

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import os
from src.utils.logger import setup_logger
from src.utils.helpers import calculate_date_range

class DataLoader:
    def __init__(self):
        # Step 1: Initialize data loader with logger
        self.logger = setup_logger("DataLoader")
        self.supported_formats = ['.csv', '.xlsx', '.json']
    
    def load_transaction_data(self, file_path: str) -> pd.DataFrame:
        # Step 1: Validate file exists and format is supported
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Step 2: Load data based on file format
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            
            self.logger.info(f"Loaded {len(df)} records from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
        
        # Step 3: Validate required columns
        required_columns = [
            'transaction_id', 'customer_id', 'transaction_date',
            'amount', 'category', 'merchant'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def load_customer_data(self, file_path: str) -> pd.DataFrame:
        # Step 1: Load customer demographic data
        df = self._load_file(file_path)
        
        # Step 2: Validate customer data columns
        required_columns = ['customer_id', 'first_name', 'last_name', 'email']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required customer columns: {missing_columns}")
        
        return df
    
    def _load_file(self, file_path: str) -> pd.DataFrame:
        # Step 1: Generic file loading method
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension == '.xlsx':
            return pd.read_excel(file_path)
        elif file_extension == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def validate_data_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        # Step 1: Perform data quality checks
        quality_report = {
            'total_records': len(df),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Step 2: Add specific validations based on data type
        if data_type == 'transactions':
            quality_report.update({
                'negative_amounts': (df['amount'] < 0).sum(),
                'zero_amounts': (df['amount'] == 0).sum(),
                'future_dates': (pd.to_datetime(df['transaction_date']) > pd.Timestamp.now()).sum()
            })
        
        # Step 3: Log quality issues
        if quality_report['duplicate_records'] > 0:
            self.logger.warning(f"Found {quality_report['duplicate_records']} duplicate records")
        
        for col, null_count in quality_report['null_counts'].items():
            if null_count > 0:
                self.logger.warning(f"Column '{col}' has {null_count} null values")
        
        return quality_report
    
    def load_sample_data(self) -> Dict[str, pd.DataFrame]:
        # Step 1: Load sample data from the data directory
        data_dir = "data/raw"
        data_files = {}
        
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(('.csv', '.xlsx', '.json')):
                    file_path = os.path.join(data_dir, file)
                    try:
                        if 'transaction' in file.lower():
                            data_files['transactions'] = self.load_transaction_data(file_path)
                        elif 'customer' in file.lower():
                            data_files['customers'] = self.load_customer_data(file_path)
                    except Exception as e:
                        self.logger.error(f"Error loading {file}: {str(e)}")
        
        return data_files 