# src/utils/validation.py
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import re
from functools import wraps
import traceback
from src.utils.logger import setup_logger


class ValidationError(Exception):
    pass


class DataValidationError(ValidationError):
    pass


class BusinessRuleValidationError(ValidationError):
    pass


def validate_input(**validators):
    """Decorator for input validation. Usage: @validate_input(df=validate_dataframe)"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = setup_logger(f"{func.__module__}.{func.__name__}")
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    try:
                        validator(value, param_name)
                    except ValidationError as e:
                        logger.error(
                            f"Validation failed for {param_name}",
                            extra={"error": str(e)}
                        )
                        raise ValidationError(f"Invalid {param_name}: {e}") from e
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Function execution failed",
                    extra={
                        "function": func.__name__,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise

        return wrapper

    return decorator


def validate_dataframe(df: Any, param_name: str = "dataframe") -> None:
    if df is None:
        raise DataValidationError(f"{param_name} cannot be None")
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            f"{param_name} must be a pandas DataFrame, got {type(df)}"
        )
    if df.empty:
        raise DataValidationError(f"{param_name} cannot be empty")


def validate_customer_id(customer_id: Any, param_name: str = "customer_id") -> None:
    if not customer_id:
        raise ValidationError(f"{param_name} cannot be empty or None")
    if not isinstance(customer_id, str):
        raise ValidationError(f"{param_name} must be a string, got {type(customer_id)}")
    if not re.match(r"^[A-Z0-9_]+$", customer_id):
        raise ValidationError(f"{param_name} contains invalid characters")


def validate_positive_number(value: Any, param_name: str) -> None:
    if value is None:
        raise ValidationError(f"{param_name} cannot be None")
    try:
        num_value = float(value)
        if num_value <= 0:
            raise ValidationError(f"{param_name} must be positive, got {num_value}")
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"{param_name} must be a valid number, got {value}"
        ) from e


def validate_date_range(start_date: Any, end_date: Any) -> None:
    if start_date and end_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        if start_date >= end_date:
            raise ValidationError("Start date must be before end date")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or pd.isna(denominator) or np.isnan(denominator):
        return default
    if pd.isna(numerator) or np.isnan(numerator):
        return default
    try:
        result = float(numerator) / float(denominator)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError, ZeroDivisionError):
        return default


def safe_mean(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    clean_values = [
        v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    if not clean_values:
        return default
    try:
        return float(np.mean(clean_values))
    except (ValueError, TypeError):
        return default


class DataFrameValidator:
    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

    @staticmethod
    def validate_data_types(df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        validated_df = df.copy()
        for column, expected in column_types.items():
            if column not in validated_df.columns:
                continue
            try:
                if expected == "datetime":
                    validated_df[column] = pd.to_datetime(validated_df[column], errors="coerce")
                elif expected == "numeric":
                    validated_df[column] = pd.to_numeric(validated_df[column], errors="coerce")
                elif expected == "string":
                    validated_df[column] = validated_df[column].astype(str)
                elif expected == "category":
                    validated_df[column] = validated_df[column].astype("category")
            except Exception as e:
                raise DataValidationError(
                    f"Failed to convert column {column} to {expected}: {e}"
                )
        return validated_df

    @staticmethod
    def validate_business_rules(df: pd.DataFrame, rules: Dict[str, Callable]) -> None:
        for rule_name, rule_func in rules.items():
            try:
                if not rule_func(df):
                    raise BusinessRuleValidationError(
                        f"Business rule violation: {rule_name}"
                    )
            except Exception as e:
                raise BusinessRuleValidationError(
                    f"Error checking business rule {rule_name}: {e}"
                )
