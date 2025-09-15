# config_loader.py
# Purpose: Load and manage configuration files

import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration with sensible defaults and error handling."""
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}. Using defaults.")
        return _get_default_config()

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error reading configuration file {config_path}: {e}")
        return _get_default_config()

    if config is None:
        logger.warning(f"Empty configuration file: {config_path}. Using defaults.")
        return _get_default_config()

    # When loading the main config, ensure required sections exist by merging with defaults
    if os.path.basename(config_path) == 'config.yaml':
        default = _get_default_config()
        config = _merge_configs(default, config)
    return config

def load_database_config() -> Dict[str, Any]:
    """Load database-specific configuration with fallback defaults."""
    path = "config/database_config.yaml"
    if not os.path.exists(path):
        logger.warning(f"Database config missing at {path}. Using defaults.")
        return {
            'database': {
                'type': 'sqlite',
                'path': 'data/customer_insights.db',
                'timeout': 30,
                'check_same_thread': False
            }
        }
    try:
        return load_config(path)
    except Exception as e:
        logger.error(f"Failed to load database config: {e}. Using defaults.")
        return {
            'database': {
                'type': 'sqlite',
                'path': 'data/customer_insights.db',
                'timeout': 30,
                'check_same_thread': False
            }
        }

def load_ml_config() -> Dict[str, Any]:
    """Load ML model configuration with fallback defaults."""
    path = "config/ml_config.yaml"
    if not os.path.exists(path):
        logger.warning(f"ML config missing at {path}. Using defaults.")
        return _get_default_config()['ml_models']
    try:
        cfg = load_config(path)
        # If file exists but empty/invalid, ensure we have ml_models keys
        default_ml = _get_default_config()['ml_models']
        return _merge_configs(default_ml, cfg)
    except Exception as e:
        logger.error(f"Failed to load ML config: {e}. Using defaults.")
        return _get_default_config()['ml_models']

def _get_default_config() -> Dict[str, Any]:
    """Return default configuration used when files are missing."""
    return {
        'database': {
            'path': 'data/customer_insights.db',
            'backup_enabled': True,
            'backup_interval_days': 7
        },
        'ml_models': {
            'segmentation': {
                'algorithm': 'kmeans',
                'max_clusters': 10,
                'random_state': 42,
                'n_init': 10
            },
            'recommendation': {
                'algorithm': 'collaborative_filtering',
                'min_interactions': 5,
                'top_n_recommendations': 10
            }
        },
        'dashboard': {
            'title': 'Customer Spending Insights Dashboard',
            'theme': 'light',
            'page_size': 20,
            'refresh_interval': 300
        },
        'rules_engine': {
            'campaign_types': ['loyalty_reward', 'win_back', 'cross_sell', 'seasonal_promotion'],
            'min_discount_amount': 5.0,
            'max_discount_amount': 100.0,
            'campaign_validity_days': 30
        },
        'logging': {
            'level': 'INFO',
            'file_rotation': 'daily',
            'max_file_size_mb': 10,
            'backup_count': 7
        }
    }

def _merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge user config into defaults."""
    result = default.copy()
    for key, value in (user or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result