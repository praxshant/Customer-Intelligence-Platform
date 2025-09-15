# model_utils.py
# Purpose: Model persistence and evaluation utilities

import pickle
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from src.utils.logger import setup_logger

class ModelUtils:
    def __init__(self, models_dir: str = "data/models"):
        # Step 1: Initialize model utilities
        self.logger = setup_logger("ModelUtils")
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, metadata: Optional[Dict[str, Any]] = None):
        # Step 1: Create model file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # Step 2: Prepare model data
        model_data = {
            'model': model,
            'metadata': metadata or {},
            'created_at': timestamp,
            'model_name': model_name
        }
        
        # Step 3: Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Step 4: Save metadata as JSON for easy reading
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(model_data['metadata'], f, indent=2, default=str)
        
        self.logger.info(f"Model saved: {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        # Step 1: Load model from file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.logger.info(f"Model loaded: {model_path}")
        return model_data
    
    def load_latest_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        # Step 1: Find the latest model file for the given name
        model_files = [f for f in os.listdir(self.models_dir) 
                      if f.startswith(model_name) and f.endswith('.pkl')]
        
        if not model_files:
            self.logger.warning(f"No model files found for {model_name}")
            return None
        
        # Step 2: Sort by timestamp and get the latest
        model_files.sort(reverse=True)
        latest_model_path = os.path.join(self.models_dir, model_files[0])
        
        return self.load_model(latest_model_path)
    
    def list_models(self) -> pd.DataFrame:
        # Step 1: Get all model files
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        # Step 2: Extract model information
        model_info = []
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            try:
                model_data = self.load_model(model_path)
                info = {
                    'filename': model_file,
                    'model_name': model_data.get('model_name', 'Unknown'),
                    'created_at': model_data.get('created_at', 'Unknown'),
                    'file_size_mb': round(os.path.getsize(model_path) / (1024 * 1024), 2)
                }
                model_info.append(info)
            except Exception as e:
                self.logger.error(f"Error reading model {model_file}: {str(e)}")
        
        return pd.DataFrame(model_info)
    
    def delete_model(self, model_path: str):
        # Step 1: Delete model file
        if os.path.exists(model_path):
            os.remove(model_path)
            
            # Step 2: Delete metadata file if it exists
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            self.logger.info(f"Model deleted: {model_path}")
        else:
            self.logger.warning(f"Model file not found: {model_path}")
    
    def evaluate_clustering_model(self, model, X: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        # Step 1: Calculate clustering evaluation metrics
        metrics = {}
        
        # Silhouette score
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except Exception as e:
            self.logger.warning(f"Could not calculate silhouette score: {str(e)}")
            metrics['silhouette_score'] = None
        
        # Calinski-Harabasz score
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except Exception as e:
            self.logger.warning(f"Could not calculate Calinski-Harabasz score: {str(e)}")
            metrics['calinski_harabasz_score'] = None
        
        # Davies-Bouldin score (if available)
        try:
            from sklearn.metrics import davies_bouldin_score
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except ImportError:
            self.logger.info("Davies-Bouldin score not available (requires scikit-learn >= 0.20)")
            metrics['davies_bouldin_score'] = None
        
        # Number of clusters
        metrics['n_clusters'] = len(np.unique(labels))
        
        # Inertia (for K-means)
        if hasattr(model, 'inertia_'):
            metrics['inertia'] = model.inertia_
        
        return metrics
    
    def create_model_report(self, model_name: str, evaluation_metrics: Dict[str, float], 
                          training_params: Dict[str, Any]) -> Dict[str, Any]:
        # Step 1: Create comprehensive model report
        report = {
            'model_name': model_name,
            'evaluation_metrics': evaluation_metrics,
            'training_parameters': training_params,
            'report_generated_at': datetime.now().isoformat(),
            'model_performance_summary': self._generate_performance_summary(evaluation_metrics)
        }
        
        # Step 2: Save report
        report_filename = f"{model_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(self.models_dir, report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Model report saved: {report_path}")
        return report
    
    def _generate_performance_summary(self, metrics: Dict[str, float]) -> str:
        # Step 1: Generate human-readable performance summary
        summary_parts = []
        
        if metrics.get('silhouette_score') is not None:
            sil_score = metrics['silhouette_score']
            if sil_score > 0.7:
                summary_parts.append("Excellent clustering quality")
            elif sil_score > 0.5:
                summary_parts.append("Good clustering quality")
            elif sil_score > 0.3:
                summary_parts.append("Fair clustering quality")
            else:
                summary_parts.append("Poor clustering quality")
        
        if metrics.get('n_clusters') is not None:
            summary_parts.append(f"{metrics['n_clusters']} clusters identified")
        
        if metrics.get('inertia') is not None:
            summary_parts.append(f"Inertia: {metrics['inertia']:.2f}")
        
        return "; ".join(summary_parts) if summary_parts else "No performance metrics available"
    
    def backup_models(self, backup_dir: str = "data/models/backup"):
        # Step 1: Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Step 2: Copy all model files to backup
        import shutil
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        for model_file in model_files:
            src_path = os.path.join(self.models_dir, model_file)
            dst_path = os.path.join(backup_dir, model_file)
            shutil.copy2(src_path, dst_path)
        
        self.logger.info(f"Backed up {len(model_files)} models to {backup_dir}")
    
    def cleanup_old_models(self, keep_latest: int = 3):
        # Step 1: Get all model files grouped by model name
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        # Step 2: Group by model name
        model_groups = {}
        for model_file in model_files:
            model_name = model_file.split('_')[0]  # Extract model name from filename
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(model_file)
        
        # Step 3: Keep only the latest models for each type
        for model_name, files in model_groups.items():
            files.sort(reverse=True)  # Sort by timestamp (newest first)
            
            # Delete older models
            for old_file in files[keep_latest:]:
                old_path = os.path.join(self.models_dir, old_file)
                self.delete_model(old_path)
        
        self.logger.info(f"Cleaned up old models, keeping {keep_latest} latest for each model type") 