"""
Advanced Machine Learning Models for Customer Intelligence Platform
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib
import os
from datetime import datetime
import logging

from .customer_segmentation import CustomerSegmentation
from src.utils.logger import setup_logger


class AdvancedClusteringModel:
    """Advanced clustering model with multiple algorithms and ensemble methods"""
    
    def __init__(self, algorithm: str = 'ensemble', hyperparameters: Optional[Dict[str, Any]] = None):
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters or {}
        self.logger = setup_logger(__name__)
        self.models = {}
        self.ensemble_weights = None
        
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit multiple clustering models and create ensemble"""
        results = {}
        
        # Fit individual models
        if self.algorithm == 'ensemble' or 'kmeans' in self.algorithm:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.hyperparameters.get('n_clusters', 5), 
                           random_state=42, n_init=10)
            kmeans.fit(data)
            self.models['kmeans'] = kmeans
            results['kmeans'] = kmeans.labels_
        
        if self.algorithm == 'ensemble' or 'dbscan' in self.algorithm:
            dbscan = DBSCAN(eps=self.hyperparameters.get('eps', 0.5),
                           min_samples=self.hyperparameters.get('min_samples', 5))
            dbscan.fit(data)
            self.models['dbscan'] = dbscan
            results['dbscan'] = dbscan.labels_
        
        if self.algorithm == 'ensemble' or 'hierarchical' in self.algorithm:
            hierarchical = AgglomerativeClustering(
                n_clusters=self.hyperparameters.get('n_clusters', 5),
                linkage=self.hyperparameters.get('linkage', 'ward')
            )
            hierarchical.fit(data)
            self.models['hierarchical'] = hierarchical
            results['hierarchical'] = hierarchical.labels_
        
        if self.algorithm == 'ensemble' or 'spectral' in self.algorithm:
            spectral = SpectralClustering(
                n_clusters=self.hyperparameters.get('n_clusters', 5),
                random_state=42
            )
            spectral.fit(data)
            self.models['spectral'] = spectral
            results['spectral'] = spectral.labels_
        
        # Create ensemble if multiple models
        if len(self.models) > 1:
            ensemble_labels = self._create_ensemble(data, results)
            results['ensemble'] = ensemble_labels
        
        return results
    
    def _create_ensemble(self, data: pd.DataFrame, individual_results: Dict[str, np.ndarray]) -> np.ndarray:
        """Create ensemble clustering using voting mechanism"""
        # Convert labels to similarity matrix
        similarity_matrices = []
        for model_name, labels in individual_results.items():
            if model_name != 'ensemble':
                similarity_matrix = self._labels_to_similarity_matrix(labels)
                similarity_matrices.append(similarity_matrix)
        
        # Average similarity matrices
        ensemble_similarity = np.mean(similarity_matrices, axis=0)
        
        # Apply final clustering to ensemble similarity
        final_clustering = AgglomerativeClustering(
            n_clusters=self.hyperparameters.get('n_clusters', 5),
            linkage='ward'
        )
        final_labels = final_clustering.fit_predict(ensemble_similarity)
        
        return final_labels
    
    def _labels_to_similarity_matrix(self, labels: np.ndarray) -> np.ndarray:
        """Convert cluster labels to similarity matrix"""
        n_samples = len(labels)
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                similarity_matrix[i, j] = 1 if labels[i] == labels[j] else 0
        
        return similarity_matrix
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict clusters for new data"""
        if 'ensemble' in self.models:
            return self.models['ensemble'].predict(data)
        elif len(self.models) == 1:
            model_name = list(self.models.keys())[0]
            return self.models[model_name].predict(data)
        else:
            raise ValueError("No trained models available")


class DeepLearningModel:
    """Deep learning model for customer behavior prediction"""
    
    def __init__(self, model_type: str = 'neural_network', architecture: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.architecture = architecture or {}
        self.logger = setup_logger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self, input_dim: int, output_dim: int) -> Any:
        """Build deep learning model architecture"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            
            model = models.Sequential()
            
            # Input layer
            model.add(layers.Dense(
                self.architecture.get('hidden_layers', [128, 64, 32])[0],
                activation='relu',
                input_shape=(input_dim,)
            ))
            model.add(layers.Dropout(self.architecture.get('dropout_rate', 0.3)))
            
            # Hidden layers
            for units in self.architecture.get('hidden_layers', [128, 64, 32])[1:]:
                model.add(layers.Dense(units, activation='relu'))
                model.add(layers.Dropout(self.architecture.get('dropout_rate', 0.3)))
            
            # Output layer
            if output_dim == 1:
                model.add(layers.Dense(1, activation='sigmoid'))
            else:
                model.add(layers.Dense(output_dim, activation='softmax'))
            
            # Compile model
            model.compile(
                optimizer=self.architecture.get('optimizer', 'adam'),
                loss=self.architecture.get('loss', 'binary_crossentropy'),
                metrics=self.architecture.get('metrics', ['accuracy'])
            )
            
            self.model = model
            return model
            
        except ImportError:
            self.logger.warning("TensorFlow not available, using sklearn MLP instead")
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(
                hidden_layer_sizes=self.architecture.get('hidden_layers', [128, 64, 32]),
                max_iter=1000,
                random_state=42
            )
            return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Any:
        """Fit the deep learning model"""
        if self.model is None:
            self.build_model(X.shape[1], len(np.unique(y)))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if hasattr(self.model, 'fit'):
            return self.model.fit(X_scaled, y, **kwargs)
        else:
            raise ValueError("Model not properly initialized")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For models without predict_proba, return dummy probabilities
            predictions = self.model.predict(X_scaled)
            probas = np.zeros((len(predictions), 2))
            probas[np.arange(len(predictions)), predictions] = 1
            return probas


class EnsembleModel:
    """Ensemble model combining multiple algorithms"""
    
    def __init__(self, base_models: Optional[List[str]] = None, voting_method: str = 'soft'):
        self.base_models = base_models or ['random_forest', 'gradient_boosting', 'neural_network']
        self.voting_method = voting_method
        self.logger = setup_logger(__name__)
        self.models = {}
        self.ensemble = None
        
    def build_models(self, input_dim: int, output_dim: int) -> Dict[str, Any]:
        """Build base models for ensemble"""
        for model_name in self.base_models:
            if model_name == 'random_forest':
                self.models[model_name] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            elif model_name == 'gradient_boosting':
                self.models[model_name] = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
            elif model_name == 'neural_network':
                nn_model = DeepLearningModel(
                    model_type='neural_network',
                    architecture={'hidden_layers': [64, 32], 'dropout_rate': 0.2}
                )
                nn_model.build_model(input_dim, output_dim)
                self.models[model_name] = nn_model
        
        return self.models
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit all base models and create ensemble"""
        # Build models if not already built
        if not self.models:
            self.build_models(X.shape[1], len(np.unique(y)))
        
        # Fit individual models
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")
            if hasattr(model, 'fit'):
                model.fit(X, y)
            else:
                self.logger.warning(f"Model {name} does not have fit method")
        
        # Create voting ensemble
        if len(self.models) > 1:
            self.ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in self.models.items()],
                voting=self.voting_method
            )
            self.ensemble.fit(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if self.ensemble is not None:
            return self.ensemble.predict(X)
        elif len(self.models) == 1:
            model_name = list(self.models.keys())[0]
            return self.models[model_name].predict(X)
        else:
            # Average predictions from all models
            predictions = []
            for model in self.models.values():
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    predictions.append(pred)
            
            if predictions:
                return np.mean(predictions, axis=0)
            else:
                raise ValueError("No valid models for prediction")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ensemble probabilities"""
        if self.ensemble is not None:
            return self.ensemble.predict_proba(X)
        elif len(self.models) == 1:
            model_name = list(self.models.keys())[0]
            if hasattr(self.models[model_name], 'predict_proba'):
                return self.models[model_name].predict_proba(X)
        
        # Average probabilities from all models
        probas = []
        for model in self.models.values():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probas.append(proba)
        
        if probas:
            return np.mean(probas, axis=0)
        else:
            raise ValueError("No valid models for probability prediction")


class AutoMLModel:
    """Automated Machine Learning model with hyperparameter optimization"""
    
    def __init__(self, task: str = 'classification', optimization_metric: str = 'accuracy'):
        self.task = task
        self.optimization_metric = optimization_metric
        self.logger = setup_logger(__name__)
        self.best_model = None
        self.best_score = 0
        self.optimization_history = []
        
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                               model_type: str = 'ensemble') -> Dict[str, Any]:
        """Optimize hyperparameters using grid search or Bayesian optimization"""
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import make_scorer, accuracy_score, f1_score
        
        # Define scoring metric
        if self.optimization_metric == 'accuracy':
            scorer = make_scorer(accuracy_score)
        elif self.optimization_metric == 'f1':
            scorer = make_scorer(f1_score, average='weighted')
        else:
            scorer = make_scorer(accuracy_score)
        
        # Define parameter grids
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
        elif model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            model = GradientBoostingClassifier(random_state=42)
        else:
            # Default to ensemble
            model = EnsembleModel()
            param_grid = {}
        
        # Perform hyperparameter optimization
        if param_grid:
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring=scorer, n_jobs=-1
            )
            grid_search.fit(X, y)
            
            self.best_model = grid_search.best_estimator_
            self.best_score = grid_search.best_score_
            self.optimization_history.append({
                'model_type': model_type,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            })
        else:
            # For ensemble models, use default parameters
            model.fit(X, y)
            self.best_model = model
            self.best_score = 0.8  # Placeholder score
        
        return {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history
        }
    
    def get_model_recommendations(self, X: pd.DataFrame, y: pd.Series) -> List[Dict[str, Any]]:
        """Get recommendations for best model types based on data characteristics"""
        recommendations = []
        
        # Analyze data characteristics
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Rule-based model recommendations
        if n_samples < 1000:
            recommendations.append({
                'model_type': 'random_forest',
                'reason': 'Small dataset - Random Forest handles small samples well',
                'expected_performance': 'Good'
            })
        elif n_samples > 10000:
            recommendations.append({
                'model_type': 'gradient_boosting',
                'reason': 'Large dataset - Gradient Boosting scales well',
                'expected_performance': 'Excellent'
            })
        
        if n_features > 100:
            recommendations.append({
                'model_type': 'ensemble',
                'reason': 'High-dimensional data - Ensemble methods reduce overfitting',
                'expected_performance': 'Very Good'
            })
        
        if n_classes > 10:
            recommendations.append({
                'model_type': 'neural_network',
                'reason': 'Many classes - Neural networks handle multi-class well',
                'expected_performance': 'Good'
            })
        
        return recommendations


# Factory function for creating models
def create_model(model_type: str, **kwargs) -> Any:
    """Factory function to create different types of models"""
    if model_type == 'advanced_clustering':
        return AdvancedClusteringModel(**kwargs)
    elif model_type == 'deep_learning':
        return DeepLearningModel(**kwargs)
    elif model_type == 'ensemble':
        return EnsembleModel(**kwargs)
    elif model_type == 'automl':
        return AutoMLModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
