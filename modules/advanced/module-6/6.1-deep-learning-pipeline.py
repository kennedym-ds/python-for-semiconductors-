"""Production Deep Learning Pipeline Script for Module 6.1

Provides a CLI to train, evaluate, and predict using deep learning models (MLPs)
for semiconductor manufacturing datasets with PyTorch or TensorFlow backends.

Features:
- Multi-layer perceptron (MLP) architectures for tabular data
- Backend selection: PyTorch or TensorFlow (with graceful fallback)
- Manufacturing metrics: MAE, RMSE, R², PWS, Estimated Loss
- Regularization: dropout, batch norm, weight decay
- Optimization: SGD/Adam with learning rate scheduling
- Model persistence (save/load)
- Reproducibility via deterministic flags and seeds
- Cost-sensitive learning for imbalanced datasets

Example usage:
    python 6.1-deep-learning-pipeline.py train --dataset synthetic_yield --backend pytorch --save model.pth
    python 6.1-deep-learning-pipeline.py evaluate --model-path model.pth --dataset synthetic_yield
    python 6.1-deep-learning-pipeline.py predict --model-path model.pth --input-json '{"temperature":455, "pressure":2.6}'
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Optional deep learning imports with feature flags
HAS_TORCH = False
HAS_TF = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    # Set deterministic flags for reproducibility
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    HAS_TORCH = True
except ImportError:
    pass

try:
    import tensorflow as tf
    # Set TensorFlow random seeds
    tf.random.set_seed(RANDOM_SEED)
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
    HAS_TF = True
except ImportError:
    pass

# ===============================================================================
# Data Generation Functions
# ===============================================================================

def generate_synthetic_tabular(
    n: int = 1000, 
    n_features: int = 20, 
    task: str = 'regression',
    noise_level: float = 0.1,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate synthetic tabular data for semiconductor process modeling."""
    rng = np.random.default_rng(seed)
    
    # Generate base features (process parameters)
    features = {}
    feature_names = []
    
    # Core process parameters
    features['temperature'] = rng.normal(450, 15, n)
    features['pressure'] = rng.normal(2.5, 0.3, n)
    features['flow_rate'] = rng.normal(120, 10, n)
    features['time'] = rng.normal(60, 5, n)
    features['rf_power'] = rng.normal(1500, 100, n)
    
    # Additional engineered features
    for i in range(n_features - 5):
        if i < 5:
            # Derived ratios and interactions
            features[f'ratio_{i}'] = features['temperature'] / features['pressure'] + rng.normal(0, 1, n)
        elif i < 10:
            # Sensor readings with drift
            features[f'sensor_{i}'] = rng.normal(100, 10, n) + 0.01 * np.arange(n)
        else:
            # Random process parameters
            features[f'param_{i}'] = rng.normal(0, 1, n)
    
    df = pd.DataFrame(features)
    
    # Generate target based on physics-inspired relationships
    if task == 'regression':
        # Yield prediction (0-100%)
        target = (
            85 + 
            0.02 * (features['temperature'] - 450) - 
            0.0001 * (features['temperature'] - 450)**2 +
            5 * (features['pressure'] - 2.5) +
            0.1 * (features['flow_rate'] - 120) +
            rng.normal(0, noise_level * 10, n)
        )
        target = np.clip(target, 0, 100)
        df['target'] = target
    else:
        # Pass/fail classification
        yield_score = (
            0.02 * (features['temperature'] - 450) - 
            0.0001 * (features['temperature'] - 450)**2 +
            0.2 * (features['pressure'] - 2.5) +
            rng.normal(0, noise_level, n)
        )
        # Create imbalanced dataset (typical for defect detection)
        threshold = np.percentile(yield_score, 85)  # 15% failure rate
        df['target'] = (yield_score > threshold).astype(int)
    
    return df

# ===============================================================================
# PyTorch Models
# ===============================================================================

if HAS_TORCH:
    class PyTorchMLP(nn.Module):
        """Multi-layer perceptron implementation in PyTorch."""
        
        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [64, 32, 16],
            output_dim: int = 1,
            dropout_rate: float = 0.2,
            use_batch_norm: bool = True,
            activation: str = 'relu'
        ):
            super().__init__()
            self.layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.dropout = nn.Dropout(dropout_rate)
            self.use_batch_norm = use_batch_norm
            
            # Activation function
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'elu':
                self.activation = nn.ELU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            # Build layers
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                prev_dim = hidden_dim
            
            # Output layer
            self.output_layer = nn.Linear(prev_dim, output_dim)
            
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if self.use_batch_norm:
                    x = self.batch_norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
            return self.output_layer(x)
else:
    # Placeholder class when PyTorch is not available
    class PyTorchMLP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")

# ===============================================================================
# TensorFlow Models
# ===============================================================================

def create_tensorflow_mlp(
    input_dim: int,
    hidden_dims: List[int] = [64, 32, 16],
    output_dim: int = 1,
    dropout_rate: float = 0.2,
    use_batch_norm: bool = True,
    activation: str = 'relu',
    task: str = 'regression'
) -> 'tf.keras.Model':
    """Create MLP model using TensorFlow/Keras."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow not available")
    
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = inputs
    
    for hidden_dim in hidden_dims:
        x = tf.keras.layers.Dense(hidden_dim)(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    if task == 'regression':
        outputs = tf.keras.layers.Dense(output_dim, activation='linear')(x)
    else:
        outputs = tf.keras.layers.Dense(output_dim, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# ===============================================================================
# Unified Pipeline Classes
# ===============================================================================

@dataclass
class DeepLearningMetadata:
    """Metadata for deep learning pipeline."""
    backend: str
    task: str
    hidden_dims: List[int]
    dropout_rate: float
    learning_rate: float
    batch_size: int
    epochs: int
    early_stopping_patience: int
    model_architecture: Dict[str, Any]
    training_params: Dict[str, Any]
    feature_names: List[str]
    target_mean: Optional[float] = None
    target_std: Optional[float] = None

class DeepLearningPipeline:
    """Unified deep learning pipeline supporting PyTorch and TensorFlow backends."""
    
    def __init__(
        self,
        backend: str = 'pytorch',
        task: str = 'regression',
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        device: str = 'auto'
    ):
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
            
        self.backend = backend.lower()
        self.task = task.lower()
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        
        # Validate backend availability
        if self.backend == 'pytorch' and not HAS_TORCH:
            if HAS_TF:
                print("PyTorch not available, falling back to TensorFlow")
                self.backend = 'tensorflow'
            else:
                raise RuntimeError("Neither PyTorch nor TensorFlow available")
        elif self.backend == 'tensorflow' and not HAS_TF:
            if HAS_TORCH:
                print("TensorFlow not available, falling back to PyTorch")
                self.backend = 'pytorch'
            else:
                raise RuntimeError("Neither PyTorch nor TensorFlow available")
        
        # Device selection
        if device == 'auto':
            if self.backend == 'pytorch' and HAS_TORCH:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.model = None
        self.scaler = StandardScaler()
        self.metadata = None
        self.feature_names = None
        
    def _prepare_data(self, X: pd.DataFrame, y: np.ndarray = None, fit_scaler: bool = True):
        """Prepare data for training or inference."""
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = list(X.columns)
        else:
            X_scaled = self.scaler.transform(X)
            
        if self.backend == 'pytorch' and HAS_TORCH:
            X_tensor = torch.FloatTensor(X_scaled)
            if y is not None:
                y_tensor = torch.FloatTensor(y.reshape(-1, 1))
                return X_tensor, y_tensor
            return X_tensor
        else:
            if y is not None:
                return X_scaled, y.reshape(-1, 1)
            return X_scaled
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'DeepLearningPipeline':
        """Train the deep learning model."""
        # Prepare data
        X_prep, y_prep = self._prepare_data(X, y, fit_scaler=True)
        
        # Split data for validation
        if self.backend == 'pytorch':
            X_train, X_val, y_train, y_val = train_test_split(
                X_prep, y_prep, test_size=0.2, random_state=RANDOM_SEED
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_prep, y_prep, test_size=0.2, random_state=RANDOM_SEED
            )
        
        input_dim = X.shape[1]
        output_dim = 1 if self.task == 'regression' else 1  # Binary classification
        
        if self.backend == 'pytorch':
            self._fit_pytorch(X_train, y_train, X_val, y_val, input_dim, output_dim)
        else:
            self._fit_tensorflow(X_train, y_train, X_val, y_val, input_dim, output_dim)
        
        # Store metadata
        self.metadata = DeepLearningMetadata(
            backend=self.backend,
            task=self.task,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_patience,
            model_architecture={
                'input_dim': input_dim,
                'output_dim': output_dim,
                'use_batch_norm': self.use_batch_norm,
                'activation': self.activation
            },
            training_params={
                'n_samples': len(X),
                'n_features': input_dim
            },
            feature_names=self.feature_names,
            target_mean=float(np.mean(y)) if self.task == 'regression' else None,
            target_std=float(np.std(y)) if self.task == 'regression' else None
        )
        
        return self
    
    def _fit_pytorch(self, X_train, y_train, X_val, y_val, input_dim, output_dim):
        """Train PyTorch model."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
            
        # Create model
        self.model = PyTorchMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            activation=self.activation
        )
        
        if HAS_TORCH and torch.cuda.is_available():
            self.model = self.model.cuda()
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_val = X_val.cuda()
            y_val = y_val.cuda()
        
        # Loss function and optimizer
        if self.task == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break
    
    def _fit_tensorflow(self, X_train, y_train, X_val, y_val, input_dim, output_dim):
        """Train TensorFlow model."""
        self.model = create_tensorflow_mlp(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            activation=self.activation,
            task=self.task
        )
        
        # Compile model
        if self.task == 'regression':
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
        else:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=0
            )
        ]
        
        # Train model
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_prep = self._prepare_data(X, fit_scaler=False)
        
        if self.backend == 'pytorch':
            if not HAS_TORCH:
                raise RuntimeError("PyTorch not available")
            self.model.eval()
            with torch.no_grad():
                if HAS_TORCH and torch.cuda.is_available():
                    X_prep = X_prep.cuda()
                outputs = self.model(X_prep)
                if self.task == 'classification':
                    outputs = torch.sigmoid(outputs)
                predictions = outputs.cpu().numpy().flatten()
        else:
            if not HAS_TF:
                raise RuntimeError("TensorFlow not available") 
            predictions = self.model.predict(X_prep, verbose=0).flatten()
            
        return predictions
    
    @staticmethod
    def compute_manufacturing_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        task: str = 'regression',
        tolerance: float = 1.0,
        cost_per_unit: float = 100.0
    ) -> Dict[str, float]:
        """Compute manufacturing-specific metrics."""
        metrics = {}
        
        if task == 'regression':
            # Standard regression metrics
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics['r2'] = float(r2_score(y_true, y_pred))
            
            # Manufacturing-specific metrics
            # Prediction Within Spec (PWS)
            within_spec = np.abs(y_true - y_pred) <= tolerance
            metrics['pws'] = float(np.mean(within_spec))
            
            # Estimated Loss
            loss_per_sample = np.maximum(0, np.abs(y_true - y_pred) - tolerance) * cost_per_unit
            metrics['estimated_loss'] = float(np.sum(loss_per_sample))
            metrics['avg_loss_per_unit'] = float(np.mean(loss_per_sample))
        else:
            # Classification metrics
            from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
            
            # Convert predictions to binary if needed
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred))
            metrics['pr_auc'] = float(average_precision_score(y_true, y_pred))
            metrics['f1'] = float(f1_score(y_true, y_pred_binary))
            
            # Manufacturing-specific metrics for classification
            tp = np.sum((y_true == 1) & (y_pred_binary == 1))
            tn = np.sum((y_true == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true == 1) & (y_pred_binary == 0))
            
            metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            
            # PWS for classification (correct predictions)
            metrics['pws'] = float(np.mean(y_true == y_pred_binary))
            
            # Estimated cost (FP = false alarms, FN = missed defects)
            false_alarm_cost = 10.0  # Cost of unnecessary inspection
            missed_defect_cost = 1000.0  # Cost of missing a defect
            
            metrics['estimated_loss'] = float(fp * false_alarm_cost + fn * missed_defect_cost)
            metrics['avg_loss_per_unit'] = float(metrics['estimated_loss'] / len(y_true))
        
        return metrics
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        return self.compute_manufacturing_metrics(y, predictions, self.task)
    
    def save(self, path: Path):
        """Save model and metadata."""
        save_data = {
            'metadata': asdict(self.metadata) if self.metadata else None,
            'scaler': self.scaler,
            'backend': self.backend,
            'task': self.task
        }
        
        if self.backend == 'pytorch':
            save_data['model_state_dict'] = self.model.state_dict()
            save_data['model_class'] = 'PyTorchMLP'
        else:
            # For TensorFlow, save model separately
            model_dir = path.parent / f"{path.stem}_tf_model"
            self.model.save(model_dir)
            save_data['model_path'] = str(model_dir)
        
        joblib.dump(save_data, path)
    
    @staticmethod
    def load(path: Path) -> 'DeepLearningPipeline':
        """Load model and metadata."""
        save_data = joblib.load(path)
        metadata = save_data['metadata']
        
        # Recreate pipeline
        pipeline = DeepLearningPipeline(
            backend=save_data['backend'],
            task=save_data['task'],
            hidden_dims=metadata['hidden_dims'],
            dropout_rate=metadata['dropout_rate'],
            learning_rate=metadata['learning_rate'],
            batch_size=metadata['batch_size'],
            epochs=metadata['epochs'],
            early_stopping_patience=metadata['early_stopping_patience']
        )
        
        pipeline.scaler = save_data['scaler']
        pipeline.metadata = DeepLearningMetadata(**metadata)
        pipeline.feature_names = metadata['feature_names']
        
        # Load model
        if save_data['backend'] == 'pytorch':
            if not HAS_TORCH:
                raise RuntimeError("PyTorch not available for loading model")
            input_dim = metadata['model_architecture']['input_dim']
            output_dim = metadata['model_architecture']['output_dim']
            
            pipeline.model = PyTorchMLP(
                input_dim=input_dim,
                hidden_dims=metadata['hidden_dims'],
                output_dim=output_dim,
                dropout_rate=metadata['dropout_rate'],
                use_batch_norm=metadata['model_architecture']['use_batch_norm'],
                activation=metadata['model_architecture']['activation']
            )
            pipeline.model.load_state_dict(save_data['model_state_dict'])
            pipeline.model.eval()
        else:
            if not HAS_TF:
                raise RuntimeError("TensorFlow not available for loading model")
            model_path = save_data['model_path']
            pipeline.model = tf.keras.models.load_model(model_path)
        
        return pipeline

# ===============================================================================
# CLI Implementation
# ===============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Module 6.1 Deep Learning Pipeline for Semiconductor Manufacturing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a PyTorch model for regression
  python 6.1-deep-learning-pipeline.py train --dataset synthetic_yield --backend pytorch --save model.pth
  
  # Train a TensorFlow model for classification  
  python 6.1-deep-learning-pipeline.py train --dataset synthetic_defects --backend tensorflow --task classification
  
  # Evaluate model
  python 6.1-deep-learning-pipeline.py evaluate --model-path model.pth --dataset synthetic_yield
  
  # Make predictions
  python 6.1-deep-learning-pipeline.py predict --model-path model.pth --input-json '{"temperature":455, "pressure":2.6}'
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a deep learning model')
    train_parser.add_argument('--dataset', required=True, 
                             choices=['synthetic_yield', 'synthetic_defects'],
                             help='Dataset to use for training')
    train_parser.add_argument('--backend', default='pytorch',
                             choices=['pytorch', 'tensorflow'],
                             help='Deep learning backend')
    train_parser.add_argument('--task', default='regression',
                             choices=['regression', 'classification'],
                             help='Learning task type')
    train_parser.add_argument('--hidden-dims', nargs='+', type=int, default=[64, 32, 16],
                             help='Hidden layer dimensions')
    train_parser.add_argument('--dropout-rate', type=float, default=0.2,
                             help='Dropout rate')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Maximum number of epochs')
    train_parser.add_argument('--early-stopping-patience', type=int, default=10,
                             help='Early stopping patience')
    train_parser.add_argument('--save', type=str,
                             help='Path to save trained model')
    train_parser.add_argument('--n-samples', type=int, default=1000,
                             help='Number of samples for synthetic data')
    train_parser.set_defaults(func=action_train)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', required=True,
                            help='Path to saved model')
    eval_parser.add_argument('--dataset', required=True,
                            choices=['synthetic_yield', 'synthetic_defects'],
                            help='Dataset to evaluate on')
    eval_parser.add_argument('--n-samples', type=int, default=500,
                            help='Number of samples for synthetic data')
    eval_parser.set_defaults(func=action_evaluate)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model-path', required=True,
                               help='Path to saved model')
    predict_parser.add_argument('--input-json', required=True,
                               help='JSON string with input features')
    predict_parser.set_defaults(func=action_predict)
    
    return parser

def action_train(args):
    """Train a deep learning model."""
    try:
        # Generate or load data
        if args.dataset == 'synthetic_yield':
            task = 'regression'
            df = generate_synthetic_tabular(n=args.n_samples, task=task)
        elif args.dataset == 'synthetic_defects':
            task = 'classification'
            df = generate_synthetic_tabular(n=args.n_samples, task=task)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        # Override task if specified
        if hasattr(args, 'task') and args.task:
            task = args.task
        
        # Prepare data
        y = df['target'].values
        X = df.drop('target', axis=1)
        
        # Create and train pipeline
        pipeline = DeepLearningPipeline(
            backend=args.backend,
            task=task,
            hidden_dims=args.hidden_dims,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience
        )
        
        pipeline.fit(X, y)
        
        # Save model if requested
        if args.save:
            pipeline.save(Path(args.save))
        
        # Evaluate on training data for reporting
        metrics = pipeline.evaluate(X, y)
        
        # Output JSON
        result = {
            'status': 'trained',
            'backend': args.backend,
            'task': task,
            'dataset': args.dataset,
            'model_architecture': {
                'hidden_dims': args.hidden_dims,
                'dropout_rate': args.dropout_rate,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size
            },
            'training_metrics': metrics,
            'metadata': asdict(pipeline.metadata) if pipeline.metadata else None
        }
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {'status': 'error', 'message': str(e)}
        print(json.dumps(error_result))
        sys.exit(1)

def action_evaluate(args):
    """Evaluate a trained model."""
    try:
        # Load model
        pipeline = DeepLearningPipeline.load(Path(args.model_path))
        
        # Generate evaluation data
        task = pipeline.task
        if args.dataset == 'synthetic_yield':
            if task != 'regression':
                task = 'regression'  # Override for consistency
            df = generate_synthetic_tabular(n=args.n_samples, task=task, seed=RANDOM_SEED + 1)
        elif args.dataset == 'synthetic_defects':
            if task != 'classification':
                task = 'classification'  # Override for consistency
            df = generate_synthetic_tabular(n=args.n_samples, task=task, seed=RANDOM_SEED + 1)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        # Prepare data
        y = df['target'].values
        X = df.drop('target', axis=1)
        
        # Evaluate
        metrics = pipeline.evaluate(X, y)
        
        # Output JSON
        result = {
            'status': 'evaluated',
            'backend': pipeline.backend,
            'task': pipeline.task,
            'dataset': args.dataset,
            'metrics': metrics,
            'metadata': asdict(pipeline.metadata) if pipeline.metadata else None
        }
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {'status': 'error', 'message': str(e)}
        print(json.dumps(error_result))
        sys.exit(1)

def action_predict(args):
    """Make predictions with a trained model."""
    try:
        # Load model
        pipeline = DeepLearningPipeline.load(Path(args.model_path))
        
        # Parse input JSON
        input_data = json.loads(args.input_json)
        
        # Create DataFrame with expected features
        if pipeline.feature_names:
            # Ensure all expected features are present
            for feature in pipeline.feature_names:
                if feature not in input_data:
                    input_data[feature] = 0.0  # Default value
        
        df = pd.DataFrame([input_data])
        
        # Make prediction
        predictions = pipeline.predict(df)
        
        # Output JSON
        result = {
            'status': 'predicted',
            'backend': pipeline.backend,
            'task': pipeline.task,
            'input': input_data,
            'predictions': predictions.tolist(),
            'prediction_value': float(predictions[0])
        }
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {'status': 'error', 'message': str(e)}
        print(json.dumps(error_result))
        sys.exit(1)

def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()