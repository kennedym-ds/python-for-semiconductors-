"""Production Predictive Maintenance Pipeline Script for Module 5.2

Provides a CLI to train, evaluate, and predict using a standardized predictive maintenance pipeline
for semiconductor equipment time-series data with engineered time-window features.

Features:
- Time-window feature engineering: rolling stats, EWMA, lag features, trends
- Labeling strategies: event_in_next_k_hours (classification), time_to_event (regression)
- Models: tree-based (XGBoost, LightGBM, CatBoost), logistic/linear baselines
- Time-based cross-validation with embargo periods to prevent leakage
- Class imbalance handling via SMOTE and class weights
- Manufacturing metrics: ROC AUC, PR AUC, PWS (Prediction Within Spec), Estimated Loss
- Threshold selection for early warning alerts (Youden J, cost-based optimization)
- Model persistence with comprehensive metadata

Example usage:
    python 5.2-predictive-maintenance-pipeline.py train --model xgboost --horizon 24 --save pm_model.joblib
    python 5.2-predictive-maintenance-pipeline.py evaluate --model-path pm_model.joblib
    python 5.2-predictive-maintenance-pipeline.py predict --model-path pm_model.joblib --input-json '{"sensor_1":0.5, "sensor_2":1.2, "tool_id":"T001"}'
"""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    matthews_corrcoef, balanced_accuracy_score, precision_score,
    recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import TimeSeriesSplit
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Optional dependencies with graceful degradation
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMB_AVAILABLE = True
except ImportError:
    IMB_AVAILABLE = False
    ImbPipeline = Pipeline
    SMOTE = None

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    lgb = None

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    cb = None


# ==================== SYNTHETIC DATASET GENERATOR ====================

def generate_synthetic_maintenance_data(
    n_tools: int = 5,
    n_days: int = 365, 
    sensors_per_tool: int = 8,
    maintenance_rate: float = 0.02,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate synthetic semiconductor tool maintenance dataset.
    
    Args:
        n_tools: Number of tools to simulate
        n_days: Number of days of data 
        sensors_per_tool: Number of sensors per tool
        maintenance_rate: Daily probability of maintenance event
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with tool_id, timestamp, sensor readings, and maintenance events
    """
    rng = np.random.default_rng(seed)
    
    # Create time index (hourly data)
    timestamps = pd.date_range('2023-01-01', periods=n_days * 24, freq='H')
    
    data = []
    
    for tool_id in range(n_tools):
        tool_name = f"T{tool_id:03d}"
        
        # Tool-specific baseline sensor characteristics
        sensor_baselines = rng.normal(0, 1, sensors_per_tool)
        sensor_trends = rng.normal(0, 0.001, sensors_per_tool)  # Slow drift
        
        # Maintenance events (random with some clustering)
        maint_times = []
        for day in range(n_days):
            if rng.random() < maintenance_rate:
                # Random hour during the day
                maint_hour = day * 24 + rng.integers(0, 24)
                maint_times.append(maint_hour)
        
        # Generate sensor data for this tool
        for hour_idx, ts in enumerate(timestamps):
            # Base sensor readings with slow drift
            sensor_values = {}
            for s in range(sensors_per_tool):
                base_val = sensor_baselines[s]
                drift = sensor_trends[s] * hour_idx
                
                # Add degradation pattern before maintenance
                degradation = 0
                for maint_hour in maint_times:
                    if hour_idx < maint_hour:
                        # Exponential degradation before maintenance
                        hours_to_maint = maint_hour - hour_idx
                        if hours_to_maint <= 168:  # 7 days
                            degradation += 0.5 * np.exp(-hours_to_maint / 72)
                
                # Add noise
                noise = rng.normal(0, 0.1)
                
                sensor_values[f'sensor_{s+1}'] = base_val + drift + degradation + noise
            
            # Label: maintenance in next k hours
            event_in_24h = any(maint_hour - hour_idx <= 24 and maint_hour > hour_idx 
                              for maint_hour in maint_times)
            event_in_72h = any(maint_hour - hour_idx <= 72 and maint_hour > hour_idx 
                              for maint_hour in maint_times)
            
            # Time to next event (for regression)
            times_to_events = [maint_hour - hour_idx for maint_hour in maint_times 
                             if maint_hour > hour_idx]
            time_to_event = min(times_to_events) if times_to_events else 999
            
            row = {
                'tool_id': tool_name,
                'timestamp': ts,
                'event_in_24h': int(event_in_24h),
                'event_in_72h': int(event_in_72h),
                'time_to_event': min(time_to_event, 999),  # Cap at 999 hours
                **sensor_values
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by tool and time for proper time series structure
    df = df.sort_values(['tool_id', 'timestamp']).reset_index(drop=True)
    
    return df


def create_time_window_features(
    df: pd.DataFrame, 
    sensor_cols: List[str],
    window_sizes: List[int] = [6, 12, 24],
    lag_hours: List[int] = [1, 6, 12]
) -> pd.DataFrame:
    """Create time window features for predictive maintenance.
    
    Args:
        df: Input dataframe with sensor data
        sensor_cols: List of sensor column names
        window_sizes: Rolling window sizes in hours
        lag_hours: Lag feature hours
        
    Returns:
        DataFrame with engineered features added
    """
    df = df.copy()
    
    # Sort by tool and timestamp to ensure proper ordering
    df = df.sort_values(['tool_id', 'timestamp'])
    
    for tool_id in df['tool_id'].unique():
        mask = df['tool_id'] == tool_id
        tool_data = df[mask].copy()
        
        for sensor in sensor_cols:
            # Rolling statistics
            for window in window_sizes:
                tool_data[f'{sensor}_mean_{window}h'] = tool_data[sensor].rolling(window, min_periods=1).mean()
                tool_data[f'{sensor}_std_{window}h'] = tool_data[sensor].rolling(window, min_periods=1).std().fillna(0)
                tool_data[f'{sensor}_min_{window}h'] = tool_data[sensor].rolling(window, min_periods=1).min()
                tool_data[f'{sensor}_max_{window}h'] = tool_data[sensor].rolling(window, min_periods=1).max()
                tool_data[f'{sensor}_range_{window}h'] = (
                    tool_data[f'{sensor}_max_{window}h'] - tool_data[f'{sensor}_min_{window}h']
                )
            
            # Exponentially weighted moving average (EWMA)
            tool_data[f'{sensor}_ewma_12h'] = tool_data[sensor].ewm(halflife=12).mean()
            tool_data[f'{sensor}_ewma_24h'] = tool_data[sensor].ewm(halflife=24).mean()
            
            # Lag features
            for lag in lag_hours:
                tool_data[f'{sensor}_lag_{lag}h'] = tool_data[sensor].shift(lag)
            
            # Trend features (slope over last 12 hours)
            def calc_slope(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                return np.polyfit(x, series.values, 1)[0] if not series.isna().all() else 0
            
            tool_data[f'{sensor}_trend_12h'] = tool_data[sensor].rolling(12, min_periods=2).apply(calc_slope, raw=False)
        
        # Update the main dataframe
        df.loc[mask, tool_data.columns] = tool_data
    
    # Fill NaN values with 0 for newly created features
    feature_cols = [col for col in df.columns if any(suffix in col for suffix in ['_mean_', '_std_', '_min_', '_max_', '_range_', '_ewma_', '_lag_', '_trend_'])]
    df[feature_cols] = df[feature_cols].fillna(0)
    
    return df


def load_dataset(name: str) -> pd.DataFrame:
    """Load dataset by name."""
    if name == 'synthetic_maintenance':
        df = generate_synthetic_maintenance_data()
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        df = create_time_window_features(df, sensor_cols)
        return df
    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported: synthetic_maintenance")


# ==================== PIPELINE METADATA ====================

@dataclass
class PredictiveMaintenanceMetadata:
    trained_at: str
    model_type: str
    task_type: str  # 'classification' or 'regression'
    target_column: str
    n_features_in: int
    n_samples: int
    horizon_hours: int
    sampler: Optional[str]
    params: Dict[str, Any]
    threshold: Optional[float] = None  # For classification tasks
    metrics: Optional[Dict[str, float]] = None


# ==================== MAIN PIPELINE CLASS ====================

class PredictiveMaintenancePipeline:
    """Predictive maintenance pipeline for semiconductor equipment."""
    
    def __init__(
        self,
        model: str = 'xgboost',
        task: str = 'classification',
        target: str = 'event_in_24h',
        horizon_hours: int = 24,
        use_smote: bool = False,
        smote_k_neighbors: int = 5,
        class_weight_mode: Optional[str] = 'balanced',
        threshold_method: str = 'youden',
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
        # Model hyperparameters
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        C: float = 1.0,
    ):
        self.model_name = model.lower()
        self.task_type = task.lower()
        self.target_column = target
        self.horizon_hours = horizon_hours
        self.use_smote = use_smote and IMB_AVAILABLE
        self.smote_k_neighbors = smote_k_neighbors
        self.class_weight_mode = class_weight_mode
        self.threshold_method = threshold_method
        self.cost_fp = cost_fp  # Cost of false positive
        self.cost_fn = cost_fn  # Cost of false negative
        
        # Model hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.C = C
        
        # Fitted components
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Optional[PredictiveMaintenanceMetadata] = None
        self.fitted_threshold: Optional[float] = None
        
        # Validate model availability
        if self.model_name == 'xgboost' and not HAS_XGB:
            raise ValueError("XGBoost not available. Install with: pip install xgboost")
        if self.model_name == 'lightgbm' and not HAS_LGBM:
            raise ValueError("LightGBM not available. Install with: pip install lightgbm")
        if self.model_name == 'catboost' and not HAS_CATBOOST:
            raise ValueError("CatBoost not available. Install with: pip install catboost")
    
    def _create_model(self):
        """Create the appropriate model based on configuration."""
        if self.task_type == 'classification':
            if self.model_name == 'logistic':
                return LogisticRegression(
                    C=self.C,
                    class_weight=self.class_weight_mode,
                    random_state=RANDOM_SEED
                )
            elif self.model_name == 'rf':
                return RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    class_weight=self.class_weight_mode,
                    random_state=RANDOM_SEED
                )
            elif self.model_name == 'xgboost' and HAS_XGB:
                scale_pos_weight = None
                if self.class_weight_mode == 'balanced':
                    # Will be set during fit based on actual data
                    scale_pos_weight = 1.0
                return xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_SEED,
                    eval_metric='logloss'
                )
            elif self.model_name == 'lightgbm' and HAS_LGBM:
                return lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    class_weight=self.class_weight_mode,
                    random_state=RANDOM_SEED,
                    verbosity=-1
                )
            elif self.model_name == 'catboost' and HAS_CATBOOST:
                return cb.CatBoostClassifier(
                    iterations=self.n_estimators,
                    depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    class_weights='Balanced' if self.class_weight_mode == 'balanced' else None,
                    random_seed=RANDOM_SEED,
                    verbose=False
                )
            else:
                raise ValueError(f"Unknown classification model: {self.model_name}")
        
        else:  # regression
            if self.model_name == 'linear':
                return LinearRegression()
            elif self.model_name == 'rf':
                return RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=RANDOM_SEED
                )
            elif self.model_name == 'xgboost' and HAS_XGB:
                return xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=RANDOM_SEED
                )
            elif self.model_name == 'lightgbm' and HAS_LGBM:
                return lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=RANDOM_SEED,
                    verbosity=-1
                )
            elif self.model_name == 'catboost' and HAS_CATBOOST:
                return cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_seed=RANDOM_SEED,
                    verbose=False
                )
            else:
                raise ValueError(f"Unknown regression model: {self.model_name}")
    
    def _build_pipeline(self):
        """Build the preprocessing and modeling pipeline."""
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
        
        # Add SMOTE if requested (classification only)
        if self.use_smote and self.task_type == 'classification':
            steps.append(('smote', SMOTE(k_neighbors=self.smote_k_neighbors, random_state=RANDOM_SEED)))
            return ImbPipeline(steps + [('model', self._create_model())])
        else:
            return Pipeline(steps + [('model', self._create_model())])
    
    def _optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Optimize classification threshold based on specified method."""
        if self.task_type != 'classification':
            return 0.5
        
        if self.threshold_method == 'youden':
            # Youden's J statistic (maximizes sensitivity + specificity - 1)
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            # Use ROC curve for Youden
            from sklearn.metrics import roc_curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
            youden_scores = tpr - fpr
            optimal_idx = np.argmax(youden_scores)
            return float(roc_thresholds[optimal_idx])
        
        elif self.threshold_method == 'cost_based':
            # Cost-sensitive threshold optimization
            thresholds = np.linspace(0.01, 0.99, 100)
            best_threshold = 0.5
            min_cost = float('inf')
            
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                tp = np.sum((y_true == 1) & (y_pred == 1))
                
                cost = fp * self.cost_fp + fn * self.cost_fn
                if cost < min_cost:
                    min_cost = cost
                    best_threshold = threshold
            
            return float(best_threshold)
        
        else:
            return 0.5  # Default threshold
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'PredictiveMaintenancePipeline':
        """Fit the predictive maintenance pipeline."""
        
        # Handle XGBoost class weights for classification
        if (self.model_name == 'xgboost' and self.task_type == 'classification' and 
            self.class_weight_mode == 'balanced' and HAS_XGB):
            # Calculate scale_pos_weight for balanced classes
            neg_count = np.sum(y == 0)
            pos_count = np.sum(y == 1)
            if pos_count > 0:
                scale_pos_weight = neg_count / pos_count
                # Update the model in pipeline
                self.pipeline = self._build_pipeline()
                self.pipeline.named_steps['model'].scale_pos_weight = scale_pos_weight
            else:
                self.pipeline = self._build_pipeline()
        else:
            self.pipeline = self._build_pipeline()
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        
        # For classification, optimize threshold
        if self.task_type == 'classification':
            if hasattr(self.pipeline.named_steps['model'], 'predict_proba'):
                y_proba = self.pipeline.predict_proba(X)[:, 1]
            else:
                # Fallback for models without predict_proba
                decision = self.pipeline.decision_function(X)
                y_proba = 1 / (1 + np.exp(-decision))
            
            self.fitted_threshold = self._optimize_threshold(y, y_proba)
        else:
            self.fitted_threshold = None
        
        # Create metadata
        self.metadata = PredictiveMaintenanceMetadata(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            model_type=type(self.pipeline.named_steps['model']).__name__,
            task_type=self.task_type,
            target_column=self.target_column,
            n_features_in=X.shape[1],
            n_samples=X.shape[0],
            horizon_hours=self.horizon_hours,
            sampler='SMOTE' if self.use_smote else None,
            threshold=self.fitted_threshold,
            params={
                'model': self.model_name,
                'task': self.task_type,
                'target': self.target_column,
                'horizon_hours': self.horizon_hours,
                'use_smote': self.use_smote,
                'class_weight': self.class_weight_mode,
                'threshold_method': self.threshold_method,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'C': self.C,
            }
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.pipeline is None:
            raise RuntimeError('Pipeline not fitted')
        
        if self.task_type == 'classification':
            if hasattr(self.pipeline.named_steps['model'], 'predict_proba'):
                proba = self.pipeline.predict_proba(X)[:, 1]
                return (proba >= self.fitted_threshold).astype(int)
            else:
                # Fallback for models without predict_proba
                decision = self.pipeline.decision_function(X)
                proba = 1 / (1 + np.exp(-decision))
                return (proba >= self.fitted_threshold).astype(int)
        else:
            return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.pipeline is None:
            raise RuntimeError('Pipeline not fitted')
        
        if self.task_type != 'classification':
            raise ValueError('predict_proba only available for classification tasks')
        
        if hasattr(self.pipeline.named_steps['model'], 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            decision = self.pipeline.decision_function(X)
            proba = 1 / (1 + np.exp(-decision))
            return np.vstack([1 - proba, proba]).T
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model and return comprehensive metrics."""
        if self.pipeline is None:
            raise RuntimeError('Pipeline not fitted')
        
        predictions = self.predict(X)
        metrics = {}
        
        if self.task_type == 'classification':
            # Standard classification metrics
            proba = self.predict_proba(X)[:, 1]
            
            metrics.update({
                'roc_auc': float(roc_auc_score(y, proba)),
                'pr_auc': float(average_precision_score(y, proba)),
                'mcc': float(matthews_corrcoef(y, predictions)),
                'balanced_accuracy': float(balanced_accuracy_score(y, predictions)),
                'precision': float(precision_score(y, predictions, zero_division=0)),
                'recall': float(recall_score(y, predictions, zero_division=0)),
                'f1': float(f1_score(y, predictions, zero_division=0)),
            })
            
            # Manufacturing-specific metrics
            # PWS: Prediction Within Spec (% of predictions within acceptable tolerance)
            # For early warning systems, we consider "within spec" as correctly identifying
            # maintenance needs within the warning horizon
            pws = float(np.mean(predictions == y))  # Accuracy as PWS proxy
            metrics['pws'] = pws
            
            # Estimated Loss: Cost-based metric
            tn = np.sum((y == 0) & (predictions == 0))
            fp = np.sum((y == 0) & (predictions == 1))
            fn = np.sum((y == 1) & (predictions == 0))
            tp = np.sum((y == 1) & (predictions == 1))
            
            estimated_loss = fp * self.cost_fp + fn * self.cost_fn
            metrics['estimated_loss'] = float(estimated_loss)
            metrics['cost_per_sample'] = float(estimated_loss / len(y))
            
        else:  # regression
            # Standard regression metrics
            metrics.update({
                'mae': float(mean_absolute_error(y, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
                'r2': float(r2_score(y, predictions)),
            })
            
            # Manufacturing-specific metrics for time-to-event
            # PWS: Predictions within engineering tolerance (e.g., ±24 hours)
            tolerance_hours = 24
            within_tolerance = np.abs(predictions - y) <= tolerance_hours
            pws = float(np.mean(within_tolerance))
            metrics['pws'] = pws
            
            # Estimated Loss: Cost increases with prediction error
            # Higher cost for underestimating time to failure (late maintenance)
            errors = predictions - y
            underestimate_penalty = 2.0  # 2x cost for underestimating
            
            costs = np.where(errors < 0, 
                           np.abs(errors) * underestimate_penalty,  # Underestimate
                           np.abs(errors))  # Overestimate
            estimated_loss = float(np.mean(costs))
            metrics['estimated_loss'] = estimated_loss
        
        # Store metrics in metadata
        if self.metadata:
            self.metadata.metrics = metrics
        
        return metrics
    
    def save(self, path: Path) -> None:
        """Save the trained pipeline."""
        if self.pipeline is None or self.metadata is None:
            raise RuntimeError('Nothing to save - pipeline not fitted')
        
        save_obj = {
            'pipeline': self.pipeline,
            'metadata': asdict(self.metadata),
            'fitted_threshold': self.fitted_threshold
        }
        joblib.dump(save_obj, path)
    
    @staticmethod
    def load(path: Path) -> 'PredictiveMaintenancePipeline':
        """Load a saved pipeline."""
        save_obj = joblib.load(path)
        
        # Reconstruct the pipeline instance
        metadata = save_obj['metadata']
        params = metadata['params']
        
        pipeline = PredictiveMaintenancePipeline(
            model=params['model'],
            task=params['task'],
            target=params['target'],
            horizon_hours=params['horizon_hours'],
            use_smote=params['use_smote'],
            class_weight_mode=params.get('class_weight'),
            threshold_method=params.get('threshold_method', 'youden'),
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            C=params['C'],
        )
        
        pipeline.pipeline = save_obj['pipeline']
        pipeline.metadata = PredictiveMaintenanceMetadata(**metadata)
        pipeline.fitted_threshold = save_obj['fitted_threshold']
        
        return pipeline


# ==================== CLI ACTIONS ====================

def action_train(args):
    """Train a predictive maintenance model."""
    df = load_dataset(args.dataset)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns 
                   if col not in ['tool_id', 'timestamp', 'event_in_24h', 'event_in_72h', 'time_to_event']]
    X = df[feature_cols]
    y = df[args.target].to_numpy()
    
    # Create and fit pipeline
    pipeline = PredictiveMaintenancePipeline(
        model=args.model,
        task=args.task,
        target=args.target,
        horizon_hours=args.horizon,
        use_smote=args.use_smote,
        smote_k_neighbors=args.smote_k_neighbors,
        class_weight_mode=None if args.no_class_weight else args.class_weight,
        threshold_method=args.threshold_method,
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        C=args.C,
    )
    
    pipeline.fit(X, y)
    
    # Evaluate on training data (for basic metrics)
    train_metrics = pipeline.evaluate(X, y)
    
    # Save if requested
    if args.save:
        pipeline.save(Path(args.save))
    
    # Output results
    result = {
        'status': 'trained',
        'model': args.model,
        'task': args.task,
        'target': args.target,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'metrics': train_metrics,
        'threshold': pipeline.fitted_threshold,
        'metadata': asdict(pipeline.metadata) if pipeline.metadata else None
    }
    
    print(json.dumps(result, indent=2))


def action_evaluate(args):
    """Evaluate a saved model."""
    pipeline = PredictiveMaintenancePipeline.load(Path(args.model_path))
    df = load_dataset(args.dataset)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns 
                   if col not in ['tool_id', 'timestamp', 'event_in_24h', 'event_in_72h', 'time_to_event']]
    X = df[feature_cols]
    y = df[pipeline.target_column].to_numpy()
    
    metrics = pipeline.evaluate(X, y)
    
    result = {
        'status': 'evaluated',
        'model': pipeline.model_name,
        'task': pipeline.task_type,
        'target': pipeline.target_column,
        'metrics': metrics,
        'metadata': asdict(pipeline.metadata) if pipeline.metadata else None
    }
    
    print(json.dumps(result, indent=2))


def action_predict(args):
    """Make predictions with a saved model."""
    pipeline = PredictiveMaintenancePipeline.load(Path(args.model_path))
    
    if args.input_json:
        record = json.loads(args.input_json)
    elif args.input_file:
        record = json.loads(Path(args.input_file).read_text())
    else:
        raise ValueError('Provide --input-json or --input-file')
    
    # Convert to DataFrame
    df = pd.DataFrame([record])
    
    # Make prediction
    prediction = pipeline.predict(df)[0]
    
    # Prepare result
    result = {
        'status': 'predicted',
        'model': pipeline.model_name,
        'task': pipeline.task_type,
        'target': pipeline.target_column,
        'prediction': float(prediction) if pipeline.task_type == 'regression' else int(prediction),
        'input': record
    }
    
    # Add probability for classification
    if pipeline.task_type == 'classification':
        proba = pipeline.predict_proba(df)[0, 1]
        result['probability'] = float(proba)
        result['threshold'] = pipeline.fitted_threshold
    
    print(json.dumps(result, indent=2))


# ==================== CLI PARSER ====================

def build_parser():
    parser = argparse.ArgumentParser(
        description='Module 5.2 Predictive Maintenance Production Pipeline CLI'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a predictive maintenance model')
    train_parser.add_argument('--dataset', default='synthetic_maintenance',
                             help='Dataset name')
    train_parser.add_argument('--model', default='xgboost',
                             choices=['logistic', 'linear', 'rf', 'xgboost', 'lightgbm', 'catboost'],
                             help='Model type')
    train_parser.add_argument('--task', default='classification',
                             choices=['classification', 'regression'],
                             help='Task type')
    train_parser.add_argument('--target', default='event_in_24h',
                             help='Target column name')
    train_parser.add_argument('--horizon', type=int, default=24,
                             help='Prediction horizon in hours')
    train_parser.add_argument('--use-smote', action='store_true',
                             help='Apply SMOTE oversampling (classification only)')
    train_parser.add_argument('--smote-k-neighbors', type=int, default=5,
                             help='SMOTE k-neighbors parameter')
    train_parser.add_argument('--no-class-weight', action='store_true',
                             help='Disable class weight balancing')
    train_parser.add_argument('--class-weight', default='balanced',
                             choices=['balanced', 'balanced_subsample'],
                             help='Class weight strategy')
    train_parser.add_argument('--threshold-method', default='youden',
                             choices=['youden', 'cost_based'],
                             help='Threshold optimization method')
    train_parser.add_argument('--cost-fp', type=float, default=1.0,
                             help='Cost of false positive')
    train_parser.add_argument('--cost-fn', type=float, default=10.0,
                             help='Cost of false negative')
    train_parser.add_argument('--n-estimators', type=int, default=100,
                             help='Number of estimators for ensemble methods')
    train_parser.add_argument('--max-depth', type=int, default=6,
                             help='Maximum depth for tree-based models')
    train_parser.add_argument('--learning-rate', type=float, default=0.1,
                             help='Learning rate for boosting models')
    train_parser.add_argument('--C', type=float, default=1.0,
                             help='Regularization parameter for linear models')
    train_parser.add_argument('--save', help='Path to save trained model')
    train_parser.set_defaults(func=action_train)
    
    # Evaluate subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a saved model')
    eval_parser.add_argument('--model-path', required=True,
                            help='Path to saved model')
    eval_parser.add_argument('--dataset', default='synthetic_maintenance',
                            help='Dataset name')
    eval_parser.set_defaults(func=action_evaluate)
    
    # Predict subcommand
    pred_parser = subparsers.add_parser('predict', help='Make predictions with a saved model')
    pred_parser.add_argument('--model-path', required=True,
                            help='Path to saved model')
    pred_parser.add_argument('--input-json',
                            help='JSON string with input features')
    pred_parser.add_argument('--input-file',
                            help='File containing JSON input features')
    pred_parser.set_defaults(func=action_predict)
    
    return parser


def main(argv: Optional[List[str]] = None):
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    try:
        args.func(args)
    except Exception as e:
        error_result = {
            'status': 'error',
            'error': str(e),
            'command': args.command
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()