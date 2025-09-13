#!/usr/bin/env python3
"""
Equipment Anomaly Detection Pipeline for Semiconductor Manufacturing

This production-ready script provides comprehensive anomaly detection capabilities
for semiconductor equipment monitoring using unsupervised learning approaches.
It implements Isolation Forest and Gaussian Mixture Models with time-series
specific features and threshold optimization for equipment monitoring.

Features:
- Unsupervised anomaly detection (Isolation Forest, GMM)
- Time-series aware feature engineering
- Threshold tuning with ROC analysis
- Manufacturing-specific metrics and cost analysis
- Synthetic equipment data generation
- Export detected intervals and anomaly scores
- Model persistence and reproducibility

Usage:
    python anomaly_detection_pipeline.py train --dataset synthetic_equipment --method isolation_forest --save model.joblib
    python anomaly_detection_pipeline.py evaluate --model-path model.joblib --dataset synthetic_equipment
    python anomaly_detection_pipeline.py predict --model-path model.joblib --input-json '{"temperature":455, "pressure":2.6, "vibration":0.8}'

Author: Machine Learning for Semiconductor Engineers
Date: 2025
License: MIT
"""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------- Synthetic Data Generators ---------------- #

def generate_equipment_timeseries(
    n_samples: int = 2000, 
    anomaly_rate: float = 0.05,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate synthetic equipment monitoring time series with injected anomalies."""
    rng = np.random.default_rng(seed)
    
    # Generate timestamp sequence (1 minute intervals)
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = pd.date_range(start_time, periods=n_samples, freq='1min')
    
    # Generate normal equipment behavior
    # Temperature follows daily pattern with noise
    daily_cycle = np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60))  # 24-hour cycle
    temperature = 450 + 10 * daily_cycle + rng.normal(0, 2, n_samples)
    
    # Pressure with some correlation to temperature
    pressure = 2.5 + 0.01 * (temperature - 450) + rng.normal(0, 0.1, n_samples)
    
    # Vibration levels (normally low)
    vibration = np.abs(rng.normal(0.5, 0.15, n_samples))
    
    # Flow rate with process variations
    flow = 120 + rng.normal(0, 5, n_samples)
    
    # Power consumption correlated with temperature
    power = 1000 + 2 * (temperature - 450) + rng.normal(0, 20, n_samples)
    
    # Generate anomalies
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = rng.choice(n_samples, n_anomalies, replace=False)
    
    # Create ground truth labels (0=normal, 1=anomaly)
    labels = np.zeros(n_samples)
    labels[anomaly_indices] = 1
    
    # Inject different types of anomalies
    for idx in anomaly_indices:
        anomaly_type = rng.choice(['temp_spike', 'pressure_drop', 'vibration_high', 'flow_anomaly'])
        
        if anomaly_type == 'temp_spike':
            temperature[idx] += rng.uniform(20, 50)
        elif anomaly_type == 'pressure_drop':
            pressure[idx] -= rng.uniform(0.5, 1.0)
        elif anomaly_type == 'vibration_high':
            vibration[idx] += rng.uniform(1.0, 3.0)
        elif anomaly_type == 'flow_anomaly':
            flow[idx] += rng.uniform(-30, 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'pressure': pressure,
        'vibration': vibration,
        'flow': flow,
        'power': power,
        'is_anomaly': labels  # Ground truth for evaluation
    })
    
    return df

def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-series specific features for anomaly detection."""
    df = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    feature_cols = ['temperature', 'pressure', 'vibration', 'flow', 'power']
    
    # Rolling statistics (windows: 5, 15, 60 minutes)
    for col in feature_cols:
        if col in df.columns:
            # Rolling mean and std
            for window in [5, 15, 60]:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            
            # First and second differences
            df[f'{col}_diff1'] = df[col].diff()
            df[f'{col}_diff2'] = df[col].diff().diff()
            
            # Rate of change
            df[f'{col}_pct_change'] = df[col].pct_change()
    
    # Cross-correlations
    if all(col in df.columns for col in ['temperature', 'pressure']):
        df['temp_pressure_ratio'] = df['temperature'] / (df['pressure'] + 1e-8)
    
    if all(col in df.columns for col in ['vibration', 'power']):
        df['vibration_power_ratio'] = df['vibration'] / (df['power'] + 1e-8)
    
    # Time-based features
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

# ---------------- Data Loading ---------------- #

def load_dataset(name: str) -> pd.DataFrame:
    """Load dataset by name."""
    if name == "synthetic_equipment":
        df = generate_equipment_timeseries()
        df = add_time_series_features(df)
        return df
    else:
        raise ValueError(f"Unknown dataset '{name}'. Currently supported: synthetic_equipment")

# ---------------- Pipeline Configuration ---------------- #

@dataclass
class AnomalyDetectionMetadata:
    """Metadata for anomaly detection pipeline."""
    trained_at: str
    method: str
    n_features_in: int
    contamination: float
    threshold: Optional[float]
    params: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None
    feature_names: Optional[List[str]] = None

class AnomalyDetectionPipeline:
    """Main pipeline for equipment anomaly detection."""
    
    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
        n_estimators: int = 100,
        n_components: int = 2,
        max_features: float = 1.0,
        random_state: int = RANDOM_SEED
    ) -> None:
        self.method = method.lower()
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state
        
        # Runtime objects
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Optional[AnomalyDetectionMetadata] = None
        self.threshold: Optional[float] = None
        self.feature_names: Optional[List[str]] = None
    
    def _build_detector(self):
        """Build the anomaly detection model."""
        if self.method == "isolation_forest":
            return IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.method == "gmm":
            # For GMM, we'll use decision_function to get scores
            return GaussianMixture(
                n_components=self.n_components,
                random_state=self.random_state,
                covariance_type='full'
            )
        else:
            raise ValueError(f"Unsupported method '{self.method}'. Choose from: isolation_forest, gmm")
    
    def build(self, feature_names: List[str]):
        """Build the complete pipeline."""
        steps = [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("detector", self._build_detector())
        ]
        
        self.pipeline = Pipeline(steps)
        self.feature_names = feature_names
        return self
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Fit the anomaly detection pipeline."""
        if self.pipeline is None:
            feature_names = [col for col in X.columns if col not in ['timestamp', 'is_anomaly']]
            self.build(feature_names)
        
        # Select features (exclude timestamp and labels)
        feature_cols = [col for col in X.columns if col not in ['timestamp', 'is_anomaly']]
        X_features = X[feature_cols]
        
        assert self.pipeline is not None
        self.pipeline.fit(X_features)
        
        # Store metadata
        self.metadata = AnomalyDetectionMetadata(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            method=self.method,
            n_features_in=X_features.shape[1],
            contamination=self.contamination,
            threshold=self.threshold,
            params={
                "method": self.method,
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "n_components": self.n_components,
                "max_features": self.max_features,
                "random_state": self.random_state
            },
            feature_names=feature_cols
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (1 for anomaly, -1 for normal)."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted")
        
        feature_cols = self.feature_names or [col for col in X.columns if col not in ['timestamp', 'is_anomaly']]
        X_features = X[feature_cols]
        
        if self.method == "isolation_forest":
            return self.pipeline.predict(X_features)
        elif self.method == "gmm":
            # For GMM, we need to implement anomaly detection logic
            scores = self.get_anomaly_scores(X)
            threshold = self.threshold or np.percentile(scores, (1 - self.contamination) * 100)
            return np.where(scores > threshold, -1, 1)  # -1 for anomaly, 1 for normal
    
    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores (higher = more anomalous)."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted")
        
        feature_cols = self.feature_names or [col for col in X.columns if col not in ['timestamp', 'is_anomaly']]
        X_features = X[feature_cols]
        
        if self.method == "isolation_forest":
            # For Isolation Forest, decision_function gives anomaly scores
            # More negative = more anomalous, so we negate
            return -self.pipeline.decision_function(X_features)
        elif self.method == "gmm":
            # For GMM, use negative log-likelihood as anomaly score
            return -self.pipeline.score_samples(X_features)
    
    def optimize_threshold(self, X: pd.DataFrame, y_true: np.ndarray) -> float:
        """Optimize anomaly detection threshold using ROC curve."""
        scores = self.get_anomaly_scores(X)
        
        # Convert predictions to binary (1 for anomaly, 0 for normal)
        y_binary = (y_true == 1).astype(int)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_binary, scores)
        
        # Find optimal threshold using Youden's J statistic
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        self.threshold = optimal_threshold
        
        # Update metadata
        if self.metadata:
            self.metadata.threshold = optimal_threshold
        
        return optimal_threshold
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate anomaly detection performance."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted")
        
        # Get predictions and scores
        y_pred = self.predict(X)
        scores = self.get_anomaly_scores(X)
        
        # Convert to binary format for evaluation
        y_binary_true = (y_true == 1).astype(int)
        y_binary_pred = (y_pred == -1).astype(int)  # -1 indicates anomaly
        
        # Calculate metrics
        metrics = self.compute_metrics(y_binary_true, y_binary_pred, scores)
        
        # Store metrics in metadata
        if self.metadata:
            self.metadata.metrics = metrics
        
        return metrics
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive anomaly detection metrics."""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        
        # Basic classification metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC and PR AUC
        roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
        pr_auc = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
        
        # Manufacturing-specific metrics
        total_samples = len(y_true)
        true_anomalies = np.sum(y_true)
        detected_anomalies = np.sum(y_pred)
        false_alarms = np.sum((y_pred == 1) & (y_true == 0))
        missed_anomalies = np.sum((y_pred == 0) & (y_true == 1))
        
        # Detection rate and false alarm rate
        detection_rate = recall
        false_alarm_rate = false_alarms / (total_samples - true_anomalies) if (total_samples - true_anomalies) > 0 else 0.0
        
        # Manufacturing cost estimates (assuming costs)
        cost_false_alarm = 100  # Cost of investigating false alarm
        cost_missed_anomaly = 1000  # Cost of missing real equipment failure
        
        estimated_cost = false_alarms * cost_false_alarm + missed_anomalies * cost_missed_anomaly
        cost_per_sample = estimated_cost / total_samples
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "detection_rate": float(detection_rate),
            "false_alarm_rate": float(false_alarm_rate),
            "true_anomalies": int(true_anomalies),
            "detected_anomalies": int(detected_anomalies),
            "false_alarms": int(false_alarms),
            "missed_anomalies": int(missed_anomalies),
            "estimated_cost": float(estimated_cost),
            "cost_per_sample": float(cost_per_sample),
            "total_samples": int(total_samples)
        }
    
    def export_intervals(self, X: pd.DataFrame, output_path: Path) -> Dict[str, Any]:
        """Export detected anomaly intervals and scores."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted")
        
        # Get predictions and scores
        y_pred = self.predict(X)
        scores = self.get_anomaly_scores(X)
        
        # Create results DataFrame
        results_df = X.copy()
        results_df['anomaly_score'] = scores
        results_df['is_predicted_anomaly'] = (y_pred == -1).astype(int)
        
        # Find anomaly intervals
        intervals = []
        in_anomaly = False
        start_idx = None
        
        for idx, is_anomaly in enumerate(y_pred == -1):
            if is_anomaly and not in_anomaly:
                # Start of anomaly interval
                start_idx = idx
                in_anomaly = True
            elif not is_anomaly and in_anomaly:
                # End of anomaly interval
                end_idx = idx - 1
                intervals.append({
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'start_timestamp': results_df.iloc[start_idx]['timestamp'] if 'timestamp' in results_df.columns else start_idx,
                    'end_timestamp': results_df.iloc[end_idx]['timestamp'] if 'timestamp' in results_df.columns else end_idx,
                    'duration_minutes': end_idx - start_idx + 1,
                    'max_score': scores[start_idx:end_idx+1].max(),
                    'mean_score': scores[start_idx:end_idx+1].mean()
                })
                in_anomaly = False
        
        # Handle case where anomaly extends to end of data
        if in_anomaly and start_idx is not None:
            end_idx = len(y_pred) - 1
            intervals.append({
                'start_index': start_idx,
                'end_index': end_idx,
                'start_timestamp': results_df.iloc[start_idx]['timestamp'] if 'timestamp' in results_df.columns else start_idx,
                'end_timestamp': results_df.iloc[end_idx]['timestamp'] if 'timestamp' in results_df.columns else end_idx,
                'duration_minutes': end_idx - start_idx + 1,
                'max_score': scores[start_idx:end_idx+1].max(),
                'mean_score': scores[start_idx:end_idx+1].mean()
            })
        
        # Export results
        export_data = {
            'metadata': asdict(self.metadata) if self.metadata else {},
            'summary': {
                'total_samples': len(X),
                'detected_anomalies': int(np.sum(y_pred == -1)),
                'anomaly_rate': float(np.mean(y_pred == -1)),
                'num_intervals': len(intervals),
                'mean_anomaly_score': float(scores.mean()),
                'max_anomaly_score': float(scores.max())
            },
            'intervals': intervals
        }
        
        # Save detailed results
        detailed_output_path = output_path.parent / f"{output_path.stem}_detailed.csv"
        results_df.to_csv(detailed_output_path, index=False)
        
        # Save summary
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return export_data
    
    def save(self, path: Path):
        """Save the trained pipeline."""
        data = {
            'pipeline': self.pipeline,
            'metadata': self.metadata,
            'threshold': self.threshold,
            'feature_names': self.feature_names
        }
        joblib.dump(data, path)
    
    @staticmethod
    def load(path: Path) -> 'AnomalyDetectionPipeline':
        """Load a trained pipeline."""
        data = joblib.load(path)
        
        # Create instance
        instance = AnomalyDetectionPipeline()
        instance.pipeline = data['pipeline']
        instance.metadata = data['metadata']
        instance.threshold = data['threshold']
        instance.feature_names = data['feature_names']
        
        return instance

# ---------------- CLI Actions ---------------- #

def action_train(args):
    """Train the anomaly detection pipeline."""
    df = load_dataset(args.dataset)
    
    # Split data if we have labels for validation
    if 'is_anomaly' in df.columns:
        # Use labels for threshold optimization
        y = df['is_anomaly'].values
        X_train, X_val, y_train, y_val = train_test_split(
            df, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
        )
    else:
        X_train = df
        X_val = None
        y_val = None
    
    # Initialize and train pipeline
    pipeline = AnomalyDetectionPipeline(
        method=args.method,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        n_components=args.n_components,
        max_features=args.max_features
    )
    
    pipeline.fit(X_train)
    
    # Optimize threshold if validation data available
    if X_val is not None and y_val is not None:
        optimal_threshold = pipeline.optimize_threshold(X_val, y_val)
        metrics = pipeline.evaluate(X_val, y_val)
    else:
        metrics = {}
        optimal_threshold = None
    
    # Save model if requested
    if args.save:
        pipeline.save(Path(args.save))
    
    # Output results
    result = {
        "status": "trained",
        "method": args.method,
        "optimal_threshold": optimal_threshold,
        "metrics": metrics,
        "metadata": asdict(pipeline.metadata) if pipeline.metadata else None
    }
    
    print(json.dumps(result, indent=2))

def action_evaluate(args):
    """Evaluate a trained anomaly detection model."""
    pipeline = AnomalyDetectionPipeline.load(Path(args.model_path))
    df = load_dataset(args.dataset)
    
    if 'is_anomaly' not in df.columns:
        raise ValueError("Dataset must contain 'is_anomaly' column for evaluation")
    
    y_true = df['is_anomaly'].values
    metrics = pipeline.evaluate(df, y_true)
    
    result = {
        "status": "evaluated",
        "metrics": metrics,
        "metadata": asdict(pipeline.metadata) if pipeline.metadata else None
    }
    
    print(json.dumps(result, indent=2))

def action_predict(args):
    """Make predictions with a trained model."""
    pipeline = AnomalyDetectionPipeline.load(Path(args.model_path))
    
    # Handle input
    if args.input_json:
        record = json.loads(args.input_json)
        df = pd.DataFrame([record])
        # For single records, add minimal required features
        base_features = ['temperature', 'pressure', 'vibration', 'flow', 'power']
        for col in base_features:
            if col not in df.columns:
                df[col] = 0.0  # Default value
        
        # For single record prediction, create a minimal time series to generate features
        # Duplicate the record to enable rolling calculations
        df_extended = pd.concat([df] * 70, ignore_index=True)  # 70 rows for 60-min rolling window
        df_extended['timestamp'] = pd.date_range('2024-01-01', periods=len(df_extended), freq='1min')
        df_extended = add_time_series_features(df_extended)
        # Take the last row (with full features)
        df = df_extended.tail(1).copy()
        
    elif args.input_file:
        df = pd.read_csv(args.input_file)
        # Add time series features for file input
        df = add_time_series_features(df)
    else:
        raise ValueError("Provide --input-json or --input-file")
    
    # Make predictions
    predictions = pipeline.predict(df)
    scores = pipeline.get_anomaly_scores(df)
    
    # Export intervals if requested and output path provided
    if args.export_intervals and len(df) > 1:
        export_data = pipeline.export_intervals(df, Path(args.export_intervals))
        result = {
            "status": "predicted",
            "anomaly_count": int(np.sum(predictions == -1)),
            "export_path": args.export_intervals,
            "export_summary": export_data['summary']
        }
    else:
        # Single prediction or no export requested
        result = {
            "status": "predicted",
            "predictions": [int(p) for p in predictions],
            "anomaly_scores": [float(s) for s in scores],
            "anomalies_detected": int(np.sum(predictions == -1))
        }
    
    print(json.dumps(result, indent=2))

# ---------------- Argument Parsing ---------------- #

def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced Equipment Anomaly Detection Pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    
    # Train subcommand
    p_train = sub.add_parser("train", help="Train an anomaly detection model")
    p_train.add_argument("--dataset", default="synthetic_equipment", help="Dataset name")
    p_train.add_argument(
        "--method", 
        default="isolation_forest", 
        choices=["isolation_forest", "gmm"],
        help="Anomaly detection method"
    )
    p_train.add_argument(
        "--contamination", 
        type=float, 
        default=0.05, 
        help="Expected proportion of anomalies"
    )
    p_train.add_argument(
        "--n-estimators", 
        type=int, 
        default=100, 
        help="Number of estimators (for Isolation Forest)"
    )
    p_train.add_argument(
        "--n-components", 
        type=int, 
        default=2, 
        help="Number of components (for GMM)"
    )
    p_train.add_argument(
        "--max-features", 
        type=float, 
        default=1.0, 
        help="Max features to use (for Isolation Forest)"
    )
    p_train.add_argument("--save", help="Path to save trained model")
    p_train.set_defaults(func=action_train)
    
    # Evaluate subcommand
    p_eval = sub.add_parser("evaluate", help="Evaluate a trained model")
    p_eval.add_argument("--model-path", required=True, help="Path to saved model")
    p_eval.add_argument("--dataset", default="synthetic_equipment", help="Dataset name")
    p_eval.set_defaults(func=action_evaluate)
    
    # Predict subcommand
    p_pred = sub.add_parser("predict", help="Make predictions with a trained model")
    p_pred.add_argument("--model-path", required=True, help="Path to saved model")
    p_pred.add_argument("--input-json", help="Single JSON record string")
    p_pred.add_argument("--input-file", help="Path to CSV file with data")
    p_pred.add_argument(
        "--export-intervals", 
        help="Path to export detected anomaly intervals"
    )
    p_pred.set_defaults(func=action_predict)
    
    return parser

def main(argv: Optional[List[str]] = None):
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()