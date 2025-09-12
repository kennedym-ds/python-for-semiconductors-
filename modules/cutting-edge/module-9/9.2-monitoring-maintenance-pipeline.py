"""Production Monitoring & Maintenance Pipeline Script for Module 9.2

Provides a CLI for model monitoring and operational maintenance including:
- Experiment tracking with MLflow
- Data/concept drift detection (PSI, KS-test, Wasserstein distance)
- Model performance monitoring over time
- Alert thresholds and trend analysis
- Manufacturing metrics with PWS and Estimated Loss tracking

Features:
- MLflow integration for experiment tracking, metrics, and artifacts
- Multiple drift detection methods with configurable thresholds
- Performance monitoring with trend analysis
- JSON output for all operations
- Model registry integration
- Alert generation based on configurable thresholds

Example usage:
    python 9.2-monitoring-maintenance-pipeline.py train --dataset synthetic_yield \\
        --enable-mlflow --save model.joblib
    python 9.2-monitoring-maintenance-pipeline.py evaluate --model-path model.joblib \\
        --dataset synthetic_yield --enable-mlflow
    python 9.2-monitoring-maintenance-pipeline.py predict --model-path model.joblib \\
        --input-json '{"temperature":455, "pressure":2.6, "flow":118, "time":62}'
"""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, wasserstein_distance
import joblib

# Optional MLflow import with graceful fallback
HAS_MLFLOW = True
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    HAS_MLFLOW = False
    warnings.warn("MLflow not available. Install with: pip install mlflow")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Manufacturing tolerances for PWS calculation
DEFAULT_TOLERANCE = 0.1  # 10% tolerance for predictions


@dataclass
class DriftConfig:
    """Configuration for drift detection thresholds."""

    psi_threshold: float = 0.2  # Population Stability Index threshold
    ks_p_threshold: float = 0.05  # Kolmogorov-Smirnov p-value threshold
    wasserstein_threshold: float = 0.5  # Wasserstein distance threshold
    performance_degradation_threshold: float = 0.1  # 10% performance drop
    alert_consecutive_violations: int = 2  # Alerts after N consecutive violations


@dataclass
class MonitoringMetrics:
    """Comprehensive monitoring metrics for semiconductor manufacturing."""

    mae: float
    rmse: float
    r2: float
    pws_percent: float  # Prediction Within Spec percentage
    estimated_loss: float  # Estimated cost impact of prediction errors
    drift_scores: Dict[str, float]
    alert_flags: Dict[str, bool]
    trend_direction: str  # "improving", "stable", "degrading"


def generate_yield_process(n=800, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic semiconductor yield process data."""
    rng = np.random.default_rng(seed)
    temp = rng.normal(450, 15, n)
    pressure = rng.normal(2.5, 0.3, n)
    flow = rng.normal(120, 10, n)
    time = rng.normal(60, 5, n)
    noise = rng.normal(0, 3, n)

    # Complex yield relationship
    yield_base = (
        70
        + 0.2 * (temp - 450)
        - 0.001 * (temp - 450) ** 2
        + 10 * pressure
        - 2 * pressure**2
        + 0.1 * flow
        - 0.0005 * flow**2
        + 0.3 * time
        - 0.002 * time**2
        + 0.01 * temp * flow / 1000
        + noise
    )

    # Add some process drift for testing drift detection
    drift_factor = np.linspace(0, 0.1, n)
    yield_final = yield_base + drift_factor * temp * 0.001

    df = pd.DataFrame({"temperature": temp, "pressure": pressure, "flow": flow, "time": time, "target": yield_final})

    # Add engineered features
    df["temp_centered"] = df["temperature"] - df["temperature"].mean()
    df["pressure_sq"] = df["pressure"] ** 2
    df["flow_time_inter"] = df["flow"] * df["time"]
    df["temp_flow_inter"] = df["temperature"] * df["flow"]

    return df


def generate_drift_data(base_data: pd.DataFrame, drift_strength: float = 0.5) -> pd.DataFrame:
    """Generate data with injected drift for testing drift detection."""
    drifted_data = base_data.copy()
    n = len(drifted_data)

    # Inject gradual drift in key features
    drift_factor = np.linspace(0, drift_strength, n)
    drifted_data["temperature"] += drift_factor * 20  # Temperature drift
    drifted_data["pressure"] += drift_factor * 0.5  # Pressure drift

    # Recalculate engineered features
    drifted_data["temp_centered"] = drifted_data["temperature"] - drifted_data["temperature"].mean()
    drifted_data["pressure_sq"] = drifted_data["pressure"] ** 2
    drifted_data["temp_flow_inter"] = drifted_data["temperature"] * drifted_data["flow"]

    return drifted_data


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between reference and current distributions."""
    # Create bins based on reference data
    _, bin_edges = np.histogram(reference, bins=bins)

    # Calculate distributions
    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    cur_hist, _ = np.histogram(current, bins=bin_edges)

    # Convert to proportions
    ref_prop = ref_hist / len(reference)
    cur_prop = cur_hist / len(current)

    # Avoid division by zero
    ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
    cur_prop = np.where(cur_prop == 0, 0.0001, cur_prop)

    # Calculate PSI
    psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
    return psi


def calculate_manufacturing_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = DEFAULT_TOLERANCE
) -> Dict[str, float]:
    """Calculate manufacturing-specific metrics including PWS and estimated loss."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Prediction Within Spec (PWS) - percentage of predictions within tolerance
    relative_error = np.abs((y_pred - y_true) / y_true)
    pws_percent = (relative_error <= tolerance).mean() * 100

    # Estimated Loss - simplified cost model based on prediction errors
    # Assume cost increases quadratically with error magnitude
    cost_per_unit_error = 1000  # $1000 per unit error squared
    estimated_loss = np.sum((y_pred - y_true) ** 2) * cost_per_unit_error

    return {"mae": mae, "rmse": rmse, "r2": r2, "pws_percent": pws_percent, "estimated_loss": estimated_loss}


def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, config: DriftConfig) -> Dict[str, Any]:
    """Comprehensive drift detection using multiple methods."""
    drift_results = {}
    alert_flags = {}

    # Select numeric columns for drift detection
    numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "target"]

    for col in numeric_cols:
        if col in current_data.columns:
            ref_values = reference_data[col].dropna()
            cur_values = current_data[col].dropna()

            if len(ref_values) > 0 and len(cur_values) > 0:
                # PSI calculation
                psi_score = calculate_psi(ref_values.values, cur_values.values)

                # KS test
                ks_stat, ks_p_value = ks_2samp(ref_values, cur_values)

                # Wasserstein distance
                w_distance = wasserstein_distance(ref_values, cur_values)

                drift_results[f"{col}_psi"] = psi_score
                drift_results[f"{col}_ks_stat"] = ks_stat
                drift_results[f"{col}_ks_p"] = ks_p_value
                drift_results[f"{col}_wasserstein"] = w_distance

                # Alert flags
                alert_flags[f"{col}_psi_alert"] = psi_score > config.psi_threshold
                alert_flags[f"{col}_ks_alert"] = ks_p_value < config.ks_p_threshold
                alert_flags[f"{col}_wasserstein_alert"] = w_distance > config.wasserstein_threshold

    return {
        "drift_scores": drift_results,
        "alert_flags": alert_flags,
        "overall_drift_detected": any(alert_flags.values()),
    }


class MonitoringPipeline:
    """Production monitoring and maintenance pipeline for semiconductor manufacturing."""

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.model = None
        self.preprocessing_pipeline = None
        self.reference_data = None
        self.performance_history = []
        self.mlflow_enabled = False

    def enable_mlflow(self, experiment_name: str = "semiconductor_monitoring"):
        """Enable MLflow tracking for experiments."""
        if not HAS_MLFLOW:
            print("Warning: MLflow not available, skipping MLflow tracking")
            return False

        self.mlflow_enabled = True
        mlflow.set_experiment(experiment_name)
        return True

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, model_type: str = "ridge", alpha: float = 1.0
    ) -> "MonitoringPipeline":
        """Train model with monitoring setup."""
        if self.mlflow_enabled and HAS_MLFLOW:
            with mlflow.start_run():
                return self._fit_with_mlflow(X, y, model_type, alpha)
        else:
            return self._fit_without_mlflow(X, y, model_type, alpha)

    def _fit_with_mlflow(self, X: pd.DataFrame, y: np.ndarray, model_type: str, alpha: float) -> "MonitoringPipeline":
        """Internal fit method with MLflow tracking."""
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])

        # Store reference data for drift detection
        self.reference_data = X.copy()

        # Create preprocessing pipeline
        self.preprocessing_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        # Create model
        if model_type == "ridge":
            self.model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        elif model_type == "lasso":
            self.model = Lasso(alpha=alpha, random_state=RANDOM_SEED)
        elif model_type == "elastic_net":
            self.model = ElasticNet(alpha=alpha, random_state=RANDOM_SEED)
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Fit preprocessing and model
        X_processed = self.preprocessing_pipeline.fit_transform(X)
        self.model.fit(X_processed, y)

        # Calculate and log initial metrics
        y_pred = self.model.predict(X_processed)
        metrics = calculate_manufacturing_metrics(y, y_pred)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model
        mlflow.sklearn.log_model(self.model, "model")
        mlflow.sklearn.log_model(self.preprocessing_pipeline, "preprocessing")

        return self

    def _fit_without_mlflow(
        self, X: pd.DataFrame, y: np.ndarray, model_type: str, alpha: float
    ) -> "MonitoringPipeline":
        """Internal fit method without MLflow tracking."""
        # Store reference data for drift detection
        self.reference_data = X.copy()

        # Create preprocessing pipeline
        self.preprocessing_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        # Create model
        if model_type == "ridge":
            self.model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        elif model_type == "lasso":
            self.model = Lasso(alpha=alpha, random_state=RANDOM_SEED)
        elif model_type == "elastic_net":
            self.model = ElasticNet(alpha=alpha, random_state=RANDOM_SEED)
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Fit preprocessing and model
        X_processed = self.preprocessing_pipeline.fit_transform(X)
        self.model.fit(X_processed, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None or self.preprocessing_pipeline is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_processed = self.preprocessing_pipeline.transform(X)
        return self.model.predict(X_processed)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model with comprehensive monitoring metrics."""
        y_pred = self.predict(X)

        # Calculate basic metrics
        metrics = calculate_manufacturing_metrics(y, y_pred)

        # Detect drift if reference data available
        drift_info = {}
        if self.reference_data is not None:
            drift_info = detect_drift(self.reference_data, X, self.config)

        # Determine trend direction based on performance history
        trend_direction = self._analyze_trend(metrics["rmse"])

        # Create comprehensive monitoring metrics
        monitoring_metrics = MonitoringMetrics(
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            r2=metrics["r2"],
            pws_percent=metrics["pws_percent"],
            estimated_loss=metrics["estimated_loss"],
            drift_scores=drift_info.get("drift_scores", {}),
            alert_flags=drift_info.get("alert_flags", {}),
            trend_direction=trend_direction,
        )

        # Log to MLflow if enabled
        if self.mlflow_enabled and HAS_MLFLOW:
            with mlflow.start_run():
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log drift scores
                for score_name, score_value in drift_info.get("drift_scores", {}).items():
                    mlflow.log_metric(f"drift_{score_name}", score_value)

        # Store performance for trend analysis
        self.performance_history.append(metrics["rmse"])

        # Convert to dict and ensure JSON serializable
        result = asdict(monitoring_metrics)

        # Convert numpy types to Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj

        return convert_numpy_types(result)

    def _analyze_trend(self, current_rmse: float) -> str:
        """Analyze performance trend based on historical data."""
        if len(self.performance_history) < 3:
            return "insufficient_data"

        recent_performance = self.performance_history[-3:]
        if len(recent_performance) >= 3:
            # Simple trend analysis
            if all(recent_performance[i] <= recent_performance[i - 1] for i in range(1, len(recent_performance))):
                return "improving"
            elif all(recent_performance[i] >= recent_performance[i - 1] for i in range(1, len(recent_performance))):
                return "degrading"
            else:
                return "stable"
        return "stable"

    def save(self, path: Path) -> None:
        """Save the complete pipeline."""
        save_data = {
            "model": self.model,
            "preprocessing_pipeline": self.preprocessing_pipeline,
            "reference_data": self.reference_data,
            "config": self.config,
            "performance_history": self.performance_history,
        }
        joblib.dump(save_data, path)

    @staticmethod
    def load(path: Path) -> "MonitoringPipeline":
        """Load a saved pipeline."""
        save_data = joblib.load(path)
        pipeline = MonitoringPipeline(config=save_data["config"])
        pipeline.model = save_data["model"]
        pipeline.preprocessing_pipeline = save_data["preprocessing_pipeline"]
        pipeline.reference_data = save_data["reference_data"]
        pipeline.performance_history = save_data.get("performance_history", [])
        return pipeline


def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


def build_parser():
    parser = argparse.ArgumentParser(description="Module 9.2 Monitoring & Maintenance Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    p_train = subparsers.add_parser("train", help="Train a model with monitoring")
    p_train.add_argument(
        "--dataset", choices=["synthetic_yield"], default="synthetic_yield", help="Dataset to use for training"
    )
    p_train.add_argument(
        "--model",
        choices=["ridge", "lasso", "elastic_net", "random_forest"],
        default="ridge",
        help="Model type to train",
    )
    p_train.add_argument("--alpha", type=float, default=1.0, help="Regularization parameter")
    p_train.add_argument("--enable-mlflow", action="store_true", help="Enable MLflow tracking")
    p_train.add_argument("--save", type=str, help="Path to save trained model")
    p_train.add_argument("--inject-drift", action="store_true", help="Inject drift for testing")
    p_train.set_defaults(func=action_train)

    # Evaluate subcommand
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model with monitoring")
    p_eval.add_argument("--model-path", type=str, required=True, help="Path to saved model")
    p_eval.add_argument(
        "--dataset", choices=["synthetic_yield"], default="synthetic_yield", help="Dataset to use for evaluation"
    )
    p_eval.add_argument("--enable-mlflow", action="store_true", help="Enable MLflow tracking")
    p_eval.add_argument("--inject-drift", action="store_true", help="Inject drift for testing")
    p_eval.set_defaults(func=action_evaluate)

    # Predict subcommand
    p_pred = subparsers.add_parser("predict", help="Make predictions")
    p_pred.add_argument("--model-path", type=str, required=True, help="Path to saved model")
    p_pred.add_argument("--input-json", type=str, help="JSON input for single prediction")
    p_pred.add_argument("--input-file", type=str, help="CSV file for batch predictions")
    p_pred.set_defaults(func=action_predict)

    return parser


def action_train(args):
    """Train a model with monitoring capabilities."""
    try:
        # Generate data
        if args.dataset == "synthetic_yield":
            data = generate_yield_process(n=800, seed=RANDOM_SEED)
            if args.inject_drift:
                # Split data and inject drift in second half
                split_idx = len(data) // 2
                data_part1 = data.iloc[:split_idx]
                data_part2 = generate_drift_data(data.iloc[split_idx:], drift_strength=0.3)
                data = pd.concat([data_part1, data_part2], ignore_index=True)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        # Prepare features and target
        feature_cols = [col for col in data.columns if col != "target"]
        X = data[feature_cols]
        y = data["target"].values

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        # Create and train pipeline
        pipeline = MonitoringPipeline()

        if args.enable_mlflow:
            pipeline.enable_mlflow("semiconductor_monitoring")

        pipeline.fit(X_train, y_train, model_type=args.model, alpha=args.alpha)

        # Evaluate on test set
        eval_results = pipeline.evaluate(X_test, y_test)

        # Save model if requested
        if args.save:
            pipeline.save(Path(args.save))

        # Return results
        result = {
            "status": "trained",
            "model_type": args.model,
            "alpha": args.alpha,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "metrics": eval_results,
            "mlflow_enabled": pipeline.mlflow_enabled,
            "drift_injected": args.inject_drift,
        }

        print(json.dumps(convert_numpy_types(result), indent=2))

    except Exception as e:
        error_result = {"status": "error", "message": str(e)}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def action_evaluate(args):
    """Evaluate a saved model with monitoring."""
    try:
        # Load model
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {args.model_path}")

        pipeline = MonitoringPipeline.load(Path(args.model_path))

        if args.enable_mlflow:
            pipeline.enable_mlflow("semiconductor_monitoring")

        # Generate evaluation data
        if args.dataset == "synthetic_yield":
            data = generate_yield_process(n=400, seed=RANDOM_SEED + 1)
            if args.inject_drift:
                data = generate_drift_data(data, drift_strength=0.5)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        # Prepare features and target
        feature_cols = [col for col in data.columns if col != "target"]
        X = data[feature_cols]
        y = data["target"].values

        # Evaluate
        eval_results = pipeline.evaluate(X, y)

        result = {
            "status": "evaluated",
            "n_samples": len(X),
            "metrics": eval_results,
            "mlflow_enabled": pipeline.mlflow_enabled,
            "drift_injected": args.inject_drift,
        }

        print(json.dumps(convert_numpy_types(result), indent=2))

    except Exception as e:
        error_result = {"status": "error", "message": str(e)}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def action_predict(args):
    """Make predictions with a saved model."""
    try:
        # Load model
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {args.model_path}")

        pipeline = MonitoringPipeline.load(Path(args.model_path))

        if args.input_json:
            # Single prediction from JSON
            input_data = json.loads(args.input_json)
            df = pd.DataFrame([input_data])
            predictions = pipeline.predict(df)

            result = {
                "status": "predicted",
                "input": input_data,
                "prediction": float(predictions[0]),
                "n_predictions": 1,
            }

        elif args.input_file:
            # Batch predictions from CSV
            if not Path(args.input_file).exists():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")

            df = pd.read_csv(args.input_file)
            predictions = pipeline.predict(df)

            result = {"status": "predicted", "predictions": predictions.tolist(), "n_predictions": len(predictions)}

        else:
            raise ValueError("Either --input-json or --input-file must be provided")

        print(json.dumps(convert_numpy_types(result), indent=2))

    except Exception as e:
        error_result = {"status": "error", "message": str(e)}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
