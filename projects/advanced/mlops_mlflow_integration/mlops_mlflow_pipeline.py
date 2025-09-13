"""MLOps MLflow Integration Pipeline for Advanced Semiconductor Manufacturing

This pipeline demonstrates comprehensive MLflow tracking integration including:
- Experiment organization and run management
- Parameter, metric, and artifact logging
- Model registry integration with versioning
- Deployment status tracking and model lifecycle management
- Semiconductor-specific monitoring with manufacturing metrics

Key MLflow features demonstrated:
- Automatic experiment creation and run management
- Parameter and hyperparameter tracking
- Performance metrics logging with manufacturing context
- Model artifacts storage and versioning
- Custom tags for semiconductor process tracking
- Model registry for production deployment workflow

Example usage:
    python mlops_mlflow_pipeline.py train --dataset synthetic_yield --model ridge --enable-mlflow
    python mlops_mlflow_pipeline.py evaluate --model-path model.joblib --enable-mlflow
    python mlops_mlflow_pipeline.py predict --model-path model.joblib --input-json '{"temperature":455, "pressure":2.6}'
    python mlops_mlflow_pipeline.py start-tracking --experiment "fab_west_yield_prediction"
    python mlops_mlflow_pipeline.py stop-tracking
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tempfile
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Optional MLflow import with graceful fallback
HAS_MLFLOW = True
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
except ImportError:
    HAS_MLFLOW = False
    mlflow = None
    MlflowClient = None
    ViewType = None
    warnings.warn("MLflow not available. Install with: pip install mlflow")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Manufacturing tolerances for PWS calculation  
DEFAULT_TOLERANCE = 0.1  # 10% tolerance for predictions


@dataclass
class MLflowConfig:
    """Configuration for MLflow tracking."""
    
    experiment_name: str = "semiconductor_mlops_demo"
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    enable_autolog: bool = True
    log_model_signature: bool = True
    log_input_example: bool = True


@dataclass
class ManufacturingMetrics:
    """Manufacturing-specific metrics for semiconductor processes."""
    
    mae: float
    rmse: float
    r2: float
    pws_percent: float  # Prediction Within Spec percentage
    estimated_loss: float  # Estimated cost impact of prediction errors
    yield_rate: float  # Percentage of predictions above yield threshold


def check_mlflow_availability() -> bool:
    """Check if MLflow is available and warn if not."""
    if not HAS_MLFLOW:
        print("❌ MLflow not available. Install with: pip install mlflow")
        return False
    print("✅ MLflow is available")
    return True


def generate_semiconductor_data(n: int = 800, inject_drift: bool = False, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic semiconductor yield process data."""
    rng = np.random.default_rng(seed)
    
    # Base process parameters
    temp = rng.normal(450, 15, n)
    pressure = rng.normal(2.5, 0.3, n)
    flow = rng.normal(120, 10, n)
    time = rng.normal(60, 5, n)
    
    # Inject drift if requested (for monitoring examples)
    if inject_drift:
        drift_factor = np.linspace(1.0, 1.3, n)  # 30% drift over time
        temp = temp * drift_factor
        pressure = pressure * rng.uniform(0.9, 1.1, n)  # Random drift
    
    # Feature engineering
    temp_centered = temp - 450
    pressure_sq = pressure ** 2
    flow_time_inter = flow * time
    temp_flow_inter = temp * flow
    
    # Complex yield relationship with noise
    noise = rng.normal(0, 3, n)
    yield_base = (
        70
        + 0.3 * temp_centered
        + 5 * pressure
        - 0.1 * pressure_sq
        + 0.02 * flow
        + 0.01 * time
        + 0.001 * flow_time_inter
        + 0.0001 * temp_flow_inter
        + noise
    )
    
    # Ensure realistic yield range (0-100%)
    yield_pct = np.clip(yield_base, 0, 100)
    
    return pd.DataFrame({
        'temperature': temp,
        'pressure': pressure,
        'flow': flow,
        'time': time,
        'temp_centered': temp_centered,
        'pressure_sq': pressure_sq,
        'flow_time_inter': flow_time_inter,
        'temp_flow_inter': temp_flow_inter,
        'yield': yield_pct
    })


def calculate_manufacturing_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 tolerance: float = DEFAULT_TOLERANCE) -> Dict[str, float]:
    """Calculate semiconductor manufacturing metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Prediction Within Spec (PWS) - percentage within tolerance
    within_spec = np.abs(y_true - y_pred) <= (tolerance * y_true)
    pws_percent = np.mean(within_spec) * 100
    
    # Estimated loss (cost impact of prediction errors)
    cost_per_unit = 1.0  # Normalized cost
    loss_components = np.maximum(0, np.abs(y_true - y_pred) - tolerance * y_true)
    estimated_loss = np.sum(loss_components) * cost_per_unit
    
    # Yield rate (percentage above acceptable yield threshold)
    yield_threshold = 70.0  # 70% yield threshold
    yield_rate = np.mean(y_pred >= yield_threshold) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pws_percent': pws_percent,
        'estimated_loss': estimated_loss,
        'yield_rate': yield_rate
    }


class MLOpsMLflowPipeline:
    """Production MLOps pipeline with comprehensive MLflow integration."""
    
    def __init__(self, config: Optional[MLflowConfig] = None):
        self.config = config or MLflowConfig()
        self.model = None
        self.preprocessing_pipeline = None
        self.mlflow_enabled = False
        self.experiment_id = None
        self.run_id = None
        
    def enable_mlflow_tracking(self, experiment_name: Optional[str] = None, verbose: bool = True) -> bool:
        """Enable MLflow tracking with comprehensive setup."""
        if not HAS_MLFLOW:
            if verbose:
                print("Warning: MLflow not available, skipping MLflow tracking")
            return False
            
        try:
            # Set tracking URI if specified
            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Create or set experiment
            exp_name = experiment_name or self.config.experiment_name
            experiment = mlflow.set_experiment(exp_name)
            self.experiment_id = experiment.experiment_id
            
            # Enable autolog if configured
            if self.config.enable_autolog:
                mlflow.sklearn.autolog(
                    log_input_examples=self.config.log_input_example,
                    log_model_signatures=self.config.log_model_signature,
                    log_models=True
                )
            
            self.mlflow_enabled = True
            if verbose:
                print(f"✅ MLflow tracking enabled for experiment: {exp_name}")
                print(f"   Experiment ID: {self.experiment_id}")
                print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
            return True
            
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to enable MLflow tracking: {e}")
            return False
    
    def start_tracking_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None, verbose: bool = True) -> Optional[str]:
        """Start a new MLflow tracking run."""
        if not self.mlflow_enabled:
            return None
            
        try:
            # Start run with optional name and tags
            mlflow.start_run(run_name=run_name, tags=tags)
            self.run_id = mlflow.active_run().info.run_id
            if verbose:
                print(f"🚀 Started MLflow run: {self.run_id}")
            return self.run_id
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to start MLflow run: {e}")
            return None
    
    def stop_tracking_run(self, verbose: bool = True) -> None:
        """Stop the current MLflow tracking run."""
        if self.mlflow_enabled and mlflow.active_run():
            try:
                mlflow.end_run()
                if verbose:
                    print(f"🏁 Ended MLflow run: {self.run_id}")
                self.run_id = None
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to end MLflow run: {e}")
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, model_type: str = "ridge", 
            alpha: float = 1.0, run_name: Optional[str] = None, verbose: bool = True) -> "MLOpsMLflowPipeline":
        """Train model with comprehensive MLflow tracking."""
        
        # Start tracking run if MLflow enabled
        tags = {
            "model_type": model_type,
            "phase": "training",
            "fab_location": "demo_fab",
            "process_node": "7nm"
        }
        
        if self.mlflow_enabled:
            self.start_tracking_run(run_name=run_name, tags=tags, verbose=verbose)
        
        try:
            return self._fit_with_logging(X, y, model_type, alpha, verbose=verbose)
        finally:
            if self.mlflow_enabled:
                self.stop_tracking_run(verbose=verbose)
    
    def _fit_with_logging(self, X: pd.DataFrame, y: np.ndarray, model_type: str, alpha: float, verbose: bool = True) -> "MLOpsMLflowPipeline":
        """Internal fit method with comprehensive MLflow logging."""
        
        # Log parameters
        if self.mlflow_enabled:
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("tolerance", DEFAULT_TOLERANCE)
            
            # Log dataset info
            mlflow.log_param("feature_columns", list(X.columns))
            mlflow.log_param("target_column", "yield")
            
        # Create preprocessing pipeline
        self.preprocessing_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        # Create model based on type
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
        
        # Calculate training metrics
        y_pred = self.model.predict(X_processed)
        metrics = calculate_manufacturing_metrics(y, y_pred)
        
        # Log metrics to MLflow
        if self.mlflow_enabled:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                registered_model_name=f"semiconductor_yield_predictor_{model_type}"
            )
            
            # Create and log feature importance plot if supported
            if hasattr(self.model, 'feature_importances_'):
                self._log_feature_importance_plot(X.columns, self.model.feature_importances_)
            
            # Log preprocessing pipeline as artifact
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
                joblib.dump(self.preprocessing_pipeline, f.name)
                mlflow.log_artifact(f.name, "preprocessing")
                os.unlink(f.name)
        
        return self
    
    def _log_feature_importance_plot(self, feature_names, importances):
        """Create and log feature importance plot to MLflow."""
        try:
            import matplotlib.pyplot as plt
            
            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importances)[::-1]
            plt.title("Feature Importance")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            
            # Save and log to MLflow
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(f.name, "plots")
                os.unlink(f.name)
            plt.close()
            
        except ImportError:
            print("Warning: matplotlib not available, skipping feature importance plot")
        except Exception as e:
            print(f"Warning: Failed to create feature importance plot: {e}")
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray, run_name: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
        """Evaluate model with MLflow tracking."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Start evaluation run
        tags = {
            "phase": "evaluation",
            "model_type": type(self.model).__name__.lower()
        }
        
        if self.mlflow_enabled:
            self.start_tracking_run(run_name=run_name, tags=tags, verbose=verbose)
        
        try:
            return self._evaluate_with_logging(X, y, verbose=verbose)
        finally:
            if self.mlflow_enabled:
                self.stop_tracking_run(verbose=verbose)
    
    def _evaluate_with_logging(self, X: pd.DataFrame, y: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        """Internal evaluate method with MLflow logging."""
        # Preprocess and predict
        X_processed = self.preprocessing_pipeline.transform(X)
        y_pred = self.model.predict(X_processed)
        
        # Calculate metrics
        metrics = calculate_manufacturing_metrics(y, y_pred)
        
        # Log evaluation metrics
        if self.mlflow_enabled:
            mlflow.log_param("eval_n_samples", len(X))
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"eval_{metric_name}", metric_value)
            
            # Log prediction vs actual plot
            self._log_prediction_plot(y, y_pred)
        
        return {
            "status": "evaluated",
            "metrics": metrics,
            "n_samples": len(X),
            "mlflow_enabled": self.mlflow_enabled,
            "run_id": self.run_id
        }
    
    def _log_prediction_plot(self, y_true, y_pred):
        """Create and log prediction vs actual plot to MLflow."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual Yield (%)')
            plt.ylabel('Predicted Yield (%)')
            plt.title('Predicted vs Actual Yield')
            plt.grid(True, alpha=0.3)
            
            # Save and log to MLflow
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(f.name, "plots")
                os.unlink(f.name)
            plt.close()
            
        except ImportError:
            print("Warning: matplotlib not available, skipping prediction plot")
        except Exception as e:
            print(f"Warning: Failed to create prediction plot: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_processed = self.preprocessing_pipeline.transform(X)
        return self.model.predict(X_processed)
    
    def save(self, path: Path) -> None:
        """Save the complete pipeline."""
        save_data = {
            "model": self.model,
            "preprocessing_pipeline": self.preprocessing_pipeline,
            "config": self.config,
            "mlflow_enabled": self.mlflow_enabled
        }
        joblib.dump(save_data, path)
        
        # Log saved model path to MLflow if enabled
        if self.mlflow_enabled and mlflow.active_run():
            mlflow.log_param("model_save_path", str(path))
    
    @staticmethod
    def load(path: Path) -> "MLOpsMLflowPipeline":
        """Load a saved pipeline."""
        save_data = joblib.load(path)
        pipeline = MLOpsMLflowPipeline(config=save_data.get("config"))
        pipeline.model = save_data["model"]
        pipeline.preprocessing_pipeline = save_data["preprocessing_pipeline"]
        pipeline.mlflow_enabled = save_data.get("mlflow_enabled", False)
        return pipeline


def get_mlflow_experiments() -> Dict[str, Any]:
    """Get list of MLflow experiments."""
    if not HAS_MLFLOW:
        return {"error": "MLflow not available"}
    
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        
        experiment_list = []
        for exp in experiments:
            experiment_list.append({
                "id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location
            })
        
        return {
            "status": "success",
            "experiments": experiment_list,
            "total_count": len(experiment_list)
        }
    except Exception as e:
        return {"error": f"Failed to get experiments: {e}"}


def start_mlflow_server(port: int = 5000, backend_store_uri: str = "sqlite:///mlflow.db") -> Dict[str, Any]:
    """Start MLflow tracking server."""
    if not HAS_MLFLOW:
        return {"error": "MLflow not available"}
    
    try:
        import subprocess
        import threading
        
        # Create MLflow directories
        os.makedirs("mlruns", exist_ok=True)
        os.makedirs("mlflow_artifacts", exist_ok=True)
        
        # Start MLflow server in background
        cmd = [
            "mlflow", "server",
            "--backend-store-uri", backend_store_uri,
            "--default-artifact-root", "./mlflow_artifacts",
            "--host", "127.0.0.1",
            "--port", str(port)
        ]
        
        def run_server():
            subprocess.run(cmd)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        return {
            "status": "starting",
            "message": f"MLflow server starting on http://127.0.0.1:{port}",
            "backend_store_uri": backend_store_uri,
            "artifact_root": "./mlflow_artifacts"
        }
    except Exception as e:
        return {"error": f"Failed to start MLflow server: {e}"}


# CLI Command Handlers

def action_train(args) -> None:
    """Handle train command."""
    import os
    import warnings
    
    # Suppress MLflow warnings for clean JSON output
    warnings.filterwarnings("ignore")
    os.environ['MLFLOW_ENABLE_AUTOLOG_WARNINGS'] = 'False'
    
    # Generate or load data
    if args.dataset == "synthetic_yield":
        data = generate_semiconductor_data(n=800, inject_drift=args.inject_drift)
    else:
        print(f"Error: Unknown dataset '{args.dataset}'")
        sys.exit(1)
    
    # Prepare features and target
    feature_cols = [col for col in data.columns if col != 'yield']
    X = data[feature_cols]
    y = data['yield'].values
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Create and configure pipeline
    pipeline = MLOpsMLflowPipeline()
    
    # Enable MLflow if requested
    if args.enable_mlflow:
        experiment_name = args.experiment_name or "semiconductor_mlops_demo"
        pipeline.enable_mlflow_tracking(experiment_name, verbose=False)
    
    # Train model
    run_name = f"{args.model}_training_{args.dataset}"
    pipeline.fit(X_train, y_train, model_type=args.model, alpha=args.alpha, run_name=run_name, verbose=False)
    
    # Evaluate on test set
    eval_results = pipeline.evaluate(X_test, y_test, run_name=f"{args.model}_evaluation_{args.dataset}", verbose=False)
    
    # Save model if requested
    if args.save:
        pipeline.save(Path(args.save))
    
    # Prepare results
    results = {
        "status": "trained",
        "model_type": args.model,
        "dataset": args.dataset,
        "metrics": eval_results["metrics"],
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "mlflow_enabled": pipeline.mlflow_enabled,
        "run_id": pipeline.run_id,
        "drift_injected": args.inject_drift
    }
    
    print(json.dumps(results, indent=2))


def action_evaluate(args) -> None:
    """Handle evaluate command."""
    # Load model
    try:
        pipeline = MLOpsMLflowPipeline.load(Path(args.model_path))
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {e}"}))
        sys.exit(1)
    
    # Enable MLflow if requested
    if args.enable_mlflow:
        experiment_name = args.experiment_name or "semiconductor_mlops_demo"
        pipeline.enable_mlflow_tracking(experiment_name, verbose=False)
    
    # Generate or load evaluation data
    if args.dataset == "synthetic_yield":
        data = generate_semiconductor_data(n=400, inject_drift=args.inject_drift)
    else:
        print(json.dumps({"error": f"Unknown dataset '{args.dataset}'"}))
        sys.exit(1)
    
    # Prepare features and target
    feature_cols = [col for col in data.columns if col != 'yield']
    X = data[feature_cols]
    y = data['yield'].values
    
    # Evaluate
    run_name = f"evaluation_{args.dataset}"
    results = pipeline.evaluate(X, y, run_name=run_name, verbose=False)
    results["drift_injected"] = args.inject_drift
    
    print(json.dumps(results, indent=2))


def action_predict(args) -> None:
    """Handle predict command."""
    # Load model
    try:
        pipeline = MLOpsMLflowPipeline.load(Path(args.model_path))
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {e}"}))
        sys.exit(1)
    
    # Parse input data
    try:
        if args.input_json:
            input_data = json.loads(args.input_json)
            X = pd.DataFrame([input_data])
        elif args.input_csv:
            X = pd.read_csv(args.input_csv)
        else:
            print(json.dumps({"error": "Must provide either --input-json or --input-csv"}))
            sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse input data: {e}"}))
        sys.exit(1)
    
    # Make predictions
    try:
        predictions = pipeline.predict(X)
        results = {
            "status": "predicted",
            "predictions": predictions.tolist(),
            "n_samples": len(X)
        }
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(json.dumps({"error": f"Prediction failed: {e}"}))
        sys.exit(1)


def action_start_tracking(args) -> None:
    """Handle start-tracking command."""
    if not HAS_MLFLOW:
        print(json.dumps({"error": "MLflow not available"}))
        sys.exit(1)
    
    pipeline = MLOpsMLflowPipeline()
    success = pipeline.enable_mlflow_tracking(args.experiment, verbose=False)
    
    if success:
        print(json.dumps({
            "status": "tracking_enabled",
            "experiment_name": args.experiment,
            "experiment_id": pipeline.experiment_id,
            "tracking_uri": mlflow.get_tracking_uri()
        }))
    else:
        print(json.dumps({"error": "Failed to enable MLflow tracking"}))
        sys.exit(1)


def action_stop_tracking(args) -> None:
    """Handle stop-tracking command."""
    if HAS_MLFLOW and mlflow.active_run():
        mlflow.end_run()
        print(json.dumps({"status": "tracking_stopped"}))
    else:
        print(json.dumps({"status": "no_active_run"}))


def action_list_experiments(args) -> None:
    """Handle list-experiments command."""
    results = get_mlflow_experiments()
    print(json.dumps(results, indent=2))


def action_start_server(args) -> None:
    """Handle start-server command."""
    results = start_mlflow_server(port=args.port, backend_store_uri=args.backend_store_uri)
    print(json.dumps(results, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description='MLOps MLflow Integration Pipeline for Semiconductor Manufacturing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model with MLflow tracking
  python mlops_mlflow_pipeline.py train --dataset synthetic_yield --model ridge --enable-mlflow

  # Evaluate saved model  
  python mlops_mlflow_pipeline.py evaluate --model-path model.joblib --enable-mlflow

  # Make predictions
  python mlops_mlflow_pipeline.py predict --model-path model.joblib --input-json '{"temperature":455, "pressure":2.6, "flow":118, "time":62, "temp_centered":5, "pressure_sq":6.76, "flow_time_inter":7316, "temp_flow_inter":53690}'

  # Start/stop MLflow tracking
  python mlops_mlflow_pipeline.py start-tracking --experiment "fab_west_yield_prediction"
  python mlops_mlflow_pipeline.py stop-tracking

  # List experiments
  python mlops_mlflow_pipeline.py list-experiments

  # Start MLflow server
  python mlops_mlflow_pipeline.py start-server --port 5000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model with MLflow tracking')
    train_parser.add_argument('--dataset', choices=['synthetic_yield'], default='synthetic_yield',
                             help='Dataset to use for training')
    train_parser.add_argument('--model', choices=['ridge', 'lasso', 'elastic_net', 'random_forest'], 
                             default='ridge', help='Model type to train')
    train_parser.add_argument('--alpha', type=float, default=1.0, help='Regularization parameter')
    train_parser.add_argument('--enable-mlflow', action='store_true', help='Enable MLflow tracking')
    train_parser.add_argument('--experiment-name', help='MLflow experiment name')
    train_parser.add_argument('--inject-drift', action='store_true', help='Inject synthetic drift in data')
    train_parser.add_argument('--save', help='Path to save trained model')
    train_parser.set_defaults(func=action_train)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', required=True, help='Path to saved model')
    eval_parser.add_argument('--dataset', choices=['synthetic_yield'], default='synthetic_yield',
                            help='Dataset to use for evaluation')
    eval_parser.add_argument('--enable-mlflow', action='store_true', help='Enable MLflow tracking')
    eval_parser.add_argument('--experiment-name', help='MLflow experiment name')
    eval_parser.add_argument('--inject-drift', action='store_true', help='Inject synthetic drift in data')
    eval_parser.set_defaults(func=action_evaluate)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with trained model')
    predict_parser.add_argument('--model-path', required=True, help='Path to saved model')
    input_group = predict_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-json', help='JSON string with input data')
    input_group.add_argument('--input-csv', help='CSV file with input data')
    predict_parser.set_defaults(func=action_predict)
    
    # MLflow management commands
    start_tracking_parser = subparsers.add_parser('start-tracking', help='Start MLflow tracking')
    start_tracking_parser.add_argument('--experiment', default='semiconductor_mlops_demo',
                                      help='Experiment name')
    start_tracking_parser.set_defaults(func=action_start_tracking)
    
    stop_tracking_parser = subparsers.add_parser('stop-tracking', help='Stop MLflow tracking')
    stop_tracking_parser.set_defaults(func=action_stop_tracking)
    
    list_experiments_parser = subparsers.add_parser('list-experiments', help='List MLflow experiments')
    list_experiments_parser.set_defaults(func=action_list_experiments)
    
    start_server_parser = subparsers.add_parser('start-server', help='Start MLflow tracking server')
    start_server_parser.add_argument('--port', type=int, default=5000, help='Server port')
    start_server_parser.add_argument('--backend-store-uri', default='sqlite:///mlflow.db',
                                    help='Backend store URI')
    start_server_parser.set_defaults(func=action_start_server)
    
    return parser


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()