"""Production Scaling & Optimization Pipeline Script for Module 10.4

Provides a CLI to demonstrate and apply scaling and optimization techniques
for semiconductor manufacturing ML workloads.

Features:
- Vectorization demonstrations with NumPy/Pandas vs Python loops
- Parallel processing with joblib for batch operations
- Memory and time profiling utilities
- Caching mechanisms with joblib.Memory
- Incremental/partial fit patterns for large datasets
- Batch processing strategies
- Manufacturing-specific optimization examples

Example usage:
    python 10.4-scaling-optimization-pipeline.py train --strategy vectorized --parallel --cache-dir /tmp/cache
    python 10.4-scaling-optimization-pipeline.py evaluate --model-path model.joblib --batch-size 1000
    python 10.4-scaling-optimization-pipeline.py predict --model-path model.joblib --input-json '{"features":[1,2,3,4,5]}'
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from joblib import Parallel, delayed, Memory

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------- Synthetic Data Generators ---------------- #


def generate_wafer_process_data(n=5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic wafer processing data for optimization demonstrations."""
    rng = np.random.default_rng(seed)

    # Process parameters
    temperature = rng.normal(450, 15, n)
    pressure = rng.normal(2.5, 0.3, n)
    flow_rate = rng.normal(120, 10, n)
    time_duration = rng.normal(60, 5, n)
    chamber_id = rng.integers(1, 9, n)  # 8 chambers

    # Additional features for complexity
    humidity = rng.normal(45, 5, n)
    gas_concentration = rng.normal(0.85, 0.05, n)
    power_consumption = rng.normal(2000, 200, n)

    # Complex yield calculation with interactions
    base_yield = (
        70
        + 0.05 * (temperature - 450)
        - 1.5 * (pressure - 2.5) ** 2
        + 0.04 * flow_rate
        + 0.2 * time_duration
        + 0.0005 * (temperature - 450) * (flow_rate - 120)
        - 0.1 * (humidity - 45) ** 2
        + 10 * (gas_concentration - 0.85)
        - 0.001 * (power_consumption - 2000)
    )

    # Add chamber effects
    chamber_effects = np.array([0, -2, 1, -1, 2, 0, 1, -1])[chamber_id - 1]

    # Add noise
    noise = rng.normal(0, 3, n)
    yield_pct = np.clip(base_yield + chamber_effects + noise, 0, 100)

    df = pd.DataFrame(
        {
            "temperature": temperature,
            "pressure": pressure,
            "flow_rate": flow_rate,
            "time_duration": time_duration,
            "chamber_id": chamber_id,
            "humidity": humidity,
            "gas_concentration": gas_concentration,
            "power_consumption": power_consumption,
            "yield_pct": yield_pct,
        }
    )

    return df


# ---------------- Optimization Utilities ---------------- #


def vectorized_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized feature engineering using NumPy operations."""
    df_new = df.copy()

    # Vectorized operations
    df_new["temp_centered"] = df["temperature"] - df["temperature"].mean()
    df_new["pressure_sq"] = df["pressure"] ** 2
    df_new["flow_temp_inter"] = df["flow_rate"] * df["temperature"]
    df_new["power_efficiency"] = df["power_consumption"] / df["power_consumption"].max()
    df_new["normalized_time"] = df["time_duration"] / df["time_duration"].max()

    # Complex vectorized calculations
    df_new["stability_index"] = np.sqrt(
        (df["temperature"] - df["temperature"].mean()) ** 2 + (df["pressure"] - df["pressure"].mean()) ** 2
    )

    return df_new


def loop_based_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Non-vectorized feature engineering using Python loops (slow)."""
    df_new = df.copy()
    n = len(df)

    # Initialize new columns
    df_new["temp_centered"] = 0.0
    df_new["pressure_sq"] = 0.0
    df_new["flow_temp_inter"] = 0.0
    df_new["power_efficiency"] = 0.0
    df_new["normalized_time"] = 0.0
    df_new["stability_index"] = 0.0

    temp_mean = df["temperature"].mean()
    pressure_mean = df["pressure"].mean()
    time_max = df["time_duration"].max()
    power_max = df["power_consumption"].max()

    # Loop-based operations (inefficient)
    for i in range(n):
        df_new.iloc[i, df_new.columns.get_loc("temp_centered")] = df.iloc[i]["temperature"] - temp_mean
        df_new.iloc[i, df_new.columns.get_loc("pressure_sq")] = df.iloc[i]["pressure"] ** 2
        df_new.iloc[i, df_new.columns.get_loc("flow_temp_inter")] = df.iloc[i]["flow_rate"] * df.iloc[i]["temperature"]
        df_new.iloc[i, df_new.columns.get_loc("power_efficiency")] = df.iloc[i]["power_consumption"] / power_max
        df_new.iloc[i, df_new.columns.get_loc("normalized_time")] = df.iloc[i]["time_duration"] / time_max
        df_new.iloc[i, df_new.columns.get_loc("stability_index")] = np.sqrt(
            (df.iloc[i]["temperature"] - temp_mean) ** 2 + (df.iloc[i]["pressure"] - pressure_mean) ** 2
        )

    return df_new


def parallel_batch_processing(data: np.ndarray, batch_size: int = 1000, n_jobs: int = -1) -> np.ndarray:
    """Process data in parallel batches."""

    def process_batch(batch):
        # Simulate complex processing (feature transformations)
        return np.sqrt(np.abs(batch)) + np.log1p(np.abs(batch))

    # Split data into batches
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    # Process batches in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_batch)(batch) for batch in batches)

    return np.concatenate(results)


class TimingProfiler:
    """Simple timing profiler for performance measurement."""

    def __init__(self):
        self.times = {}

    def time_function(self, name: str, func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        self.times[name] = end_time - start_time
        return result

    def get_times(self) -> Dict[str, float]:
        return self.times.copy()


class MemoryProfiler:
    """Simple memory profiler for memory usage measurement."""

    def __init__(self):
        self.memory_usage = {}

    def profile_function(self, name: str, func, *args, **kwargs):
        """Profile memory usage of a function."""
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.memory_usage[name] = {"current_mb": current / 1024 / 1024, "peak_mb": peak / 1024 / 1024}
        return result

    def get_memory_usage(self) -> Dict[str, Dict[str, float]]:
        return self.memory_usage.copy()


# ---------------- Pipeline Wrapper ---------------- #


@dataclass
class PipelineMetadata:
    trained_at: str
    model_type: str
    optimization_strategy: str
    n_features_in: int
    training_time: float
    memory_usage: Dict[str, Any]
    params: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None


class ScalingOptimizationPipeline:
    def __init__(
        self,
        strategy: str = "vectorized",
        use_parallel: bool = False,
        batch_size: int = 1000,
        n_jobs: int = -1,
        cache_dir: Optional[str] = None,
        use_incremental: bool = False,
    ) -> None:
        self.strategy = strategy
        self.use_parallel = use_parallel
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.cache_dir = cache_dir
        self.use_incremental = use_incremental

        # Initialize components
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Any] = None
        self.metadata: Optional[PipelineMetadata] = None
        self.timing_profiler = TimingProfiler()
        self.memory_profiler = MemoryProfiler()

        # Setup caching if specified
        self.memory_cache = None
        if cache_dir:
            self.memory_cache = Memory(cache_dir, verbose=0)

    def _get_cached_or_compute(self, func, *args, **kwargs):
        """Use cached computation if available, otherwise compute."""
        if self.memory_cache:
            cached_func = self.memory_cache.cache(func)
            return cached_func(*args, **kwargs)
        return func(*args, **kwargs)

    def _build_model(self):
        """Build model based on strategy."""
        if self.use_incremental:
            # Use SGD for incremental learning
            return SGDRegressor(random_state=RANDOM_SEED, max_iter=1000)
        else:
            # Use RandomForest for batch learning
            return RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=self.n_jobs if self.use_parallel else 1
            )

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with chosen optimization strategy."""
        if self.strategy == "vectorized":
            return self._get_cached_or_compute(vectorized_feature_engineering, df)
        elif self.strategy == "loops":
            return self._get_cached_or_compute(loop_based_feature_engineering, df)
        else:
            return df

    def fit(self, X: pd.DataFrame, y: Any):
        """Fit the pipeline with optimization demonstrations."""
        start_time = time.perf_counter()

        # Preprocess with profiling
        X_processed = self.timing_profiler.time_function("preprocessing", self._preprocess_data, X)
        X_processed = self.memory_profiler.profile_function("preprocessing", lambda x: x, X_processed)

        # Prepare features and target
        feature_cols = [col for col in X_processed.columns if col != "yield_pct"]
        X_features = X_processed[feature_cols]
        y_array = np.asarray(y)

        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.timing_profiler.time_function("scaling", self.scaler.fit_transform, X_features)

        # Build and train model
        self.model = self._build_model()

        if self.use_incremental and hasattr(self.model, "partial_fit"):
            # Incremental training in batches
            for i in range(0, len(X_scaled), self.batch_size):
                end_idx = min(i + self.batch_size, len(X_scaled))
                X_batch = X_scaled[i:end_idx]
                y_batch = y_array[i:end_idx]

                if i == 0:
                    self.model.partial_fit(X_batch, y_batch)
                else:
                    self.model.partial_fit(X_batch, y_batch)
        else:
            # Standard batch training
            self.timing_profiler.time_function("training", self.model.fit, X_scaled, y_array)

        # Store metadata
        training_time = time.perf_counter() - start_time
        self.metadata = PipelineMetadata(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            model_type=type(self.model).__name__,
            optimization_strategy=self.strategy,
            n_features_in=X_features.shape[1],
            training_time=training_time,
            memory_usage=self.memory_profiler.get_memory_usage(),
            params={
                "strategy": self.strategy,
                "use_parallel": self.use_parallel,
                "batch_size": self.batch_size,
                "n_jobs": self.n_jobs,
                "cache_dir": self.cache_dir,
                "use_incremental": self.use_incremental,
            },
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with optimization."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Pipeline not fitted")

        # Preprocess
        X_processed = self._preprocess_data(X)
        feature_cols = [col for col in X_processed.columns if col != "yield_pct"]
        X_features = X_processed[feature_cols]

        # Scale
        X_scaled = self.scaler.transform(X_features)

        # Predict in batches if needed
        if len(X_scaled) > self.batch_size and self.use_parallel:
            batches = [X_scaled[i : i + self.batch_size] for i in range(0, len(X_scaled), self.batch_size)]
            predictions = Parallel(n_jobs=self.n_jobs)(delayed(self.model.predict)(batch) for batch in batches)
            return np.concatenate(predictions)
        else:
            return self.model.predict(X_scaled)

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tolerance: float = 2.0,
        spec_low: float = 60,
        spec_high: float = 100,
        cost_per_unit: float = 1.0,
    ) -> Dict[str, float]:
        """Compute manufacturing metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        pws = ((y_pred >= spec_low) & (y_pred <= spec_high)).mean()
        loss_components = np.maximum(0, np.abs(y_true - y_pred) - tolerance)
        est_loss = float(np.sum(loss_components) * cost_per_unit)
        return {"MAE": mae, "RMSE": rmse, "R2": r2, "PWS": pws, "Estimated_Loss": est_loss}

    def evaluate(self, X: pd.DataFrame, y: Any) -> Dict[str, float]:
        """Evaluate pipeline performance."""
        y_array = np.asarray(y)
        preds = self.predict(X)
        metrics = self.compute_metrics(y_array, preds)

        # Add timing and memory metrics
        timing_metrics = {f"time_{k}": v for k, v in self.timing_profiler.get_times().items()}
        metrics.update(timing_metrics)

        if self.metadata:
            self.metadata.metrics = metrics
        return metrics

    def save(self, path: Path):
        """Save pipeline with all optimization metadata."""
        if self.model is None or self.metadata is None:
            raise RuntimeError("Nothing to save; fit the pipeline first")

        save_data = {
            "model": self.model,
            "scaler": self.scaler,
            "metadata": asdict(self.metadata),
            "timing_profile": self.timing_profiler.get_times(),
            "memory_profile": self.memory_profiler.get_memory_usage(),
        }
        joblib.dump(save_data, path)

    @staticmethod
    def load(path: Path) -> "ScalingOptimizationPipeline":
        """Load pipeline with optimization metadata."""
        data = joblib.load(path)

        metadata = data["metadata"]
        params = metadata["params"]

        pipeline = ScalingOptimizationPipeline(
            strategy=params["strategy"],
            use_parallel=params["use_parallel"],
            batch_size=params["batch_size"],
            n_jobs=params["n_jobs"],
            cache_dir=params["cache_dir"],
            use_incremental=params["use_incremental"],
        )

        pipeline.model = data["model"]
        pipeline.scaler = data["scaler"]
        pipeline.metadata = PipelineMetadata(**metadata)

        return pipeline


# ---------------- Data Loading ---------------- #


def load_dataset(name: str, size: int = 5000) -> pd.DataFrame:
    """Load dataset for optimization demonstrations."""
    if name == "wafer_process":
        return generate_wafer_process_data(n=size)
    raise ValueError(f"Unknown dataset '{name}'. Currently supported: wafer_process")


TARGET_COLUMN = "yield_pct"

# ---------------- CLI Actions ---------------- #


def action_train(args):
    """Train pipeline with optimization strategies."""
    df = load_dataset(args.dataset, args.size)
    y = df[TARGET_COLUMN].values
    X = df.drop(columns=[TARGET_COLUMN])

    pipeline = ScalingOptimizationPipeline(
        strategy=args.strategy,
        use_parallel=args.parallel,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        cache_dir=args.cache_dir,
        use_incremental=args.incremental,
    )

    pipeline.fit(X, y)
    metrics = pipeline.evaluate(X, y)

    if args.save:
        pipeline.save(Path(args.save))

    meta_dict = asdict(pipeline.metadata) if pipeline.metadata else None
    print(
        json.dumps(
            {
                "status": "trained",
                "metrics": metrics,
                "metadata": meta_dict,
                "optimization_summary": {
                    "strategy": args.strategy,
                    "parallel": args.parallel,
                    "batch_size": args.batch_size,
                    "cache_enabled": args.cache_dir is not None,
                },
            },
            indent=2,
        )
    )


def action_evaluate(args):
    """Evaluate existing pipeline."""
    pipeline = ScalingOptimizationPipeline.load(Path(args.model_path))
    df = load_dataset(args.dataset, args.size)
    y = df[TARGET_COLUMN].values
    X = df.drop(columns=[TARGET_COLUMN])

    metrics = pipeline.evaluate(X, y)
    meta_dict = asdict(pipeline.metadata) if pipeline.metadata else None

    print(json.dumps({"status": "evaluated", "metrics": metrics, "metadata": meta_dict}, indent=2))


def action_predict(args):
    """Make predictions with optimization."""
    pipeline = ScalingOptimizationPipeline.load(Path(args.model_path))

    if args.input_json:
        record = json.loads(args.input_json)
        df = pd.DataFrame([record])
    elif args.input_file:
        df = pd.read_json(args.input_file)
    else:
        raise ValueError("Provide --input-json or --input-file")

    pred = pipeline.predict(df)
    meta_dict = asdict(pipeline.metadata) if pipeline.metadata else None

    result = {"predictions": pred.tolist(), "input_shape": df.shape, "model_meta": meta_dict}
    print(json.dumps(result, indent=2))


def action_benchmark(args):
    """Benchmark vectorized vs loop-based operations."""
    df = load_dataset("wafer_process", args.size)

    profiler = TimingProfiler()
    memory_profiler = MemoryProfiler()

    # Benchmark vectorized approach
    vec_result = profiler.time_function("vectorized", vectorized_feature_engineering, df)
    vec_result = memory_profiler.profile_function("vectorized", lambda x: x, vec_result)

    # Benchmark loop-based approach (only for smaller datasets)
    if args.size <= 1000:
        loop_result = profiler.time_function("loops", loop_based_feature_engineering, df)
        loop_result = memory_profiler.profile_function("loops", lambda x: x, loop_result)

    results = {
        "dataset_size": args.size,
        "timing": profiler.get_times(),
        "memory_usage": memory_profiler.get_memory_usage(),
        "speedup": profiler.get_times().get("loops", 0) / profiler.get_times().get("vectorized", 1)
        if "loops" in profiler.get_times()
        else None,
    }

    print(json.dumps(results, indent=2))


# ---------------- Argument Parsing ---------------- #


def build_parser():
    parser = argparse.ArgumentParser(description="Module 10.4 Scaling & Optimization Pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    p_train = sub.add_parser("train", help="Train optimization pipeline")
    p_train.add_argument("--dataset", default="wafer_process", help="Dataset name")
    p_train.add_argument("--size", type=int, default=5000, help="Dataset size")
    p_train.add_argument(
        "--strategy", default="vectorized", choices=["vectorized", "loops"], help="Optimization strategy"
    )
    p_train.add_argument("--parallel", action="store_true", help="Use parallel processing")
    p_train.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    p_train.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")
    p_train.add_argument("--cache-dir", help="Directory for caching computations")
    p_train.add_argument("--incremental", action="store_true", help="Use incremental learning")
    p_train.add_argument("--save", help="Path to save trained model")
    p_train.set_defaults(func=action_train)

    # Evaluate subcommand
    p_eval = sub.add_parser("evaluate", help="Evaluate existing pipeline")
    p_eval.add_argument("--model-path", required=True, help="Path to saved model")
    p_eval.add_argument("--dataset", default="wafer_process", help="Dataset name")
    p_eval.add_argument("--size", type=int, default=5000, help="Dataset size")
    p_eval.set_defaults(func=action_evaluate)

    # Predict subcommand
    p_pred = sub.add_parser("predict", help="Make predictions")
    p_pred.add_argument("--model-path", required=True, help="Path to saved model")
    p_pred.add_argument("--input-json", help="JSON string with input data")
    p_pred.add_argument("--input-file", help="Path to JSON file with input data")
    p_pred.set_defaults(func=action_predict)

    # Benchmark subcommand
    p_bench = sub.add_parser("benchmark", help="Benchmark optimization strategies")
    p_bench.add_argument("--size", type=int, default=1000, help="Dataset size for benchmark")
    p_bench.set_defaults(func=action_benchmark)

    return parser


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
