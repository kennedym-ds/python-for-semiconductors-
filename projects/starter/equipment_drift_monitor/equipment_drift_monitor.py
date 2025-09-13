"""Production Equipment Drift Monitor Pipeline Script

Provides a CLI to train, evaluate, and predict using time series models
for semiconductor equipment drift detection and forecasting.

Features:
- Sliding window feature extraction for drift detection
- ARIMA/SARIMA modeling with anomaly detection
- Statistical process control with configurable thresholds
- Manufacturing metrics: MAE, MAPE, Anomaly Rate, PWS, Estimated Loss
- Model persistence with comprehensive metadata
- Synthetic equipment drift data generation

Example usage:
    python equipment_drift_monitor.py train --data synthetic_equipment \\
        --window-size 24 --horizon 12 --save drift_model.joblib
    python equipment_drift_monitor.py evaluate --model-path drift_model.joblib \\
        --tolerance 2.0 --cost-per-unit 1.5
    python equipment_drift_monitor.py predict --model-path drift_model.joblib \\
        --horizon 24 --output forecasts.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Core time series dependencies
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
TARGET_COLUMN = "target"

# Optional dependencies with graceful fallback
try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False
    pm = None  # type: ignore

# ---------------- Synthetic Data Generators ---------------- #

def generate_equipment_drift_data(
    n_periods: int = 500, 
    freq: str = "h", 
    seed: int = RANDOM_SEED,
    include_drift: bool = True,
    include_failures: bool = True
) -> pd.DataFrame:
    """Generate synthetic semiconductor equipment time series with drift patterns.
    
    Simulates realistic equipment behavior including:
    - Normal operation with seasonal patterns
    - Gradual drift in process parameters
    - Sudden shifts from equipment adjustments
    - Failure events requiring maintenance
    - Multiple correlated sensors
    """
    rng = np.random.default_rng(seed)
    
    # Create time index
    start_date = pd.Timestamp("2023-01-01")
    time_index = pd.date_range(start=start_date, periods=n_periods, freq=freq)
    
    # Generate base signals
    t = np.arange(n_periods)
    
    # Chamber temperature with drift and seasonality
    daily_cycle = 3 * np.sin(2 * np.pi * t / 24)  # 24-hour cycle
    weekly_cycle = 1 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly maintenance cycle
    
    if include_drift:
        # Gradual drift over time
        drift = 0.01 * t + 0.0005 * t**2 / n_periods  # Accelerating drift
        # Sudden shifts (equipment adjustments)
        shift_points = rng.choice(n_periods, size=n_periods//100, replace=False)
        shifts = np.zeros(n_periods)
        for shift_point in shift_points:
            shift_magnitude = rng.normal(0, 2)
            shifts[shift_point:] += shift_magnitude
    else:
        drift = np.zeros(n_periods)
        shifts = np.zeros(n_periods)
    
    temp_noise = rng.normal(0, 0.5, n_periods)
    temperature = 450 + daily_cycle + weekly_cycle + drift + shifts + temp_noise
    
    # Pressure with random walk and correlation to temperature
    pressure_walk = np.cumsum(rng.normal(0, 0.02, n_periods))
    pressure_temp_correlation = 0.3 * (temperature - 450) / 10
    pressure_noise = rng.normal(0, 0.05, n_periods)
    pressure = 2.5 + pressure_walk + pressure_temp_correlation + pressure_noise
    
    # Flow rate with equipment degradation
    flow_baseline = 120
    if include_drift:
        flow_degradation = -0.005 * t  # Gradual decrease due to wear
    else:
        flow_degradation = np.zeros(n_periods)
    
    flow_noise = rng.normal(0, 1, n_periods)
    flow_rate = flow_baseline + flow_degradation + flow_noise
    
    # Power consumption (derived parameter)
    power_base = 1500
    power_temp_effect = 2 * (temperature - 450)
    power_flow_effect = 0.5 * (flow_rate - 120)
    power_noise = rng.normal(0, 20, n_periods)
    power = power_base + power_temp_effect + power_flow_effect + power_noise
    
    # Target: equipment health score (0-100)
    # Based on deviations from nominal conditions
    temp_penalty = np.abs(temperature - 450) * 0.1
    pressure_penalty = np.abs(pressure - 2.5) * 5
    flow_penalty = np.abs(flow_rate - 120) * 0.05
    
    health_score = 100 - temp_penalty - pressure_penalty - flow_penalty
    
    # Add failure events
    if include_failures:
        failure_periods = rng.choice(n_periods, size=n_periods//200, replace=False)
        for failure_start in failure_periods:
            failure_duration = rng.integers(4, 24)  # 4-24 hour failures
            failure_end = min(failure_start + failure_duration, n_periods)
            health_score[failure_start:failure_end] *= rng.uniform(0.3, 0.7)
    
    # Add target noise and ensure valid range
    target_noise = rng.normal(0, 2, n_periods)
    target = np.clip(health_score + target_noise, 0, 100)
    
    return pd.DataFrame({
        "timestamp": time_index,
        "temperature": temperature,
        "pressure": pressure,
        "flow_rate": flow_rate,
        "power": power,
        "target": target,
    }).set_index("timestamp")


# ---------------- Feature Engineering ---------------- #

def extract_sliding_window_features(
    df: pd.DataFrame, 
    window_size: int = 24, 
    target_col: str = TARGET_COLUMN
) -> pd.DataFrame:
    """Extract sliding window features for drift detection."""
    features = df.copy()
    
    # Rolling statistics for each numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == target_col:
            continue
            
        # Rolling mean and std
        features[f"{col}_rolling_mean_{window_size}h"] = df[col].rolling(window_size).mean()
        features[f"{col}_rolling_std_{window_size}h"] = df[col].rolling(window_size).std()
        
        # Rolling min/max
        features[f"{col}_rolling_min_{window_size}h"] = df[col].rolling(window_size).min()
        features[f"{col}_rolling_max_{window_size}h"] = df[col].rolling(window_size).max()
        
        # Trend (slope of linear regression over window)
        def rolling_slope(series):
            if len(series) < 2:
                return np.nan
            x = np.arange(len(series))
            return np.polyfit(x, series, 1)[0]
        
        features[f"{col}_trend_{window_size}h"] = df[col].rolling(window_size).apply(
            rolling_slope, raw=False
        )
        
        # Lag features
        for lag in [1, 6, 12, 24]:
            if lag < len(df):
                features[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    
    # Cross-correlation features
    if "temperature" in df.columns and "pressure" in df.columns:
        features["temp_pressure_ratio"] = df["temperature"] / (df["pressure"] + 1e-8)
    
    if "power" in df.columns and "flow_rate" in df.columns:
        features["power_efficiency"] = df["power"] / (df["flow_rate"] + 1e-8)
    
    return features


# ---------------- Equipment Drift Pipeline Class ---------------- #

@dataclass
class DriftMonitorMetadata:
    """Metadata for equipment drift monitor pipeline."""
    model_type: str
    window_size: int
    horizon: int
    order: Optional[Tuple[int, int, int]] = None
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
    drift_threshold: float = 2.0
    confidence_level: float = 0.95
    feature_columns: Optional[List[str]] = None
    training_periods: int = 0
    metrics: Optional[Dict[str, float]] = None


class EquipmentDriftMonitor:
    """Equipment Drift Monitor Pipeline for semiconductor manufacturing."""
    
    def __init__(
        self,
        window_size: int = 24,
        horizon: int = 12,
        drift_threshold: float = 2.0,
        confidence_level: float = 0.95,
        model_type: str = "arima",
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto_arima: bool = True,
    ) -> None:
        """Initialize equipment drift monitor.
        
        Args:
            window_size: Hours for sliding window feature extraction
            horizon: Forecast horizon in hours
            drift_threshold: Standard deviations for anomaly detection
            confidence_level: Confidence level for prediction intervals
            model_type: Type of model ('arima', 'sarima', 'auto_arima')
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            auto_arima: Use pmdarima auto_arima if available
        """
        self.window_size = window_size
        self.horizon = horizon
        self.drift_threshold = drift_threshold
        self.confidence_level = confidence_level
        self.model_type = model_type
        self.order = order or (1, 1, 1)
        self.seasonal_order = seasonal_order
        self.auto_arima = auto_arima and HAS_PMDARIMA
        
        # Fitted attributes
        self.model = None
        self.fitted_model = None
        self.metadata: Optional[DriftMonitorMetadata] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self._last_obs_index = None
        self._training_stats = None
    
    def _check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Perform ADF test for stationarity."""
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "critical_values": dict(result[4]),
            "is_stationary": result[1] < 0.05,
        }
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.Series, pd.DataFrame]:
        """Prepare time series data with feature engineering."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        # Extract features
        features_df = extract_sliding_window_features(df, self.window_size, target_col)
        
        # Extract target series
        y = features_df[target_col].copy()
        
        # Select feature columns (exclude target and original raw columns)
        feature_cols = [col for col in features_df.columns 
                       if col != target_col and 
                       ("rolling" in col or "trend" in col or "lag" in col or 
                        "ratio" in col or "efficiency" in col)]
        
        if feature_cols:
            X = features_df[feature_cols].copy()
            # Drop rows with NaN (from rolling windows)
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]
            
            # Scale features
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
                X_scaled = self.feature_scaler.fit_transform(X)
            else:
                X_scaled = self.feature_scaler.transform(X)
                
            X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        else:
            X = pd.DataFrame(index=y.index)
        
        return y, X
    
    def fit(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> "EquipmentDriftMonitor":
        """Fit the drift monitoring model."""
        y, X = self._prepare_data(df, target_col)
        
        # Store last observation index for forecasting
        self._last_obs_index = y.index[-1]
        
        # Store training statistics for anomaly detection
        self._training_stats = {
            "mean": y.mean(),
            "std": y.std(),
            "min": y.min(),
            "max": y.max(),
        }
        
        # Check stationarity
        stationarity = self._check_stationarity(y)
        if not stationarity["is_stationary"]:
            warnings.warn(
                f"Series may not be stationary (ADF p-value: {stationarity['p_value']:.4f})"
            )
        
        try:
            # Use exogenous features if available
            exog = X if len(X.columns) > 0 else None
            
            if self.auto_arima and HAS_PMDARIMA:
                # Use pmdarima auto_arima
                self.model = pm.auto_arima(
                    y,
                    exogenous=exog,
                    seasonal=self.seasonal_order is not None,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                )
                fitted_order = self.model.order
                fitted_seasonal = getattr(self.model, "seasonal_order", None)
            else:
                # Use statsmodels ARIMA/SARIMAX
                if self.seasonal_order:
                    self.model = SARIMAX(
                        y,
                        exog=exog,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                    )
                else:
                    self.model = ARIMA(
                        y,
                        exog=exog,
                        order=self.order,
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                    )
                fitted_order = self.order
                fitted_seasonal = self.seasonal_order
            
            self.fitted_model = self.model.fit()
            
            # Store metadata
            self.metadata = DriftMonitorMetadata(
                model_type=self.model_type,
                window_size=self.window_size,
                horizon=self.horizon,
                order=fitted_order,
                seasonal_order=fitted_seasonal,
                drift_threshold=self.drift_threshold,
                confidence_level=self.confidence_level,
                feature_columns=list(X.columns) if len(X.columns) > 0 else None,
                training_periods=len(y),
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to fit drift monitor model: {str(e)}")
        
        return self
    
    def predict(
        self,
        df: Optional[pd.DataFrame] = None,
        horizon: Optional[int] = None,
        return_conf_int: bool = True,
    ) -> Dict[str, Any]:
        """Generate forecasts and anomaly detection."""
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        forecast_horizon = horizon or self.horizon
        
        try:
            # Prepare exogenous data if needed
            exog_fc = None
            if self.metadata.feature_columns:
                if df is not None:
                    _, X = self._prepare_data(df, TARGET_COLUMN)
                    if len(X.columns) > 0 and len(X) > 0:
                        # Use last available values repeated for forecast horizon
                        last_values = X.iloc[-1:].values
                        exog_fc = np.repeat(last_values, forecast_horizon, axis=0)
                        exog_fc = pd.DataFrame(
                            exog_fc, 
                            columns=X.columns,
                            index=pd.date_range(
                                start=self._last_obs_index, 
                                periods=forecast_horizon + 1, 
                                freq="h"
                            )[1:]
                        )
                    else:
                        # No valid data, use zeros for exogenous features
                        exog_fc = pd.DataFrame(
                            np.zeros((forecast_horizon, len(self.metadata.feature_columns))),
                            columns=self.metadata.feature_columns,
                            index=pd.date_range(
                                start=self._last_obs_index, 
                                periods=forecast_horizon + 1, 
                                freq="h"
                            )[1:]
                        )
                else:
                    # No data provided, use zeros for exogenous features
                    exog_fc = pd.DataFrame(
                        np.zeros((forecast_horizon, len(self.metadata.feature_columns))),
                        columns=self.metadata.feature_columns,
                        index=pd.date_range(
                            start=self._last_obs_index, 
                            periods=forecast_horizon + 1, 
                            freq="h"
                        )[1:]
                    )
            
            # Generate forecast
            if self.auto_arima and HAS_PMDARIMA:
                forecasts, conf_int = self.fitted_model.predict(
                    n_periods=forecast_horizon,
                    exogenous=exog_fc,
                    return_conf_int=return_conf_int,
                )
                
                # Create proper index for forecasts
                freq = pd.infer_freq(self.fitted_model.arima_res_.data.dates)
                forecast_index = pd.date_range(
                    start=self._last_obs_index, 
                    periods=forecast_horizon + 1, 
                    freq=freq or "h"
                )[1:]
                
                forecasts = pd.Series(forecasts, index=forecast_index)
                if return_conf_int:
                    conf_int = pd.DataFrame(
                        conf_int, index=forecast_index, columns=["lower", "upper"]
                    )
            else:
                # statsmodels forecast
                forecast_result = self.fitted_model.forecast(
                    steps=forecast_horizon, exog=exog_fc
                )
                forecasts = forecast_result
                
                if return_conf_int:
                    conf_int_result = self.fitted_model.get_forecast(
                        steps=forecast_horizon, exog=exog_fc
                    ).conf_int()
                    conf_int = conf_int_result
                else:
                    conf_int = None
            
            # Anomaly detection
            anomaly_flags = []
            if self._training_stats:
                threshold = self._training_stats["std"] * self.drift_threshold
                for forecast in forecasts:
                    deviation = abs(forecast - self._training_stats["mean"])
                    anomaly_flags.append(bool(deviation > threshold))
            
            result = {
                "forecasts": (
                    forecasts.tolist() if isinstance(forecasts, pd.Series) else 
                    [float(x) for x in forecasts]
                ),
                "forecast_index": (
                    [str(ts) for ts in forecasts.index]
                    if isinstance(forecasts, pd.Series)
                    else None
                ),
                "anomaly_flags": anomaly_flags,
            }
            
            if return_conf_int and conf_int is not None:
                result["confidence_intervals"] = {
                    "lower": (
                        [float(x) for x in conf_int.iloc[:, 0].tolist()]
                        if hasattr(conf_int, "iloc")
                        else [float(x) for x in conf_int[:, 0].tolist()]
                    ),
                    "upper": (
                        [float(x) for x in conf_int.iloc[:, 1].tolist()]
                        if hasattr(conf_int, "iloc")
                        else [float(x) for x in conf_int[:, 1].tolist()]
                    ),
                }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
        test_size: int = 48,  # 48 hours for equipment monitoring
        tolerance: float = 2.0,
        cost_per_unit: float = 1.0,
    ) -> Dict[str, float]:
        """Evaluate model using time series cross-validation."""
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before evaluation")
        
        y, X = self._prepare_data(df, target_col)
        
        if len(y) < test_size + self.window_size + 10:
            raise ValueError("Insufficient data for evaluation")
        
        # Split data maintaining temporal order
        train_data = y.iloc[:-test_size]
        test_data = y.iloc[-test_size:]
        
        # Fit temporary model on training data
        temp_monitor = EquipmentDriftMonitor(
            window_size=self.window_size,
            horizon=self.horizon,
            drift_threshold=self.drift_threshold,
            confidence_level=self.confidence_level,
            model_type=self.model_type,
            order=self.order,
            seasonal_order=self.seasonal_order,
            auto_arima=self.auto_arima,
        )
        
        # Create temporary training dataframe
        train_df = df.iloc[:-test_size].copy()
        temp_monitor.fit(train_df, target_col)
        
        # Generate step-ahead predictions
        predictions = []
        confidence_intervals = {"lower": [], "upper": []}
        
        for i in range(len(test_data)):
            # Get data up to current point
            current_df = df.iloc[:-(test_size-i)].copy()
            
            pred_result = temp_monitor.predict(
                df=current_df, horizon=1, return_conf_int=True
            )
            predictions.append(pred_result["forecasts"][0])
            if "confidence_intervals" in pred_result:
                confidence_intervals["lower"].append(pred_result["confidence_intervals"]["lower"][0])
                confidence_intervals["upper"].append(pred_result["confidence_intervals"]["upper"][0])
        
        predictions = np.array(predictions)
        y_true = test_data.values
        
        # Calculate standard metrics
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        r2 = r2_score(y_true, predictions)
        
        # MAPE (handle zero values)
        mape = (
            np.mean(np.abs((y_true - predictions) / np.maximum(np.abs(y_true), 1e-8)))
            * 100
        )
        
        # Manufacturing-specific metrics
        # PWS: Prediction Within Spec
        within_spec = np.abs(predictions - y_true) <= tolerance
        pws = np.mean(within_spec) * 100
        
        # Estimated Loss
        errors = np.abs(predictions - y_true)
        estimated_loss = np.sum(errors * cost_per_unit)
        
        # Anomaly detection metrics
        if self._training_stats:
            threshold = self._training_stats["std"] * self.drift_threshold
            predicted_anomalies = np.abs(predictions - self._training_stats["mean"]) > threshold
            actual_anomalies = np.abs(y_true - self._training_stats["mean"]) > threshold
            
            anomaly_rate = np.mean(predicted_anomalies) * 100
            
            # Calculate anomaly detection accuracy if there are any anomalies
            if np.sum(actual_anomalies) > 0:
                from sklearn.metrics import precision_score, recall_score, f1_score
                anomaly_precision = precision_score(actual_anomalies, predicted_anomalies, zero_division=0)
                anomaly_recall = recall_score(actual_anomalies, predicted_anomalies, zero_division=0)
                anomaly_f1 = f1_score(actual_anomalies, predicted_anomalies, zero_division=0)
            else:
                anomaly_precision = 1.0 if np.sum(predicted_anomalies) == 0 else 0.0
                anomaly_recall = 1.0
                anomaly_f1 = 1.0 if np.sum(predicted_anomalies) == 0 else 0.0
        else:
            anomaly_rate = 0.0
            anomaly_precision = 0.0
            anomaly_recall = 0.0
            anomaly_f1 = 0.0
        
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "pws": pws,
            "estimated_loss": estimated_loss,
            "anomaly_rate": anomaly_rate,
            "anomaly_precision": anomaly_precision,
            "anomaly_recall": anomaly_recall,
            "anomaly_f1": anomaly_f1,
            "test_size": len(y_true),
        }
        
        if self.metadata:
            self.metadata.metrics = metrics
        
        return metrics
    
    def save(self, path: Path) -> None:
        """Save the fitted model and metadata."""
        if self.fitted_model is None or self.metadata is None:
            raise RuntimeError("Nothing to save; fit the pipeline first")
        
        save_data = {
            "fitted_model": self.fitted_model,
            "metadata": asdict(self.metadata),
            "model_params": {
                "window_size": self.window_size,
                "horizon": self.horizon,
                "drift_threshold": self.drift_threshold,
                "confidence_level": self.confidence_level,
                "model_type": self.model_type,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "auto_arima": self.auto_arima,
            },
            "feature_scaler": self.feature_scaler,
            "last_obs_index": self._last_obs_index,
            "training_stats": self._training_stats,
        }
        
        joblib.dump(save_data, path)
    
    @staticmethod
    def load(path: Path) -> "EquipmentDriftMonitor":
        """Load a saved model."""
        save_data = joblib.load(path)
        
        # Reconstruct pipeline
        params = save_data["model_params"]
        monitor = EquipmentDriftMonitor(
            window_size=params["window_size"],
            horizon=params["horizon"],
            drift_threshold=params["drift_threshold"],
            confidence_level=params["confidence_level"],
            model_type=params["model_type"],
            order=params["order"],
            seasonal_order=params["seasonal_order"],
            auto_arima=params["auto_arima"],
        )
        
        monitor.fitted_model = save_data["fitted_model"]
        monitor.metadata = DriftMonitorMetadata(**save_data["metadata"])
        monitor.feature_scaler = save_data.get("feature_scaler")
        monitor._last_obs_index = save_data.get("last_obs_index")
        monitor._training_stats = save_data.get("training_stats")
        
        return monitor


# ---------------- Data Loading Utilities ---------------- #

def load_equipment_dataset(dataset_name: str) -> pd.DataFrame:
    """Load equipment dataset with fallback to synthetic data."""
    
    if dataset_name == "synthetic_equipment":
        return generate_equipment_drift_data()
    
    # Try to load from datasets directory using standardized path resolution
    # Resolve relative to this script's location for proper DATA_DIR
    # For projects/starter/equipment_drift_monitor/, path to datasets is ../../../datasets
    script_dir = Path(__file__).parent
    data_dir = (script_dir / "../../../datasets").resolve()
    dataset_path = data_dir / "equipment" / f"{dataset_name}.csv"
    
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        # Try to parse timestamp column
        timestamp_cols = [
            col for col in df.columns if "time" in col.lower() or "date" in col.lower()
        ]
        if timestamp_cols:
            df[timestamp_cols[0]] = pd.to_datetime(df[timestamp_cols[0]])
            df = df.set_index(timestamp_cols[0])
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        return df
    else:
        print(
            f"Dataset {dataset_name} not found at {dataset_path}, using synthetic data", 
            file=sys.stderr
        )
        return generate_equipment_drift_data()


# ---------------- CLI Actions ---------------- #

def action_train(args):
    """Train an equipment drift monitor."""
    try:
        df = load_equipment_dataset(args.data)
        
        # Initialize monitor
        monitor = EquipmentDriftMonitor(
            window_size=args.window_size,
            horizon=args.horizon,
            drift_threshold=args.drift_threshold,
            confidence_level=args.confidence_level,
            model_type=args.model,
            auto_arima=args.auto_arima,
        )
        
        # Fit model
        monitor.fit(df, args.target)
        
        # Save if requested
        if args.save:
            monitor.save(Path(args.save))
        
        # Output results
        result = {
            "status": "trained",
            "model_type": monitor.model_type,
            "window_size": monitor.window_size,
            "horizon": monitor.horizon,
            "training_periods": len(df),
            "metadata": asdict(monitor.metadata) if monitor.metadata else None,
        }
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(
            json.dumps({"status": "error", "message": str(e)}, indent=2),
            file=sys.stderr,
        )
        sys.exit(1)


def action_evaluate(args):
    """Evaluate a trained drift monitor."""
    try:
        monitor = EquipmentDriftMonitor.load(Path(args.model_path))
        df = load_equipment_dataset(args.data)
        
        metrics = monitor.evaluate(
            df,
            args.target,
            test_size=args.test_size,
            tolerance=args.tolerance,
            cost_per_unit=args.cost_per_unit,
        )
        
        result = {
            "status": "evaluated",
            "metrics": metrics,
            "metadata": asdict(monitor.metadata) if monitor.metadata else None,
        }
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(
            json.dumps({"status": "error", "message": str(e)}, indent=2),
            file=sys.stderr,
        )
        sys.exit(1)


def action_predict(args):
    """Generate predictions with a trained drift monitor."""
    try:
        monitor = EquipmentDriftMonitor.load(Path(args.model_path))
        
        # Load data for feature extraction if needed
        df = None
        if args.data:
            df = load_equipment_dataset(args.data)
        
        # Generate predictions
        predictions = monitor.predict(
            df=df, 
            horizon=args.horizon, 
            return_conf_int=True
        )
        
        result = {
            "status": "predicted",
            "horizon": args.horizon,
            "predictions": predictions,
        }
        
        # Save to output file if specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(
            json.dumps({"status": "error", "message": str(e)}, indent=2),
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------- Argument Parsing ---------------- #

def build_parser():
    parser = argparse.ArgumentParser(
        description="Equipment Drift Monitor Pipeline CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    
    # train subcommand
    p_train = sub.add_parser("train", help="Train an equipment drift monitor")
    p_train.add_argument(
        "--data", default="synthetic_equipment", help="Dataset name or path"
    )
    p_train.add_argument("--target", default=TARGET_COLUMN, help="Target column name")
    p_train.add_argument(
        "--window-size", type=int, default=24, help="Sliding window size in hours"
    )
    p_train.add_argument(
        "--horizon", type=int, default=12, help="Forecast horizon in hours"
    )
    p_train.add_argument(
        "--drift-threshold", type=float, default=2.0, 
        help="Standard deviations for anomaly detection"
    )
    p_train.add_argument(
        "--confidence-level", type=float, default=0.95,
        help="Confidence level for prediction intervals"
    )
    p_train.add_argument(
        "--model", default="arima", choices=["arima", "sarima", "auto_arima"],
        help="Model type"
    )
    p_train.add_argument(
        "--auto-arima", action="store_true", default=True,
        help="Use pmdarima auto_arima (if available)"
    )
    p_train.add_argument(
        "--no-auto-arima", dest="auto_arima", action="store_false",
        help="Disable auto_arima"
    )
    p_train.add_argument("--save", help="Path to save trained model")
    p_train.set_defaults(func=action_train)
    
    # evaluate subcommand
    p_eval = sub.add_parser("evaluate", help="Evaluate a trained drift monitor")
    p_eval.add_argument("--model-path", required=True, help="Path to saved model")
    p_eval.add_argument(
        "--data", default="synthetic_equipment", help="Dataset name or path"
    )
    p_eval.add_argument("--target", default=TARGET_COLUMN, help="Target column name")
    p_eval.add_argument(
        "--test-size", type=int, default=48, 
        help="Number of periods for evaluation (hours)"
    )
    p_eval.add_argument(
        "--tolerance", type=float, default=2.0, help="Tolerance for PWS calculation"
    )
    p_eval.add_argument(
        "--cost-per-unit", type=float, default=1.0,
        help="Cost per unit for loss calculation"
    )
    p_eval.set_defaults(func=action_evaluate)
    
    # predict subcommand
    p_pred = sub.add_parser("predict", help="Generate drift forecasts")
    p_pred.add_argument("--model-path", required=True, help="Path to saved model")
    p_pred.add_argument("--data", help="Dataset for feature extraction")
    p_pred.add_argument("--horizon", type=int, default=24, help="Forecast horizon")
    p_pred.add_argument("--output", help="Output file for predictions")
    p_pred.set_defaults(func=action_predict)
    
    return parser


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()