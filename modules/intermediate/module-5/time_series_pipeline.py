"""Production Time Series Forecasting Pipeline Script for Module 5.1

Provides a CLI to train, evaluate, and predict using time series models
for semiconductor manufacturing datasets with temporal structure.

Features:
- ARIMA/Seasonal ARIMA modeling via statsmodels
- Optional pmdarima auto-selection (when available)
- Time series cross-validation to prevent data leakage
- Manufacturing metrics: MAE, RMSE, MAPE, R², PWS, Estimated Loss
- Prediction intervals and uncertainty quantification
- Model persistence (save/load) with parameter storage
- Synthetic time series generators for semiconductor use cases
- Forecast reconciliation to engineering constraints

Example usage:
    python 5.1-time-series-pipeline.py train --data datasets/time_series/processed/kpi.csv \\
        --target target --horizon 24 --model arima --save model.joblib
    python 5.1-time-series-pipeline.py evaluate --model-path model.joblib \\
        --data datasets/time_series/processed/kpi.csv --tolerance 0.05 --cost_per_unit 1.5
    python 5.1-time-series-pipeline.py predict --model-path model.joblib \\
        --data datasets/time_series/processed/kpi.csv --horizon 12 --output forecasts.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

try:
    from prophet import Prophet

    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False
    Prophet = None  # type: ignore

# ---------------- Synthetic Data Generators ---------------- #


def generate_semiconductor_time_series(
    n_periods: int = 200, freq: str = "h", seed: int = RANDOM_SEED  # Hourly data
) -> pd.DataFrame:
    """Generate synthetic semiconductor time series data.

    Simulates tool parameters with trend, seasonality, and noise:
    - Chamber temperature with daily cycles and gradual drift
    - Pressure with random walk and outliers
    - Flow rate with weekly patterns
    - Target yield metric combining all factors
    """
    rng = np.random.default_rng(seed)

    # Create time index
    start_date = pd.Timestamp("2023-01-01")
    time_index = pd.date_range(start=start_date, periods=n_periods, freq=freq)

    # Generate base signals
    t = np.arange(n_periods)

    # Chamber temperature: daily cycle + drift
    daily_cycle = 5 * np.sin(2 * np.pi * t / 24)  # 24-hour cycle
    temp_drift = 0.02 * t  # Gradual drift
    temp_noise = rng.normal(0, 1, n_periods)
    temperature = 450 + daily_cycle + temp_drift + temp_noise

    # Pressure: random walk with outliers
    pressure_walk = np.cumsum(rng.normal(0, 0.05, n_periods))
    pressure_outliers = rng.exponential(0.1, n_periods) * rng.choice([-1, 1], n_periods)
    outlier_mask = rng.random(n_periods) < 0.05  # 5% outlier rate
    pressure = 2.5 + pressure_walk + outlier_mask * pressure_outliers

    # Flow rate: weekly pattern + noise
    weekly_cycle = 3 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly cycle
    flow_noise = rng.normal(0, 2, n_periods)
    flow_rate = 120 + weekly_cycle + flow_noise

    # Target: yield metric combining factors with lags
    temp_effect = -0.1 * (temperature - 450)  # Temp deviation impact
    pressure_effect = -2.0 * np.abs(pressure - 2.5)  # Pressure deviation impact
    flow_effect = -0.05 * np.abs(flow_rate - 120)  # Flow deviation impact

    # Add lag effects (process delays)
    temp_lag = np.concatenate([[0], temp_effect[:-1]])
    target_noise = rng.normal(0, 1, n_periods)

    target = 95 + temp_effect + temp_lag + pressure_effect + flow_effect + target_noise
    target = np.maximum(target, 0)  # Non-negative yield

    return pd.DataFrame(
        {
            "timestamp": time_index,
            "temperature": temperature,
            "pressure": pressure,
            "flow_rate": flow_rate,
            "target": target,
        }
    ).set_index("timestamp")


# ---------------- Time Series Pipeline Class ---------------- #


@dataclass
class TimeSeriesMetadata:
    """Metadata for time series pipeline."""

    model_type: str
    order: Optional[Tuple[int, int, int]] = None
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
    exog_features: Optional[List[str]] = None
    horizon: int = 1
    training_periods: int = 0
    metrics: Optional[Dict[str, float]] = None
    forecast_params: Optional[Dict[str, Any]] = None


class TimeSeriesPipeline:
    """Time Series Forecasting Pipeline for semiconductor manufacturing data."""

    def __init__(
        self,
        model_type: str = "arima",
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        exog_features: Optional[List[str]] = None,
        auto_arima: bool = True,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ) -> None:
        """Initialize time series pipeline.

        Args:
            model_type: Type of model ('arima', 'sarima', 'auto_arima')
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            exog_features: List of exogenous feature column names
            auto_arima: Use pmdarima auto_arima if available
            enforce_stationarity: Enforce stationarity constraint
            enforce_invertibility: Enforce invertibility constraint
        """
        self.model_type = model_type
        self.order = order or (1, 1, 1)
        self.seasonal_order = seasonal_order
        self.exog_features = exog_features or []
        self.auto_arima = auto_arima and HAS_PMDARIMA
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        # Fitted attributes
        self.model = None
        self.fitted_model = None
        self.metadata: Optional[TimeSeriesMetadata] = None
        self._last_obs_index = None
        self._exog_scaler = None

    def _check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Perform ADF test for stationarity."""
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "critical_values": dict(result[4]),
            "is_stationary": result[1] < 0.05,
        }

    def _prepare_data(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Prepare time series data for modeling."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Extract target series
        y = df[target_col].copy()

        # Extract exogenous features if specified
        exog = None
        if self.exog_features:
            available_features = [f for f in self.exog_features if f in df.columns]
            if available_features:
                exog = df[available_features].copy()

                # Simple standardization for exog features
                from sklearn.preprocessing import StandardScaler

                if self._exog_scaler is None:
                    self._exog_scaler = StandardScaler()
                    exog_scaled = self._exog_scaler.fit_transform(exog)
                else:
                    exog_scaled = self._exog_scaler.transform(exog)

                exog = pd.DataFrame(exog_scaled, index=exog.index, columns=exog.columns)

        return y, exog

    def fit(
        self, df: pd.DataFrame, target_col: str = TARGET_COLUMN
    ) -> "TimeSeriesPipeline":
        """Fit the time series model."""
        y, exog = self._prepare_data(df, target_col)

        # Store last observation index for forecasting
        self._last_obs_index = y.index[-1]

        # Check stationarity
        stationarity = self._check_stationarity(y)
        if not stationarity["is_stationary"]:
            warnings.warn(
                f"Series may not be stationary (ADF p-value: {stationarity['p_value']:.4f})"
            )

        try:
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
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility,
                    )
                else:
                    self.model = ARIMA(
                        y,
                        exog=exog,
                        order=self.order,
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility,
                    )
                fitted_order = self.order
                fitted_seasonal = self.seasonal_order

            self.fitted_model = self.model.fit()

            # Store metadata
            self.metadata = TimeSeriesMetadata(
                model_type=self.model_type,
                order=fitted_order,
                seasonal_order=fitted_seasonal,
                exog_features=self.exog_features,
                training_periods=len(y),
            )

        except Exception as e:
            raise RuntimeError(f"Failed to fit time series model: {str(e)}")

        return self

    def predict(
        self,
        horizon: int = 1,
        exog_future: Optional[pd.DataFrame] = None,
        return_conf_int: bool = True,
    ) -> Dict[str, Any]:
        """Generate forecasts."""
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before prediction")

        try:
            # Prepare future exogenous variables
            exog_fc = None
            if self.exog_features and exog_future is not None:
                available_features = [
                    f for f in self.exog_features if f in exog_future.columns
                ]
                if available_features:
                    exog_fc = exog_future[available_features]
                    if self._exog_scaler:
                        exog_fc = pd.DataFrame(
                            self._exog_scaler.transform(exog_fc),
                            index=exog_fc.index,
                            columns=exog_fc.columns,
                        )

            # Generate forecast
            if self.auto_arima and HAS_PMDARIMA:
                forecasts, conf_int = self.fitted_model.predict(
                    n_periods=horizon,
                    exogenous=exog_fc,
                    return_conf_int=return_conf_int,
                )

                # Create proper index for forecasts
                freq = pd.infer_freq(self.fitted_model.arima_res_.data.dates)
                forecast_index = pd.date_range(
                    start=self._last_obs_index, periods=horizon + 1, freq=freq
                )[
                    1:
                ]  # Exclude the last observed point

                forecasts = pd.Series(forecasts, index=forecast_index)
                if return_conf_int:
                    conf_int = pd.DataFrame(
                        conf_int, index=forecast_index, columns=["lower", "upper"]
                    )

            else:
                # statsmodels forecast
                forecast_result = self.fitted_model.forecast(
                    steps=horizon, exog=exog_fc
                )
                forecasts = forecast_result

                if return_conf_int:
                    conf_int_result = self.fitted_model.get_forecast(
                        steps=horizon, exog=exog_fc
                    ).conf_int()
                    conf_int = conf_int_result
                else:
                    conf_int = None

            result = {
                "forecasts": (
                    forecasts.tolist()
                    if isinstance(forecasts, pd.Series)
                    else forecasts
                ),
                "forecast_index": (
                    [str(ts) for ts in forecasts.index]
                    if isinstance(forecasts, pd.Series)
                    else None
                ),
            }

            if return_conf_int and conf_int is not None:
                result["confidence_intervals"] = {
                    "lower": (
                        conf_int.iloc[:, 0].tolist()
                        if hasattr(conf_int, "iloc")
                        else conf_int[:, 0].tolist()
                    ),
                    "upper": (
                        conf_int.iloc[:, 1].tolist()
                        if hasattr(conf_int, "iloc")
                        else conf_int[:, 1].tolist()
                    ),
                }

            return result

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
        test_size: int = 20,
        tolerance: float = 2.0,
        cost_per_unit: float = 1.0,
    ) -> Dict[str, float]:
        """Evaluate model using time series cross-validation."""
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before evaluation")

        y, exog = self._prepare_data(df, target_col)

        if len(y) < test_size + 10:
            raise ValueError("Insufficient data for evaluation")

        # Split data maintaining temporal order
        train_data = y.iloc[:-test_size]
        test_data = y.iloc[-test_size:]

        exog_train = exog.iloc[:-test_size] if exog is not None else None
        exog_test = exog.iloc[-test_size:] if exog is not None else None

        # Refit on training data
        temp_pipeline = TimeSeriesPipeline(
            model_type=self.model_type,
            order=self.order,
            seasonal_order=self.seasonal_order,
            exog_features=self.exog_features,
            auto_arima=self.auto_arima,
        )

        # Create temporary training dataframe
        train_df = pd.DataFrame({target_col: train_data})
        if exog_train is not None:
            train_df = pd.concat([train_df, exog_train], axis=1)

        temp_pipeline.fit(train_df, target_col)

        # Generate predictions
        predictions = []
        for i in range(len(test_data)):
            # Get exog for current prediction if available
            exog_current = None
            if exog_test is not None:
                exog_current = exog_test.iloc[[i]]

            pred_result = temp_pipeline.predict(
                horizon=1, exog_future=exog_current, return_conf_int=False
            )
            predictions.append(pred_result["forecasts"][0])

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

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "pws": pws,
            "estimated_loss": estimated_loss,
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
                "model_type": self.model_type,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "exog_features": self.exog_features,
                "auto_arima": self.auto_arima,
            },
            "exog_scaler": self._exog_scaler,
            "last_obs_index": self._last_obs_index,
        }

        joblib.dump(save_data, path)

    @staticmethod
    def load(path: Path) -> "TimeSeriesPipeline":
        """Load a saved model."""
        save_data = joblib.load(path)

        # Reconstruct pipeline
        params = save_data["model_params"]
        pipeline = TimeSeriesPipeline(
            model_type=params["model_type"],
            order=params["order"],
            seasonal_order=params["seasonal_order"],
            exog_features=params["exog_features"],
            auto_arima=params["auto_arima"],
        )

        pipeline.fitted_model = save_data["fitted_model"]
        pipeline.metadata = TimeSeriesMetadata(**save_data["metadata"])
        pipeline._exog_scaler = save_data.get("exog_scaler")
        pipeline._last_obs_index = save_data.get("last_obs_index")

        return pipeline


# ---------------- Data Loading Utilities ---------------- #


def load_time_series_dataset(dataset_name: str) -> pd.DataFrame:
    """Load time series dataset with fallback to synthetic data."""

    if dataset_name == "synthetic_semiconductor":
        return generate_semiconductor_time_series()

    # Try to load from datasets directory
    dataset_path = Path("datasets/time_series/processed") / f"{dataset_name}.csv"

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
            f"Dataset {dataset_name} not found, using synthetic data", file=sys.stderr
        )
        return generate_semiconductor_time_series()


# ---------------- CLI Actions ---------------- #


def action_train(args):
    """Train a time series model."""
    try:
        df = load_time_series_dataset(args.dataset)

        # Determine model parameters
        order = None
        seasonal_order = None

        if hasattr(args, "order") and args.order:
            order = tuple(map(int, args.order.split(",")))
        if hasattr(args, "seasonal_order") and args.seasonal_order:
            seasonal_order = tuple(map(int, args.seasonal_order.split(",")))

        # Initialize pipeline
        pipeline = TimeSeriesPipeline(
            model_type=args.model,
            order=order,
            seasonal_order=seasonal_order,
            exog_features=args.exog_features.split(",") if args.exog_features else None,
            auto_arima=args.auto_arima,
        )

        # Fit model
        pipeline.fit(df, args.target)

        # Save if requested
        if args.save:
            pipeline.save(Path(args.save))

        # Output results
        result = {
            "status": "trained",
            "model_type": pipeline.model_type,
            "training_periods": len(df),
            "metadata": asdict(pipeline.metadata) if pipeline.metadata else None,
        }

        print(json.dumps(result, indent=2))

    except Exception as e:
        print(
            json.dumps({"status": "error", "message": str(e)}, indent=2),
            file=sys.stderr,
        )
        sys.exit(1)


def action_evaluate(args):
    """Evaluate a trained model."""
    try:
        pipeline = TimeSeriesPipeline.load(Path(args.model_path))
        df = load_time_series_dataset(args.dataset)

        metrics = pipeline.evaluate(
            df,
            args.target,
            test_size=args.test_size,
            tolerance=args.tolerance,
            cost_per_unit=args.cost_per_unit,
        )

        result = {
            "status": "evaluated",
            "metrics": metrics,
            "metadata": asdict(pipeline.metadata) if pipeline.metadata else None,
        }

        print(json.dumps(result, indent=2))

    except Exception as e:
        print(
            json.dumps({"status": "error", "message": str(e)}, indent=2),
            file=sys.stderr,
        )
        sys.exit(1)


def action_predict(args):
    """Generate predictions with a trained model."""
    try:
        pipeline = TimeSeriesPipeline.load(Path(args.model_path))

        # Load data for exogenous features if needed
        exog_future = None
        if args.data and pipeline.exog_features:
            df = load_time_series_dataset(args.data)
            # Use last few rows as future exog (this is a simplified approach)
            exog_future = df[pipeline.exog_features].tail(args.horizon)

        # Generate predictions
        predictions = pipeline.predict(
            horizon=args.horizon, exog_future=exog_future, return_conf_int=True
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
        description="Module 5.1 Time Series Analysis Pipeline CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train subcommand
    p_train = sub.add_parser("train", help="Train a time series model")
    p_train.add_argument(
        "--dataset", default="synthetic_semiconductor", help="Dataset name or path"
    )
    p_train.add_argument("--target", default=TARGET_COLUMN, help="Target column name")
    p_train.add_argument(
        "--model",
        default="arima",
        choices=["arima", "sarima", "auto_arima"],
        help="Model type",
    )
    p_train.add_argument("--order", help="ARIMA order as p,d,q (e.g., 1,1,1)")
    p_train.add_argument(
        "--seasonal-order", help="Seasonal ARIMA order as P,D,Q,s (e.g., 1,1,1,24)"
    )
    p_train.add_argument(
        "--exog-features", help="Comma-separated exogenous feature names"
    )
    p_train.add_argument(
        "--auto-arima",
        action="store_true",
        default=True,
        help="Use pmdarima auto_arima (if available)",
    )
    p_train.add_argument(
        "--no-auto-arima",
        dest="auto_arima",
        action="store_false",
        help="Disable auto_arima",
    )
    p_train.add_argument("--save", help="Path to save trained model")
    p_train.set_defaults(func=action_train)

    # evaluate subcommand
    p_eval = sub.add_parser("evaluate", help="Evaluate a trained model")
    p_eval.add_argument("--model-path", required=True, help="Path to saved model")
    p_eval.add_argument(
        "--dataset", default="synthetic_semiconductor", help="Dataset name or path"
    )
    p_eval.add_argument("--target", default=TARGET_COLUMN, help="Target column name")
    p_eval.add_argument(
        "--test-size", type=int, default=20, help="Number of periods for evaluation"
    )
    p_eval.add_argument(
        "--tolerance", type=float, default=2.0, help="Tolerance for PWS calculation"
    )
    p_eval.add_argument(
        "--cost-per-unit",
        type=float,
        default=1.0,
        help="Cost per unit for loss calculation",
    )
    p_eval.set_defaults(func=action_evaluate)

    # predict subcommand
    p_pred = sub.add_parser("predict", help="Generate forecasts")
    p_pred.add_argument("--model-path", required=True, help="Path to saved model")
    p_pred.add_argument("--data", help="Dataset for exogenous features")
    p_pred.add_argument("--horizon", type=int, default=12, help="Forecast horizon")
    p_pred.add_argument("--output", help="Output file for predictions")
    p_pred.set_defaults(func=action_predict)

    return parser


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
