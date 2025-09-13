"""Unit tests for Equipment Drift Monitor Pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np
import pandas as pd

from equipment_drift_monitor import (
    EquipmentDriftMonitor,
    generate_equipment_drift_data,
    extract_sliding_window_features,
    load_equipment_dataset,
    TARGET_COLUMN,
)


class TestEquipmentDriftMonitor(unittest.TestCase):
    """Test cases for EquipmentDriftMonitor class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = generate_equipment_drift_data(n_periods=200, seed=42)
        self.monitor = EquipmentDriftMonitor(
            window_size=12, horizon=6, auto_arima=False
        )

    def test_synthetic_data_generation(self):
        """Test synthetic equipment drift data generation."""
        df = generate_equipment_drift_data(n_periods=100)
        
        # Check basic structure
        self.assertEqual(len(df), 100)
        self.assertIn("temperature", df.columns)
        self.assertIn("pressure", df.columns)
        self.assertIn("flow_rate", df.columns)
        self.assertIn("power", df.columns)
        self.assertIn(TARGET_COLUMN, df.columns)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        
        # Check value ranges are reasonable
        self.assertTrue(df["temperature"].mean() > 440)  # Around 450
        self.assertTrue(df["pressure"].mean() > 2.0)  # Around 2.5
        self.assertTrue(df["flow_rate"].mean() > 100)  # Around 120
        self.assertTrue(df[TARGET_COLUMN].min() >= 0)  # Non-negative
        self.assertTrue(df[TARGET_COLUMN].max() <= 100)  # Max 100

    def test_synthetic_data_with_options(self):
        """Test synthetic data generation with different options."""
        # Without drift
        df_no_drift = generate_equipment_drift_data(
            n_periods=50, include_drift=False, include_failures=False
        )
        self.assertEqual(len(df_no_drift), 50)
        
        # With failures
        df_with_failures = generate_equipment_drift_data(
            n_periods=100, include_failures=True
        )
        self.assertEqual(len(df_with_failures), 100)

    def test_sliding_window_features(self):
        """Test sliding window feature extraction."""
        features_df = extract_sliding_window_features(self.df, window_size=12)
        
        # Check that feature columns were created
        feature_cols = [col for col in features_df.columns if "rolling" in col or "trend" in col or "lag" in col]
        self.assertGreater(len(feature_cols), 0)
        
        # Check for specific feature types
        rolling_mean_cols = [col for col in features_df.columns if "rolling_mean" in col]
        rolling_std_cols = [col for col in features_df.columns if "rolling_std" in col]
        trend_cols = [col for col in features_df.columns if "trend" in col]
        lag_cols = [col for col in features_df.columns if "lag" in col]
        
        self.assertGreater(len(rolling_mean_cols), 0)
        self.assertGreater(len(rolling_std_cols), 0)
        self.assertGreater(len(trend_cols), 0)
        self.assertGreater(len(lag_cols), 0)
        
        # Check cross-correlation features
        if "temperature" in self.df.columns and "pressure" in self.df.columns:
            self.assertIn("temp_pressure_ratio", features_df.columns)
        if "power" in self.df.columns and "flow_rate" in self.df.columns:
            self.assertIn("power_efficiency", features_df.columns)

    def test_monitor_initialization(self):
        """Test monitor initialization with different parameters."""
        # Default initialization
        monitor1 = EquipmentDriftMonitor()
        self.assertEqual(monitor1.window_size, 24)
        self.assertEqual(monitor1.horizon, 12)
        self.assertEqual(monitor1.drift_threshold, 2.0)
        self.assertEqual(monitor1.confidence_level, 0.95)
        
        # Custom initialization
        monitor2 = EquipmentDriftMonitor(
            window_size=48,
            horizon=24,
            drift_threshold=3.0,
            confidence_level=0.99,
            model_type="sarima",
        )
        self.assertEqual(monitor2.window_size, 48)
        self.assertEqual(monitor2.horizon, 24)
        self.assertEqual(monitor2.drift_threshold, 3.0)
        self.assertEqual(monitor2.confidence_level, 0.99)
        self.assertEqual(monitor2.model_type, "sarima")

    def test_fit_basic(self):
        """Test basic model fitting."""
        monitor = self.monitor.fit(self.df, TARGET_COLUMN)
        
        self.assertIsNotNone(monitor.fitted_model)
        self.assertIsNotNone(monitor.metadata)
        self.assertEqual(monitor.metadata.model_type, "arima")
        self.assertEqual(monitor.metadata.window_size, 12)
        self.assertEqual(monitor.metadata.horizon, 6)
        self.assertIsNotNone(monitor._training_stats)

    def test_fit_with_features(self):
        """Test fitting with sliding window features."""
        monitor = EquipmentDriftMonitor(window_size=12, auto_arima=False)
        monitor.fit(self.df, TARGET_COLUMN)
        
        self.assertIsNotNone(monitor.fitted_model)
        self.assertIsNotNone(monitor.feature_scaler)
        self.assertIsNotNone(monitor.metadata.feature_columns)

    def test_predict_basic(self):
        """Test basic prediction functionality."""
        self.monitor.fit(self.df, TARGET_COLUMN)
        
        # Test prediction with default horizon
        result = self.monitor.predict(return_conf_int=True)
        
        self.assertIn("forecasts", result)
        self.assertIn("forecast_index", result)
        self.assertIn("confidence_intervals", result)
        self.assertIn("anomaly_flags", result)
        self.assertEqual(len(result["forecasts"]), 6)  # Default horizon
        self.assertEqual(len(result["anomaly_flags"]), 6)
        
        # Test prediction with custom horizon
        result_custom = self.monitor.predict(horizon=3, return_conf_int=False)
        self.assertEqual(len(result_custom["forecasts"]), 3)
        self.assertNotIn("confidence_intervals", result_custom)

    def test_predict_with_data(self):
        """Test prediction with new data for feature extraction."""
        self.monitor.fit(self.df, TARGET_COLUMN)
        
        # Generate new data for prediction
        new_data = generate_equipment_drift_data(n_periods=50, seed=123)
        
        result = self.monitor.predict(df=new_data, horizon=4)
        self.assertEqual(len(result["forecasts"]), 4)
        self.assertIn("anomaly_flags", result)

    def test_evaluate(self):
        """Test model evaluation."""
        self.monitor.fit(self.df, TARGET_COLUMN)
        
        metrics = self.monitor.evaluate(
            self.df, 
            TARGET_COLUMN, 
            test_size=20, 
            tolerance=2.0, 
            cost_per_unit=1.5
        )
        
        # Check required metrics are present
        required_metrics = [
            "mae", "rmse", "r2", "mape", "pws", "estimated_loss",
            "anomaly_rate", "anomaly_precision", "anomaly_recall", "anomaly_f1",
            "test_size"
        ]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Check metric values are reasonable
        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)
        self.assertGreaterEqual(metrics["mape"], 0)
        self.assertGreaterEqual(metrics["pws"], 0)
        self.assertLessEqual(metrics["pws"], 100)
        self.assertGreaterEqual(metrics["anomaly_rate"], 0)
        self.assertLessEqual(metrics["anomaly_rate"], 100)
        self.assertEqual(metrics["test_size"], 20)

    def test_save_load_round_trip(self):
        """Test saving and loading the monitor."""
        self.monitor.fit(self.df, TARGET_COLUMN)
        
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save the model
            self.monitor.save(temp_path)
            
            # Load the model
            loaded_monitor = EquipmentDriftMonitor.load(temp_path)
            
            # Check loaded model attributes
            self.assertEqual(loaded_monitor.window_size, self.monitor.window_size)
            self.assertEqual(loaded_monitor.horizon, self.monitor.horizon)
            self.assertEqual(loaded_monitor.drift_threshold, self.monitor.drift_threshold)
            self.assertIsNotNone(loaded_monitor.fitted_model)
            self.assertIsNotNone(loaded_monitor.metadata)
            self.assertIsNotNone(loaded_monitor._training_stats)
            
            # Test that loaded model can make predictions
            original_pred = self.monitor.predict(horizon=3, return_conf_int=False)
            loaded_pred = loaded_monitor.predict(horizon=3, return_conf_int=False)
            
            # Predictions should be very close
            np.testing.assert_array_almost_equal(
                original_pred["forecasts"], loaded_pred["forecasts"], decimal=5
            )
            
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_error_cases(self):
        """Test error handling."""
        # Test prediction before fitting
        with self.assertRaises(RuntimeError):
            self.monitor.predict(horizon=1)
        
        # Test evaluation before fitting
        with self.assertRaises(RuntimeError):
            self.monitor.evaluate(self.df, TARGET_COLUMN)
        
        # Test saving before fitting
        with self.assertRaises(RuntimeError):
            self.monitor.save(Path("/tmp/test.joblib"))
        
        # Test with insufficient data for evaluation
        small_df = self.df.head(20)
        self.monitor.fit(small_df, TARGET_COLUMN)
        with self.assertRaises(ValueError):
            self.monitor.evaluate(small_df, TARGET_COLUMN, test_size=30)

    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        # Create data with known anomalies
        df_with_anomalies = self.df.copy()
        # Inject some obvious anomalies
        df_with_anomalies.iloc[-10:, -1] = 10  # Very low target values
        
        self.monitor.fit(df_with_anomalies, TARGET_COLUMN)
        
        # Predict and check for anomaly detection
        result = self.monitor.predict(horizon=5)
        self.assertIn("anomaly_flags", result)
        self.assertEqual(len(result["anomaly_flags"]), 5)
        
        # Some predictions might be flagged as anomalies
        self.assertIsInstance(result["anomaly_flags"][0], bool)


class TestDataLoading(unittest.TestCase):
    """Test data loading utilities."""

    def test_load_synthetic_dataset(self):
        """Test loading synthetic dataset."""
        df = load_equipment_dataset("synthetic_equipment")
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn(TARGET_COLUMN, df.columns)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_load_nonexistent_dataset(self):
        """Test loading nonexistent dataset falls back to synthetic."""
        with patch("sys.stderr"):  # Suppress warning message
            df = load_equipment_dataset("nonexistent_dataset")
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn(TARGET_COLUMN, df.columns)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration through direct function calls."""

    def setUp(self):
        """Set up for CLI tests."""
        self.temp_model_path = Path("/tmp/test_drift_model.joblib")

    def tearDown(self):
        """Clean up test files."""
        if self.temp_model_path.exists():
            self.temp_model_path.unlink()

    @patch("sys.stdout")
    def test_cli_train(self, mock_stdout):
        """Test CLI training command."""
        from equipment_drift_monitor import action_train
        
        # Mock arguments
        class Args:
            data = "synthetic_equipment"
            target = TARGET_COLUMN
            window_size = 12
            horizon = 6
            drift_threshold = 2.0
            confidence_level = 0.95
            model = "arima"
            auto_arima = False
            save = str(self.temp_model_path)
        
        # Should not raise an exception
        action_train(Args())
        
        # Check that model was saved
        self.assertTrue(self.temp_model_path.exists())

    @patch("sys.stdout")
    def test_cli_evaluate(self, mock_stdout):
        """Test CLI evaluation command."""
        from equipment_drift_monitor import action_train, action_evaluate
        
        # First train a model
        class TrainArgs:
            data = "synthetic_equipment"
            target = TARGET_COLUMN
            window_size = 12
            horizon = 6
            drift_threshold = 2.0
            confidence_level = 0.95
            model = "arima"
            auto_arima = False
            save = str(self.temp_model_path)
        
        action_train(TrainArgs())
        
        # Then evaluate it
        class EvalArgs:
            model_path = str(self.temp_model_path)
            data = "synthetic_equipment"
            target = TARGET_COLUMN
            test_size = 20
            tolerance = 2.0
            cost_per_unit = 1.0
        
        # Should not raise an exception
        action_evaluate(EvalArgs())

    @patch("sys.stdout")
    def test_cli_predict(self, mock_stdout):
        """Test CLI prediction command."""
        from equipment_drift_monitor import action_train, action_predict
        
        # First train a model
        class TrainArgs:
            data = "synthetic_equipment"
            target = TARGET_COLUMN
            window_size = 12
            horizon = 6
            drift_threshold = 2.0
            confidence_level = 0.95
            model = "arima"
            auto_arima = False
            save = str(self.temp_model_path)
        
        action_train(TrainArgs())
        
        # Then predict with it
        class PredArgs:
            model_path = str(self.temp_model_path)
            data = "synthetic_equipment"
            horizon = 8
            output = None
        
        # Should not raise an exception
        action_predict(PredArgs())

    @patch("sys.stdout")
    def test_cli_predict_with_output_file(self, mock_stdout):
        """Test CLI prediction with output file."""
        from equipment_drift_monitor import action_train, action_predict
        
        # Train a model
        class TrainArgs:
            data = "synthetic_equipment"
            target = TARGET_COLUMN
            window_size = 12
            horizon = 6
            drift_threshold = 2.0
            confidence_level = 0.95
            model = "arima"
            auto_arima = False
            save = str(self.temp_model_path)
        
        action_train(TrainArgs())
        
        # Predict with output file
        output_file = Path("/tmp/test_predictions.json")
        try:
            class PredArgs:
                model_path = str(self.temp_model_path)
                data = "synthetic_equipment"
                horizon = 5
                output = str(output_file)
            
            action_predict(PredArgs())
            
            # Check that output file was created and contains valid JSON
            self.assertTrue(output_file.exists())
            with open(output_file, "r") as f:
                result = json.load(f)
            
            self.assertEqual(result["status"], "predicted")
            self.assertEqual(result["horizon"], 5)
            self.assertIn("predictions", result)
            self.assertEqual(len(result["predictions"]["forecasts"]), 5)
            
        finally:
            if output_file.exists():
                output_file.unlink()


class TestManufacturingMetrics(unittest.TestCase):
    """Test manufacturing-specific metrics and features."""

    def setUp(self):
        """Set up test data with known patterns."""
        np.random.seed(42)
        self.df = generate_equipment_drift_data(n_periods=200, seed=42)
        self.monitor = EquipmentDriftMonitor(window_size=12, auto_arima=False)

    def test_manufacturing_tolerance_metrics(self):
        """Test PWS (Prediction Within Spec) calculation."""
        self.monitor.fit(self.df, TARGET_COLUMN)
        
        # Test with tight tolerance
        metrics_tight = self.monitor.evaluate(
            self.df, TARGET_COLUMN, test_size=20, tolerance=0.5
        )
        
        # Test with loose tolerance
        metrics_loose = self.monitor.evaluate(
            self.df, TARGET_COLUMN, test_size=20, tolerance=5.0
        )
        
        # Loose tolerance should have higher PWS
        self.assertGreaterEqual(metrics_loose["pws"], metrics_tight["pws"])

    def test_cost_calculation(self):
        """Test estimated loss calculation with different costs."""
        self.monitor.fit(self.df, TARGET_COLUMN)
        
        # Test with different cost per unit
        metrics_low_cost = self.monitor.evaluate(
            self.df, TARGET_COLUMN, test_size=20, cost_per_unit=1.0
        )
        
        metrics_high_cost = self.monitor.evaluate(
            self.df, TARGET_COLUMN, test_size=20, cost_per_unit=5.0
        )
        
        # Higher cost should result in higher estimated loss
        self.assertGreaterEqual(
            metrics_high_cost["estimated_loss"], 
            metrics_low_cost["estimated_loss"]
        )

    def test_equipment_specific_features(self):
        """Test equipment-specific feature generation."""
        features_df = extract_sliding_window_features(self.df, window_size=12)
        
        # Check for semiconductor equipment specific features
        expected_features = [
            "temperature_rolling_mean_12h",
            "pressure_rolling_std_12h", 
            "flow_rate_trend_12h",
            "power_lag_1h",
            "temp_pressure_ratio",
            "power_efficiency",
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features_df.columns)

    def test_drift_threshold_sensitivity(self):
        """Test drift detection with different thresholds."""
        # Test with sensitive threshold
        sensitive_monitor = EquipmentDriftMonitor(
            window_size=12, drift_threshold=1.0, auto_arima=False
        )
        sensitive_monitor.fit(self.df, TARGET_COLUMN)
        
        # Test with conservative threshold
        conservative_monitor = EquipmentDriftMonitor(
            window_size=12, drift_threshold=3.0, auto_arima=False
        )
        conservative_monitor.fit(self.df, TARGET_COLUMN)
        
        # Generate predictions
        sensitive_result = sensitive_monitor.predict(horizon=10)
        conservative_result = conservative_monitor.predict(horizon=10)
        
        # Sensitive monitor should detect more anomalies
        sensitive_anomalies = sum(sensitive_result["anomaly_flags"])
        conservative_anomalies = sum(conservative_result["anomaly_flags"])
        
        self.assertGreaterEqual(sensitive_anomalies, conservative_anomalies)


if __name__ == "__main__":
    unittest.main()