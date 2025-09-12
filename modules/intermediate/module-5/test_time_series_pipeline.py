"""Unit tests for Module 5.1 Time Series Pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from time_series_pipeline import (
    TimeSeriesPipeline,
    generate_semiconductor_time_series,
    load_time_series_dataset,
    TARGET_COLUMN,
)


class TestTimeSeriesPipeline(unittest.TestCase):
    """Test cases for TimeSeriesPipeline class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = generate_semiconductor_time_series(n_periods=100, seed=42)
        self.pipeline = TimeSeriesPipeline(model_type="arima", auto_arima=False)

    def test_synthetic_data_generation(self):
        """Test synthetic time series data generation."""
        df = generate_semiconductor_time_series(n_periods=50)

        # Check basic structure
        self.assertEqual(len(df), 50)
        self.assertIn("temperature", df.columns)
        self.assertIn("pressure", df.columns)
        self.assertIn("flow_rate", df.columns)
        self.assertIn(TARGET_COLUMN, df.columns)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

        # Check value ranges are reasonable
        self.assertTrue(df["temperature"].mean() > 440)  # Around 450
        self.assertTrue(df["pressure"].mean() > 2.0)  # Around 2.5
        self.assertTrue(df[TARGET_COLUMN].min() >= 0)  # Non-negative

    def test_pipeline_initialization(self):
        """Test pipeline initialization with different parameters."""
        # Default initialization
        pipeline1 = TimeSeriesPipeline()
        self.assertEqual(pipeline1.model_type, "arima")
        self.assertEqual(pipeline1.order, (1, 1, 1))
        self.assertIsNone(pipeline1.seasonal_order)

        # Custom initialization
        pipeline2 = TimeSeriesPipeline(
            model_type="sarima",
            order=(2, 1, 2),
            seasonal_order=(1, 1, 1, 24),
            exog_features=["temperature", "pressure"],
        )
        self.assertEqual(pipeline2.model_type, "sarima")
        self.assertEqual(pipeline2.order, (2, 1, 2))
        self.assertEqual(pipeline2.seasonal_order, (1, 1, 1, 24))
        self.assertEqual(pipeline2.exog_features, ["temperature", "pressure"])

    def test_fit_basic(self):
        """Test basic model fitting."""
        # Test fitting without exogenous variables
        pipeline = self.pipeline.fit(self.df, TARGET_COLUMN)

        self.assertIsNotNone(pipeline.fitted_model)
        self.assertIsNotNone(pipeline.metadata)
        self.assertEqual(pipeline.metadata.model_type, "arima")
        self.assertEqual(pipeline.metadata.training_periods, len(self.df))

    def test_fit_with_exogenous(self):
        """Test fitting with exogenous variables."""
        pipeline = TimeSeriesPipeline(
            exog_features=["temperature", "pressure"], auto_arima=False
        )
        pipeline.fit(self.df, TARGET_COLUMN)

        self.assertIsNotNone(pipeline.fitted_model)
        self.assertEqual(pipeline.exog_features, ["temperature", "pressure"])
        self.assertIsNotNone(pipeline._exog_scaler)

    def test_predict_basic(self):
        """Test basic prediction functionality."""
        self.pipeline.fit(self.df, TARGET_COLUMN)

        # Test single-step prediction
        result = self.pipeline.predict(horizon=1, return_conf_int=True)

        self.assertIn("forecasts", result)
        self.assertIn("forecast_index", result)
        self.assertIn("confidence_intervals", result)
        self.assertEqual(len(result["forecasts"]), 1)
        self.assertEqual(len(result["confidence_intervals"]["lower"]), 1)
        self.assertEqual(len(result["confidence_intervals"]["upper"]), 1)

        # Test multi-step prediction
        result_multi = self.pipeline.predict(horizon=5, return_conf_int=False)
        self.assertEqual(len(result_multi["forecasts"]), 5)
        self.assertNotIn("confidence_intervals", result_multi)

    def test_predict_with_exogenous(self):
        """Test prediction with exogenous variables."""
        pipeline = TimeSeriesPipeline(exog_features=["temperature"], auto_arima=False)
        pipeline.fit(self.df, TARGET_COLUMN)

        # Create future exogenous data
        future_exog = pd.DataFrame(
            {"temperature": [450.0, 451.0, 452.0]},
            index=pd.date_range("2023-01-09 09:00:00", periods=3, freq="h"),
        )

        result = pipeline.predict(horizon=3, exog_future=future_exog)
        self.assertEqual(len(result["forecasts"]), 3)

    def test_evaluate(self):
        """Test model evaluation."""
        self.pipeline.fit(self.df, TARGET_COLUMN)

        metrics = self.pipeline.evaluate(
            self.df, TARGET_COLUMN, test_size=10, tolerance=1.0, cost_per_unit=2.0
        )

        # Check required metrics are present
        required_metrics = [
            "mae",
            "rmse",
            "r2",
            "mape",
            "pws",
            "estimated_loss",
            "test_size",
        ]
        for metric in required_metrics:
            self.assertIn(metric, metrics)

        # Check metric values are reasonable
        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)
        self.assertGreaterEqual(metrics["mape"], 0)
        self.assertGreaterEqual(metrics["pws"], 0)
        self.assertLessEqual(metrics["pws"], 100)
        self.assertEqual(metrics["test_size"], 10)

    def test_save_load_round_trip(self):
        """Test saving and loading the pipeline."""
        self.pipeline.fit(self.df, TARGET_COLUMN)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save the model
            self.pipeline.save(temp_path)

            # Load the model
            loaded_pipeline = TimeSeriesPipeline.load(temp_path)

            # Check loaded model attributes
            self.assertEqual(loaded_pipeline.model_type, self.pipeline.model_type)
            self.assertEqual(loaded_pipeline.order, self.pipeline.order)
            self.assertIsNotNone(loaded_pipeline.fitted_model)
            self.assertIsNotNone(loaded_pipeline.metadata)

            # Test that loaded model can make predictions
            original_pred = self.pipeline.predict(horizon=3, return_conf_int=False)
            loaded_pred = loaded_pipeline.predict(horizon=3, return_conf_int=False)

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
            self.pipeline.predict(horizon=1)

        # Test evaluation before fitting
        with self.assertRaises(RuntimeError):
            self.pipeline.evaluate(self.df, TARGET_COLUMN)

        # Test saving before fitting
        with self.assertRaises(RuntimeError):
            self.pipeline.save(Path("/tmp/test.joblib"))

        # Test with insufficient data for evaluation
        small_df = self.df.head(10)
        self.pipeline.fit(small_df, TARGET_COLUMN)
        with self.assertRaises(ValueError):
            self.pipeline.evaluate(small_df, TARGET_COLUMN, test_size=15)


class TestDataLoading(unittest.TestCase):
    """Test data loading utilities."""

    def test_load_synthetic_dataset(self):
        """Test loading synthetic dataset."""
        df = load_time_series_dataset("synthetic_semiconductor")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn(TARGET_COLUMN, df.columns)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_load_nonexistent_dataset(self):
        """Test loading nonexistent dataset falls back to synthetic."""
        with patch("sys.stderr"):  # Suppress warning message
            df = load_time_series_dataset("nonexistent_dataset")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn(TARGET_COLUMN, df.columns)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration through direct function calls."""

    def setUp(self):
        """Set up for CLI tests."""
        self.temp_model_path = Path("/tmp/test_ts_model.joblib")

    def tearDown(self):
        """Clean up test files."""
        if self.temp_model_path.exists():
            self.temp_model_path.unlink()

    @patch("sys.stdout")
    def test_cli_train(self, mock_stdout):
        """Test CLI training command."""
        from time_series_pipeline import action_train

        # Mock arguments
        class Args:
            dataset = "synthetic_semiconductor"
            target = TARGET_COLUMN
            model = "arima"
            order = None
            seasonal_order = None
            exog_features = None
            auto_arima = False
            save = str(self.temp_model_path)

        # Should not raise an exception
        action_train(Args())

        # Check that model was saved
        self.assertTrue(self.temp_model_path.exists())

    @patch("sys.stdout")
    def test_cli_evaluate(self, mock_stdout):
        """Test CLI evaluation command."""
        from time_series_pipeline import action_train, action_evaluate

        # First train a model
        class TrainArgs:
            dataset = "synthetic_semiconductor"
            target = TARGET_COLUMN
            model = "arima"
            order = None
            seasonal_order = None
            exog_features = None
            auto_arima = False
            save = str(self.temp_model_path)

        action_train(TrainArgs())

        # Then evaluate it
        class EvalArgs:
            model_path = str(self.temp_model_path)
            dataset = "synthetic_semiconductor"
            target = TARGET_COLUMN
            test_size = 10
            tolerance = 1.0
            cost_per_unit = 1.0

        # Should not raise an exception
        action_evaluate(EvalArgs())

    @patch("sys.stdout")
    def test_cli_predict(self, mock_stdout):
        """Test CLI prediction command."""
        from time_series_pipeline import action_train, action_predict

        # First train a model
        class TrainArgs:
            dataset = "synthetic_semiconductor"
            target = TARGET_COLUMN
            model = "arima"
            order = None
            seasonal_order = None
            exog_features = None
            auto_arima = False
            save = str(self.temp_model_path)

        action_train(TrainArgs())

        # Then predict with it
        class PredArgs:
            model_path = str(self.temp_model_path)
            data = None
            horizon = 5
            output = None

        # Should not raise an exception
        action_predict(PredArgs())


if __name__ == "__main__":
    unittest.main()
