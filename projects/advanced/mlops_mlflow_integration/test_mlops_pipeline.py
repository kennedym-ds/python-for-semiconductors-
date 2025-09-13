#!/usr/bin/env python3
"""
Test Suite for MLOps MLflow Integration Pipeline

Comprehensive tests covering:
- Basic pipeline functionality without MLflow
- MLflow integration with tracking enabled  
- Model training, evaluation, and prediction
- CLI interface validation
- Error handling and edge cases
- Optional dependency management
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mlops_mlflow_pipeline import (
    MLOpsMLflowPipeline,
    MLflowConfig,
    check_mlflow_availability,
    generate_semiconductor_data,
    calculate_manufacturing_metrics,
    get_mlflow_experiments
)

# Test configuration
SCRIPT = Path(__file__).parent / "mlops_mlflow_pipeline.py"
TEST_MODEL_PATH = "test_model.joblib"


class TestMLOpsMLflowPipeline(unittest.TestCase):
    """Test the core MLOps pipeline functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = MLOpsMLflowPipeline()
        self.test_data = generate_semiconductor_data(n=100, seed=42)
        self.X = self.test_data.drop('yield', axis=1)
        self.y = self.test_data['yield'].values
    
    def tearDown(self):
        """Clean up test artifacts."""
        # Clean up any test files
        test_files = [TEST_MODEL_PATH, "test_output.json"]
        for file_path in test_files:
            if Path(file_path).exists():
                Path(file_path).unlink()
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with default config."""
        pipeline = MLOpsMLflowPipeline()
        self.assertIsNotNone(pipeline.config)
        self.assertEqual(pipeline.config.experiment_name, "semiconductor_mlops_demo")
        self.assertFalse(pipeline.mlflow_enabled)
        self.assertIsNone(pipeline.model)
        self.assertIsNone(pipeline.preprocessing_pipeline)
    
    def test_pipeline_initialization_with_custom_config(self):
        """Test pipeline initialization with custom config."""
        config = MLflowConfig(
            experiment_name="test_experiment",
            tracking_uri="file:///tmp/test_mlruns",
            enable_autolog=False
        )
        pipeline = MLOpsMLflowPipeline(config=config)
        self.assertEqual(pipeline.config.experiment_name, "test_experiment")
        self.assertEqual(pipeline.config.tracking_uri, "file:///tmp/test_mlruns")
        self.assertFalse(pipeline.config.enable_autolog)
    
    def test_basic_training_without_mlflow(self):
        """Test basic model training without MLflow."""
        pipeline = self.pipeline.fit(self.X, self.y, model_type="ridge", alpha=1.0)
        
        # Check that model is trained
        self.assertIsNotNone(pipeline.model)
        self.assertIsNotNone(pipeline.preprocessing_pipeline)
        self.assertFalse(pipeline.mlflow_enabled)
        
        # Check that predictions work
        predictions = pipeline.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_different_model_types(self):
        """Test training with different model types."""
        model_types = ["ridge", "lasso", "elastic_net", "random_forest"]
        
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                pipeline = MLOpsMLflowPipeline()
                pipeline.fit(self.X, self.y, model_type=model_type)
                
                self.assertIsNotNone(pipeline.model)
                predictions = pipeline.predict(self.X)
                self.assertEqual(len(predictions), len(self.y))
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with self.assertRaises(ValueError):
            self.pipeline.fit(self.X, self.y, model_type="invalid_model")
    
    def test_evaluation_without_trained_model(self):
        """Test error handling when evaluating without trained model."""
        with self.assertRaises(ValueError):
            self.pipeline.evaluate(self.X, self.y)
    
    def test_prediction_without_trained_model(self):
        """Test error handling when predicting without trained model."""
        with self.assertRaises(ValueError):
            self.pipeline.predict(self.X)
    
    def test_save_and_load_pipeline(self):
        """Test saving and loading pipeline."""
        # Train a model
        self.pipeline.fit(self.X, self.y, model_type="ridge")
        
        # Save pipeline
        self.pipeline.save(Path(TEST_MODEL_PATH))
        self.assertTrue(Path(TEST_MODEL_PATH).exists())
        
        # Load pipeline
        loaded_pipeline = MLOpsMLflowPipeline.load(Path(TEST_MODEL_PATH))
        
        # Test that loaded pipeline works
        predictions_original = self.pipeline.predict(self.X)
        predictions_loaded = loaded_pipeline.predict(self.X)
        
        # Predictions should be identical
        import numpy as np
        np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)
    
    def test_manufacturing_metrics_calculation(self):
        """Test manufacturing metrics calculation."""
        y_true = np.array([70.0, 75.0, 80.0, 85.0, 90.0])
        y_pred = np.array([72.0, 74.0, 82.0, 83.0, 88.0])
        
        metrics = calculate_manufacturing_metrics(y_true, y_pred)
        
        # Check that all expected metrics are present
        expected_metrics = ['mae', 'rmse', 'r2', 'pws_percent', 'estimated_loss', 'yield_rate']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_semiconductor_data_generation(self):
        """Test synthetic data generation."""
        # Test normal data generation
        data_normal = generate_semiconductor_data(n=100, inject_drift=False, seed=42)
        self.assertEqual(len(data_normal), 100)
        self.assertIn('yield', data_normal.columns)
        
        # Test data generation with drift
        data_drift = generate_semiconductor_data(n=100, inject_drift=True, seed=42)
        self.assertEqual(len(data_drift), 100)
        self.assertIn('yield', data_drift.columns)
        
        # Check that drift injection produces different data
        self.assertFalse(data_normal['temperature'].equals(data_drift['temperature']))


class TestMLflowIntegration(unittest.TestCase):
    """Test MLflow integration features."""
    
    def setUp(self):
        """Set up test fixtures with temporary MLflow directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.mlflow_uri = f"file://{self.temp_dir}/mlruns"
        
        # Set environment variable for test MLflow tracking
        os.environ["MLFLOW_TRACKING_URI"] = self.mlflow_uri
        
        self.test_data = generate_semiconductor_data(n=100, seed=42)
        self.X = self.test_data.drop('yield', axis=1)
        self.y = self.test_data['yield'].values
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up environment variable
        if "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_mlflow_availability_check(self):
        """Test MLflow availability checking."""
        # This should return True in the test environment since we have MLflow installed
        is_available = check_mlflow_availability()
        self.assertTrue(is_available)
    
    def test_enable_mlflow_tracking(self):
        """Test enabling MLflow tracking."""
        pipeline = MLOpsMLflowPipeline()
        success = pipeline.enable_mlflow_tracking("test_experiment")
        
        self.assertTrue(success)
        self.assertTrue(pipeline.mlflow_enabled)
        self.assertIsNotNone(pipeline.experiment_id)
    
    def test_training_with_mlflow(self):
        """Test model training with MLflow tracking enabled."""
        config = MLflowConfig(
            experiment_name="test_training",
            tracking_uri=self.mlflow_uri
        )
        pipeline = MLOpsMLflowPipeline(config=config)
        
        # Enable MLflow tracking
        pipeline.enable_mlflow_tracking()
        
        # Train model
        pipeline.fit(self.X, self.y, model_type="ridge", run_name="test_run")
        
        self.assertIsNotNone(pipeline.model)
        self.assertTrue(pipeline.mlflow_enabled)
    
    def test_evaluation_with_mlflow(self):
        """Test model evaluation with MLflow tracking."""
        config = MLflowConfig(
            experiment_name="test_evaluation",
            tracking_uri=self.mlflow_uri
        )
        pipeline = MLOpsMLflowPipeline(config=config)
        
        # Train model first
        pipeline.fit(self.X, self.y, model_type="ridge")
        
        # Enable MLflow for evaluation
        pipeline.enable_mlflow_tracking()
        
        # Evaluate model
        results = pipeline.evaluate(self.X, self.y, run_name="test_eval")
        
        self.assertEqual(results["status"], "evaluated")
        self.assertIn("metrics", results)
        self.assertTrue(results["mlflow_enabled"])


class TestCLIInterface(unittest.TestCase):
    """Test the command-line interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_model_path = Path(self.temp_dir) / "test_model.joblib"
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def run_cmd(self, args):
        """Run CLI command and return JSON output."""
        cmd = [sys.executable, str(SCRIPT)] + args
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    
    def test_train_command_basic(self):
        """Test basic training command."""
        output = self.run_cmd([
            "train",
            "--dataset", "synthetic_yield",
            "--model", "ridge",
            "--save", str(self.test_model_path)
        ])
        
        self.assertEqual(output["status"], "trained")
        self.assertEqual(output["model_type"], "ridge")
        self.assertIn("metrics", output)
        self.assertTrue(self.test_model_path.exists())
    
    def test_train_command_with_mlflow(self):
        """Test training command with MLflow enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Set temporary MLflow tracking URI
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{tmp_dir}/mlruns"
            
            try:
                output = self.run_cmd([
                    "train",
                    "--dataset", "synthetic_yield",
                    "--model", "ridge",
                    "--enable-mlflow",
                    "--experiment-name", "test_cli_experiment",
                    "--save", str(self.test_model_path)
                ])
                
                self.assertEqual(output["status"], "trained")
                self.assertTrue(output["mlflow_enabled"])
                self.assertTrue(self.test_model_path.exists())
                
            finally:
                if "MLFLOW_TRACKING_URI" in os.environ:
                    del os.environ["MLFLOW_TRACKING_URI"]
    
    def test_train_command_with_drift(self):
        """Test training command with drift injection."""
        output = self.run_cmd([
            "train",
            "--dataset", "synthetic_yield",
            "--model", "ridge",
            "--inject-drift",
            "--save", str(self.test_model_path)
        ])
        
        self.assertEqual(output["status"], "trained")
        self.assertTrue(output["drift_injected"])
        self.assertTrue(self.test_model_path.exists())
    
    def test_evaluate_command(self):
        """Test model evaluation command."""
        # First train a model
        self.run_cmd([
            "train",
            "--dataset", "synthetic_yield",
            "--model", "ridge",
            "--save", str(self.test_model_path)
        ])
        
        # Then evaluate it
        output = self.run_cmd([
            "evaluate",
            "--model-path", str(self.test_model_path),
            "--dataset", "synthetic_yield"
        ])
        
        self.assertEqual(output["status"], "evaluated")
        self.assertIn("metrics", output)
        self.assertFalse(output["drift_injected"])
    
    def test_evaluate_command_with_drift(self):
        """Test evaluation command with drift injection."""
        # First train a model
        self.run_cmd([
            "train",
            "--dataset", "synthetic_yield",
            "--model", "ridge",
            "--save", str(self.test_model_path)
        ])
        
        # Then evaluate with drift
        output = self.run_cmd([
            "evaluate",
            "--model-path", str(self.test_model_path),
            "--dataset", "synthetic_yield",
            "--inject-drift"
        ])
        
        self.assertEqual(output["status"], "evaluated")
        self.assertTrue(output["drift_injected"])
    
    def test_predict_command_json(self):
        """Test prediction command with JSON input."""
        # First train a model
        self.run_cmd([
            "train",
            "--dataset", "synthetic_yield",
            "--model", "ridge",
            "--save", str(self.test_model_path)
        ])
        
        # Test prediction
        input_data = {
            "temperature": 455.0,
            "pressure": 2.6,
            "flow": 118.0,
            "time": 62.0,
            "temp_centered": 5.0,
            "pressure_sq": 6.76,
            "flow_time_inter": 7316.0,
            "temp_flow_inter": 53690.0
        }
        
        output = self.run_cmd([
            "predict",
            "--model-path", str(self.test_model_path),
            "--input-json", json.dumps(input_data)
        ])
        
        self.assertEqual(output["status"], "predicted")
        self.assertIn("predictions", output)
        self.assertEqual(output["n_samples"], 1)
    
    def test_predict_command_csv(self):
        """Test prediction command with CSV input."""
        # First train a model
        self.run_cmd([
            "train",
            "--dataset", "synthetic_yield", 
            "--model", "ridge",
            "--save", str(self.test_model_path)
        ])
        
        # Create test CSV file
        test_csv = Path(self.temp_dir) / "test_input.csv"
        import pandas as pd
        
        test_data = generate_semiconductor_data(n=5, seed=42)
        test_features = test_data.drop('yield', axis=1)
        test_features.to_csv(test_csv, index=False)
        
        # Test prediction
        output = self.run_cmd([
            "predict",
            "--model-path", str(self.test_model_path),
            "--input-csv", str(test_csv)
        ])
        
        self.assertEqual(output["status"], "predicted")
        self.assertIn("predictions", output)
        self.assertEqual(output["n_samples"], 5)
    
    def test_mlflow_management_commands(self):
        """Test MLflow management commands."""
        # Test list experiments
        output = self.run_cmd(["list-experiments"])
        self.assertIn("status", output)
        
        # Test start tracking
        output = self.run_cmd(["start-tracking", "--experiment", "test_cli_experiment"])
        self.assertEqual(output["status"], "tracking_enabled")
        
        # Test stop tracking
        output = self.run_cmd(["stop-tracking"])
        self.assertIn("status", output)
    
    def test_error_handling_invalid_dataset(self):
        """Test error handling for invalid dataset."""
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_cmd([
                "train",
                "--dataset", "invalid_dataset",
                "--model", "ridge"
            ])
    
    def test_error_handling_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_cmd([
                "train",
                "--dataset", "synthetic_yield",
                "--model", "invalid_model"
            ])
    
    def test_error_handling_missing_model_file(self):
        """Test error handling for missing model file."""
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_cmd([
                "evaluate",
                "--model-path", "nonexistent_model.joblib",
                "--dataset", "synthetic_yield"
            ])
    
    def test_help_commands(self):
        """Test help command functionality."""
        # Test main help
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], 
                              capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("MLOps MLflow Integration Pipeline", result.stdout)
        
        # Test subcommand help
        result = subprocess.run([sys.executable, str(SCRIPT), "train", "--help"],
                              capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Dataset to use for training", result.stdout)


class TestOptionalDependencies(unittest.TestCase):
    """Test optional dependency handling."""
    
    def test_graceful_mlflow_fallback(self):
        """Test graceful fallback when MLflow is not available."""
        # Mock MLflow as unavailable
        with patch('mlops_mlflow_pipeline.HAS_MLFLOW', False):
            from mlops_mlflow_pipeline import MLOpsMLflowPipeline
            
            pipeline = MLOpsMLflowPipeline()
            
            # Attempting to enable MLflow should fail gracefully
            success = pipeline.enable_mlflow_tracking("test_experiment")
            self.assertFalse(success)
            self.assertFalse(pipeline.mlflow_enabled)
            
            # Training should still work without MLflow
            test_data = generate_semiconductor_data(n=50, seed=42)
            X = test_data.drop('yield', axis=1)
            y = test_data['yield'].values
            
            pipeline.fit(X, y, model_type="ridge")
            self.assertIsNotNone(pipeline.model)
            
            # Predictions should work
            predictions = pipeline.predict(X)
            self.assertEqual(len(predictions), len(y))


def run_all_tests():
    """Run all tests and return results."""
    print("Running MLOps MLflow Integration Pipeline Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMLOpsMLflowPipeline,
        TestMLflowIntegration, 
        TestCLIInterface,
        TestOptionalDependencies
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success


if __name__ == "__main__":
    import numpy as np
    
    success = run_all_tests()
    sys.exit(0 if success else 1)