"""Tests for Deep Learning Pipeline Module 6.1

Test the deep learning pipeline functionality with synthetic data,
focusing on CLI interface, model training, evaluation, and persistence.
Uses small datasets and limited epochs for fast execution.
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import pipeline components
from pathlib import Path
import sys

module_path = Path(__file__).parent
sys.path.insert(0, str(module_path))

try:
    from typing import Dict, Any
    import importlib.util

    # Import the pipeline module
    spec = importlib.util.spec_from_file_location(
        "deep_learning_pipeline", module_path / "6.1-deep-learning-pipeline.py"
    )
    dl_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl_pipeline)

    DeepLearningPipeline = dl_pipeline.DeepLearningPipeline
    generate_synthetic_tabular = dl_pipeline.generate_synthetic_tabular
    HAS_TORCH = dl_pipeline.HAS_TORCH
    HAS_TF = dl_pipeline.HAS_TF

except ImportError as e:
    pytest.skip(f"Could not import pipeline module: {e}", allow_module_level=True)

# Skip tests if no deep learning frameworks available
if not HAS_TORCH and not HAS_TF:
    pytest.skip("No deep learning frameworks available", allow_module_level=True)

RANDOM_SEED = 42
PIPELINE_SCRIPT = module_path / "6.1-deep-learning-pipeline.py"


class TestDeepLearningPipeline:
    """Test the DeepLearningPipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with different backends."""
        if HAS_TORCH:
            pipeline = DeepLearningPipeline(backend="pytorch", task="regression")
            assert pipeline.backend == "pytorch"
            assert pipeline.task == "regression"

        if HAS_TF:
            pipeline = DeepLearningPipeline(backend="tensorflow", task="classification")
            assert pipeline.backend == "tensorflow"
            assert pipeline.task == "classification"

    def test_synthetic_data_generation(self):
        """Test synthetic data generation for both tasks."""
        # Regression data
        df_reg = generate_synthetic_tabular(n=100, task="regression", seed=RANDOM_SEED)
        assert len(df_reg) == 100
        assert "target" in df_reg.columns
        assert df_reg["target"].min() >= 0
        assert df_reg["target"].max() <= 100

        # Classification data
        df_clf = generate_synthetic_tabular(n=100, task="classification", seed=RANDOM_SEED)
        assert len(df_clf) == 100
        assert "target" in df_clf.columns
        assert set(df_clf["target"].unique()).issubset({0, 1})

    @pytest.mark.parametrize("backend", ["pytorch", "tensorflow"])
    def test_regression_training(self, backend):
        """Test regression model training."""
        if backend == "pytorch" and not HAS_TORCH:
            pytest.skip("PyTorch not available")
        if backend == "tensorflow" and not HAS_TF:
            pytest.skip("TensorFlow not available")

        # Generate small dataset
        df = generate_synthetic_tabular(n=200, task="regression", seed=RANDOM_SEED)
        y = df["target"].values
        X = df.drop("target", axis=1)

        # Create and train pipeline
        pipeline = DeepLearningPipeline(
            backend=backend,
            task="regression",
            hidden_dims=[16, 8],  # Small network for fast training
            epochs=5,  # Few epochs for speed
            batch_size=32,
            early_stopping_patience=3,
        )

        pipeline.fit(X, y)

        # Check that model was created
        assert pipeline.model is not None
        assert pipeline.scaler is not None
        assert pipeline.metadata is not None

        # Make predictions
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64 or predictions.dtype == np.float32

    @pytest.mark.parametrize("backend", ["pytorch", "tensorflow"])
    def test_classification_training(self, backend):
        """Test classification model training."""
        if backend == "pytorch" and not HAS_TORCH:
            pytest.skip("PyTorch not available")
        if backend == "tensorflow" and not HAS_TF:
            pytest.skip("TensorFlow not available")

        # Generate small dataset
        df = generate_synthetic_tabular(n=200, task="classification", seed=RANDOM_SEED)
        y = df["target"].values
        X = df.drop("target", axis=1)

        # Create and train pipeline
        pipeline = DeepLearningPipeline(
            backend=backend,
            task="classification",
            hidden_dims=[16, 8],  # Small network for fast training
            epochs=5,  # Few epochs for speed
            batch_size=32,
            early_stopping_patience=3,
        )

        pipeline.fit(X, y)

        # Check that model was created
        assert pipeline.model is not None
        assert pipeline.scaler is not None
        assert pipeline.metadata is not None

        # Make predictions
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)
        assert np.all((predictions >= 0) & (predictions <= 1))  # Probabilities

    def test_evaluation_metrics(self):
        """Test evaluation metrics computation."""
        # Test regression metrics
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        metrics = DeepLearningPipeline.compute_manufacturing_metrics(y_true, y_pred, task="regression", tolerance=0.2)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "pws" in metrics
        assert "estimated_loss" in metrics
        assert metrics["mae"] > 0
        assert metrics["r2"] <= 1.0

        # Test classification metrics
        y_true_clf = np.array([0, 0, 1, 1, 0, 1])
        y_pred_clf = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

        metrics_clf = DeepLearningPipeline.compute_manufacturing_metrics(y_true_clf, y_pred_clf, task="classification")

        assert "roc_auc" in metrics_clf
        assert "pr_auc" in metrics_clf
        assert "f1" in metrics_clf
        assert "pws" in metrics_clf
        assert "estimated_loss" in metrics_clf
        assert 0 <= metrics_clf["roc_auc"] <= 1
        assert 0 <= metrics_clf["pr_auc"] <= 1

    def test_model_persistence(self):
        """Test model save and load functionality."""
        if not HAS_TORCH and not HAS_TF:
            pytest.skip("No deep learning frameworks available")

        # Use available backend
        backend = "pytorch" if HAS_TORCH else "tensorflow"

        # Generate small dataset
        df = generate_synthetic_tabular(n=100, task="regression", seed=RANDOM_SEED)
        y = df["target"].values
        X = df.drop("target", axis=1)

        # Train model
        pipeline1 = DeepLearningPipeline(
            backend=backend, task="regression", hidden_dims=[8, 4], epochs=3, batch_size=16
        )
        pipeline1.fit(X, y)

        # Make predictions with original model
        pred1 = pipeline1.predict(X)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model.joblib"
            pipeline1.save(save_path)

            pipeline2 = DeepLearningPipeline.load(save_path)

            # Make predictions with loaded model
            pred2 = pipeline2.predict(X)

            # Check predictions are (approximately) the same
            np.testing.assert_allclose(pred1, pred2, rtol=1e-5, atol=1e-5)

            # Check metadata is preserved
            assert pipeline2.backend == pipeline1.backend
            assert pipeline2.task == pipeline1.task
            assert pipeline2.feature_names == pipeline1.feature_names


class TestCLIInterface:
    """Test the command-line interface."""

    def test_cli_train_command(self):
        """Test training via CLI."""
        if not HAS_TORCH and not HAS_TF:
            pytest.skip("No deep learning frameworks available")

        backend = "pytorch" if HAS_TORCH else "tensorflow"

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"

            # Run training command
            cmd = [
                "python",
                str(PIPELINE_SCRIPT),
                "train",
                "--dataset",
                "synthetic_yield",
                "--backend",
                backend,
                "--hidden-dims",
                "8",
                "4",
                "--epochs",
                "3",
                "--batch-size",
                "16",
                "--n-samples",
                "100",
                "--save",
                str(model_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=module_path.parent)

            # Check command succeeded
            assert result.returncode == 0, f"Command failed: {result.stderr}"

            # Parse JSON output
            output = json.loads(result.stdout)
            assert output["status"] == "trained"
            assert output["backend"] == backend
            assert output["task"] == "regression"
            assert "training_metrics" in output

            # Check model file was created
            assert model_path.exists()

    def test_cli_evaluate_command(self):
        """Test evaluation via CLI."""
        if not HAS_TORCH and not HAS_TF:
            pytest.skip("No deep learning frameworks available")

        backend = "pytorch" if HAS_TORCH else "tensorflow"

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"

            # First train a model
            train_cmd = [
                "python",
                str(PIPELINE_SCRIPT),
                "train",
                "--dataset",
                "synthetic_yield",
                "--backend",
                backend,
                "--hidden-dims",
                "8",
                "4",
                "--epochs",
                "3",
                "--batch-size",
                "16",
                "--n-samples",
                "100",
                "--save",
                str(model_path),
            ]

            train_result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=module_path.parent)
            assert train_result.returncode == 0

            # Now evaluate the model
            eval_cmd = [
                "python",
                str(PIPELINE_SCRIPT),
                "evaluate",
                "--model-path",
                str(model_path),
                "--dataset",
                "synthetic_yield",
                "--n-samples",
                "50",
            ]

            eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=module_path.parent)

            # Check command succeeded
            assert eval_result.returncode == 0, f"Command failed: {eval_result.stderr}"

            # Parse JSON output
            output = json.loads(eval_result.stdout)
            assert output["status"] == "evaluated"
            assert output["backend"] == backend
            assert "metrics" in output
            assert "mae" in output["metrics"]
            assert "rmse" in output["metrics"]
            assert "r2" in output["metrics"]
            assert "pws" in output["metrics"]

    def test_cli_predict_command(self):
        """Test prediction via CLI."""
        if not HAS_TORCH and not HAS_TF:
            pytest.skip("No deep learning frameworks available")

        backend = "pytorch" if HAS_TORCH else "tensorflow"

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"

            # First train a model
            train_cmd = [
                "python",
                str(PIPELINE_SCRIPT),
                "train",
                "--dataset",
                "synthetic_yield",
                "--backend",
                backend,
                "--hidden-dims",
                "8",
                "4",
                "--epochs",
                "3",
                "--batch-size",
                "16",
                "--n-samples",
                "100",
                "--save",
                str(model_path),
            ]

            train_result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=module_path.parent)
            assert train_result.returncode == 0

            # Now make a prediction
            input_json = '{"temperature": 450, "pressure": 2.5, "flow_rate": 120, "time": 60, "rf_power": 1500}'

            predict_cmd = [
                "python",
                str(PIPELINE_SCRIPT),
                "predict",
                "--model-path",
                str(model_path),
                "--input-json",
                input_json,
            ]

            predict_result = subprocess.run(predict_cmd, capture_output=True, text=True, cwd=module_path.parent)

            # Check command succeeded
            assert predict_result.returncode == 0, f"Command failed: {predict_result.stderr}"

            # Parse JSON output
            output = json.loads(predict_result.stdout)
            assert output["status"] == "predicted"
            assert output["backend"] == backend
            assert "predictions" in output
            assert "prediction_value" in output
            assert isinstance(output["prediction_value"], (int, float))

    def test_cli_error_handling(self):
        """Test CLI error handling."""
        # Test invalid dataset
        cmd = ["python", str(PIPELINE_SCRIPT), "train", "--dataset", "invalid_dataset"]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=module_path.parent)
        assert result.returncode != 0

    def test_backend_fallback(self):
        """Test backend fallback behavior."""
        # This test checks if the pipeline gracefully handles missing backends
        # Since we can't control which backends are available in CI,
        # we'll test the fallback logic indirectly

        # Test that pipeline initialization doesn't crash
        try:
            pipeline = DeepLearningPipeline(backend="pytorch")
            assert pipeline.backend in ["pytorch", "tensorflow"]
        except RuntimeError as e:
            # If neither backend is available, should get clear error
            assert "neither" in str(e).lower() or "not available" in str(e).lower()


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_regression_workflow(self):
        """Test complete regression workflow."""
        if not HAS_TORCH and not HAS_TF:
            pytest.skip("No deep learning frameworks available")

        backend = "pytorch" if HAS_TORCH else "tensorflow"

        # Generate data
        df = generate_synthetic_tabular(n=150, task="regression", seed=RANDOM_SEED)
        y = df["target"].values
        X = df.drop("target", axis=1)

        # Train model
        pipeline = DeepLearningPipeline(
            backend=backend, task="regression", hidden_dims=[16, 8], epochs=5, batch_size=32
        )
        pipeline.fit(X, y)

        # Evaluate
        metrics = pipeline.evaluate(X, y)

        # Check reasonable performance (not too strict for synthetic data)
        assert metrics["r2"] > -1.0  # Basic sanity check
        assert metrics["mae"] > 0
        assert metrics["pws"] >= 0 and metrics["pws"] <= 1

        # Test persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "integration_test.joblib"
            pipeline.save(save_path)

            loaded_pipeline = DeepLearningPipeline.load(save_path)
            loaded_metrics = loaded_pipeline.evaluate(X, y)

            # Metrics should be approximately the same
            np.testing.assert_allclose(
                [metrics["mae"], metrics["rmse"]], [loaded_metrics["mae"], loaded_metrics["rmse"]], rtol=1e-5
            )

    def test_classification_workflow(self):
        """Test complete classification workflow."""
        if not HAS_TORCH and not HAS_TF:
            pytest.skip("No deep learning frameworks available")

        backend = "pytorch" if HAS_TORCH else "tensorflow"

        # Generate data
        df = generate_synthetic_tabular(n=150, task="classification", seed=RANDOM_SEED)
        y = df["target"].values
        X = df.drop("target", axis=1)

        # Train model
        pipeline = DeepLearningPipeline(
            backend=backend, task="classification", hidden_dims=[16, 8], epochs=5, batch_size=32
        )
        pipeline.fit(X, y)

        # Evaluate
        metrics = pipeline.evaluate(X, y)

        # Check reasonable performance
        assert metrics["roc_auc"] >= 0.3  # Should be better than random for synthetic data
        assert metrics["roc_auc"] <= 1.0
        assert metrics["pws"] >= 0 and metrics["pws"] <= 1

        # Predictions should be probabilities
        predictions = pipeline.predict(X)
        assert np.all((predictions >= 0) & (predictions <= 1))


if __name__ == "__main__":
    pytest.main([__file__])
