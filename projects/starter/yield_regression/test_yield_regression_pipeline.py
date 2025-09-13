"""Tests for Starter Yield Regression Pipeline

Comprehensive test suite covering CLI functionality, pipeline methods,
model persistence, and semiconductor-specific metrics.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / "yield_regression_pipeline.py"


def run_cmd(args):
    """Run pipeline command and return JSON output."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)] + args, capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def test_train_basic():
    """Test basic training functionality."""
    out = run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge"])

    assert out["status"] == "trained"
    assert "metrics" in out
    assert "metadata" in out

    # Check required metrics
    metrics = out["metrics"]
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert "PWS" in metrics
    assert "Estimated_Loss" in metrics

    # Verify metric ranges
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] > 0
    assert -1 <= metrics["R2"] <= 1
    assert 0 <= metrics["PWS"] <= 1
    assert metrics["Estimated_Loss"] >= 0

    # Check metadata
    metadata = out["metadata"]
    assert metadata["model_type"] == "Ridge"
    assert metadata["n_features_in"] == 8  # 4 base + 4 engineered features
    assert metadata["params"]["model"] == "ridge"


def test_train_different_models():
    """Test training with different model types."""
    models = ["ridge", "lasso", "elasticnet", "linear", "rf"]

    for model in models:
        out = run_cmd(["train", "--dataset", "synthetic_yield", "--model", model])
        assert out["status"] == "trained"
        assert "metrics" in out
        # Check that all metrics are finite
        for metric_name, metric_value in out["metrics"].items():
            assert np.isfinite(metric_value), f"{metric_name} is not finite for {model}"


def test_train_with_save():
    """Test training with model saving."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"

        out = run_cmd(
            [
                "train",
                "--dataset",
                "synthetic_yield",
                "--model",
                "ridge",
                "--save",
                str(model_path),
            ]
        )

        assert out["status"] == "trained"
        assert model_path.exists()


def test_train_hyperparameters():
    """Test training with different hyperparameters."""
    out = run_cmd(
        [
            "train",
            "--dataset",
            "synthetic_yield",
            "--model",
            "ridge",
            "--alpha",
            "0.1",
            "--k-best",
            "5",
            "--pca-components",
            "0.8",
            "--no-feature-selection",
        ]
    )

    assert out["status"] == "trained"
    assert out["metadata"]["params"]["alpha"] == 0.1
    assert out["metadata"]["params"]["use_feature_selection"] is False


def test_evaluate():
    """Test model evaluation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"

        # Train and save model
        run_cmd(
            [
                "train",
                "--dataset",
                "synthetic_yield",
                "--model",
                "ridge",
                "--save",
                str(model_path),
            ]
        )

        # Evaluate model
        out = run_cmd(
            [
                "evaluate",
                "--model-path",
                str(model_path),
                "--dataset",
                "synthetic_yield",
            ]
        )

        assert out["status"] == "evaluated"
        assert "metrics" in out
        assert "metadata" in out

        # Check that metrics are consistent (same data, same model)
        metrics = out["metrics"]
        assert metrics["MAE"] > 0
        assert metrics["RMSE"] > 0
        assert -1 <= metrics["R2"] <= 1


def test_predict_json_input():
    """Test prediction with JSON input string."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"

        # Train and save model
        run_cmd(
            [
                "train",
                "--dataset",
                "synthetic_yield",
                "--model",
                "ridge",
                "--save",
                str(model_path),
            ]
        )

        # Make prediction
        input_data = {
            "temperature": 455,
            "pressure": 2.6,
            "flow": 118,
            "time": 62,
            "temp_centered": 5.0,
            "pressure_sq": 6.76,
            "flow_time_inter": 7316,
            "temp_flow_inter": 53690,
        }

        out = run_cmd(
            [
                "predict",
                "--model-path",
                str(model_path),
                "--input-json",
                json.dumps(input_data),
            ]
        )

        assert "prediction" in out
        assert "input" in out
        assert "model_meta" in out

        # Check prediction is reasonable (yield percentage)
        prediction = out["prediction"]
        assert 0 <= prediction <= 100

        # Check input echo
        assert out["input"] == input_data


def test_predict_file_input():
    """Test prediction with JSON file input."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"
        input_file = Path(tmp_dir) / "input.json"

        # Train and save model
        run_cmd(
            [
                "train",
                "--dataset",
                "synthetic_yield",
                "--model",
                "ridge",
                "--save",
                str(model_path),
            ]
        )

        # Create input file
        input_data = {
            "temperature": 450,
            "pressure": 2.5,
            "flow": 120,
            "time": 60,
            "temp_centered": 0.0,
            "pressure_sq": 6.25,
            "flow_time_inter": 7200,
            "temp_flow_inter": 54000,
        }
        input_file.write_text(json.dumps(input_data))

        # Make prediction
        out = run_cmd(
            [
                "predict",
                "--model-path",
                str(model_path),
                "--input-file",
                str(input_file),
            ]
        )

        assert "prediction" in out
        assert out["input"] == input_data


def test_pipeline_class_directly():
    """Test pipeline class methods directly."""
    from yield_regression_pipeline import (
        YieldRegressionPipeline,
        generate_yield_process,
    )

    # Generate test data
    df = generate_yield_process(n=100, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    # Create and train pipeline
    pipeline = YieldRegressionPipeline(model="ridge", alpha=1.0)
    pipeline.fit(X, y)

    # Test predictions
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds))

    # Test evaluation
    metrics = pipeline.evaluate(X, y)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert "PWS" in metrics
    assert "Estimated_Loss" in metrics

    # Test save/load
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "pipeline.joblib"
        pipeline.save(model_path)

        loaded_pipeline = YieldRegressionPipeline.load(model_path)
        loaded_preds = loaded_pipeline.predict(X)

        # Predictions should be identical
        np.testing.assert_array_almost_equal(preds, loaded_preds)


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    from yield_regression_pipeline import generate_yield_process

    df = generate_yield_process(n=100, seed=42)

    # Check expected columns
    expected_columns = [
        "temperature",
        "pressure",
        "flow",
        "time",
        "yield_pct",
        "temp_centered",
        "pressure_sq",
        "flow_time_inter",
        "temp_flow_inter",
    ]
    assert list(df.columns) == expected_columns

    # Check data ranges
    assert df["temperature"].min() > 400
    assert df["temperature"].max() < 500
    assert df["pressure"].min() > 1.5
    assert df["pressure"].max() < 3.5
    assert df["flow"].min() > 90
    assert df["flow"].max() < 150
    assert df["time"].min() > 40
    assert df["time"].max() < 80

    # Check yield range
    assert df["yield_pct"].min() >= 0
    assert df["yield_pct"].max() <= 100

    # Check reproducibility
    df2 = generate_yield_process(n=100, seed=42)
    pd.testing.assert_frame_equal(df, df2)


def test_metrics_calculation():
    """Test semiconductor metrics calculation."""
    from yield_regression_pipeline import YieldRegressionPipeline

    y_true = np.array([70, 75, 80, 85, 90])
    y_pred = np.array([72, 73, 82, 87, 88])

    metrics = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    # Check PWS (all predictions within spec)
    assert metrics["PWS"] == 1.0

    # Check estimated loss (errors exactly at tolerance = no loss)
    assert metrics["Estimated_Loss"] == 0.0

    # Check MAE calculation
    expected_mae = np.mean(np.abs(y_true - y_pred))
    assert abs(metrics["MAE"] - expected_mae) < 1e-6

    # Test with errors beyond tolerance
    y_pred_high_error = np.array([65, 70, 85, 90, 95])  # Larger errors
    metrics_loss = YieldRegressionPipeline.compute_metrics(
        y_true,
        y_pred_high_error,
        tolerance=2.0,
        spec_low=60,
        spec_high=100,
        cost_per_unit=1.0,
    )
    assert metrics_loss["Estimated_Loss"] > 0


def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test invalid dataset
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd(["train", "--dataset", "invalid_dataset"])

    # Test missing model path for evaluate
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd(["evaluate", "--model-path", "/nonexistent/path.joblib"])

    # Test missing input for predict
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"

        # Train and save model first
        run_cmd(
            [
                "train",
                "--dataset",
                "synthetic_yield",
                "--model",
                "ridge",
                "--save",
                str(model_path),
            ]
        )

        # Test predict without input
        with pytest.raises(subprocess.CalledProcessError):
            run_cmd(["predict", "--model-path", str(model_path)])


def test_reproducibility():
    """Test that results are reproducible with fixed seed."""
    out1 = run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge"])
    out2 = run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge"])

    # Metrics should be identical due to fixed seed
    metrics1 = out1["metrics"]
    metrics2 = out2["metrics"]

    for key in metrics1:
        assert (
            abs(metrics1[key] - metrics2[key]) < 1e-10
        ), f"Metric {key} not reproducible"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
