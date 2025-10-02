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
    result = subprocess.run([sys.executable, str(SCRIPT)] + args, capture_output=True, text=True, check=True)
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
        assert abs(metrics1[key] - metrics2[key]) < 1e-10, f"Metric {key} not reproducible"


# ============================================================================
# EDGE CASE TESTS (8 tests)
# ============================================================================


def test_zero_yield_prediction():
    """Test handling of extreme low yield predictions."""
    from yield_regression_pipeline import YieldRegressionPipeline

    y_true = np.array([70, 75, 80, 85, 90])
    y_pred = np.array([0, 5, 10, 15, 20])  # Extreme low predictions

    metrics = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    # All predictions below spec
    assert metrics["PWS"] == 0.0
    # High error should result in high loss
    assert metrics["Estimated_Loss"] > 50.0
    # RMSE should be high
    assert metrics["RMSE"] > 50.0


def test_hundred_percent_yield():
    """Test handling of perfect yield predictions."""
    from yield_regression_pipeline import YieldRegressionPipeline

    y_true = np.array([100, 100, 100, 100, 100])
    y_pred = np.array([100, 99, 100, 101, 100])  # Near-perfect predictions

    metrics = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    # Most predictions at spec upper limit
    assert metrics["PWS"] >= 0.8
    # Very low error
    assert metrics["MAE"] < 1.0


def test_negative_predictions():
    """Test handling of negative yield predictions (should be clipped)."""
    from yield_regression_pipeline import YieldRegressionPipeline

    y_true = np.array([70, 75, 80, 85, 90])
    y_pred = np.array([-10, -5, 0, 5, 10])  # Some negative predictions

    # Should not raise error
    metrics = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    assert metrics["PWS"] == 0.0
    assert metrics["Estimated_Loss"] > 0


def test_missing_features():
    """Test prediction with missing engineered features (should fail gracefully)."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=100, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    pipeline = YieldRegressionPipeline(model="ridge")
    pipeline.fit(X, y)

    # Try predicting with only base features (missing engineered)
    X_incomplete = X[["temperature", "pressure", "flow", "time"]]

    with pytest.raises((ValueError, KeyError)):
        pipeline.predict(X_incomplete)


def test_outlier_process_parameters():
    """Test with parameters far outside training range."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=100, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    pipeline = YieldRegressionPipeline(model="ridge")
    pipeline.fit(X, y)

    # Create extreme outlier
    X_outlier = X.iloc[:1].copy()
    X_outlier["temperature"] = 1000  # Way outside training range
    X_outlier["pressure"] = 10.0
    X_outlier["flow"] = 500

    # Should still make prediction (may be unreliable)
    preds = pipeline.predict(X_outlier)
    assert len(preds) == 1
    assert np.isfinite(preds[0])


def test_constant_predictions():
    """Test model behavior when all predictions are same."""
    from yield_regression_pipeline import YieldRegressionPipeline

    y_true = np.array([70, 75, 80, 85, 90])
    y_pred = np.array([75, 75, 75, 75, 75])  # All same

    metrics = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    # R² should be 0 for constant predictions
    assert metrics["R2"] <= 0.1
    # PWS should be 100% (all within spec)
    assert metrics["PWS"] == 1.0


def test_high_variance_predictions():
    """Test model with very scattered predictions."""
    from yield_regression_pipeline import YieldRegressionPipeline

    y_true = np.array([70, 70, 70, 70, 70])
    y_pred = np.array([50, 60, 70, 80, 90])  # High variance

    metrics = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    # R² should be negative (worse than mean baseline)
    assert metrics["R2"] < 0
    # RMSE should be high
    assert metrics["RMSE"] > 10.0


def test_pws_edge_cases():
    """Test PWS calculation with boundary values."""
    from yield_regression_pipeline import YieldRegressionPipeline

    # Test exactly at spec limits
    y_true = np.array([70, 75, 80, 85, 90])
    y_pred = np.array([60, 70, 80, 90, 100])  # At boundaries

    metrics = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    # All predictions at or within spec
    assert metrics["PWS"] == 1.0

    # Test just outside spec limits
    y_pred_out = np.array([59.9, 70, 80, 90, 100.1])  # Slightly outside

    metrics_out = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred_out, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    # 60% within spec (3 out of 5)
    assert metrics_out["PWS"] == 0.6


# ============================================================================
# MANUFACTURING SCENARIO TESTS (6 tests)
# ============================================================================


def test_process_drift_simulation():
    """Test model on data with process drift."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    # Generate training data
    df_train = generate_yield_process(n=200, seed=42)
    X_train = df_train.drop(columns=["yield_pct"])
    y_train = df_train["yield_pct"].values

    # Train model
    pipeline = YieldRegressionPipeline(model="ridge")
    pipeline.fit(X_train, y_train)

    # Generate drifted test data (shifted temperature)
    df_test = generate_yield_process(n=100, seed=43)
    df_test["temperature"] += 10  # Process drift
    df_test["temp_centered"] += 10  # Update engineered feature
    df_test["temp_flow_inter"] = df_test["temperature"] * df_test["flow"]  # Recalc

    X_test = df_test.drop(columns=["yield_pct"])
    y_test = df_test["yield_pct"].values

    # Model should still make predictions but may have degraded performance
    preds = pipeline.predict(X_test)
    assert len(preds) == len(y_test)
    assert np.all(np.isfinite(preds))

    metrics = pipeline.evaluate(X_test, y_test)
    # R² may be lower due to drift
    assert metrics["R2"] < 0.8


def test_yield_improvement_scenario():
    """Test tracking yield improvements over time."""
    from yield_regression_pipeline import YieldRegressionPipeline

    # Baseline yield
    y_baseline = np.array([65, 68, 70, 67, 69])
    # Improved yield (after process optimization)
    y_improved = np.array([75, 78, 80, 77, 79])

    # Model predicts based on baseline
    y_pred = y_baseline.copy()

    # Calculate improvement gap
    metrics_baseline = YieldRegressionPipeline.compute_metrics(
        y_baseline, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    metrics_improved = YieldRegressionPipeline.compute_metrics(
        y_improved, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    # Improvement should increase error but maintain PWS
    assert metrics_improved["MAE"] > metrics_baseline["MAE"]
    assert metrics_baseline["PWS"] == 1.0
    assert metrics_improved["PWS"] == 1.0


def test_batch_processing():
    """Test predictions on batch of wafers."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=1000, seed=42)  # Large batch
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    pipeline = YieldRegressionPipeline(model="rf")
    pipeline.fit(X, y)

    # Predict on entire batch
    preds = pipeline.predict(X)

    assert len(preds) == 1000
    assert np.all(np.isfinite(preds))

    # Check batch statistics
    assert preds.mean() > 60
    assert preds.mean() < 90
    assert preds.std() > 0


def test_spec_limit_violations():
    """Test predictions outside spec limits."""
    from yield_regression_pipeline import YieldRegressionPipeline

    y_true = np.array([70, 75, 80, 85, 90])
    y_pred = np.array([55, 65, 105, 110, 90])  # Some out of spec

    metrics = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    # PWS should be 40% (2 out of 5 within spec)
    assert metrics["PWS"] == 0.4

    # Violations should contribute to loss
    assert metrics["Estimated_Loss"] > 0


def test_high_loss_scenarios():
    """Test estimated loss with high error predictions."""
    from yield_regression_pipeline import YieldRegressionPipeline

    y_true = np.array([70, 75, 80, 85, 90])
    y_pred = np.array([50, 55, 60, 65, 70])  # Consistent 20-point error

    metrics = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=5.0
    )

    # High cost per unit should amplify loss
    assert metrics["Estimated_Loss"] > 50.0

    # Compare with lower cost
    metrics_low_cost = YieldRegressionPipeline.compute_metrics(
        y_true, y_pred, tolerance=2.0, spec_low=60, spec_high=100, cost_per_unit=1.0
    )

    assert metrics["Estimated_Loss"] > metrics_low_cost["Estimated_Loss"]


def test_parameter_optimization_scenario():
    """Test identifying optimal process parameters."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=500, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    pipeline = YieldRegressionPipeline(model="rf")
    pipeline.fit(X, y)

    # Test different parameter combinations
    test_params = pd.DataFrame(
        {
            "temperature": [440, 450, 460],
            "pressure": [2.3, 2.5, 2.7],
            "flow": [110, 120, 130],
            "time": [55, 60, 65],
        }
    )

    # Add engineered features
    test_params["temp_centered"] = test_params["temperature"] - 450
    test_params["pressure_sq"] = test_params["pressure"] ** 2
    test_params["flow_time_inter"] = test_params["flow"] * test_params["time"]
    test_params["temp_flow_inter"] = test_params["temperature"] * test_params["flow"]

    preds = pipeline.predict(test_params)

    # Find optimal parameters (highest predicted yield)
    best_idx = np.argmax(preds)
    assert 0 <= best_idx < len(preds)
    assert preds[best_idx] >= preds.min()


# ============================================================================
# INTEGRATION TESTS (6 tests)
# ============================================================================


def test_end_to_end_pipeline():
    """Test complete workflow: generate → train → evaluate → predict → save → load."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    # 1. Generate data
    df = generate_yield_process(n=200, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 2. Train
    pipeline = YieldRegressionPipeline(model="ridge", alpha=1.0)
    pipeline.fit(X_train, y_train)

    # 3. Evaluate
    metrics = pipeline.evaluate(X_test, y_test)
    assert "R2" in metrics
    assert metrics["R2"] > -1.0

    # 4. Predict
    preds = pipeline.predict(X_test)
    assert len(preds) == len(y_test)

    # 5. Save
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "e2e_model.joblib"
        pipeline.save(model_path)
        assert model_path.exists()

        # 6. Load
        loaded = YieldRegressionPipeline.load(model_path)
        loaded_preds = loaded.predict(X_test)

        # Verify consistency
        np.testing.assert_array_almost_equal(preds, loaded_preds)


def test_cli_integration():
    """Test CLI commands in sequence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "cli_model.joblib"

        # 1. Train
        train_out = run_cmd(
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
        assert train_out["status"] == "trained"
        assert model_path.exists()

        # 2. Evaluate
        eval_out = run_cmd(
            [
                "evaluate",
                "--model-path",
                str(model_path),
                "--dataset",
                "synthetic_yield",
            ]
        )
        assert eval_out["status"] == "evaluated"

        # 3. Predict
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
        pred_out = run_cmd(
            [
                "predict",
                "--model-path",
                str(model_path),
                "--input-json",
                json.dumps(input_data),
            ]
        )
        assert "prediction" in pred_out


def test_model_versioning():
    """Test saving multiple model versions."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=100, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Train and save multiple versions
        models = []
        for i, alpha in enumerate([0.1, 1.0, 10.0]):
            pipeline = YieldRegressionPipeline(model="ridge", alpha=alpha)
            pipeline.fit(X, y)

            model_path = tmp_path / f"model_v{i}_alpha{alpha}.joblib"
            pipeline.save(model_path)
            models.append((model_path, alpha))

        # Load and verify each version
        for model_path, expected_alpha in models:
            loaded = YieldRegressionPipeline.load(model_path)
            assert loaded.alpha == expected_alpha


def test_production_simulation():
    """Test realistic production scenario with batch predictions."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    # Train on historical data
    df_historical = generate_yield_process(n=500, seed=42)
    X_historical = df_historical.drop(columns=["yield_pct"])
    y_historical = df_historical["yield_pct"].values

    pipeline = YieldRegressionPipeline(model="rf")
    pipeline.fit(X_historical, y_historical)

    # Simulate incoming production batches
    for batch_id in range(5):
        df_batch = generate_yield_process(n=50, seed=100 + batch_id)
        X_batch = df_batch.drop(columns=["yield_pct"])
        y_batch = df_batch["yield_pct"].values

        # Predict
        preds_batch = pipeline.predict(X_batch)

        # Evaluate batch
        metrics_batch = pipeline.evaluate(X_batch, y_batch)

        # Check batch quality
        assert metrics_batch["PWS"] > 0.8  # Expect high PWS
        assert preds_batch.mean() > 60  # Reasonable yield


def test_cross_model_comparison():
    """Test comparing different models on same data."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=300, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    models_to_test = ["linear", "ridge", "lasso", "elasticnet", "rf"]
    results = []

    for model_type in models_to_test:
        pipeline = YieldRegressionPipeline(model=model_type)
        pipeline.fit(X_train, y_train)
        metrics = pipeline.evaluate(X_test, y_test)

        results.append(
            {
                "model": model_type,
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "PWS": metrics["PWS"],
            }
        )

    # Verify all models completed
    assert len(results) == 5

    # Find best model by R²
    best_model = max(results, key=lambda x: x["R2"])
    assert best_model["R2"] > -1.0


def test_hyperparameter_robustness():
    """Test that different hyperparameters don't break pipeline."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=100, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    # Test various hyperparameter combinations
    configs = [
        {"alpha": 0.001, "k_best": 4, "pca_components": 0.95},
        {"alpha": 100.0, "k_best": 8, "pca_components": 0.5},
        {"alpha": 1.0, "use_feature_selection": False, "use_pca": False},
    ]

    for config in configs:
        pipeline = YieldRegressionPipeline(model="ridge", **config)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        assert len(preds) == len(y)
        assert np.all(np.isfinite(preds))


# ============================================================================
# PERFORMANCE BENCHMARK TESTS (5 tests)
# ============================================================================


def test_training_speed():
    """Verify training completes within time limit."""
    import time
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=1000, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    pipeline = YieldRegressionPipeline(model="ridge")

    start_time = time.time()
    pipeline.fit(X, y)
    elapsed = time.time() - start_time

    # Training should complete in under 5 seconds
    assert elapsed < 5.0


def test_prediction_speed():
    """Verify predictions are fast enough for real-time use."""
    import time
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=500, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    pipeline = YieldRegressionPipeline(model="ridge")
    pipeline.fit(X, y)

    # Test single prediction latency
    X_single = X.iloc[:1]

    start_time = time.time()
    for _ in range(100):  # 100 predictions
        pipeline.predict(X_single)
    elapsed = time.time() - start_time

    # Average latency should be < 10ms per prediction
    avg_latency = elapsed / 100
    assert avg_latency < 0.01


def test_memory_usage():
    """Verify pipeline doesn't exceed memory limits."""
    import sys
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=5000, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    pipeline = YieldRegressionPipeline(model="rf")

    # Get size before training
    size_before = sys.getsizeof(pipeline)

    pipeline.fit(X, y)

    # Get size after training
    size_after = sys.getsizeof(pipeline)

    # Pipeline size should be reasonable (< 10 MB)
    size_increase = size_after - size_before
    assert size_increase < 10 * 1024 * 1024  # 10 MB


def test_model_file_size():
    """Verify saved model size is reasonable."""
    from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

    df = generate_yield_process(n=500, seed=42)
    X = df.drop(columns=["yield_pct"])
    y = df["yield_pct"].values

    pipeline = YieldRegressionPipeline(model="rf")
    pipeline.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "model.joblib"
        pipeline.save(model_path)

        # Check file size
        file_size = model_path.stat().st_size
        # Should be under 5 MB
        assert file_size < 5 * 1024 * 1024


def test_reproducibility_with_seed():
    """Test that results are reproducible with fixed seed."""
    from yield_regression_pipeline import generate_yield_process

    # Generate same data multiple times
    datasets = [generate_yield_process(n=100, seed=123) for _ in range(3)]

    # All should be identical
    for i in range(1, len(datasets)):
        pd.testing.assert_frame_equal(datasets[0], datasets[i])

    # Different seeds should produce different data
    df1 = generate_yield_process(n=100, seed=123)
    df2 = generate_yield_process(n=100, seed=456)

    # Should not be identical
    assert not df1.equals(df2)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
