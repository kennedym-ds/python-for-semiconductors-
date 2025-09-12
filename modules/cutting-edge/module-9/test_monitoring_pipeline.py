"""Tests for Module 9.2 Monitoring & Maintenance Pipeline"""

import json
import subprocess
import sys
import tempfile
import os
from pathlib import Path

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / "9.2-monitoring-maintenance-pipeline.py"


def run_cmd(args):
    """Run pipeline command and return JSON output."""
    result = subprocess.run([sys.executable, str(SCRIPT)] + args, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_train_basic():
    """Test basic training without MLflow."""
    out = run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge"])
    assert out["status"] == "trained"
    assert out["model_type"] == "ridge"
    assert "metrics" in out
    assert "mae" in out["metrics"]
    assert "rmse" in out["metrics"]
    assert "r2" in out["metrics"]
    assert "pws_percent" in out["metrics"]
    assert "estimated_loss" in out["metrics"]
    assert out["mlflow_enabled"] is False


def test_train_with_drift_injection():
    """Test training with synthetic drift injection."""
    out = run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge", "--inject-drift"])
    assert out["status"] == "trained"
    assert out["drift_injected"] is True
    assert "drift_scores" in out["metrics"]
    assert "alert_flags" in out["metrics"]


def test_train_with_mlflow():
    """Test training with MLflow enabled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Set MLflow tracking URI to temporary directory
        os.environ["MLFLOW_TRACKING_URI"] = f"file://{tmp_dir}/mlruns"

        out = run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge", "--enable-mlflow"])
        assert out["status"] == "trained"
        assert out["mlflow_enabled"] is True

        # Clean up environment
        if "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]


def test_train_and_save():
    """Test training and saving model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"

        out = run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge", "--save", str(model_path)])
        assert out["status"] == "trained"
        assert model_path.exists()


def test_evaluate_saved_model():
    """Test evaluating a saved model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"

        # First train and save a model
        run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge", "--save", str(model_path)])

        # Then evaluate it
        out = run_cmd(["evaluate", "--model-path", str(model_path), "--dataset", "synthetic_yield"])
        assert out["status"] == "evaluated"
        assert "metrics" in out
        assert "mae" in out["metrics"]
        assert "drift_scores" in out["metrics"]
        assert "alert_flags" in out["metrics"]


def test_evaluate_with_drift_detection():
    """Test drift detection during evaluation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"

        # Train model with normal data
        run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge", "--save", str(model_path)])

        # Evaluate with drift injection
        out = run_cmd(["evaluate", "--model-path", str(model_path), "--dataset", "synthetic_yield", "--inject-drift"])
        assert out["status"] == "evaluated"
        assert out["drift_injected"] is True

        # Check that drift was detected
        drift_scores = out["metrics"]["drift_scores"]
        alert_flags = out["metrics"]["alert_flags"]

        # Should have drift scores for temperature and pressure
        assert any("temperature" in key for key in drift_scores.keys())
        assert any("pressure" in key for key in drift_scores.keys())

        # At least some alerts should be triggered with injected drift
        assert any(alert_flags.values())


def test_predict_single_json():
    """Test single prediction with JSON input."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"

        # Train and save model
        run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge", "--save", str(model_path)])

        # Make prediction
        input_json = (
            '{"temperature": 455, "pressure": 2.6, "flow": 118, "time": 62, '
            '"temp_centered": 5, "pressure_sq": 6.76, "flow_time_inter": 7316, "temp_flow_inter": 53690}'
        )
        out = run_cmd(["predict", "--model-path", str(model_path), "--input-json", input_json])

        assert out["status"] == "predicted"
        assert "prediction" in out
        assert isinstance(out["prediction"], (int, float))
        assert out["n_predictions"] == 1


def test_predict_csv_file():
    """Test batch predictions with CSV file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"
        csv_path = Path(tmp_dir) / "test_input.csv"

        # Train and save model
        run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge", "--save", str(model_path)])

        # Create test CSV
        csv_content = """temperature,pressure,flow,time,temp_centered,pressure_sq,flow_time_inter,temp_flow_inter
455,2.6,118,62,5,6.76,7316,53690
460,2.7,120,63,10,7.29,7560,55200"""

        with open(csv_path, "w") as f:
            f.write(csv_content)

        # Make batch predictions
        out = run_cmd(["predict", "--model-path", str(model_path), "--input-file", str(csv_path)])

        assert out["status"] == "predicted"
        assert "predictions" in out
        assert len(out["predictions"]) == 2
        assert out["n_predictions"] == 2


def test_different_models():
    """Test training with different model types."""
    models = ["ridge", "lasso", "random_forest"]

    for model_type in models:
        out = run_cmd(["train", "--dataset", "synthetic_yield", "--model", model_type])
        assert out["status"] == "trained"
        assert out["model_type"] == model_type


def test_drift_metrics_calculation():
    """Test that drift metrics are properly calculated."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"

        # Train model
        run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge", "--save", str(model_path)])

        # Evaluate with drift
        out = run_cmd(["evaluate", "--model-path", str(model_path), "--dataset", "synthetic_yield", "--inject-drift"])

        drift_scores = out["metrics"]["drift_scores"]

        # Check that PSI, KS, and Wasserstein scores are calculated
        psi_scores = [k for k in drift_scores.keys() if "psi" in k]
        ks_scores = [k for k in drift_scores.keys() if "ks_stat" in k]
        wasserstein_scores = [k for k in drift_scores.keys() if "wasserstein" in k]

        assert len(psi_scores) > 0
        assert len(ks_scores) > 0
        assert len(wasserstein_scores) > 0


def test_manufacturing_metrics():
    """Test that manufacturing-specific metrics are calculated."""
    out = run_cmd(["train", "--dataset", "synthetic_yield", "--model", "ridge"])

    metrics = out["metrics"]

    # Check PWS (Prediction Within Spec) is calculated
    assert "pws_percent" in metrics
    assert 0 <= metrics["pws_percent"] <= 100

    # Check Estimated Loss is calculated
    assert "estimated_loss" in metrics
    assert metrics["estimated_loss"] >= 0


def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test with non-existent model file
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "evaluate", "--model-path", "nonexistent.joblib"], capture_output=True, text=True
    )
    assert result.returncode == 1
    output = json.loads(result.stdout)
    assert output["status"] == "error"

    # Test prediction with invalid JSON
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "predict", "--model-path", "nonexistent.joblib", "--input-json", "invalid json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_help_commands():
    """Test that help commands work without errors."""
    # Test main help
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Module 9.2 Monitoring & Maintenance Pipeline" in result.stdout

    # Test subcommand help
    for cmd in ["train", "evaluate", "predict"]:
        result = subprocess.run([sys.executable, str(SCRIPT), cmd, "--help"], capture_output=True, text=True)
        assert result.returncode == 0


if __name__ == "__main__":
    # Run all tests
    test_functions = [func for name, func in globals().items() if name.startswith("test_")]

    print(f"Running {len(test_functions)} tests...")

    for i, test_func in enumerate(test_functions, 1):
        try:
            print(f"[{i}/{len(test_functions)}] {test_func.__name__}...", end=" ")
            test_func()
            print("PASS")
        except Exception as e:
            print(f"FAIL: {e}")
            sys.exit(1)

    print("All tests passed!")
