"""Tests for Module 9.1 Model Deployment Pipeline."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import shutil

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / "9.1-model-deployment-pipeline.py"


def run_cmd(args):
    """Run CLI command and return parsed JSON output."""
    result = subprocess.run([sys.executable, str(SCRIPT)] + args, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def create_test_model(path: Path):
    """Create a simple test model for testing."""
    # Create a simple linear regression model
    X = np.random.rand(100, 4)
    y = X.sum(axis=1) + np.random.normal(0, 0.1, 100)

    model = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
    model.fit(X, y)

    # Save the model
    joblib.dump(model, path)
    return model


def test_export_model():
    """Test model export functionality."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "test_model.joblib"
        output_dir = tmp_path / "deployment"

        # Create test model
        create_test_model(model_path)

        # Test export
        result = run_cmd(
            ["export", "--model-path", str(model_path), "--output-dir", str(output_dir), "--version", "1.0.0"]
        )

        assert result["status"] == "exported"
        assert result["version"] == "1.0.0"
        assert "model_hash" in result

        # Check output files exist
        assert (output_dir / "model.joblib").exists()
        assert (output_dir / "metadata.json").exists()


def test_validate_deployment():
    """Test deployment validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "test_model.joblib"
        output_dir = tmp_path / "deployment"

        # Create and export model
        create_test_model(model_path)
        export_result = run_cmd(
            ["export", "--model-path", str(model_path), "--output-dir", str(output_dir), "--version", "1.0.0"]
        )

        # Test validation
        result = run_cmd(["validate", "--deployment-dir", str(output_dir)])

        assert result["status"] == "valid"
        assert result["issues"] == []


def test_validate_invalid_deployment():
    """Test validation with missing files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        empty_dir = tmp_path / "empty_deployment"
        empty_dir.mkdir()

        # Test validation of empty directory
        result = run_cmd(["validate", "--deployment-dir", str(empty_dir)])

        assert result["status"] == "invalid"
        assert len(result["issues"]) > 0
        assert any("Missing model.joblib" in issue for issue in result["issues"])


def test_package_model():
    """Test model packaging with automatic naming."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "test_model.joblib"

        # Create test model
        create_test_model(model_path)

        # Test packaging
        result = run_cmd(["package", "--model-path", str(model_path), "--version", "2.1.0"])

        assert result["status"] == "exported"
        assert result["version"] == "2.1.0"

        # Check auto-generated directory name
        expected_dir = tmp_path / "deployment_v2_1_0"
        assert expected_dir.exists()
        assert (expected_dir / "model.joblib").exists()
        assert (expected_dir / "metadata.json").exists()


def test_package_model_custom_name():
    """Test model packaging with custom output name."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "test_model.joblib"

        # Create test model
        create_test_model(model_path)

        # Test packaging with custom name
        result = run_cmd(
            ["package", "--model-path", str(model_path), "--version", "1.5.0", "--output-name", "my_custom_deployment"]
        )

        assert result["status"] == "exported"
        assert result["version"] == "1.5.0"

        # Check custom directory name
        expected_dir = tmp_path / "my_custom_deployment"
        assert expected_dir.exists()


def test_roundtrip_export_validate():
    """Test complete export and validate workflow."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "test_model.joblib"
        output_dir = tmp_path / "deployment"

        # Create test model
        create_test_model(model_path)

        # Export model
        export_result = run_cmd(
            ["export", "--model-path", str(model_path), "--output-dir", str(output_dir), "--version", "1.2.3"]
        )

        # Validate exported model
        validate_result = run_cmd(["validate", "--deployment-dir", str(output_dir)])

        assert export_result["status"] == "exported"
        assert validate_result["status"] == "valid"
        assert export_result["version"] == "1.2.3"

        # Check metadata content
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["version"] == "1.2.3"
        assert "model_hash" in metadata
        assert "input_schema" in metadata
        assert "output_schema" in metadata


def test_error_handling_missing_model():
    """Test error handling for missing model file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        missing_model = tmp_path / "missing_model.joblib"
        output_dir = tmp_path / "deployment"

        # Try to export non-existent model
        try:
            result = run_cmd(["export", "--model-path", str(missing_model), "--output-dir", str(output_dir)])
            assert False, "Should have failed with missing model"
        except subprocess.CalledProcessError as e:
            # Parse error output
            error_output = json.loads(e.stdout)
            assert error_output["status"] == "error"
            assert "not found" in error_output["message"].lower()


def test_metadata_generation():
    """Test that metadata is generated correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "test_model.joblib"
        output_dir = tmp_path / "deployment"

        # Create test model
        create_test_model(model_path)

        # Export model
        result = run_cmd(
            ["export", "--model-path", str(model_path), "--output-dir", str(output_dir), "--version", "3.0.0"]
        )

        # Check metadata file content
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        required_fields = [
            "model_name",
            "version",
            "created_at",
            "model_type",
            "input_schema",
            "output_schema",
            "model_hash",
            "deployment_config",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"

        assert metadata["version"] == "3.0.0"
        assert isinstance(metadata["input_schema"], dict)
        assert isinstance(metadata["output_schema"], dict)
        assert isinstance(metadata["deployment_config"], dict)
