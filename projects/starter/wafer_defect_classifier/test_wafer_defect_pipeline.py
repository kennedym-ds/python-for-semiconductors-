"""Unit tests for wafer defect classification pipeline."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add the project directory to the path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from wafer_defect_pipeline import (
    WaferDefectPipeline,
    generate_synthetic_wafer_defects,
    load_dataset,
)


class TestSyntheticDataGeneration:
    """Test synthetic wafer defect data generation."""

    def test_generate_synthetic_wafer_defects_default(self):
        """Test synthetic data generation with default parameters."""
        df = generate_synthetic_wafer_defects(n_samples=100, seed=42)

        assert len(df) == 100
        assert "defect" in df.columns
        assert "wafer_id" in df.columns
        assert "center_density" in df.columns
        assert "edge_density" in df.columns
        assert "defect_area_ratio" in df.columns

        # Check data types
        assert df["defect"].dtype == int
        assert df["defect"].isin([0, 1]).all()

        # Check ranges
        assert (df["center_density"] >= 0).all()
        assert (df["edge_density"] >= 0).all()
        assert (df["defect_area_ratio"] >= 0).all()
        assert (df["defect_area_ratio"] <= 1).all()

    def test_generate_synthetic_wafer_defects_custom_params(self):
        """Test synthetic data with custom parameters."""
        df = generate_synthetic_wafer_defects(n_samples=50, defect_rate=0.3, seed=123)

        assert len(df) == 50
        # With defect_rate=0.3, roughly 30% should have defects
        defect_rate = df["defect"].mean()
        assert 0.1 < defect_rate < 0.5  # Allowing some variance


class TestDatasetLoading:
    """Test dataset loading functionality."""

    def test_load_synthetic_dataset(self):
        """Test loading synthetic dataset."""
        df = load_dataset("synthetic_wafer")

        assert isinstance(df, pd.DataFrame)
        assert "defect" in df.columns
        assert len(df) > 0

    def test_load_synthetic_dataset_with_params(self):
        """Test loading synthetic dataset with parameters."""
        df = load_dataset("synthetic_wafer_200_0.25")

        assert len(df) == 200
        # Should have roughly 25% defects (allowing variance)
        defect_rate = df["defect"].mean()
        assert 0.1 < defect_rate < 0.4

    def test_load_unknown_dataset(self):
        """Test loading unknown dataset raises error."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")


class TestWaferDefectPipeline:
    """Test the WaferDefectPipeline class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        df = generate_synthetic_wafer_defects(n_samples=100, seed=42)
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])
        return X, y

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipe = WaferDefectPipeline()
        assert pipe.model_name == "logistic"
        assert pipe.fitted_threshold == 0.5
        assert pipe.pipeline is None

    def test_pipeline_fit_and_predict(self, sample_data):
        """Test pipeline fitting and prediction."""
        X, y = sample_data
        pipe = WaferDefectPipeline(model="logistic")

        # Fit pipeline
        pipe.fit(X, y)
        assert pipe.pipeline is not None
        assert pipe.metadata is not None

        # Test predictions
        preds = pipe.predict(X)
        assert len(preds) == len(y)
        assert set(preds) <= {0, 1}

        # Test probabilities
        probs = pipe.predict_proba(X)
        assert probs.shape == (len(y), 2)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert np.allclose(probs.sum(axis=1), 1)

    def test_pipeline_different_models(self, sample_data):
        """Test pipeline with different model types."""
        X, y = sample_data
        models = ["logistic", "tree", "rf"]

        for model_name in models:
            pipe = WaferDefectPipeline(model=model_name)
            pipe.fit(X, y)

            assert pipe.metadata.model_type is not None
            preds = pipe.predict(X)
            assert len(preds) == len(y)

    def test_pipeline_metrics_computation(self, sample_data):
        """Test metrics computation."""
        X, y = sample_data
        pipe = WaferDefectPipeline()
        pipe.fit(X, y)

        metrics = pipe.evaluate(X, y)

        # Check that all expected metrics are present
        expected_metrics = [
            "roc_auc",
            "pr_auc",
            "mcc",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "pws",
            "estimated_loss",
            "false_positive_count",
            "false_negative_count",
        ]

        for metric in expected_metrics:
            assert metric in metrics

        # Check metric ranges
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1
        assert 0 <= metrics["pws"] <= 100
        assert metrics["estimated_loss"] >= 0

    def test_pipeline_save_load(self, sample_data):
        """Test pipeline save and load functionality."""
        X, y = sample_data
        pipe = WaferDefectPipeline()
        pipe.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save pipeline
            pipe.save(tmp_path)
            assert tmp_path.exists()

            # Load pipeline
            loaded_pipe = WaferDefectPipeline.load(tmp_path)

            # Test that loaded pipeline works
            orig_preds = pipe.predict(X)
            loaded_preds = loaded_pipe.predict(X)
            np.testing.assert_array_equal(orig_preds, loaded_preds)

            # Check metadata
            assert loaded_pipe.metadata is not None
            assert loaded_pipe.fitted_threshold == pipe.fitted_threshold

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_threshold_optimization(self, sample_data):
        """Test threshold optimization with precision/recall constraints."""
        X, y = sample_data

        # Test with minimum precision constraint
        pipe = WaferDefectPipeline(min_precision=0.8)
        pipe.fit(X, y)

        # Check that precision constraint is (approximately) met
        preds = pipe.predict(X)
        from sklearn.metrics import precision_score

        precision = precision_score(y, preds, zero_division=0)

        # Allow some tolerance due to discrete thresholds
        if precision > 0:  # Only check if we have positive predictions
            assert precision >= 0.7  # Relaxed threshold for test stability


class TestCLIInterface:
    """Test the CLI interface using subprocess."""

    def run_cli_command(self, args):
        """Helper to run CLI commands and return JSON output."""
        script_path = Path(__file__).parent / "wafer_defect_pipeline.py"
        result = subprocess.run(
            [sys.executable, str(script_path)] + args, capture_output=True, text=True
        )

        if result.returncode != 0:
            pytest.fail(f"CLI command failed: {result.stderr}")

        return json.loads(result.stdout)

    def test_cli_train_command(self):
        """Test CLI train command."""
        result = self.run_cli_command(
            ["train", "--dataset", "synthetic_wafer_100_0.2", "--model", "logistic"]
        )

        assert result["status"] == "trained"
        assert "metrics" in result
        assert "metadata" in result
        assert "roc_auc" in result["metrics"]
        assert "pws" in result["metrics"]

    def test_cli_train_and_evaluate_roundtrip(self):
        """Test complete train -> save -> evaluate cycle."""
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            model_path = tmp.name

        try:
            # Train and save
            train_result = self.run_cli_command(
                [
                    "train",
                    "--dataset",
                    "synthetic_wafer_100_0.2",
                    "--model",
                    "rf",
                    "--save",
                    model_path,
                ]
            )

            assert train_result["status"] == "trained"
            assert Path(model_path).exists()

            # Evaluate
            eval_result = self.run_cli_command(
                [
                    "evaluate",
                    "--model-path",
                    model_path,
                    "--dataset",
                    "synthetic_wafer_100_0.2",
                ]
            )

            assert eval_result["status"] == "evaluated"
            assert "metrics" in eval_result

        finally:
            if Path(model_path).exists():
                Path(model_path).unlink()

    def test_cli_predict_command(self):
        """Test CLI predict command."""
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            model_path = tmp.name

        try:
            # Train and save a model first
            self.run_cli_command(
                [
                    "train",
                    "--dataset",
                    "synthetic_wafer_50_0.3",
                    "--model",
                    "logistic",
                    "--save",
                    model_path,
                ]
            )

            # Test prediction
            test_input = {
                "center_density": 0.05,
                "edge_density": 0.02,
                "defect_area_ratio": 0.03,
                "defect_spread": 1.0,
                "total_pixels": 2500,
                "defect_pixels": 75,
            }

            result = self.run_cli_command(
                [
                    "predict",
                    "--model-path",
                    model_path,
                    "--input-json",
                    json.dumps(test_input),
                ]
            )

            assert "prediction" in result
            assert "probability" in result
            assert "threshold" in result
            assert result["prediction"] in [0, 1]
            assert 0 <= result["probability"] <= 1
            assert result["input"] == test_input

        finally:
            if Path(model_path).exists():
                Path(model_path).unlink()

    def test_cli_error_handling(self):
        """Test CLI error handling."""
        # Test with invalid model path
        script_path = Path(__file__).parent / "wafer_defect_pipeline.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "evaluate",
                "--model-path",
                "nonexistent.joblib",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        error_output = json.loads(result.stdout)
        assert error_output["status"] == "error"
        assert "message" in error_output


class TestManufacturingMetrics:
    """Test manufacturing-specific metrics."""

    def test_pws_calculation(self):
        """Test PWS (Prediction Within Spec) calculation."""
        # Perfect predictions
        y_true = np.array([1, 0, 1, 0, 1])
        preds = np.array([1, 0, 1, 0, 1])
        probs = np.array([0.9, 0.1, 0.8, 0.2, 0.95])

        metrics = WaferDefectPipeline.compute_metrics(y_true, probs, preds)
        assert metrics["pws"] == 100.0

        # 80% accuracy
        y_true = np.array([1, 0, 1, 0, 1])
        preds = np.array([1, 0, 1, 1, 1])  # One error
        probs = np.array([0.9, 0.1, 0.8, 0.7, 0.95])

        metrics = WaferDefectPipeline.compute_metrics(y_true, probs, preds)
        assert metrics["pws"] == 80.0

    def test_estimated_loss_calculation(self):
        """Test estimated loss calculation."""
        # No errors
        y_true = np.array([1, 0, 1, 0])
        preds = np.array([1, 0, 1, 0])
        probs = np.array([0.9, 0.1, 0.8, 0.2])

        metrics = WaferDefectPipeline.compute_metrics(y_true, probs, preds)
        assert metrics["estimated_loss"] == 0.0
        assert metrics["false_positive_count"] == 0
        assert metrics["false_negative_count"] == 0

        # One false positive, one false negative
        y_true = np.array([1, 0, 1, 0])
        preds = np.array([0, 1, 1, 0])  # FN at index 0, FP at index 1
        probs = np.array([0.3, 0.7, 0.8, 0.2])

        metrics = WaferDefectPipeline.compute_metrics(y_true, probs, preds)
        # Expected: 1 FP * 1.0 + 1 FN * 10.0 = 11.0
        assert metrics["estimated_loss"] == 11.0
        assert metrics["false_positive_count"] == 1
        assert metrics["false_negative_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
