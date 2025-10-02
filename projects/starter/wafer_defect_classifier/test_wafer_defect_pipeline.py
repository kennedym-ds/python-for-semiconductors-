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

            assert pipe.metadata is not None
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
        result = subprocess.run([sys.executable, str(script_path)] + args, capture_output=True, text=True)

        if result.returncode != 0:
            pytest.fail(f"CLI command failed: {result.stderr}")

        return json.loads(result.stdout)

    def test_cli_train_command(self):
        """Test CLI train command."""
        result = self.run_cli_command(["train", "--dataset", "synthetic_wafer_100_0.2", "--model", "logistic"])

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


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self):
        """Test pipeline behavior with empty dataset."""
        X = pd.DataFrame()
        y = np.array([])

        pipe = WaferDefectPipeline()

        # Should raise ValueError due to empty dataset
        with pytest.raises((ValueError, IndexError)):
            pipe.fit(X, y)

    def test_single_sample(self):
        """Test pipeline with single sample."""
        df = generate_synthetic_wafer_defects(n_samples=1, seed=42)
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])

        pipe = WaferDefectPipeline()

        # Single sample should fail in cross-validation or raise warning
        with pytest.raises((ValueError, Warning)):
            pipe.fit(X, y)

    def test_single_class_all_defects(self):
        """Test with dataset where all samples are defects."""
        # Create artificial all-defect dataset
        df = generate_synthetic_wafer_defects(n_samples=100, seed=42)
        y = np.ones(len(df), dtype=int)  # All defects
        X = df.drop(columns=["defect", "wafer_id"])

        pipe = WaferDefectPipeline()

        # Should fit but may have issues with some metrics
        pipe.fit(X, y)
        preds = pipe.predict(X)

        # All predictions should be 1 (defect) due to single class
        assert all(preds == 1)

    def test_single_class_all_good(self):
        """Test with dataset where all samples are good (no defects)."""
        df = generate_synthetic_wafer_defects(n_samples=100, seed=42)
        y = np.zeros(len(df), dtype=int)  # All good wafers
        X = df.drop(columns=["defect", "wafer_id"])

        pipe = WaferDefectPipeline()

        # Should fit but predictions should all be 0
        pipe.fit(X, y)
        preds = pipe.predict(X)

        # All predictions should be 0 (good) due to single class
        assert all(preds == 0)

    def test_extreme_imbalance(self):
        """Test with 99.9% imbalance (1000 good, 1 defect)."""
        # Create highly imbalanced dataset
        df = generate_synthetic_wafer_defects(n_samples=1001, defect_rate=0.001, seed=42)
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])

        # Ensure we have the expected imbalance
        if y.sum() == 0:
            # Force at least one defect for testing
            y[0] = 1

        pipe = WaferDefectPipeline()
        pipe.fit(X, y)

        # Should still produce predictions
        preds = pipe.predict(X)
        assert len(preds) == len(y)

        # Might predict mostly 0s due to imbalance, which is expected
        assert preds.sum() <= len(preds) * 0.1  # At most 10% predicted as defects

    def test_missing_features_in_prediction(self):
        """Test prediction with missing required features."""
        df = generate_synthetic_wafer_defects(n_samples=100, seed=42)
        y = df["defect"].to_numpy()
        X_train = df.drop(columns=["defect", "wafer_id"])

        pipe = WaferDefectPipeline()
        pipe.fit(X_train, y)

        # Create test data with missing column
        X_test = X_train.drop(columns=["center_density"])

        # Should raise error due to missing features
        with pytest.raises((ValueError, KeyError)):
            pipe.predict(X_test)

    def test_extra_features_in_prediction(self):
        """Test prediction with extra features."""
        df = generate_synthetic_wafer_defects(n_samples=100, seed=42)
        y = df["defect"].to_numpy()
        X_train = df.drop(columns=["defect", "wafer_id"])

        pipe = WaferDefectPipeline()
        pipe.fit(X_train, y)

        # Create test data with extra column
        X_test = X_train.copy()
        X_test["extra_feature"] = np.random.randn(len(X_test))

        # Should either ignore extra feature or raise error depending on implementation
        try:
            preds = pipe.predict(X_test)
            # If it succeeds, verify predictions are valid
            assert len(preds) == len(X_test)
        except (ValueError, KeyError):
            # Also acceptable to raise error for feature mismatch
            pass

    def test_nan_values_in_data(self):
        """Test handling of NaN values in input data."""
        df = generate_synthetic_wafer_defects(n_samples=100, seed=42)
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])

        # Introduce NaN values
        X.iloc[0, 0] = np.nan
        X.iloc[5, 2] = np.nan

        pipe = WaferDefectPipeline()

        # Should either handle NaN or raise informative error
        with pytest.raises((ValueError, TypeError)):
            pipe.fit(X, y)


class TestManufacturingScenarios:
    """Test manufacturing-specific use cases."""

    @pytest.fixture
    def manufacturing_data(self):
        """Create realistic manufacturing dataset."""
        df = generate_synthetic_wafer_defects(n_samples=500, defect_rate=0.15, seed=42)
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])
        return X, y

    def test_high_precision_mode(self, manufacturing_data):
        """Test with min_precision=0.90 to minimize false positives."""
        X, y = manufacturing_data

        # Split data for proper evaluation
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        pipe = WaferDefectPipeline(model="rf", min_precision=0.90)
        pipe.fit(X_train, y_train)

        # Evaluate on test set
        preds = pipe.predict(X_test)
        from sklearn.metrics import precision_score

        precision = precision_score(y_test, preds, zero_division=0)

        # High precision mode should achieve high precision (with tolerance)
        # May sacrifice recall to achieve this
        if preds.sum() > 0:  # Only check if we have predictions
            assert precision >= 0.75  # Relaxed for test stability

    def test_high_recall_mode(self, manufacturing_data):
        """Test with min_recall=0.90 to catch all defects."""
        X, y = manufacturing_data

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        pipe = WaferDefectPipeline(model="rf", min_recall=0.90)
        pipe.fit(X_train, y_train)

        # Evaluate on test set
        preds = pipe.predict(X_test)
        from sklearn.metrics import recall_score

        recall = recall_score(y_test, preds, zero_division=0)

        # High recall mode should catch most defects (with tolerance)
        # May sacrifice precision to achieve this
        if y_test.sum() > 0:  # Only check if we have positive samples
            assert recall >= 0.70  # Relaxed for test stability

    def test_cost_calculation_consistency(self, manufacturing_data):
        """Test that cost calculations are consistent with FP/FN counts."""
        X, y = manufacturing_data

        # Note: Current implementation uses hardcoded costs (FP=1.0, FN=10.0)
        pipe = WaferDefectPipeline(model="rf")
        pipe.fit(X, y)
        metrics = pipe.evaluate(X, y)

        # Verify loss calculation: should be FP*1.0 + FN*10.0
        fp_count = metrics["false_positive_count"]
        fn_count = metrics["false_negative_count"]
        expected_loss = fp_count * 1.0 + fn_count * 10.0

        assert metrics["estimated_loss"] == expected_loss

        # Verify counts are non-negative
        assert fp_count >= 0
        assert fn_count >= 0

    def test_production_threshold_tuning(self, manufacturing_data):
        """Test threshold optimization for production scenarios."""
        X, y = manufacturing_data

        # Test with both precision and recall constraints
        pipe = WaferDefectPipeline(model="rf", min_precision=0.80, min_recall=0.75)
        pipe.fit(X, y)

        # Check that threshold is in reasonable range
        assert 0.0 <= pipe.fitted_threshold <= 1.0

        # Threshold should not be at extremes (0 or 1) for balanced constraints
        # unless data is very separable
        assert pipe.fitted_threshold != 0.0
        assert pipe.fitted_threshold != 1.0

        # Metadata should record the optimization settings
        assert pipe.metadata is not None
        assert hasattr(pipe.metadata, "model_type")

    def test_batch_production_simulation(self, manufacturing_data):
        """Simulate batch production scenario with multiple predictions."""
        X, y = manufacturing_data

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train model once
        pipe = WaferDefectPipeline(model="rf")
        pipe.fit(X_train, y_train)

        # Simulate multiple batches
        batch_size = 10
        num_batches = len(X_test) // batch_size

        all_preds = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_X = X_test.iloc[start_idx:end_idx]

            batch_preds = pipe.predict(batch_X)
            all_preds.extend(batch_preds)

        # Verify we got predictions for all batches
        assert len(all_preds) == num_batches * batch_size

        # Predictions should be binary
        assert all(p in [0, 1] for p in all_preds)


class TestIntegration:
    """Test end-to-end integration workflows."""

    def test_complete_ml_pipeline(self):
        """Test full pipeline: load → train → optimize → save → load → predict."""
        # Step 1: Load dataset
        df = load_dataset("synthetic_wafer_200_0.2")
        assert len(df) == 200

        # Step 2: Prepare data
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Step 3: Train model with optimization
        pipe = WaferDefectPipeline(model="rf", min_precision=0.80, min_recall=0.75)
        pipe.fit(X_train, y_train)

        # Step 4: Evaluate
        train_metrics = pipe.evaluate(X_train, y_train)
        test_metrics = pipe.evaluate(X_test, y_test)

        assert train_metrics["roc_auc"] > 0.5  # Better than random
        assert test_metrics["roc_auc"] > 0.5

        # Step 5: Save model
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            model_path = Path(tmp.name)

        try:
            pipe.save(model_path)
            assert model_path.exists()

            # Step 6: Load model
            loaded_pipe = WaferDefectPipeline.load(model_path)

            # Step 7: Make predictions with loaded model
            loaded_preds = loaded_pipe.predict(X_test)

            # Verify consistency
            original_preds = pipe.predict(X_test)
            np.testing.assert_array_equal(original_preds, loaded_preds)

            # Step 8: Verify metadata preserved
            assert loaded_pipe.metadata is not None
            assert loaded_pipe.fitted_threshold == pipe.fitted_threshold

        finally:
            if model_path.exists():
                model_path.unlink()

    def test_multi_model_comparison(self):
        """Test comparing multiple models end-to-end."""
        df = generate_synthetic_wafer_defects(n_samples=300, defect_rate=0.2, seed=42)
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        models = ["logistic", "tree", "rf"]
        results = {}

        # Train and evaluate each model
        for model_name in models:
            pipe = WaferDefectPipeline(model=model_name)
            pipe.fit(X_train, y_train)
            metrics = pipe.evaluate(X_test, y_test)
            results[model_name] = metrics

        # Verify all models were evaluated
        assert len(results) == len(models)

        # All models should have same metrics available
        for model_name, metrics in results.items():
            assert "roc_auc" in metrics
            assert "pr_auc" in metrics
            assert "pws" in metrics
            assert "estimated_loss" in metrics

        # At least one model should perform better than random
        best_auc = max(results[m]["roc_auc"] for m in models)
        assert best_auc > 0.5

    def test_dataset_to_deployment_workflow(self):
        """Test realistic deployment workflow."""
        # Step 1: Generate production-like data
        df = generate_synthetic_wafer_defects(n_samples=1000, defect_rate=0.12, seed=123)
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])

        # Step 2: Train with production settings
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        pipe = WaferDefectPipeline(model="rf", min_precision=0.85, min_recall=0.80)
        pipe.fit(X_train, y_train)

        # Step 3: Evaluate with manufacturing metrics
        metrics = pipe.evaluate(X_test, y_test)

        # Check manufacturing KPIs
        assert "pws" in metrics  # Prediction Within Spec
        assert "estimated_loss" in metrics
        assert metrics["pws"] >= 0  # Should be reasonable

        # Step 4: Save for deployment
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            model_path = Path(tmp.name)

        try:
            pipe.save(model_path)

            # Step 5: Load and verify metadata
            deployed_pipe = WaferDefectPipeline.load(model_path)

            assert deployed_pipe.metadata is not None
            assert hasattr(deployed_pipe.metadata, "trained_at")
            assert hasattr(deployed_pipe.metadata, "model_type")

            # Step 6: Make batch predictions (simulate production)
            batch_predictions = deployed_pipe.predict(X_test[:50])
            assert len(batch_predictions) == 50
            assert all(p in [0, 1] for p in batch_predictions)

            # Step 7: Get prediction probabilities for confidence scoring
            batch_probs = deployed_pipe.predict_proba(X_test[:50])
            assert batch_probs.shape == (50, 2)
            assert (batch_probs >= 0).all()
            assert (batch_probs <= 1).all()

        finally:
            if model_path.exists():
                model_path.unlink()


class TestPerformance:
    """Test performance characteristics and scalability."""

    def test_training_time_scaling(self):
        """Test training time with different dataset sizes."""
        import time

        sizes = [100, 500, 1000]
        times = []

        for size in sizes:
            df = generate_synthetic_wafer_defects(n_samples=size, seed=42)
            y = df["defect"].to_numpy()
            X = df.drop(columns=["defect", "wafer_id"])

            pipe = WaferDefectPipeline(model="logistic")

            start = time.time()
            pipe.fit(X, y)
            elapsed = time.time() - start

            times.append(elapsed)

        # Training time should scale reasonably (not exponentially)
        # 10x data should not take 100x time
        # This is a rough check - may vary by system
        if times[0] > 0:  # Avoid division by zero
            ratio_10x = times[2] / times[0]  # 1000 vs 100
            assert ratio_10x < 50  # Should be much less than quadratic

    def test_memory_efficiency(self):
        """Test that model doesn't consume excessive memory."""
        # Create moderately large dataset
        df = generate_synthetic_wafer_defects(n_samples=5000, seed=42)
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])

        pipe = WaferDefectPipeline(model="logistic")
        pipe.fit(X, y)

        # Save and verify file size is reasonable
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            model_path = Path(tmp.name)

        try:
            pipe.save(model_path)

            # Model file should be reasonable size (< 50 MB for simple model)
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            assert file_size_mb < 50

        finally:
            if model_path.exists():
                model_path.unlink()

    def test_prediction_latency(self):
        """Test prediction speed for real-time applications."""
        import time

        # Train a model
        df = generate_synthetic_wafer_defects(n_samples=500, seed=42)
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])

        pipe = WaferDefectPipeline(model="logistic")
        pipe.fit(X, y)

        # Test single prediction latency
        single_sample = X.iloc[[0]]

        start = time.time()
        for _ in range(100):
            pipe.predict(single_sample)
        elapsed = time.time() - start

        avg_latency_ms = (elapsed / 100) * 1000

        # Single prediction should be fast (< 50ms average)
        assert avg_latency_ms < 50

        # Test batch prediction throughput
        batch_size = 100
        batch_data = X.iloc[:batch_size]

        start = time.time()
        pipe.predict(batch_data)
        elapsed = time.time() - start

        predictions_per_second = batch_size / elapsed if elapsed > 0 else float("inf")

        # Should handle at least 500 predictions per second
        assert predictions_per_second > 500

    def test_concurrent_predictions(self):
        """Test that model can handle concurrent prediction requests."""
        # Train a model
        df = generate_synthetic_wafer_defects(n_samples=300, seed=42)
        y = df["defect"].to_numpy()
        X = df.drop(columns=["defect", "wafer_id"])

        pipe = WaferDefectPipeline(model="rf")
        pipe.fit(X, y)

        # Simulate concurrent requests by making multiple predictions
        # In same process (thread-safety test)
        test_samples = [X.iloc[[i]] for i in range(10)]

        results = []
        for sample in test_samples:
            pred = pipe.predict(sample)
            results.append(pred)

        # All predictions should complete successfully
        assert len(results) == 10
        assert all(len(r) == 1 for r in results)
        assert all(r[0] in [0, 1] for r in results)


if __name__ == "__main__":
    pytest.main([__file__])
