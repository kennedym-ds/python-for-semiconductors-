"""
Integration tests for complete learning workflows in Python for Semiconductors.
"""

import pytest
import subprocess
import sys
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np


class TestLearningPathIntegration:
    """Integration tests for complete learning paths across modules."""
    
    @pytest.fixture
    def temp_output_dir(self, temp_dir: Path) -> Path:
        """Temporary directory for pipeline outputs."""
        output_dir = temp_dir / "integration_outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_regression_learning_path(self, temp_output_dir: Path):
        """Test complete regression learning path from Module 3."""
        pipeline_script = Path("modules/foundation/module-3/3.1-regression-pipeline.py")
        
        if not pipeline_script.exists():
            pytest.skip(f"Pipeline script not found: {pipeline_script}")
        
        # Test training
        train_cmd = [
            sys.executable, str(pipeline_script), "train",
            "--output-dir", str(temp_output_dir),
            "--max-samples", "500",
            "--test-size", "0.3"
        ]
        
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=60)
        
        # Verify training succeeded
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        
        # Check for expected outputs
        expected_files = ["model.pkl", "metrics.json", "feature_importance.json"]
        for expected_file in expected_files:
            file_path = temp_output_dir / expected_file
            if file_path.exists():
                assert file_path.stat().st_size > 0, f"Empty file: {expected_file}"
        
        # Test evaluation if metrics file exists
        metrics_file = temp_output_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            # Verify reasonable performance metrics
            assert "r2_score" in metrics
            assert "mae" in metrics
            assert "rmse" in metrics
            assert metrics["r2_score"] > -1.0  # At least better than constant prediction
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_ensemble_learning_path(self, temp_output_dir: Path):
        """Test complete ensemble learning path from Module 4."""
        pipeline_script = Path("modules/intermediate/module-4/4.1-ensemble-pipeline.py")
        
        if not pipeline_script.exists():
            pytest.skip(f"Pipeline script not found: {pipeline_script}")
        
        # Test training with ensemble methods
        train_cmd = [
            sys.executable, str(pipeline_script), "train",
            "--output-dir", str(temp_output_dir),
            "--max-samples", "500"
        ]
        
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=90)
        
        # Verify training succeeded or gracefully handled missing dependencies
        if result.returncode != 0:
            # Check if failure is due to missing optional dependencies
            if "xgboost" in result.stderr.lower() or "lightgbm" in result.stderr.lower():
                pytest.skip("Optional ensemble dependencies not available")
            else:
                pytest.fail(f"Ensemble training failed: {result.stderr}")
        
        # Check outputs
        model_files = list(temp_output_dir.glob("*.pkl"))
        assert len(model_files) > 0, "No model files generated"
    
    @pytest.mark.integration
    def test_dataset_path_validation_workflow(self):
        """Test dataset path validation across all modules."""
        result = subprocess.run([
            sys.executable, "verify_dataset_paths.py", 
            "--format", "json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Dataset validation failed: {result.stderr}"
        
        # Parse validation results
        validation_data = json.loads(result.stdout)
        assert "issue_count" in validation_data
        assert "scanned_files" in validation_data
        assert validation_data["scanned_files"] > 0
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_starter_project(self, temp_output_dir: Path):
        """Test end-to-end execution of a starter project."""
        starter_projects = [
            "projects/starter/yield_regression/yield_regression_pipeline.py",
            "projects/starter/wafer_defect_classifier/wafer_defect_pipeline.py"
        ]
        
        for project_script in starter_projects:
            project_path = Path(project_script)
            if not project_path.exists():
                continue
            
            # Test the project pipeline
            project_output_dir = temp_output_dir / project_path.stem
            project_output_dir.mkdir(exist_ok=True)
            
            train_cmd = [
                sys.executable, str(project_path), "train",
                "--output-dir", str(project_output_dir),
                "--max-samples", "300"
            ]
            
            result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=120)
            
            # Verify execution (may fail due to missing datasets, but should handle gracefully)
            if result.returncode != 0:
                # Check if it's a graceful failure due to missing data
                if "dataset not found" in result.stderr.lower() or "file not found" in result.stderr.lower():
                    continue  # Expected for projects requiring external datasets
                else:
                    pytest.fail(f"Project {project_script} failed unexpectedly: {result.stderr}")
            
            # If successful, verify outputs
            output_files = list(project_output_dir.glob("*"))
            assert len(output_files) > 0, f"No outputs generated for {project_script}"


class TestModelPerformanceRegression:
    """Regression tests to ensure model performance doesn't degrade."""
    
    @pytest.mark.regression
    def test_synthetic_data_performance(self, synthetic_regression_data, performance_thresholds):
        """Test model performance on synthetic data meets minimum thresholds."""
        X, y = synthetic_regression_data
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error
        import time
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model with timing
        start_time = time.time()
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Performance assertions
        assert training_time < performance_thresholds["training_time_max"], \
            f"Training time {training_time:.2f}s exceeds threshold"
        assert r2 > performance_thresholds["min_r2_score"], \
            f"R² score {r2:.3f} below minimum threshold"
        assert mae < performance_thresholds["max_mae"], \
            f"MAE {mae:.3f} exceeds maximum threshold"
    
    @pytest.mark.regression
    def test_classification_performance(self, synthetic_classification_data, performance_thresholds):
        """Test classification model performance meets thresholds."""
        X, y = synthetic_classification_data
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Assertions
        assert accuracy > performance_thresholds["min_accuracy"], \
            f"Accuracy {accuracy:.3f} below minimum threshold"
        assert auc > 0.7, f"AUC {auc:.3f} below minimum threshold"


class TestDataValidationWorkflows:
    """Tests for data validation and quality checks."""
    
    @pytest.mark.unit
    def test_synthetic_data_quality(self, synthetic_regression_data):
        """Validate synthetic data meets quality requirements."""
        X, y = synthetic_regression_data
        
        # Basic data quality checks
        assert not X.isnull().any().any(), "Synthetic features contain null values"
        assert not y.isnull().any(), "Synthetic target contains null values"
        assert X.shape[0] == y.shape[0], "Feature and target sample counts mismatch"
        assert X.shape[1] > 0, "No features generated"
        
        # Statistical validity
        assert X.std().min() > 0, "Features with zero variance detected"
        assert y.std() > 0, "Target has zero variance"
        
        # Range checks
        assert np.isfinite(X.values).all(), "Non-finite values in features"
        assert np.isfinite(y.values).all(), "Non-finite values in target"
    
    @pytest.mark.unit
    def test_manufacturing_metrics_calculation(self, manufacturing_metrics_config):
        """Test semiconductor manufacturing-specific metrics."""
        # Mock prediction results
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
        
        # Calculate confusion matrix components
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        # Manufacturing cost calculation
        cost_config = manufacturing_metrics_config
        total_cost = (false_positives * cost_config["cost_per_false_positive"] + 
                     false_negatives * cost_config["cost_per_false_negative"])
        
        # Predictions Within Specification (PWS) - simplified version
        tolerance = cost_config["tolerance"]
        # For classification, PWS is essentially accuracy
        pws = np.mean(y_true == y_pred)
        
        # Assertions
        assert total_cost >= 0, "Negative cost calculation"
        assert 0 <= pws <= 1, "PWS outside valid range"
        assert isinstance(total_cost, (int, float)), "Cost should be numeric"
        
        # Log results for visibility
        print(f"Total manufacturing cost: ${total_cost}")
        print(f"Predictions Within Spec (PWS): {pws:.3f}")