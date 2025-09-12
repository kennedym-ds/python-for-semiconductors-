import json
import subprocess
import sys
from pathlib import Path
import pytest

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / "10.2-testing-qa-pipeline.py"


def run_cmd(args):
    """Execute the QA pipeline CLI command and return JSON response."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)] + args, capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def test_train_basic_qa_suite():
    """Test basic QA suite execution."""
    out = run_cmd(["train", "--test-suite", "smoke", "--coverage-threshold", "50"])
    assert out["status"] == "trained"
    assert "metrics" in out
    assert "quality_gates" in out


def test_train_with_specific_modules():
    """Test QA suite with specific target modules."""
    out = run_cmd(
        ["train", "--test-suite", "smoke", "--target-modules", "modules/foundation"]
    )
    assert out["status"] == "trained"
    assert out["metadata"]["target_modules"] == ["modules/foundation"]


def test_train_and_save_pipeline(tmp_path):
    """Test training and saving QA pipeline state."""
    qa_path = tmp_path / "qa_pipeline.joblib"
    out = run_cmd(["train", "--test-suite", "smoke", "--save", str(qa_path)])
    assert out["status"] == "trained"
    assert qa_path.exists()


def test_evaluate_code_quality():
    """Test code quality evaluation."""
    out = run_cmd(["evaluate", "--check-type", "lint", "--target-path", str(THIS_DIR)])
    assert out["status"] == "evaluated"
    assert "checks" in out
    assert "code_quality" in out["checks"]


def test_evaluate_dataset_paths():
    """Test dataset path validation."""
    out = run_cmd(["evaluate", "--check-type", "paths"])
    assert out["status"] == "evaluated"
    assert "checks" in out
    assert "dataset_paths" in out["checks"]

    # Should contain expected dataset entries
    datasets = out["checks"]["dataset_paths"]["datasets"]
    expected_datasets = {"secom", "steel_plates", "wm811k", "synthetic"}
    assert set(datasets.keys()) >= expected_datasets


def test_evaluate_smoke_tests():
    """Test smoke test execution via evaluate command."""
    out = run_cmd(["evaluate", "--check-type", "smoke"])
    assert out["status"] == "evaluated"
    assert "checks" in out
    assert "smoke_tests" in out["checks"]

    smoke_results = out["checks"]["smoke_tests"]
    assert "import_tests" in smoke_results
    assert "basic_functionality" in smoke_results


def test_evaluate_performance_benchmarks():
    """Test performance benchmark execution."""
    out = run_cmd(["evaluate", "--check-type", "performance"])
    assert out["status"] == "evaluated"
    assert "checks" in out
    assert "performance" in out["checks"]

    perf_results = out["checks"]["performance"]
    assert "benchmarks" in perf_results
    assert "overall_score" in perf_results


def test_evaluate_all_checks():
    """Test running all quality checks together."""
    out = run_cmd(["evaluate", "--check-type", "all"])
    assert out["status"] == "evaluated"
    assert "checks" in out

    # Should contain all check types
    expected_checks = {"code_quality", "dataset_paths", "smoke_tests", "performance"}
    assert set(out["checks"].keys()) >= expected_checks


def test_predict_system_health():
    """Test system health validation."""
    out = run_cmd(["predict"])
    assert out["status"] == "predicted"
    assert out["target"] == "system"
    assert "health_check" in out
    assert "recommendation" in out


def test_predict_specific_module():
    """Test validation of a specific module."""
    out = run_cmd(["predict", "--target-module", "json"])  # Use built-in module
    assert out["status"] == "predicted"
    assert out["target"] == "json"
    assert "health_check" in out
    assert out["health_check"]["module_import"]["status"] == "success"


def test_invalid_test_suite():
    """Test handling of invalid test suite specification."""
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd(["train", "--test-suite", "invalid_suite"])


def test_invalid_check_type():
    """Test handling of invalid check type."""
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd(["evaluate", "--check-type", "invalid_check"])


def test_nonexistent_target_module():
    """Test handling of nonexistent module validation."""
    out = run_cmd(["predict", "--target-module", "nonexistent.module.name"])
    assert out["status"] == "predicted"
    assert out["health_check"]["module_import"]["status"] == "failed"


def test_coverage_threshold_validation():
    """Test that coverage thresholds are properly applied."""
    out = run_cmd(["train", "--test-suite", "smoke", "--coverage-threshold", "95"])
    assert out["status"] == "trained"
    assert out["metadata"]["quality_gates"]["coverage"] == 95.0


def test_performance_threshold_validation():
    """Test that performance thresholds are properly applied."""
    out = run_cmd(["train", "--test-suite", "smoke", "--performance-threshold", "10"])
    assert out["status"] == "trained"
    assert out["metadata"]["quality_gates"]["performance"] == 10.0


def test_json_output_schema():
    """Test that all commands produce valid JSON with expected schema."""
    # Test train command output schema
    train_out = run_cmd(["train", "--test-suite", "smoke"])
    assert isinstance(train_out, dict)
    assert set(train_out.keys()) >= {"status", "metrics", "metadata", "quality_gates"}

    # Test evaluate command output schema
    eval_out = run_cmd(["evaluate", "--check-type", "smoke"])
    assert isinstance(eval_out, dict)
    assert set(eval_out.keys()) >= {"status", "checks", "summary"}

    # Test predict command output schema
    pred_out = run_cmd(["predict"])
    assert isinstance(pred_out, dict)
    assert set(pred_out.keys()) >= {
        "status",
        "target",
        "health_check",
        "recommendation",
    }


def test_cli_help_commands():
    """Test that help commands work without errors."""
    # Test main help
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Testing & QA Pipeline CLI" in result.stdout

    # Test subcommand help
    for subcommand in ["train", "evaluate", "predict"]:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), subcommand, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_error_handling_and_exit_codes():
    """Test proper error handling and exit codes for invalid inputs."""
    # Test missing required arguments
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode != 0

    # Test invalid command
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "invalid_command"], capture_output=True, text=True
    )
    assert result.returncode != 0


def test_train_evaluate_predict_roundtrip(tmp_path):
    """Test complete workflow: train -> save -> evaluate -> predict."""
    qa_path = tmp_path / "qa_pipeline.joblib"

    # Step 1: Train and save
    train_out = run_cmd(["train", "--test-suite", "smoke", "--save", str(qa_path)])
    assert train_out["status"] == "trained"
    assert qa_path.exists()

    # Step 2: Evaluate using saved pipeline
    eval_out = run_cmd(
        ["evaluate", "--qa-pipeline-path", str(qa_path), "--check-type", "smoke"]
    )
    assert eval_out["status"] == "evaluated"

    # Step 3: Predict system health
    pred_out = run_cmd(["predict"])
    assert pred_out["status"] == "predicted"


def test_manufacturing_specific_validations():
    """Test semiconductor manufacturing specific validation checks."""
    # Test dataset path validation includes semiconductor datasets
    out = run_cmd(["evaluate", "--check-type", "paths"])
    datasets = out["checks"]["dataset_paths"]["datasets"]

    # Should include key semiconductor datasets
    assert "secom" in datasets  # Semiconductor dataset
    assert "steel_plates" in datasets  # Materials dataset
    assert "synthetic" in datasets  # Generated data


def test_performance_benchmarks_completeness():
    """Test that performance benchmarks cover key operations."""
    out = run_cmd(["evaluate", "--check-type", "performance"])
    benchmarks = out["checks"]["performance"]["benchmarks"]

    # Should benchmark critical pipeline operations
    expected_operations = {"data_generation", "model_training", "prediction"}
    assert set(benchmarks.keys()) >= expected_operations

    # Each benchmark should have timing information
    for operation, metrics in benchmarks.items():
        assert "execution_time" in metrics
        assert "status" in metrics


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])
