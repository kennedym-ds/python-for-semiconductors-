"""Tests for Module 8.2 LLM Manufacturing NLP Pipeline"""
import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / "8.2-llm-manufacturing-nlp-pipeline.py"


def run_cmd(args):
    """Run pipeline command and return JSON output."""
    result = subprocess.run([sys.executable, str(SCRIPT)] + args, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_train_classification_classical():
    """Test training classification model with classical backend."""
    out = run_cmd(["train", "--task", "classification", "--backend", "classical", "--n-samples", "20"])
    assert out["status"] == "trained"
    assert out["task"] == "classification"
    assert out["backend"] == "classical"
    assert "metrics" in out
    assert "accuracy" in out["metrics"]
    assert "f1_score" in out["metrics"]
    assert "estimated_loss" in out["metrics"]
    assert "pws_percent" in out["metrics"]


def test_train_summarization_classical():
    """Test training summarization model with classical backend."""
    out = run_cmd(["train", "--task", "summarization", "--backend", "classical", "--n-samples", "20"])
    assert out["status"] == "trained"
    assert out["task"] == "summarization"
    assert out["backend"] == "classical"
    assert "metrics" in out
    assert "word_overlap" in out["metrics"]
    assert "length_similarity" in out["metrics"]
    assert "estimated_loss" in out["metrics"]


def test_train_and_evaluate_roundtrip(tmp_path):
    """Test training, saving, loading, and evaluating a model."""
    model_path = tmp_path / "test_model.joblib"

    # Train and save
    train_out = run_cmd(
        ["train", "--task", "classification", "--backend", "classical", "--n-samples", "30", "--save", str(model_path)]
    )
    assert train_out["status"] == "trained"
    assert model_path.exists()

    # Evaluate
    eval_out = run_cmd(["evaluate", "--model-path", str(model_path), "--n-samples", "20"])
    assert eval_out["status"] == "evaluated"
    assert eval_out["task"] == "classification"
    assert eval_out["backend"] == "classical"
    assert "metrics" in eval_out


def test_predict_classification(tmp_path):
    """Test making predictions with classification model."""
    model_path = tmp_path / "classification_model.joblib"

    # Train and save model
    run_cmd(
        ["train", "--task", "classification", "--backend", "classical", "--n-samples", "30", "--save", str(model_path)]
    )

    # Test prediction with high severity text
    high_severity_text = '{"text":"Reactor R-204 emergency shutdown due to critical overheating"}'
    pred_out = run_cmd(["predict", "--model-path", str(model_path), "--input-json", high_severity_text])
    assert pred_out["status"] == "predicted"
    assert pred_out["task"] == "classification"
    assert "prediction" in pred_out
    assert "value" in pred_out["prediction"]
    assert "label" in pred_out["prediction"]
    # Should predict high severity (2)
    assert pred_out["prediction"]["value"] in [0, 1, 2]  # Valid severity levels

    # Test prediction with low severity text
    low_severity_text = '{"text":"Pump P-101 completed routine maintenance check successfully"}'
    pred_out = run_cmd(["predict", "--model-path", str(model_path), "--input-json", low_severity_text])
    assert pred_out["status"] == "predicted"
    assert pred_out["prediction"]["value"] in [0, 1, 2]


def test_predict_summarization(tmp_path):
    """Test making predictions with summarization model."""
    model_path = tmp_path / "summarization_model.joblib"

    # Train and save model
    run_cmd(
        ["train", "--task", "summarization", "--backend", "classical", "--n-samples", "20", "--save", str(model_path)]
    )

    # Test prediction
    text_to_summarize = """{"text":"Day Shift Report - Lithography Area\\n\\nAll lithography tools operating within normal parameters. Completed 12 wafer lots successfully. Tool A experienced minor alarm, resolved by technician. Overall yield: 96.2%"}"""
    pred_out = run_cmd(["predict", "--model-path", str(model_path), "--input-json", text_to_summarize])
    assert pred_out["status"] == "predicted"
    assert pred_out["task"] == "summarization"
    assert "prediction" in pred_out
    assert "summary" in pred_out["prediction"]
    assert len(pred_out["prediction"]["summary"]) > 0


def test_different_target_types(tmp_path):
    """Test classification with different target types."""
    model_path = tmp_path / "tool_area_model.joblib"

    # Train with tool_area target
    train_out = run_cmd(
        [
            "train",
            "--task",
            "classification",
            "--backend",
            "classical",
            "--target-type",
            "tool_area",
            "--n-samples",
            "30",
            "--save",
            str(model_path),
        ]
    )
    assert train_out["status"] == "trained"
    assert train_out["target_type"] == "tool_area"

    # Test prediction
    text = '{"text":"Etcher E-301 showing pressure fluctuations during processing"}'
    pred_out = run_cmd(["predict", "--model-path", str(model_path), "--input-json", text])
    assert pred_out["status"] == "predicted"
    # Should predict a valid tool area (0-4)
    assert pred_out["prediction"]["value"] in [0, 1, 2, 3, 4]


def test_transformers_backend_fallback():
    """Test that transformers backend falls back to classical when transformers unavailable."""
    # This test assumes transformers is not available in the test environment
    # If transformers becomes available, the test logic may need adjustment
    out = run_cmd(["train", "--task", "classification", "--backend", "transformers", "--n-samples", "20"])
    # Should either work with transformers or fall back to classical
    assert out["status"] == "trained"
    assert out["backend"] in ["classical", "transformers"]


def test_help_commands():
    """Test that help commands work without errors."""
    # Test main help
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Module 8.2 - LLM Manufacturing NLP Pipeline" in result.stdout

    # Test subcommand help
    result = subprocess.run([sys.executable, str(SCRIPT), "train", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Task type" in result.stdout


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
