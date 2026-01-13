import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / "4.2-unsupervised-pipeline.py"


def run_cmd(args):
    result = subprocess.run([sys.executable, str(SCRIPT)] + args, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_train_kmeans():
    out = run_cmd(["train", "--model", "kmeans", "--k", "4"])
    assert out["status"] == "trained"
    assert "metrics" in out
    assert "model" in out and out["model"] == "kmeans"


def test_train_iso_forest():
    out = run_cmd(["train", "--model", "iso_forest"])
    assert out["status"] == "trained"
    assert out["model"] == "iso_forest"


def test_train_and_evaluate_roundtrip(tmp_path):
    model_path = tmp_path / "model.joblib"
    train_out = run_cmd(["train", "--model", "kmeans", "--k", "3", "--save", str(model_path)])
    assert model_path.exists()
    eval_out = run_cmd(["evaluate", "--model-path", str(model_path)])
    assert eval_out["status"] == "evaluated"
    assert eval_out["model"] == "kmeans"


def test_predict_single_record(tmp_path):
    model_path = tmp_path / "model.joblib"
    run_cmd(["train", "--model", "kmeans", "--k", "3", "--save", str(model_path)])

    # Generate correct number of features (18 features in synthetic dataset)
    record_dict = {f"f{i+1}": 0.1 for i in range(18)}
    record = json.dumps(record_dict)

    pred_out = run_cmd(["predict", "--model-path", str(model_path), "--input-json", record])
    assert pred_out["status"] == "predicted"
    assert pred_out["model"] == "kmeans"
