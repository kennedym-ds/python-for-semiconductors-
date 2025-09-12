import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / '10.4-scaling-optimization-pipeline.py'


def run_cmd(args):
    result = subprocess.run([sys.executable, str(SCRIPT)] + args, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_train_vectorized():
    out = run_cmd(['train', '--strategy', 'vectorized', '--size', '500'])
    assert out['status'] == 'trained'
    assert 'metrics' in out
    assert 'optimization_summary' in out
    assert out['optimization_summary']['strategy'] == 'vectorized'


def test_train_parallel():
    out = run_cmd(['train', '--strategy', 'vectorized', '--parallel', '--size', '500'])
    assert out['status'] == 'trained'
    assert out['optimization_summary']['parallel'] is True


def test_train_with_caching(tmp_path):
    cache_dir = str(tmp_path / 'cache')
    out = run_cmd(['train', '--strategy', 'vectorized', '--cache-dir', cache_dir, '--size', '500'])
    assert out['status'] == 'trained'
    assert out['optimization_summary']['cache_enabled'] is True


def test_train_incremental():
    out = run_cmd(['train', '--strategy', 'vectorized', '--incremental', '--size', '500'])
    assert out['status'] == 'trained'
    assert 'metrics' in out


def test_train_and_evaluate_roundtrip(tmp_path):
    model_path = tmp_path / 'model.joblib'
    train_out = run_cmd(['train', '--strategy', 'vectorized', '--size', '500', '--save', str(model_path)])
    assert model_path.exists()
    assert train_out['status'] == 'trained'
    
    eval_out = run_cmd(['evaluate', '--model-path', str(model_path), '--size', '500'])
    assert eval_out['status'] == 'evaluated'
    assert 'metrics' in eval_out


def test_predict_single_record(tmp_path):
    model_path = tmp_path / 'model.joblib'
    run_cmd(['train', '--strategy', 'vectorized', '--size', '500', '--save', str(model_path)])
    
    # Create test input matching the wafer process data structure
    record = json.dumps({
        "temperature": 450.0,
        "pressure": 2.5,
        "flow_rate": 120.0,
        "time_duration": 60.0,
        "chamber_id": 1,
        "humidity": 45.0,
        "gas_concentration": 0.85,
        "power_consumption": 2000.0
    })
    
    pred_out = run_cmd(['predict', '--model-path', str(model_path), '--input-json', record])
    assert 'predictions' in pred_out
    assert len(pred_out['predictions']) == 1
    assert pred_out['input_shape'] == [1, 8]


def test_benchmark_small():
    out = run_cmd(['benchmark', '--size', '100'])
    assert 'dataset_size' in out
    assert out['dataset_size'] == 100
    assert 'timing' in out
    assert 'memory_usage' in out
    assert 'vectorized' in out['timing']


def test_benchmark_vectorized_only():
    out = run_cmd(['benchmark', '--size', '2000'])  # Too large for loops
    assert 'dataset_size' in out
    assert out['dataset_size'] == 2000
    assert 'timing' in out
    assert 'vectorized' in out['timing']
    assert out['speedup'] is None  # No loops benchmark for large datasets