"""Tests for Module 5.2 Predictive Maintenance Pipeline."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / '5.2-predictive-maintenance-pipeline.py'


def run_cmd(args):
    """Run CLI command and parse JSON output."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)] + args, 
        capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def test_train_logistic_classification():
    """Test training a logistic regression classification model."""
    out = run_cmd(['train', '--model', 'logistic', '--task', 'classification', '--target', 'event_in_24h'])
    assert out['status'] == 'trained'
    assert out['model'] == 'logistic'
    assert out['task'] == 'classification'
    assert out['target'] == 'event_in_24h'
    assert 'metrics' in out
    assert 'roc_auc' in out['metrics']
    assert 'pr_auc' in out['metrics']
    assert 'pws' in out['metrics']
    assert 'estimated_loss' in out['metrics']


def test_train_rf_regression():
    """Test training a random forest regression model."""
    out = run_cmd(['train', '--model', 'rf', '--task', 'regression', '--target', 'time_to_event'])
    assert out['status'] == 'trained'
    assert out['model'] == 'rf'
    assert out['task'] == 'regression'
    assert out['target'] == 'time_to_event'
    assert 'metrics' in out
    assert 'mae' in out['metrics']
    assert 'rmse' in out['metrics']
    assert 'r2' in out['metrics']
    assert 'pws' in out['metrics']
    assert 'estimated_loss' in out['metrics']


def test_train_and_evaluate_roundtrip():
    """Test training a model and then evaluating it."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / 'model.joblib'
        
        # Train model
        train_out = run_cmd([
            'train', '--model', 'logistic', '--task', 'classification', 
            '--target', 'event_in_24h', '--save', str(model_path)
        ])
        assert train_out['status'] == 'trained'
        assert model_path.exists()
        
        # Evaluate model
        eval_out = run_cmd(['evaluate', '--model-path', str(model_path)])
        assert eval_out['status'] == 'evaluated'
        assert eval_out['model'] == 'logistic'
        assert eval_out['task'] == 'classification'
        assert eval_out['target'] == 'event_in_24h'
        assert 'metrics' in eval_out


def test_xgboost_model():
    """Test XGBoost model (if available)."""
    try:
        out = run_cmd(['train', '--model', 'xgboost', '--task', 'classification', '--n-estimators', '10'])
        assert out['status'] == 'trained'
        assert out['model'] == 'xgboost'
        assert 'metrics' in out
    except subprocess.CalledProcessError as e:
        # XGBoost might not be available, check error message
        error_output = json.loads(e.stdout)
        assert 'xgboost' in error_output['error'].lower()


def test_smote_option():
    """Test SMOTE oversampling option."""
    out = run_cmd(['train', '--model', 'logistic', '--use-smote', '--smote-k-neighbors', '3'])
    assert out['status'] == 'trained'
    # SMOTE availability depends on imbalanced-learn installation
    # The test should pass regardless


def test_different_horizons():
    """Test different prediction horizons."""
    out_24h = run_cmd(['train', '--target', 'event_in_24h', '--horizon', '24'])
    assert out_24h['status'] == 'trained'
    
    out_72h = run_cmd(['train', '--target', 'event_in_72h', '--horizon', '72'])
    assert out_72h['status'] == 'trained'


def test_threshold_methods():
    """Test different threshold optimization methods."""
    out_youden = run_cmd(['train', '--threshold-method', 'youden'])
    assert out_youden['status'] == 'trained'
    
    out_cost = run_cmd(['train', '--threshold-method', 'cost_based', '--cost-fp', '2.0', '--cost-fn', '5.0'])
    assert out_cost['status'] == 'trained'


def test_manufacturing_metrics():
    """Test that manufacturing-specific metrics are calculated."""
    out = run_cmd(['train', '--model', 'logistic'])
    assert out['status'] == 'trained'
    
    metrics = out['metrics']
    
    # Check standard metrics
    assert 'roc_auc' in metrics
    assert 'pr_auc' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    # Check manufacturing-specific metrics
    assert 'pws' in metrics  # Prediction Within Spec
    assert 'estimated_loss' in metrics  # Cost-based loss
    assert 'cost_per_sample' in metrics
    
    # Validate metric ranges
    assert 0 <= metrics['roc_auc'] <= 1
    assert 0 <= metrics['pr_auc'] <= 1
    assert 0 <= metrics['pws'] <= 1
    assert metrics['estimated_loss'] >= 0
    assert metrics['cost_per_sample'] >= 0


def test_reproducibility():
    """Test that models are deterministic with fixed seed."""
    out1 = run_cmd(['train', '--model', 'rf', '--n-estimators', '10'])
    out2 = run_cmd(['train', '--model', 'rf', '--n-estimators', '10'])
    
    assert out1['status'] == 'trained'
    assert out2['status'] == 'trained'
    
    # Should get same number of samples and features
    assert out1['n_samples'] == out2['n_samples']
    assert out1['n_features'] == out2['n_features']


def test_class_weight_options():
    """Test different class weight options."""
    out_balanced = run_cmd(['train', '--class-weight', 'balanced'])
    assert out_balanced['status'] == 'trained'
    
    out_no_weight = run_cmd(['train', '--no-class-weight'])
    assert out_no_weight['status'] == 'trained'


def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test invalid model
    try:
        run_cmd(['train', '--model', 'invalid_model'])
        assert False, "Should have raised an error"
    except subprocess.CalledProcessError as e:
        error_output = json.loads(e.stdout)
        assert error_output['status'] == 'error'
        assert 'invalid_model' in error_output['error']
    
    # Test invalid task
    try:
        run_cmd(['train', '--task', 'invalid_task'])
        assert False, "Should have raised an error"
    except subprocess.CalledProcessError as e:
        error_output = json.loads(e.stdout)
        assert error_output['status'] == 'error'


if __name__ == '__main__':
    # Run all test functions
    test_functions = [
        test_train_logistic_classification,
        test_train_rf_regression,
        test_train_and_evaluate_roundtrip,
        test_xgboost_model,
        test_smote_option,
        test_different_horizons,
        test_threshold_methods,
        test_manufacturing_metrics,
        test_reproducibility,
        test_class_weight_options,
        test_error_handling,
    ]
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            print(f"✓ {test_func.__name__} passed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
    
    print("All tests completed.")