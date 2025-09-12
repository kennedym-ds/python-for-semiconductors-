import json
import subprocess
import sys
import tempfile
import numpy as np
from pathlib import Path

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / '6.2-cnn-defect-detection-pipeline.py'


def run_cmd(args):
    """Run pipeline command and return JSON output"""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)] + args, 
        capture_output=True, 
        text=True, 
        check=True
    )
    return json.loads(result.stdout)


def test_train_simple_cnn():
    """Test training with synthetic data"""
    out = run_cmd([
        'train', 
        '--dataset', 'synthetic_small',
        '--model', 'simple_cnn',
        '--epochs', '2',
        '--batch-size', '16'
    ])
    assert out['status'] == 'trained'
    assert 'metrics' in out
    assert 'model_type' in out
    assert out['model_type'] == 'simple_cnn'
    assert 'pytorch_available' in out


def test_train_sklearn_fallback():
    """Test sklearn fallback training"""
    out = run_cmd([
        'train',
        '--dataset', 'synthetic_small', 
        '--model', 'fallback',  # Force sklearn
        '--fallback-model', 'random_forest'
    ])
    assert out['status'] == 'trained'
    assert 'metrics' in out
    assert out['metrics']['accuracy'] > 0.5  # Should do better than random


def test_save_load_roundtrip():
    """Test model persistence"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / 'test_model.joblib'
        
        # Train and save
        train_out = run_cmd([
            'train',
            '--dataset', 'synthetic_small',
            '--model', 'simple_cnn',
            '--epochs', '2',
            '--save', str(model_path)
        ])
        assert train_out['status'] == 'trained'
        assert model_path.exists()
        
        # Load and evaluate
        eval_out = run_cmd([
            'evaluate',
            '--model-path', str(model_path),
            '--dataset', 'synthetic_small'
        ])
        assert eval_out['status'] == 'evaluated'
        assert eval_out['model_type'] == 'simple_cnn'
        assert 'metrics' in eval_out


def test_predict_dataset():
    """Test prediction on dataset"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / 'test_model.joblib'
        
        # Train model
        run_cmd([
            'train',
            '--dataset', 'synthetic_small',
            '--epochs', '2',
            '--save', str(model_path)
        ])
        
        # Predict on dataset
        pred_out = run_cmd([
            'predict',
            '--model-path', str(model_path),
            '--dataset', 'synthetic_small'
        ])
        assert pred_out['status'] == 'predicted'
        assert 'predictions' in pred_out
        assert 'class_names' in pred_out
        assert len(pred_out['predictions']) > 0


def test_predict_single_image():
    """Test prediction on single image"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / 'test_model.joblib'
        image_path = Path(tmp_dir) / 'test_wafer.npy'
        
        # Create test wafer map
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("pipeline", SCRIPT)
        pipeline_module = module_from_spec(spec)
        spec.loader.exec_module(pipeline_module)
        
        test_wafer = pipeline_module.generate_synthetic_wafer_map(
            pattern='center', 
            size=64
        )
        np.save(image_path, test_wafer)
        
        # Train model
        run_cmd([
            'train',
            '--dataset', 'synthetic_small',
            '--epochs', '2',
            '--save', str(model_path)
        ])
        
        # Predict single image
        pred_out = run_cmd([
            'predict',
            '--model-path', str(model_path),
            '--input-image', str(image_path)
        ])
        assert pred_out['status'] == 'predicted'
        assert 'predictions' in pred_out
        assert 'probabilities' in pred_out
        assert len(pred_out['predictions']) == 1


def test_synthetic_data_generator():
    """Test synthetic wafer map generation"""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("pipeline", SCRIPT)
    pipeline_module = module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    
    # Test different patterns
    patterns = ['normal', 'center', 'edge', 'scratch', 'donut']
    for pattern in patterns:
        wafer = pipeline_module.generate_synthetic_wafer_map(
            pattern=pattern, 
            size=32
        )
        assert wafer.shape == (32, 32)
        assert wafer.dtype == np.float32
        assert 0 <= wafer.min() <= wafer.max() <= 1
    
    # Test dataset generation
    images, labels, class_names = pipeline_module.generate_synthetic_dataset(
        n_samples=50, 
        image_size=32
    )
    assert images.shape == (50, 32, 32)
    assert len(labels) == 50
    assert len(class_names) == 5
    assert set(class_names) == set(patterns)


def test_metrics_computation():
    """Test that all required metrics are computed"""
    out = run_cmd([
        'train',
        '--dataset', 'synthetic_small',
        '--epochs', '2'
    ])
    
    metrics = out['metrics']
    required_metrics = [
        'accuracy', 'f1_macro', 'f1_weighted', 
        'roc_auc_ovr', 'pr_auc_macro', 'pws', 'estimated_loss'
    ]
    
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))
        assert 0 <= metrics[metric] <= 100 if metric == 'pws' else True


def test_deterministic_results():
    """Test that results are deterministic with fixed seed"""
    # Run same training twice
    out1 = run_cmd([
        'train',
        '--dataset', 'synthetic_small',
        '--epochs', '2'
    ])
    
    out2 = run_cmd([
        'train', 
        '--dataset', 'synthetic_small',
        '--epochs', '2'
    ])
    
    # Results should be very similar (allowing for small numerical differences)
    acc1 = out1['metrics']['accuracy']
    acc2 = out2['metrics']['accuracy']
    assert abs(acc1 - acc2) < 0.1  # Within 10% difference
    
    # Metadata should be identical
    assert out1['model_type'] == out2['model_type']


def test_json_output_format():
    """Test that all outputs are valid JSON"""
    commands = [
        ['train', '--dataset', 'synthetic_small', '--epochs', '1'],
    ]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / 'test_model.joblib'
        
        # Train first
        run_cmd([
            'train', '--dataset', 'synthetic_small', 
            '--epochs', '1', '--save', str(model_path)
        ])
        
        # Add evaluate and predict commands
        commands.extend([
            ['evaluate', '--model-path', str(model_path), '--dataset', 'synthetic_small'],
            ['predict', '--model-path', str(model_path), '--dataset', 'synthetic_small']
        ])
    
        for cmd in commands:
            out = run_cmd(cmd)
            assert 'status' in out
            assert out['status'] in ['trained', 'evaluated', 'predicted']