import json
import subprocess
import sys
import tempfile
import numpy as np
from pathlib import Path

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / 'pipeline.py'


def run_cmd(args):
    """Run pipeline command and return JSON output"""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)] + args, 
        capture_output=True, 
        text=True, 
        check=True
    )
    return json.loads(result.stdout)


def test_train_synthetic():
    """Test training with synthetic data"""
    out = run_cmd([
        'train', 
        '--dataset', 'synthetic',
        '--model', 'fallback',
        '--fallback-model', 'random_forest',
        '--n-samples', '50',
        '--epochs', '2'
    ])
    assert out['status'] == 'trained'
    assert 'metrics' in out
    assert 'mIoU' in out['metrics']
    assert out['metrics']['mIoU'] >= 0.0
    assert out['metrics']['mIoU'] <= 1.0
    assert out['metadata']['model_type'] == 'fallback'
    assert out['metadata']['n_samples'] == 50


def test_train_with_save():
    """Test training and saving model"""
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        model_path = Path(tmp.name)
    
    try:
        out = run_cmd([
            'train',
            '--dataset', 'synthetic',
            '--model', 'fallback',
            '--n-samples', '20',
            '--epochs', '1',
            '--save', str(model_path)
        ])
        
        assert out['status'] == 'trained'
        assert model_path.exists()
        
        # Test loading the saved model
        out_eval = run_cmd([
            'evaluate',
            '--model-path', str(model_path),
            '--dataset', 'synthetic',
            '--n-samples', '10'
        ])
        
        assert out_eval['status'] == 'evaluated'
        assert 'metrics' in out_eval
        
    finally:
        if model_path.exists():
            model_path.unlink()


def test_evaluate():
    """Test model evaluation"""
    # First train a model
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        model_path = Path(tmp.name)
    
    try:
        # Train
        run_cmd([
            'train',
            '--dataset', 'synthetic',
            '--model', 'fallback',
            '--n-samples', '20',
            '--epochs', '1',
            '--save', str(model_path)
        ])
        
        # Evaluate
        out = run_cmd([
            'evaluate',
            '--model-path', str(model_path),
            '--dataset', 'synthetic',
            '--n-samples', '10'
        ])
        
        assert out['status'] == 'evaluated'
        assert 'metrics' in out
        
        # Check required metrics
        required_metrics = ['mIoU', 'pixel_accuracy', 'defect_coverage', 'false_positive_rate']
        for metric in required_metrics:
            assert metric in out['metrics']
            assert 0.0 <= out['metrics'][metric] <= 1.0
        
        assert out['metadata']['n_test_samples'] == 10
        
    finally:
        if model_path.exists():
            model_path.unlink()


def test_predict():
    """Test prediction functionality"""
    # Train a model first
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_model:
        model_path = Path(tmp_model.name)
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_image:
        image_path = Path(tmp_image.name)
    
    try:
        # Train model
        run_cmd([
            'train',
            '--dataset', 'synthetic',
            '--model', 'fallback',
            '--n-samples', '20',
            '--epochs', '1',
            '--save', str(model_path)
        ])
        
        # Create test image
        test_image = np.random.random((128, 128)).astype(np.float32)
        np.save(image_path, test_image)
        
        # Test prediction
        out = run_cmd([
            'predict',
            '--model-path', str(model_path),
            '--input', str(image_path)
        ])
        
        assert out['status'] == 'predicted'
        assert 'predictions' in out
        assert len(out['predictions']) == 1
        
        pred = out['predictions'][0]
        assert 'filename' in pred
        assert 'defect_detected' in pred
        assert 'defect_area_percentage' in pred
        assert isinstance(pred['defect_detected'], bool)
        assert 0.0 <= pred['defect_area_percentage'] <= 100.0
        
    finally:
        if model_path.exists():
            model_path.unlink()
        if image_path.exists():
            image_path.unlink()


def test_synthetic_data_generation():
    """Test synthetic data generation functionality"""
    from pipeline import generate_synthetic_die, generate_synthetic_dataset, DEFECT_TYPES
    
    # Test single die generation
    for defect_type in DEFECT_TYPES:
        image, mask = generate_synthetic_die(
            size=64,
            defect_type=defect_type,
            severity=0.5,
            seed=42
        )
        
        assert image.shape == (64, 64)
        assert mask.shape == (64, 64)
        assert image.dtype == np.float32
        assert mask.dtype == np.uint8
        assert np.all(image >= 0) and np.all(image <= 1)
        assert np.all((mask == 0) | (mask == 1))
        
        if defect_type == 'clean':
            assert np.sum(mask) == 0
        else:
            # Should have some defect pixels for non-clean types
            assert np.sum(mask) > 0
    
    # Test dataset generation
    images, masks = generate_synthetic_dataset(n_samples=10, size=32)
    assert images.shape == (10, 32, 32)
    assert masks.shape == (10, 32, 32)
    assert images.dtype == np.float32
    assert masks.dtype == np.uint8


def test_segmentation_metrics():
    """Test segmentation metric calculations"""
    from pipeline import compute_segmentation_metrics
    
    # Create test data
    true_masks = np.array([
        [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    ], dtype=np.uint8)
    
    # Perfect predictions
    pred_masks = true_masks.astype(np.float32)
    metrics = compute_segmentation_metrics(pred_masks, true_masks)
    
    assert metrics['mIoU'] == 1.0
    assert metrics['pixel_accuracy'] == 1.0
    assert metrics['defect_coverage'] == 1.0
    assert metrics['false_positive_rate'] == 0.0
    
    # All zeros prediction
    pred_masks = np.zeros_like(true_masks, dtype=np.float32)
    metrics = compute_segmentation_metrics(pred_masks, true_masks)
    
    assert metrics['mIoU'] < 1.0
    assert metrics['defect_coverage'] == 0.0


def test_cli_help():
    """Test that CLI help works"""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'Die Defect Segmentation Pipeline' in result.stdout
    
    # Test subcommand help
    for command in ['train', 'evaluate', 'predict']:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), command, '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0


def test_error_handling():
    """Test error handling for invalid inputs"""
    # Test missing model file
    result = subprocess.run(
        [sys.executable, str(SCRIPT), 'evaluate', '--model-path', 'nonexistent.joblib'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1
    
    # Find JSON in stderr - it might span multiple lines
    stderr_text = result.stderr.strip()
    start_idx = stderr_text.find('{')
    end_idx = stderr_text.rfind('}')
    
    assert start_idx != -1 and end_idx != -1, f"No JSON found in stderr: {stderr_text}"
    json_text = stderr_text[start_idx:end_idx+1]
    
    error_output = json.loads(json_text)
    assert error_output['status'] == 'error'
    assert 'not found' in error_output['error'].lower()
    
    # Test missing input file
    result = subprocess.run(
        [sys.executable, str(SCRIPT), 'predict', '--model-path', 'model.joblib', '--input', 'nonexistent.npy'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1
    
    # Find JSON in stderr - it might span multiple lines
    stderr_text = result.stderr.strip()
    start_idx = stderr_text.find('{')
    end_idx = stderr_text.rfind('}')
    
    assert start_idx != -1 and end_idx != -1, f"No JSON found in stderr: {stderr_text}"
    json_text = stderr_text[start_idx:end_idx+1]
    
    error_output = json.loads(json_text)
    assert error_output['status'] == 'error'


if __name__ == '__main__':
    # Run basic smoke test
    test_synthetic_data_generation()
    test_segmentation_metrics()
    test_cli_help()
    print("Basic tests passed!")