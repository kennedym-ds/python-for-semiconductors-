import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

THIS_DIR = Path(__file__).parent
SCRIPT = THIS_DIR / '7.2-pattern-recognition-pipeline.py'


def run_cmd(args):
    """Run the CLI script and parse JSON output."""
    result = subprocess.run([sys.executable, str(SCRIPT)] + args, 
                          capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_train_classical_svm():
    """Test training classical SVM model."""
    start_time = time.time()
    out = run_cmd(['train', '--approach', 'classical', '--model', 'svm', '--epochs', '2', '--dataset', 'synthetic_wafer_small'])
    duration = time.time() - start_time
    
    assert out['status'] == 'trained'
    assert out['approach'] == 'classical'
    assert out['model'] == 'svm'
    assert 'metrics' in out
    assert 'metadata' in out
    assert out['metadata']['n_classes'] == 5
    assert out['metadata']['class_names'] == ['Normal', 'Center', 'Edge', 'Scratch', 'Ring']
    assert duration < 45  # Runtime requirement


def test_train_classical_rf():
    """Test training classical Random Forest model."""
    start_time = time.time()
    out = run_cmd(['train', '--approach', 'classical', '--model', 'rf', '--n-estimators', '50', '--dataset', 'synthetic_wafer_small'])
    duration = time.time() - start_time
    
    assert out['status'] == 'trained'
    assert out['approach'] == 'classical'
    assert out['model'] == 'rf'
    assert 'metrics' in out
    assert duration < 45


def test_train_deep_learning_cnn():
    """Test training deep learning CNN model."""
    start_time = time.time()
    out = run_cmd(['train', '--approach', 'deep_learning', '--model', 'cnn', 
                   '--epochs', '2', '--batch-size', '64', '--dataset', 'synthetic_wafer_small'])
    duration = time.time() - start_time
    
    assert out['status'] == 'trained'
    assert out['approach'] == 'deep_learning'
    assert out['model'] == 'cnn'
    assert 'metrics' in out
    assert out['metadata']['n_features_in'] is None  # CNN doesn't use extracted features
    assert duration < 45


def test_save_load_roundtrip_classical():
    """Test save/load roundtrip for classical model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / 'classical_model.joblib'
        
        # Train and save
        train_out = run_cmd(['train', '--approach', 'classical', '--model', 'svm', 
                           '--save', str(model_path), '--dataset', 'synthetic_wafer_small'])
        assert model_path.exists()
        assert train_out['status'] == 'trained'
        
        # Load and evaluate
        eval_out = run_cmd(['evaluate', '--model-path', str(model_path), '--dataset', 'synthetic_wafer_small'])
        assert eval_out['status'] == 'evaluated'
        assert eval_out['approach'] == 'classical'
        assert eval_out['model'] == 'svm'
        
        # Check metrics consistency (should be similar since same dataset)
        train_acc = train_out['metrics']['accuracy']
        eval_acc = eval_out['metrics']['accuracy']
        assert abs(train_acc - eval_acc) < 0.1


def test_save_load_roundtrip_deep_learning():
    """Test save/load roundtrip for deep learning model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / 'dl_model.joblib'
        
        # Train and save
        train_out = run_cmd(['train', '--approach', 'deep_learning', '--model', 'cnn',
                           '--epochs', '2', '--save', str(model_path), '--dataset', 'synthetic_wafer_small'])
        assert model_path.exists()
        assert train_out['status'] == 'trained'
        
        # Load and evaluate
        eval_out = run_cmd(['evaluate', '--model-path', str(model_path), '--dataset', 'synthetic_wafer_small'])
        assert eval_out['status'] == 'evaluated'
        assert eval_out['approach'] == 'deep_learning'
        assert eval_out['model'] == 'cnn'


def test_predict_single_wafer_classical():
    """Test prediction on single wafer map with classical model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / 'model.joblib'
        
        # Train and save
        run_cmd(['train', '--approach', 'classical', '--model', 'svm', 
                '--save', str(model_path), '--dataset', 'synthetic_wafer_small'])
        
        # Create a simple test wafer map (center defect pattern)
        wafer_map = [[0 for _ in range(32)] for _ in range(32)]
        # Add center defects
        for i in range(14, 18):
            for j in range(14, 18):
                wafer_map[i][j] = 1
        
        # Predict
        input_json = json.dumps({"wafer_map": wafer_map})
        pred_out = run_cmd(['predict', '--model-path', str(model_path), 
                          '--input-json', input_json])
        
        assert pred_out['status'] == 'predicted'
        assert pred_out['approach'] == 'classical'
        assert 'prediction' in pred_out
        assert 'probabilities' in pred_out
        assert len(pred_out['probabilities']) == 5  # 5 classes


def test_predict_single_wafer_deep_learning():
    """Test prediction on single wafer map with deep learning model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / 'model.joblib'
        
        # Train and save
        run_cmd(['train', '--approach', 'deep_learning', '--model', 'cnn',
                '--epochs', '2', '--save', str(model_path), '--dataset', 'synthetic_wafer_small'])
        
        # Create a simple test wafer map (edge defect pattern)  
        wafer_map = [[0 for _ in range(32)] for _ in range(32)]
        # Add edge defects
        for i in range(32):
            if i < 2 or i > 29:  # Top and bottom edges
                for j in range(32):
                    wafer_map[i][j] = 1
            else:  # Left and right edges
                for j in [0, 1, 30, 31]:
                    wafer_map[i][j] = 1
        
        # Predict
        input_json = json.dumps({"wafer_map": wafer_map})
        pred_out = run_cmd(['predict', '--model-path', str(model_path), 
                          '--input-json', input_json])
        
        assert pred_out['status'] == 'predicted'
        assert pred_out['approach'] == 'deep_learning'
        assert 'prediction' in pred_out
        assert 'probabilities' in pred_out


def test_manufacturing_metrics():
    """Test that manufacturing-specific metrics are computed."""
    out = run_cmd(['train', '--approach', 'classical', '--model', 'svm', '--dataset', 'synthetic_wafer_small'])
    
    metrics = out['metrics']
    
    # Standard ML metrics
    assert 'roc_auc_weighted' in metrics
    assert 'pr_auc_weighted' in metrics
    assert 'f1_weighted' in metrics
    assert 'f1_macro' in metrics
    assert 'accuracy' in metrics
    
    # Manufacturing-specific metrics
    assert 'pws' in metrics  # Prediction Within Spec
    assert 'estimated_loss' in metrics  # Estimated cost
    
    # Check metric ranges
    assert 0 <= metrics['roc_auc_weighted'] <= 1
    assert 0 <= metrics['pr_auc_weighted'] <= 1
    assert 0 <= metrics['f1_weighted'] <= 1
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['pws'] <= 1
    assert metrics['estimated_loss'] >= 0


def test_class_imbalance_handling():
    """Test that class imbalance is properly handled."""
    # Train models and check they handle the imbalanced synthetic data
    classical_out = run_cmd(['train', '--approach', 'classical', '--model', 'svm', '--dataset', 'synthetic_wafer_small'])
    dl_out = run_cmd(['train', '--approach', 'deep_learning', '--model', 'cnn', '--epochs', '2', '--dataset', 'synthetic_wafer_small'])
    
    # Both should achieve reasonable performance despite imbalance
    assert classical_out['metrics']['f1_macro'] > 0.3
    assert dl_out['metrics']['f1_macro'] > 0.2  # Lower threshold for limited training
    
    # Check that class names are properly preserved
    assert len(classical_out['metadata']['class_names']) == 5
    assert len(dl_out['metadata']['class_names']) == 5


def test_deterministic_behavior():
    """Test that results are deterministic with fixed random seed."""
    # Train two identical models
    out1 = run_cmd(['train', '--approach', 'classical', '--model', 'svm', '--dataset', 'synthetic_wafer_small'])
    out2 = run_cmd(['train', '--approach', 'classical', '--model', 'svm', '--dataset', 'synthetic_wafer_small'])
    
    # Results should be identical due to random seed
    assert out1['metrics']['accuracy'] == out2['metrics']['accuracy']
    assert out1['metadata']['n_features_in'] == out2['metadata']['n_features_in']


def test_error_handling():
    """Test error handling for invalid inputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / 'model.joblib'
        
        # Train a model
        run_cmd(['train', '--approach', 'classical', '--model', 'svm', 
                '--save', str(model_path), '--dataset', 'synthetic_wafer_small'])
        
        # Test invalid JSON input
        try:
            run_cmd(['predict', '--model-path', str(model_path), 
                    '--input-json', '{"invalid": "input"}'])
            assert False, "Should have raised error for missing wafer_map"
        except subprocess.CalledProcessError:
            pass  # Expected
        
        # Test malformed wafer map
        try:
            run_cmd(['predict', '--model-path', str(model_path), 
                    '--input-json', '{"wafer_map": "not_an_array"}'])
            assert False, "Should have raised error for invalid wafer_map"
        except subprocess.CalledProcessError:
            pass  # Expected