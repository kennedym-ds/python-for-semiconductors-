#!/usr/bin/env python3
"""
Test script for Anomaly Detection Pipeline

This script tests all functionality of the anomaly detection pipeline
to ensure it meets the acceptance criteria.
"""

import json
import subprocess
import tempfile
from pathlib import Path
import sys

def run_command(cmd):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return json.loads(result.stdout) if result.stdout.strip() else {}

def test_anomaly_detection_pipeline():
    """Test the complete anomaly detection pipeline."""
    
    # Test 1: Train Isolation Forest model
    print("=== Testing Isolation Forest Training ===")
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        model_path = f.name
    
    train_cmd = [
        'python', 'projects/advanced/anomaly_detection_equipment/anomaly_detection_pipeline.py',
        'train', '--save', model_path, '--method', 'isolation_forest'
    ]
    train_result = run_command(train_cmd)
    
    if not train_result or train_result['status'] != 'trained':
        print("❌ Isolation Forest training failed")
        return False
    
    print(f"✅ Isolation Forest trained successfully")
    print(f"   ROC AUC: {train_result['metrics']['roc_auc']:.3f}")
    print(f"   F1 Score: {train_result['metrics']['f1_score']:.3f}")
    print(f"   Detection Rate: {train_result['metrics']['detection_rate']:.3f}")
    
    # Test 2: Evaluate model
    print("\n=== Testing Model Evaluation ===")
    eval_cmd = [
        'python', 'projects/advanced/anomaly_detection_equipment/anomaly_detection_pipeline.py',
        'evaluate', '--model-path', model_path
    ]
    eval_result = run_command(eval_cmd)
    
    if not eval_result or eval_result['status'] != 'evaluated':
        print("❌ Model evaluation failed")
        return False
    
    print(f"✅ Model evaluation successful")
    print(f"   ROC AUC: {eval_result['metrics']['roc_auc']:.3f}")
    print(f"   Precision: {eval_result['metrics']['precision']:.3f}")
    print(f"   Recall: {eval_result['metrics']['recall']:.3f}")
    
    # Test 3: Single prediction
    print("\n=== Testing Single Prediction ===")
    pred_cmd = [
        'python', 'projects/advanced/anomaly_detection_equipment/anomaly_detection_pipeline.py',
        'predict', '--model-path', model_path,
        '--input-json', '{"temperature":455, "pressure":2.6, "vibration":0.8, "flow":120, "power":1000}'
    ]
    pred_result = run_command(pred_cmd)
    
    if not pred_result or pred_result['status'] != 'predicted':
        print("❌ Single prediction failed")
        return False
    
    print(f"✅ Single prediction successful")
    print(f"   Anomaly detected: {'Yes' if pred_result['anomalies_detected'] > 0 else 'No'}")
    print(f"   Anomaly score: {pred_result['anomaly_scores'][0]:.4f}")
    
    # Test 4: Interval export
    print("\n=== Testing Interval Export ===")
    
    # First create test data
    create_data_cmd = [
        'python', '-c',
        '''
from projects.advanced.anomaly_detection_equipment.anomaly_detection_pipeline import generate_equipment_timeseries, add_time_series_features
df = generate_equipment_timeseries(n_samples=500)
df = add_time_series_features(df)
df.to_csv("/tmp/test_data.csv", index=False)
print(f"Created test data: {len(df)} samples, {df['is_anomaly'].sum()} anomalies")
        '''
    ]
    subprocess.run(create_data_cmd, shell=True)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        export_path = f.name
    
    export_cmd = [
        'python', 'projects/advanced/anomaly_detection_equipment/anomaly_detection_pipeline.py',
        'predict', '--model-path', model_path,
        '--input-file', '/tmp/test_data.csv',
        '--export-intervals', export_path
    ]
    export_result = run_command(export_cmd)
    
    if not export_result or export_result['status'] != 'predicted':
        print("❌ Interval export failed")
        return False
    
    print(f"✅ Interval export successful")
    print(f"   Total samples: {export_result['export_summary']['total_samples']}")
    print(f"   Detected anomalies: {export_result['export_summary']['detected_anomalies']}")
    print(f"   Anomaly intervals: {export_result['export_summary']['num_intervals']}")
    print(f"   Export file: {export_path}")
    
    # Test 5: GMM method
    print("\n=== Testing GMM Method ===")
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        gmm_model_path = f.name
    
    gmm_train_cmd = [
        'python', 'projects/advanced/anomaly_detection_equipment/anomaly_detection_pipeline.py',
        'train', '--save', gmm_model_path, '--method', 'gmm', '--n-components', '3'
    ]
    gmm_result = run_command(gmm_train_cmd)
    
    if not gmm_result or gmm_result['status'] != 'trained':
        print("❌ GMM training failed")
        return False
    
    print(f"✅ GMM training successful")
    print(f"   ROC AUC: {gmm_result['metrics']['roc_auc']:.3f}")
    print(f"   F1 Score: {gmm_result['metrics']['f1_score']:.3f}")
    
    # Cleanup
    Path(model_path).unlink(missing_ok=True)
    Path(gmm_model_path).unlink(missing_ok=True)
    Path(export_path).unlink(missing_ok=True)
    Path('/tmp/test_data.csv').unlink(missing_ok=True)
    
    print("\n🎉 All tests passed! Anomaly detection pipeline is working correctly.")
    return True

if __name__ == "__main__":
    success = test_anomaly_detection_pipeline()
    sys.exit(0 if success else 1)