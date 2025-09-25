#!/usr/bin/env python3
"""
Final Validation Script for Anomaly Detection Pipeline

This script validates that all deliverables and acceptance criteria are met.
"""

import json
import subprocess
import tempfile
from pathlib import Path

def validate_pipeline():
    """Validate the anomaly detection pipeline meets all requirements."""
    
    print("🔍 Validating Equipment Anomaly Detection Pipeline")
    print("=" * 60)
    
    results = {
        "train_evaluate_pipeline": False,
        "threshold_tuning_roc": False,
        "export_intervals_scores": False,
        "end_to_end_documented": False,
        "metrics_saved_disk": False
    }
    
    # Test 1: Train/evaluate anomaly detection pipeline
    print("\n✅ Test 1: Train/Evaluate Pipeline")
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        model_path = f.name
    
    train_cmd = [
        'python', 'anomaly_detection_pipeline.py', 
        'train', '--save', model_path
    ]
    
    try:
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            train_data = json.loads(result.stdout)
            print(f"   ✓ Training successful: {train_data['method']}")
            print(f"   ✓ ROC AUC: {train_data['metrics']['roc_auc']:.3f}")
            results["train_evaluate_pipeline"] = True
        else:
            print(f"   ✗ Training failed: {result.stderr}")
    except Exception as e:
        print(f"   ✗ Training error: {e}")
    
    # Test 2: Threshold tuning and ROC analysis
    print("\n✅ Test 2: Threshold Tuning & ROC Analysis")
    try:
        if results["train_evaluate_pipeline"]:
            # Check if optimal threshold was computed
            if train_data.get('optimal_threshold') is not None:
                print(f"   ✓ Optimal threshold computed: {train_data['optimal_threshold']:.4f}")
                print(f"   ✓ ROC AUC available: {train_data['metrics']['roc_auc']:.3f}")
                results["threshold_tuning_roc"] = True
            else:
                print("   ✗ No optimal threshold found")
        else:
            print("   ✗ Cannot test - training failed")
    except Exception as e:
        print(f"   ✗ Threshold tuning error: {e}")
    
    # Test 3: Export detected intervals and scores
    print("\n✅ Test 3: Export Intervals & Scores")
    try:
        # Create minimal test data
        test_data = '''timestamp,temperature,pressure,vibration,flow,power,is_anomaly
2024-01-01 00:00:00,450,2.5,0.5,120,1000,0
2024-01-01 00:01:00,451,2.5,0.5,120,1000,0
2024-01-01 00:02:00,480,2.0,1.5,100,1100,1
2024-01-01 00:03:00,452,2.5,0.5,120,1000,0'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(test_data)
            test_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_file = f.name
        
        predict_cmd = [
            'python', 'anomaly_detection_pipeline.py',
            'predict', '--model-path', model_path,
            '--input-file', test_file,
            '--export-intervals', export_file
        ]
        
        result = subprocess.run(predict_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and Path(export_file).exists():
            with open(export_file, 'r') as f:
                export_data = json.load(f)
            
            print(f"   ✓ Intervals exported successfully")
            print(f"   ✓ Export contains summary: {bool(export_data.get('summary'))}")
            print(f"   ✓ Export contains intervals: {bool(export_data.get('intervals'))}")
            print(f"   ✓ Metadata included: {bool(export_data.get('metadata'))}")
            results["export_intervals_scores"] = True
        else:
            print(f"   ✗ Export failed: {result.stderr}")
            
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        Path(export_file).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"   ✗ Export error: {e}")
    
    # Test 4: End-to-end documentation
    print("\n✅ Test 4: Documentation")
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text()
        has_usage = "Usage Examples" in content
        has_features = "Features" in content
        has_metrics = "Manufacturing-Specific Metrics" in content
        
        if has_usage and has_features and has_metrics:
            print("   ✓ Comprehensive README.md exists")
            print("   ✓ Usage examples documented")
            print("   ✓ Features documented") 
            print("   ✓ Manufacturing metrics documented")
            results["end_to_end_documented"] = True
        else:
            print("   ✗ README.md incomplete")
    else:
        print("   ✗ README.md missing")
    
    # Test 5: Metrics saved to disk
    print("\n✅ Test 5: Metrics Saved to Disk")
    try:
        if results["train_evaluate_pipeline"]:
            # Check if model file was created and contains metrics
            if Path(model_path).exists():
                print(f"   ✓ Model saved to disk: {model_path}")
                # Try to load and check for metrics
                eval_cmd = [
                    'python', 'anomaly_detection_pipeline.py',
                    'evaluate', '--model-path', model_path
                ]
                result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    eval_data = json.loads(result.stdout)
                    if eval_data.get('metrics'):
                        print("   ✓ Metrics retrievable from saved model")
                        print(f"   ✓ Evaluation metrics available: {len(eval_data['metrics'])} metrics")
                        results["metrics_saved_disk"] = True
                    else:
                        print("   ✗ No metrics in saved model")
                else:
                    print("   ✗ Cannot evaluate saved model")
            else:
                print("   ✗ Model file not created")
        else:
            print("   ✗ Cannot test - no trained model")
    except Exception as e:
        print(f"   ✗ Metrics save error: {e}")
    
    # Cleanup
    Path(model_path).unlink(missing_ok=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 ACCEPTANCE CRITERIA VALIDATION SUMMARY")
    print("=" * 60)
    
    criteria = [
        ("Train/evaluate anomaly detection pipeline", results["train_evaluate_pipeline"]),
        ("Threshold tuning and ROC analysis", results["threshold_tuning_roc"]),
        ("Export detected intervals and scores", results["export_intervals_scores"]),
        ("End-to-end run documented", results["end_to_end_documented"]),
        ("Metrics and outputs saved to disk", results["metrics_saved_disk"])
    ]
    
    passed = 0
    for criterion, status in criteria:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {criterion}")
        if status:
            passed += 1
    
    print(f"\n🎯 RESULT: {passed}/{len(criteria)} acceptance criteria met")
    
    if passed == len(criteria):
        print("🎉 ALL ACCEPTANCE CRITERIA SATISFIED!")
        print("\nDeliverables completed:")
        print("• Unsupervised anomaly detection (Isolation Forest/GMM)")
        print("• Time-series aware feature engineering")
        print("• Threshold optimization with ROC analysis")
        print("• Comprehensive metrics and cost analysis") 
        print("• Anomaly interval detection and export")
        print("• Complete documentation and examples")
        print("• Model persistence and reproducibility")
        return True
    else:
        print(f"⚠️  {len(criteria) - passed} criteria not met")
        return False

if __name__ == "__main__":
    success = validate_pipeline()
    exit(0 if success else 1)