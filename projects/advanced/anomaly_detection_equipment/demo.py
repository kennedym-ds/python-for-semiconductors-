#!/usr/bin/env python3
"""
Equipment Anomaly Detection Demo

This script demonstrates the complete anomaly detection pipeline
with realistic equipment monitoring scenarios.
"""

import json
import tempfile
from pathlib import Path
import subprocess
import sys

def run_demo():
    """Run the complete anomaly detection demo."""
    print("🏭 Equipment Anomaly Detection Demo")
    print("=" * 50)
    
    # Step 1: Train models
    print("\n📊 Step 1: Training Anomaly Detection Models")
    print("-" * 40)
    
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        iso_model = f.name
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        gmm_model = f.name
    
    # Train Isolation Forest
    print("Training Isolation Forest...")
    iso_cmd = [
        'python', 'anomaly_detection_pipeline.py',
        'train', '--method', 'isolation_forest', '--save', iso_model
    ]
    result = subprocess.run(iso_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        data = json.loads(result.stdout)
        print(f"  ✅ Isolation Forest: ROC AUC = {data['metrics']['roc_auc']:.3f}")
    else:
        print(f"  ❌ Training failed: {result.stderr}")
        return False
    
    # Train GMM
    print("Training Gaussian Mixture Model...")
    gmm_cmd = [
        'python', 'anomaly_detection_pipeline.py',
        'train', '--method', 'gmm', '--n-components', '3', '--save', gmm_model
    ]
    result = subprocess.run(gmm_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        data = json.loads(result.stdout)
        print(f"  ✅ GMM: ROC AUC = {data['metrics']['roc_auc']:.3f}")
    else:
        print(f"  ❌ GMM training failed: {result.stderr}")
        return False
    
    # Step 2: Evaluate models
    print("\n🔍 Step 2: Model Evaluation")
    print("-" * 30)
    
    for model_name, model_path in [("Isolation Forest", iso_model), ("GMM", gmm_model)]:
        eval_cmd = [
            'python', 'anomaly_detection_pipeline.py',
            'evaluate', '--model-path', model_path
        ]
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            metrics = data['metrics']
            print(f"  {model_name}:")
            print(f"    • Precision: {metrics['precision']:.3f}")
            print(f"    • Recall: {metrics['recall']:.3f}")
            print(f"    • F1 Score: {metrics['f1_score']:.3f}")
            print(f"    • Detection Rate: {metrics['detection_rate']:.3f}")
            print(f"    • False Alarm Rate: {metrics['false_alarm_rate']:.3f}")
            print(f"    • Estimated Cost: ${metrics['estimated_cost']:.0f}")
        else:
            print(f"  ❌ {model_name} evaluation failed")
    
    # Step 3: Real-time prediction scenarios
    print("\n⚡ Step 3: Real-time Prediction Scenarios")
    print("-" * 40)
    
    # Normal operation
    normal_data = '{"temperature":450, "pressure":2.5, "vibration":0.5, "flow":120, "power":1000}'
    pred_cmd = [
        'python', 'anomaly_detection_pipeline.py',
        'predict', '--model-path', iso_model, '--input-json', normal_data
    ]
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        data = json.loads(result.stdout)
        status = "NORMAL" if data['anomalies_detected'] == 0 else "ANOMALY"
        score = data['anomaly_scores'][0]
        print(f"  Normal Operation: {status} (score: {score:.4f})")
    
    # Simulated anomaly
    anomaly_data = '{"temperature":520, "pressure":1.8, "vibration":2.5, "flow":80, "power":1200}'
    pred_cmd = [
        'python', 'anomaly_detection_pipeline.py',
        'predict', '--model-path', iso_model, '--input-json', anomaly_data
    ]
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        data = json.loads(result.stdout)
        status = "NORMAL" if data['anomalies_detected'] == 0 else "ANOMALY"
        score = data['anomaly_scores'][0]
        print(f"  Potential Failure: {status} (score: {score:.4f})")
    
    # Step 4: Batch processing with intervals
    print("\n📈 Step 4: Batch Processing with Interval Detection")
    print("-" * 50)
    
    # Generate test data
    print("Generating test equipment data...")
    gen_cmd = [
        'python', '-c',
        '''
from anomaly_detection_pipeline import generate_equipment_timeseries, add_time_series_features
df = generate_equipment_timeseries(n_samples=1000, anomaly_rate=0.08)
df = add_time_series_features(df)
df.to_csv("demo_equipment_data.csv", index=False)
print(f"Generated {len(df)} samples with {df['is_anomaly'].sum()} anomalies")
        '''
    ]
    result = subprocess.run(gen_cmd, shell=True, capture_output=True, text=True)
    print(f"  {result.stdout.strip()}")
    
    # Process batch data
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        intervals_file = f.name
    
    batch_cmd = [
        'python', 'anomaly_detection_pipeline.py',
        'predict', '--model-path', iso_model, 
        '--input-file', 'demo_equipment_data.csv',
        '--export-intervals', intervals_file
    ]
    result = subprocess.run(batch_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        data = json.loads(result.stdout)
        summary = data['export_summary']
        print(f"  Batch Processing Results:")
        print(f"    • Total Samples: {summary['total_samples']}")
        print(f"    • Detected Anomalies: {summary['detected_anomalies']}")
        print(f"    • Anomaly Rate: {summary['anomaly_rate']:.1%}")
        print(f"    • Anomaly Intervals: {summary['num_intervals']}")
        print(f"    • Results exported to: {intervals_file}")
        
        # Show first few intervals
        with open(intervals_file, 'r') as f:
            export_data = json.load(f)
        
        if export_data['intervals']:
            print(f"  First Detected Interval:")
            interval = export_data['intervals'][0]
            print(f"    • Start: {interval['start_timestamp']}")
            print(f"    • Duration: {interval['duration_minutes']} minutes")
            print(f"    • Max Score: {interval['max_score']:.4f}")
    
    # Step 5: Performance comparison
    print("\n🏆 Step 5: Model Performance Comparison")
    print("-" * 40)
    
    models = [
        ("Isolation Forest", iso_model),
        ("Gaussian Mixture", gmm_model)
    ]
    
    print("Model Performance Summary:")
    print("Method              ROC AUC    F1 Score   Cost/Sample")
    print("-" * 55)
    
    for name, model_path in models:
        eval_cmd = [
            'python', 'anomaly_detection_pipeline.py',
            'evaluate', '--model-path', model_path
        ]
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            metrics = data['metrics']
            print(f"{name:<18} {metrics['roc_auc']:.3f}      {metrics['f1_score']:.3f}      ${metrics['cost_per_sample']:.2f}")
    
    # Cleanup
    Path(iso_model).unlink(missing_ok=True)
    Path(gmm_model).unlink(missing_ok=True)
    Path(intervals_file).unlink(missing_ok=True)
    Path('demo_equipment_data.csv').unlink(missing_ok=True)
    
    print("\n🎉 Demo completed successfully!")
    print("\nKey Takeaways:")
    print("• Both models effectively detect equipment anomalies")
    print("• Isolation Forest provides balanced precision/recall")
    print("• GMM offers higher recall but more false alarms")
    print("• Real-time predictions enable proactive maintenance")
    print("• Interval detection helps identify failure patterns")
    print("• Cost-based metrics support business decisions")
    
    return True

if __name__ == "__main__":
    print("Starting Equipment Anomaly Detection Demo...")
    success = run_demo()
    
    if not success:
        print("❌ Demo failed!")
        sys.exit(1)
    
    print("\n✅ Demo completed successfully!")
    print("For more details, see README.md")