#!/usr/bin/env python3
"""
MLflow Integration Example Run Script

This script demonstrates a complete MLflow lifecycle for semiconductor manufacturing ML:
1. Dependency checking and environment setup
2. MLflow server management
3. Model training with experiment tracking
4. Model evaluation with artifact logging
5. Model registry management
6. Production deployment simulation

Run this script to see MLflow tracking in action with realistic semiconductor data.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mlops_mlflow_pipeline import (
    check_mlflow_availability, 
    MLOpsMLflowPipeline,
    start_mlflow_server,
    get_mlflow_experiments
)

def run_command(cmd_list, description=""):
    """Run a command and return the JSON output."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"Command: {' '.join(cmd_list)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            try:
                output = json.loads(result.stdout)
                print(json.dumps(output, indent=2))
                return output
            except json.JSONDecodeError:
                print(result.stdout)
                return {"stdout": result.stdout}
        return {"status": "success", "output": "No output"}
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {"error": f"Command failed: {e}"}


def check_prerequisites():
    """Check if all prerequisites are available."""
    print("\n🔍 Checking Prerequisites")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check MLflow availability
    mlflow_available = check_mlflow_availability()
    
    # Check if we can import our pipeline
    try:
        from mlops_mlflow_pipeline import MLOpsMLflowPipeline
        print("✅ MLOps pipeline module is importable")
    except ImportError as e:
        print(f"❌ Cannot import pipeline module: {e}")
        return False
    
    return mlflow_available


def demonstrate_training_workflow():
    """Demonstrate the complete training workflow with MLflow."""
    print("\n🚀 Demonstrating Training Workflow")
    print("=" * 50)
    
    script_path = Path(__file__).parent / "mlops_mlflow_pipeline.py"
    
    # 1. Train a Ridge model with MLflow tracking
    result1 = run_command([
        sys.executable, str(script_path), "train",
        "--dataset", "synthetic_yield",
        "--model", "ridge",
        "--alpha", "1.0",
        "--enable-mlflow",
        "--experiment-name", "semiconductor_demo_training",
        "--save", "ridge_model.joblib"
    ], "Training Ridge model with MLflow tracking")
    
    # 2. Train a Random Forest model for comparison
    result2 = run_command([
        sys.executable, str(script_path), "train", 
        "--dataset", "synthetic_yield",
        "--model", "random_forest",
        "--enable-mlflow",
        "--experiment-name", "semiconductor_demo_training",
        "--save", "rf_model.joblib"
    ], "Training Random Forest model for comparison")
    
    # 3. Train with drift injection to show monitoring capabilities
    result3 = run_command([
        sys.executable, str(script_path), "train",
        "--dataset", "synthetic_yield", 
        "--model", "ridge",
        "--enable-mlflow",
        "--experiment-name", "semiconductor_demo_drift",
        "--inject-drift",
        "--save", "ridge_drift_model.joblib"
    ], "Training with synthetic drift injection")
    
    return [result1, result2, result3]


def demonstrate_evaluation_workflow():
    """Demonstrate model evaluation with MLflow tracking."""
    print("\n📊 Demonstrating Evaluation Workflow")
    print("=" * 50)
    
    script_path = Path(__file__).parent / "mlops_mlflow_pipeline.py"
    
    # Check if models exist
    model_files = ["ridge_model.joblib", "rf_model.joblib", "ridge_drift_model.joblib"]
    for model_file in model_files:
        if not Path(model_file).exists():
            print(f"⚠️  Model file {model_file} not found, skipping evaluation")
            continue
        
        # Evaluate on clean data
        result1 = run_command([
            sys.executable, str(script_path), "evaluate",
            "--model-path", model_file,
            "--dataset", "synthetic_yield",
            "--enable-mlflow",
            "--experiment-name", "semiconductor_demo_evaluation"
        ], f"Evaluating {model_file} on clean data")
        
        # Evaluate on data with drift
        result2 = run_command([
            sys.executable, str(script_path), "evaluate",
            "--model-path", model_file,
            "--dataset", "synthetic_yield", 
            "--inject-drift",
            "--enable-mlflow",
            "--experiment-name", "semiconductor_demo_evaluation"
        ], f"Evaluating {model_file} on drifted data")


def demonstrate_prediction_workflow():
    """Demonstrate prediction capabilities."""
    print("\n🔮 Demonstrating Prediction Workflow")
    print("=" * 50)
    
    script_path = Path(__file__).parent / "mlops_mlflow_pipeline.py"
    
    # Sample input data representing typical semiconductor process parameters
    sample_inputs = [
        {
            "temperature": 455.0,
            "pressure": 2.6,
            "flow": 118.0,
            "time": 62.0,
            "temp_centered": 5.0,
            "pressure_sq": 6.76,
            "flow_time_inter": 7316.0,
            "temp_flow_inter": 53690.0
        },
        {
            "temperature": 435.0,
            "pressure": 2.3,
            "flow": 125.0,
            "time": 58.0,
            "temp_centered": -15.0,
            "pressure_sq": 5.29,
            "flow_time_inter": 7250.0,
            "temp_flow_inter": 54375.0
        }
    ]
    
    # Test predictions with different models
    for i, input_data in enumerate(sample_inputs, 1):
        for model_file in ["ridge_model.joblib", "rf_model.joblib"]:
            if Path(model_file).exists():
                result = run_command([
                    sys.executable, str(script_path), "predict",
                    "--model-path", model_file,
                    "--input-json", json.dumps(input_data)
                ], f"Prediction {i} with {model_file}")


def demonstrate_mlflow_management():
    """Demonstrate MLflow management capabilities."""
    print("\n🎛️  Demonstrating MLflow Management")
    print("=" * 50)
    
    script_path = Path(__file__).parent / "mlops_mlflow_pipeline.py"
    
    # List experiments
    result1 = run_command([
        sys.executable, str(script_path), "list-experiments"
    ], "Listing all MLflow experiments")
    
    # Start tracking for a new experiment
    result2 = run_command([
        sys.executable, str(script_path), "start-tracking",
        "--experiment", "semiconductor_fab_monitoring"
    ], "Starting tracking for new experiment")
    
    # Stop tracking
    result3 = run_command([
        sys.executable, str(script_path), "stop-tracking"
    ], "Stopping MLflow tracking")


def demonstrate_dependency_checking():
    """Demonstrate optional dependency handling."""
    print("\n🔧 Demonstrating Dependency Checking")
    print("=" * 50)
    
    # Show MLflow availability check
    print("\nMLflow Availability Check:")
    is_available = check_mlflow_availability()
    
    if is_available:
        print("✅ All MLflow features are available")
        
        # Show MLflow tracking URI
        try:
            import mlflow
            tracking_uri = mlflow.get_tracking_uri()
            print(f"📍 MLflow Tracking URI: {tracking_uri}")
        except Exception as e:
            print(f"⚠️  Could not get tracking URI: {e}")
    else:
        print("❌ MLflow not available - graceful fallback will be used")


def show_mlflow_ui_instructions():
    """Show instructions for accessing MLflow UI."""
    print("\n🌐 MLflow UI Access Instructions")
    print("=" * 50)
    
    print("""
To view your MLflow experiments and runs:

1. Start MLflow UI server (if not already running):
   mlflow ui --port 5000

2. Open your browser and navigate to:
   http://localhost:5000

3. Browse experiments and runs to see:
   - Parameter tracking (model types, hyperparameters)
   - Metric tracking (accuracy, loss, manufacturing metrics)
   - Artifact storage (models, plots, preprocessing pipelines)
   - Model registry and versioning

4. Key features to explore:
   - Compare runs across different models
   - View feature importance plots
   - Download trained models
   - Track model lineage and versioning
   - Monitor experiment progress over time

5. For production deployment:
   - Use model registry to promote models (None → Staging → Production)
   - Set up webhooks for deployment automation
   - Monitor model performance over time
""")


def cleanup_example_files():
    """Clean up example files created during demo."""
    print("\n🧹 Cleaning up example files")
    print("=" * 50)
    
    files_to_clean = [
        "ridge_model.joblib",
        "rf_model.joblib", 
        "ridge_drift_model.joblib",
        "mlflow.db",
    ]
    
    dirs_to_clean = [
        "mlruns",
        "mlflow_artifacts"
    ]
    
    for file_path in files_to_clean:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"🗑️  Removed {file_path}")
    
    import shutil
    for dir_path in dirs_to_clean:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
            print(f"🗑️  Removed directory {dir_path}")


def main():
    """Main demonstration script."""
    print("🏭 MLflow Integration Example for Semiconductor Manufacturing")
    print("=" * 70)
    print("""
This script demonstrates comprehensive MLflow integration for semiconductor ML:

✨ Features Demonstrated:
- Optional MLflow dependency checking with graceful fallback
- Complete experiment lifecycle management  
- Parameter, metric, and artifact logging
- Model registry integration
- Manufacturing-specific metrics (PWS, Estimated Loss, Yield Rate)
- Drift detection and monitoring capabilities
- Production-ready CLI interface

🎯 Use Cases Covered:
- Model training with hyperparameter tracking
- Model comparison and evaluation
- Artifact storage (models, plots, preprocessing)
- Production deployment simulation
- Monitoring and alerting workflows
""")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install required dependencies.")
        print("Run: pip install mlflow scikit-learn pandas numpy matplotlib")
        return 1
    
    print("\n🚀 Starting MLflow Integration Demonstration...")
    
    try:
        # Run demonstration workflows
        demonstrate_dependency_checking()
        demonstrate_training_workflow()
        demonstrate_evaluation_workflow()
        demonstrate_prediction_workflow()
        demonstrate_mlflow_management()
        
        # Show UI instructions
        show_mlflow_ui_instructions()
        
        print("\n✅ MLflow Integration Demonstration Complete!")
        print("""
🎉 Summary:
- Trained multiple models with MLflow tracking
- Demonstrated evaluation with drift detection
- Showed prediction capabilities  
- Explored MLflow management features
- Generated artifacts and metrics for analysis

📊 Next Steps:
1. Start MLflow UI to explore results: mlflow ui --port 5000
2. Compare model performance across different runs
3. Use model registry for production deployment
4. Set up monitoring for ongoing model performance
""")
        
        # Ask if user wants to clean up
        response = input("\n🧹 Clean up example files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            cleanup_example_files()
            print("✅ Cleanup complete!")
        else:
            print("📁 Example files preserved for further exploration")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())