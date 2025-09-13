# MLOps with MLflow Integration

This project demonstrates comprehensive MLflow tracking integration for semiconductor manufacturing ML workflows. It showcases production-ready MLOps practices including experiment tracking, model registry, artifact management, and deployment monitoring.

## 🎯 Overview

MLflow is an open-source platform for managing the complete machine learning lifecycle, including experimentation, reproducibility, deployment, and monitoring. This project integrates MLflow with semiconductor manufacturing ML pipelines to provide:

- **Experiment Tracking**: Track parameters, metrics, and artifacts across all ML experiments
- **Model Registry**: Centralized model storage with versioning and lifecycle management  
- **Artifact Storage**: Persistent storage for models, plots, and preprocessing pipelines
- **Deployment Monitoring**: Track model performance and drift in production environments
- **Collaboration**: Share experiments and results across teams

## 🏗️ Architecture

```
MLOps MLflow Integration
├── mlops_mlflow_pipeline.py     # Core pipeline with MLflow integration
├── example_run.py               # Complete demonstration script
├── test_mlops_pipeline.py       # Comprehensive test suite
└── README.md                    # This documentation
```

### Key Components

1. **MLOpsMLflowPipeline**: Production pipeline class with full MLflow integration
2. **MLflow Tracking Helpers**: Start/stop tracking with experiment management
3. **Manufacturing Metrics**: Semiconductor-specific metrics (PWS, Estimated Loss, Yield Rate)
4. **Artifact Management**: Automated logging of models, plots, and preprocessing pipelines
5. **Optional Dependency Handling**: Graceful fallback when MLflow is not available

## 🚀 Quick Start

### Prerequisites

Install advanced dependencies including MLflow:

```bash
# From repository root
python env_setup.py --tier advanced

# Or install specific dependencies
pip install mlflow scikit-learn pandas numpy matplotlib
```

### Basic Usage

```bash
# Train model with MLflow tracking
python mlops_mlflow_pipeline.py train --dataset synthetic_yield --model ridge --enable-mlflow

# Evaluate model with tracking
python mlops_mlflow_pipeline.py evaluate --model-path model.joblib --enable-mlflow

# Make predictions
python mlops_mlflow_pipeline.py predict --model-path model.joblib --input-json '{"temperature":455, "pressure":2.6, "flow":118, "time":62, "temp_centered":5, "pressure_sq":6.76, "flow_time_inter":7316, "temp_flow_inter":53690}'

# Run complete demonstration
python example_run.py
```

### Start MLflow UI

```bash
# Start MLflow tracking server
mlflow ui --port 5000

# Access at: http://localhost:5000
```

## 📋 Features Demonstrated

### 1. Optional Dependency Checking

```python
from mlops_mlflow_pipeline import check_mlflow_availability

# Check if MLflow is available
if check_mlflow_availability():
    # MLflow features enabled
    pipeline.enable_mlflow_tracking("semiconductor_experiment")
else:
    # Graceful fallback without MLflow
    print("Running without MLflow tracking")
```

### 2. Comprehensive Experiment Tracking

```python
pipeline = MLOpsMLflowPipeline()
pipeline.enable_mlflow_tracking("fab_west_yield_prediction")

# Training automatically logs:
# - Model parameters (alpha, model_type, random_seed)
# - Dataset information (n_samples, n_features, feature_columns)
# - Training metrics (MAE, RMSE, R², PWS, Estimated Loss)
# - Model artifacts and preprocessing pipelines
# - Feature importance plots (for supported models)
pipeline.fit(X_train, y_train, model_type="ridge", alpha=1.0)
```

### 3. Model Registry Integration

Models are automatically registered with versioning:

```python
# Models registered as: "semiconductor_yield_predictor_{model_type}"
# - semiconductor_yield_predictor_ridge
# - semiconductor_yield_predictor_random_forest
# - semiconductor_yield_predictor_lasso
```

### 4. Manufacturing-Specific Metrics

```python
# Automatically calculated and logged:
metrics = {
    'mae': 2.34,                    # Mean Absolute Error
    'rmse': 3.12,                   # Root Mean Square Error  
    'r2': 0.85,                     # R-squared
    'pws_percent': 92.5,            # Prediction Within Spec %
    'estimated_loss': 145.67,       # Estimated cost impact
    'yield_rate': 88.3              # % above yield threshold
}
```

### 5. Artifact Management

Automatically logged artifacts include:

- **Models**: Serialized model with metadata
- **Preprocessing Pipelines**: Feature scaling and imputation
- **Plots**: Feature importance, prediction vs actual
- **Metadata**: Feature schemas, model configuration

### 6. Drift Detection Integration

```python
# Train with normal data
python mlops_mlflow_pipeline.py train --dataset synthetic_yield --model ridge --enable-mlflow

# Evaluate with drift injection  
python mlops_mlflow_pipeline.py evaluate --model-path model.joblib --inject-drift --enable-mlflow
```

## 🛠️ CLI Interface

### Training Commands

```bash
# Basic training
python mlops_mlflow_pipeline.py train --dataset synthetic_yield --model ridge

# With MLflow tracking
python mlops_mlflow_pipeline.py train --dataset synthetic_yield --model ridge --enable-mlflow

# Custom experiment name
python mlops_mlflow_pipeline.py train --dataset synthetic_yield --model ridge --enable-mlflow --experiment-name "fab_east_optimization"

# Save trained model
python mlops_mlflow_pipeline.py train --dataset synthetic_yield --model ridge --save my_model.joblib

# With drift injection
python mlops_mlflow_pipeline.py train --dataset synthetic_yield --model ridge --inject-drift
```

### Evaluation Commands

```bash
# Evaluate saved model
python mlops_mlflow_pipeline.py evaluate --model-path my_model.joblib --dataset synthetic_yield

# With MLflow tracking
python mlops_mlflow_pipeline.py evaluate --model-path my_model.joblib --enable-mlflow

# Evaluate with drift
python mlops_mlflow_pipeline.py evaluate --model-path my_model.joblib --inject-drift
```

### Prediction Commands

```bash
# Single prediction from JSON
python mlops_mlflow_pipeline.py predict --model-path my_model.joblib --input-json '{"temperature":455, "pressure":2.6, "flow":118, "time":62, "temp_centered":5, "pressure_sq":6.76, "flow_time_inter":7316, "temp_flow_inter":53690}'

# Batch predictions from CSV
python mlops_mlflow_pipeline.py predict --model-path my_model.joblib --input-csv batch_data.csv
```

### MLflow Management Commands

```bash
# Start tracking for experiment
python mlops_mlflow_pipeline.py start-tracking --experiment "my_experiment"

# Stop tracking
python mlops_mlflow_pipeline.py stop-tracking

# List all experiments
python mlops_mlflow_pipeline.py list-experiments

# Start MLflow server
python mlops_mlflow_pipeline.py start-server --port 5000
```

## 📊 MLflow UI Features

### Experiment Organization

```
semiconductor_mlops_demo/
├── Run 1: ridge_training_synthetic_yield
│   ├── Parameters: model_type=ridge, alpha=1.0, n_samples=640
│   ├── Metrics: train_mae=2.34, train_rmse=3.12, train_r2=0.85
│   └── Artifacts: model/, preprocessing/, plots/
├── Run 2: random_forest_training_synthetic_yield  
│   ├── Parameters: model_type=random_forest, n_estimators=100
│   ├── Metrics: train_mae=1.89, train_rmse=2.67, train_r2=0.91
│   └── Artifacts: model/, preprocessing/, plots/
└── Run 3: ridge_evaluation_synthetic_yield
    ├── Parameters: eval_n_samples=160
    ├── Metrics: eval_mae=2.41, eval_rmse=3.18, eval_r2=0.84
    └── Artifacts: plots/prediction_vs_actual.png
```

### Key UI Features

1. **Compare Runs**: Side-by-side comparison of parameters and metrics
2. **Visualizations**: Interactive plots and charts
3. **Model Registry**: Version control and stage management
4. **Search and Filter**: Find specific experiments and runs
5. **Download Artifacts**: Access models and files directly

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_mlops_pipeline.py
```

Tests cover:
- Basic pipeline functionality without MLflow
- MLflow integration with tracking enabled
- Model training, evaluation, and prediction
- Error handling and edge cases
- CLI interface validation

## 🔧 Advanced Configuration

### Custom MLflow Configuration

```python
from mlops_mlflow_pipeline import MLflowConfig, MLOpsMLflowPipeline

# Custom configuration
config = MLflowConfig(
    experiment_name="fab_west_yield_optimization",
    tracking_uri="http://mlflow-server:5000",
    artifact_location="s3://mlflow-artifacts/",
    enable_autolog=True,
    log_model_signature=True,
    log_input_example=True
)

pipeline = MLOpsMLflowPipeline(config=config)
```

### Production Deployment

```python
import mlflow

# Load model from registry
model_name = "semiconductor_yield_predictor_ridge"
model_version = "1" 
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

# Make predictions
predictions = model.predict(production_data)
```

### Custom Tags and Context

```python
# Training with manufacturing context
tags = {
    "fab_location": "west",
    "process_node": "7nm", 
    "tool_set": "ASML_EUV_001",
    "recipe_version": "R_v2.1",
    "shift": "day"
}

pipeline.fit(X, y, run_name="production_training_lot_47", tags=tags)
```

## 📈 Manufacturing Integration

### Process Parameter Tracking

```python
# Log manufacturing context
mlflow.log_param("fab_location", "west")
mlflow.log_param("process_node", "7nm")
mlflow.log_param("tool_set", "ASML_EUV_001") 
mlflow.log_param("recipe_version", "R_v2.1")
mlflow.log_param("lot_id", "LOT_2024_W47_001")
```

### Quality Metrics

```python
# Manufacturing-specific metrics automatically logged
{
    "pws_percent": 92.5,        # Prediction Within Spec
    "estimated_loss": 145.67,   # Cost impact ($)
    "yield_rate": 88.3,         # Yield percentage
    "defect_rate": 2.1,         # Estimated defect rate
    "oee_impact": 0.95          # OEE impact factor
}
```

### Production Monitoring

```python
# Monitor model performance over time
def log_production_batch(inputs, predictions, actuals=None):
    with mlflow.start_run(nested=True):
        mlflow.log_metric("batch_size", len(inputs))
        mlflow.log_metric("avg_prediction", predictions.mean())
        
        if actuals is not None:
            mae = mean_absolute_error(actuals, predictions)
            mlflow.log_metric("batch_mae", mae)
```

## 🔍 Troubleshooting

### Common Issues

**MLflow Connection Errors**
```bash
# Check MLflow server status
curl http://localhost:5000/health

# Restart MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

**Import Errors**
```bash
# Check MLflow installation
pip show mlflow

# Reinstall if needed
pip install --upgrade mlflow
```

**Permission Errors**
```bash
# Check directory permissions
ls -la mlruns/
chmod -R 755 mlruns/
```

**Database Lock Errors**
```bash
# Stop all MLflow processes
pkill -f mlflow

# Remove lock file
rm mlflow.db-wal mlflow.db-shm

# Restart server
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

## 📚 Learning Resources

### MLflow Documentation
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Projects](https://mlflow.org/docs/latest/projects.html)

### Semiconductor Manufacturing ML
- Process parameter optimization
- Yield prediction and analysis
- Quality control and SPC
- Equipment monitoring and maintenance

### Production MLOps
- CI/CD for ML models
- Model monitoring and alerting
- A/B testing frameworks
- Feature stores and data versioning

## 🤝 Contributing

To contribute improvements:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/mlflow-enhancement`
3. Make changes and add tests
4. Update documentation
5. Submit a pull request

### Development Guidelines

- Maintain backward compatibility
- Add comprehensive tests for new features
- Follow semiconductor manufacturing conventions
- Document all configuration options
- Include examples for complex features

## 📄 License

This project is part of the Python for Semiconductors learning series. See the main repository license for usage terms.

---

**🎉 Happy MLOps! 🚀**

*This project demonstrates production-ready MLOps practices specifically designed for semiconductor manufacturing environments, showcasing how to integrate MLflow tracking into existing ML workflows while maintaining manufacturing domain expertise.*