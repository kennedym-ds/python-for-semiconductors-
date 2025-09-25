# Wafer Defect Classifier - Starter Project

A baseline binary classification project for semiconductor wafer defect detection using classical machine learning approaches.

## Overview

This project provides a production-ready pipeline for wafer defect classification with standardized CLI interface and manufacturing-specific metrics. It serves as a foundation for understanding classification problems in semiconductor manufacturing.

## Features

- **Classical ML Models**: Logistic regression, SVM, decision trees, random forest, gradient boosting
- **Data Handling**: Synthetic wafer defect pattern generation for learning
- **Preprocessing**: Automatic imputation, scaling, and optional SMOTE oversampling
- **Manufacturing Metrics**: PWS (Prediction Within Spec), Estimated Loss, standard classification metrics
- **CLI Interface**: Standardized train/evaluate/predict commands with JSON output
- **Model Persistence**: Save/load trained models with metadata
- **Threshold Optimization**: Precision/recall constraint satisfaction

## Quick Start

### Training a Model

```bash
# Train with synthetic data (default)
python wafer_defect_pipeline.py train --dataset synthetic_wafer --model logistic --save wafer_model.joblib

# Train with specific parameters
python wafer_defect_pipeline.py train \
    --dataset synthetic_wafer_500_0.2 \
    --model rf \
    --min-precision 0.85 \
    --save rf_model.joblib
```

### Evaluating a Model

```bash
# Evaluate on test data
python wafer_defect_pipeline.py evaluate \
    --model-path wafer_model.joblib \
    --dataset synthetic_wafer
```

### Making Predictions

```bash
# Predict on single wafer
python wafer_defect_pipeline.py predict \
    --model-path wafer_model.joblib \
    --input-json '{"center_density":0.12, "edge_density":0.05, "defect_area_ratio":0.08, "defect_spread":2.5, "total_pixels":3000, "defect_pixels":240}'
```

## Data

### Synthetic Wafer Defects

The pipeline includes a synthetic wafer defect generator that creates realistic defect patterns:

- **Center defects**: Clustered defects in wafer center
- **Edge defects**: Ring-shaped defects near wafer edge  
- **Scratch defects**: Linear defects across wafer
- **Random defects**: Scattered random defects

### Features

Generated features include:
- `center_density`: Defect density in center region
- `edge_density`: Defect density in edge region
- `defect_area_ratio`: Ratio of defected area to total area
- `defect_spread`: Spatial spread of defects
- `total_pixels`: Total valid wafer area
- `defect_pixels`: Total defected pixels

### Datasets

- `synthetic_wafer`: Default synthetic dataset (1000 samples, 15% defect rate)
- `synthetic_wafer_<n>_<rate>`: Custom synthetic dataset (e.g., `synthetic_wafer_500_0.25`)

## Models

Supported models:
- `logistic`: Logistic Regression (default)
- `linear_svm`: Linear Support Vector Machine
- `tree`: Decision Tree
- `rf`: Random Forest
- `gb`: Gradient Boosting

## Metrics

### Standard Classification Metrics
- ROC AUC: Area under ROC curve
- PR AUC: Area under precision-recall curve  
- MCC: Matthews Correlation Coefficient
- Balanced Accuracy: Average of sensitivity and specificity
- Precision, Recall, F1-Score

### Manufacturing-Specific Metrics
- **PWS (Prediction Within Spec)**: Percentage of predictions matching true labels
- **Estimated Loss**: Cost-weighted error metric (FN cost = 10x FP cost)
- **False Positive/Negative Counts**: Direct counts for process monitoring

## CLI Reference

### Train Command

```bash
python wafer_defect_pipeline.py train [OPTIONS]

Options:
  --dataset DATASET              Dataset name (default: synthetic_wafer)
  --model {logistic,linear_svm,tree,rf,gb}  Model type (default: logistic)
  --use-smote                    Apply SMOTE oversampling
  --smote-k-neighbors INT        SMOTE k neighbors (default: 5)
  --no-class-weight              Disable class weight balancing
  --min-precision FLOAT          Minimum precision constraint
  --min-recall FLOAT             Minimum recall constraint
  --C FLOAT                      Regularization parameter (default: 1.0)
  --max-depth INT                Tree max depth (default: 6)
  --n-estimators INT             Ensemble size (default: 300)
  --save PATH                    Save model path
```

### Evaluate Command

```bash
python wafer_defect_pipeline.py evaluate [OPTIONS]

Options:
  --model-path PATH              Path to saved model (required)
  --dataset DATASET              Dataset name (default: synthetic_wafer)
```

### Predict Command

```bash
python wafer_defect_pipeline.py predict [OPTIONS]

Options:
  --model-path PATH              Path to saved model (required)
  --input-json JSON              JSON input record
  --input-file PATH              Path to JSON input file
```

## Examples

### High-Precision Classifier

```bash
# Train model optimized for high precision (fewer false alarms)
python wafer_defect_pipeline.py train \
    --dataset synthetic_wafer \
    --model rf \
    --min-precision 0.9 \
    --save high_precision_model.joblib
```

### High-Recall Classifier

```bash
# Train model optimized for high recall (catch more defects)
python wafer_defect_pipeline.py train \
    --dataset synthetic_wafer \
    --model gb \
    --min-recall 0.95 \
    --save high_recall_model.joblib
```

### Imbalanced Data Handling

```bash
# Use SMOTE for highly imbalanced data
python wafer_defect_pipeline.py train \
    --dataset synthetic_wafer_1000_0.05 \
    --model logistic \
    --use-smote \
    --save balanced_model.joblib
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest test_wafer_defect_pipeline.py -v

# Run specific test class
python -m pytest test_wafer_defect_pipeline.py::TestCLIInterface -v
```

## Output Format

All CLI commands return JSON output for programmatic consumption:

### Training Output

```json
{
  "status": "trained",
  "metrics": {
    "roc_auc": 0.92,
    "pr_auc": 0.87,
    "precision": 0.85,
    "recall": 0.78,
    "f1": 0.81,
    "pws": 89.5,
    "estimated_loss": 23.0
  },
  "metadata": {
    "trained_at": "2024-01-15T10:30:45.123456+00:00",
    "model_type": "RandomForestClassifier",
    "n_features_in": 6,
    "threshold": 0.65
  }
}
```

### Prediction Output

```json
{
  "prediction": 1,
  "probability": 0.87,
  "threshold": 0.5,
  "input": {
    "center_density": 0.12,
    "edge_density": 0.05,
    "defect_area_ratio": 0.08
  }
}
```

## Project Structure

```
wafer_defect_classifier/
├── wafer_defect_pipeline.py      # Main pipeline script
├── test_wafer_defect_pipeline.py # Unit tests
└── README.md                     # This file
```

## Dependencies

This project uses the basic tier dependencies:
- numpy>=1.24.0
- pandas>=2.0.0  
- scikit-learn>=1.3.0
- joblib>=1.3.0

Optional dependencies:
- imbalanced-learn (for SMOTE support)

## Related Modules

- Module 3.2: Classification fundamentals
- Module 6: Computer vision for wafer maps
- Module 10.1: Project architecture best practices

## Docker Deployment

### Quick Start with Docker

The easiest way to run this project is using Docker:

```bash
# Build and run the main service
docker-compose up wafer-defect-classifier

# Run with MLflow tracking
docker-compose up wafer-defect-classifier mlflow-server

# Development with Jupyter notebooks
docker-compose up jupyter

# Run all services
docker-compose up
```

### Service URLs

When running with docker-compose:
- **Main application**: http://localhost:8001
- **MLflow UI**: http://localhost:5001
- **Jupyter notebooks**: http://localhost:8888

### Docker Commands

```bash
# Train a model in Docker
docker-compose run wafer-defect-classifier python wafer_defect_pipeline.py train --dataset synthetic_wafer --model rf --enable-mlflow

# Evaluate model
docker-compose run wafer-defect-classifier python wafer_defect_pipeline.py evaluate --model-path model.joblib --dataset synthetic_wafer

# Interactive shell
docker-compose exec wafer-defect-classifier bash
```

## MLflow Experiment Tracking

### Enable MLflow Tracking

Add `--enable-mlflow` to any command to enable experiment tracking:

```bash
# Train with MLflow tracking
python wafer_defect_pipeline.py train --dataset synthetic_wafer --model rf --enable-mlflow --experiment-name "production_wafer_defects"

# Custom experiment name
python wafer_defect_pipeline.py train --enable-mlflow --experiment-name "fab_west_line_1"
```

### MLflow Features

- **Automatic Experiment Creation**: Experiments are created automatically
- **Parameter Tracking**: All hyperparameters logged automatically
- **Manufacturing Metrics**: PWS, Estimated Loss, and standard ML metrics
- **Model Registry**: Models automatically registered with versioning
- **Comparison**: Compare runs in the MLflow UI

### MLflow UI

Access the MLflow tracking UI at http://localhost:5001 to:
- Compare experiment runs
- View parameter and metric trends
- Download trained models
- Promote models to production

### Manufacturing Metrics in MLflow

This pipeline logs semiconductor-specific metrics:

- **PWS (Prediction Within Spec)**: Manufacturing quality metric
- **Estimated Loss**: Cost impact of prediction errors
- **Defect Detection Rate**: Percentage of defective wafers caught
- **False Positive Cost**: Impact of scrapping good wafers
- **False Negative Cost**: Impact of shipping defective wafers

## Notebook Tutorial

Explore the interactive tutorial:

```bash
# Run locally
jupyter notebook wafer_defect_tutorial.ipynb

# Run in Docker
docker-compose up jupyter
# Then navigate to http://localhost:8888
```

The tutorial covers:
- Data generation and exploration
- Model comparison and selection
- Manufacturing threshold optimization
- Production deployment examples

## Production Deployment

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install MLflow
pip install mlflow

# Optional: Install SMOTE support
pip install imbalanced-learn
```

### Kubernetes Deployment

Example Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wafer-defect-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wafer-defect-classifier
  template:
    metadata:
      labels:
        app: wafer-defect-classifier
    spec:
      containers:
      - name: classifier
        image: wafer-defect-classifier:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
```

## Next Steps

This baseline classifier can be extended with:
1. **Real Dataset Integration**: Connect to actual wafer map datasets
2. **Deep Learning**: CNN models for spatial pattern recognition
3. **Feature Engineering**: Advanced spatial and statistical features
4. **Model Ensemble**: Combine multiple model predictions
5. **Real-time Deployment**: API wrapper for production use
6. **Drift Monitoring**: Track model performance over time
7. **A/B Testing**: MLflow integration for production experiments
8. **Automated Retraining**: CI/CD pipelines with MLflow model registry

## Support

For questions or issues with this project, refer to:
- Module 3.2 classification materials
- Module 10.2 testing and QA guidelines
- Project architecture documentation