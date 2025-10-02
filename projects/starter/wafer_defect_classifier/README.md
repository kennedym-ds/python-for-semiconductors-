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
├── wafer_defect_pipeline.py         # Production pipeline (1000+ lines)
├── wafer_defect_tutorial.ipynb      # Interactive learning notebook with 8 exercises
├── wafer_defect_solution.ipynb      # Complete solutions and reference implementations
├── evaluate_submission.py           # Automated grading script (100-point rubric)
├── test_wafer_defect_pipeline.py    # Comprehensive test suite (37 tests, 95% coverage)
├── README.md                        # This file
├── BUG_FIXES_SUMMARY.md            # Bug resolution documentation
└── TEST_ENHANCEMENT_SUMMARY.md     # Test suite details and benchmarks
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

## Educational Materials

### For Learners

This project includes comprehensive learning materials for hands-on practice:

#### 📘 Interactive Tutorial (`wafer_defect_tutorial.ipynb`)

An interactive Jupyter notebook with 8 hands-on exercises covering the complete ML workflow:

**Section 1: Data Generation and Exploration**
- Exercise 1.1 (★ beginner): Generate synthetic wafer defect data
- Exercise 1.2 (★★ intermediate): Visualize feature distributions and identify discriminative features

**Section 2: Model Training and Comparison**
- Exercise 2.1 (★★ intermediate): Train multiple classifiers (logistic, SVM, tree, RF, GB)
- Exercise 2.2 (★★ intermediate): Visualize model performance with ROC curves

**Section 3: Manufacturing-Specific Metrics**
- Exercise 3.1 (★★★ advanced): Calculate manufacturing costs (FP vs FN asymmetry)
- Exercise 3.2 (★★★ advanced): Optimize decision thresholds for cost minimization

**Section 4: Model Deployment**
- Exercise 4.1 (★★ intermediate): Save production models with metadata
- Exercise 4.2 (★★ intermediate): Test model loading and CLI usage

**Expected Completion Time**: 2 hours for first-time learners

```bash
# Run locally
jupyter notebook wafer_defect_tutorial.ipynb

# Run in Docker
docker-compose up jupyter
# Then navigate to http://localhost:8888
```

#### 📗 Solution Notebook (`wafer_defect_solution.ipynb`)

Complete reference implementations for all tutorial exercises:
- **Exercise 1**: Data generation with statistical analysis
- **Exercise 2**: 5-model comparison with comprehensive metrics
- **Exercise 3**: Threshold optimization showing 10-30% cost savings
- **Exercise 4**: Production deployment checklist (40+ items)

Includes debugging tips, best practices, and manufacturing context throughout.

#### 🔍 Automated Grading (`evaluate_submission.py`)

Automated evaluation script for student submissions with 100-point scoring rubric:

```bash
# Grade a completed notebook
python evaluate_submission.py --notebook wafer_defect_tutorial.ipynb

# Save results to JSON
python evaluate_submission.py --notebook submission.ipynb --output-json grades.json

# Verbose feedback
python evaluate_submission.py --notebook submission.ipynb --verbose
```

**Scoring Rubric** (100 points total):
- Exercise 1 (Data Exploration): 20 points
- Exercise 2 (Model Training): 30 points
- Exercise 3 (Manufacturing Metrics): 25 points
- Exercise 4 (CLI Usage): 15 points
- Code Quality: 10 points

**Features**:
- Automated notebook execution and validation
- Code quality checks (PEP 8, documentation)
- Results validation against expected ranges
- Detailed feedback generation
- JSON output for LMS integration

### For Instructors

#### Common Student Mistakes

1. **Stratification**: Forgetting stratification in train/test split leads to imbalanced test sets
2. **Metrics Selection**: Using accuracy instead of ROC-AUC for imbalanced data
3. **Threshold Default**: Setting threshold = 0.5 without cost analysis (misses cost asymmetry)
4. **Model Metadata**: Not including timestamps/metrics when saving models (breaks versioning)

#### Teaching Tips

- **Exercise 1**: Emphasize 5-20% defect rate is typical in semiconductor manufacturing
- **Exercise 2**: Discuss why RF/GB outperform linear models (non-linear patterns)
- **Exercise 3**: Highlight FN costs 4-10x more than FP (customer returns vs inspection)
- **Exercise 4**: Connect CLI design to MES/ERP integration requirements

#### Grading Workflow

```bash
# Grade all submissions in a directory
for notebook in submissions/*.ipynb; do
    python evaluate_submission.py \
        --notebook "$notebook" \
        --output-json "grades/$(basename $notebook .ipynb)_grade.json" \
        --verbose
done

# Generate class summary
python -c "
import json
from pathlib import Path
grades = [json.load(open(f)) for f in Path('grades').glob('*.json')]
avg_score = sum(g['total_score'] for g in grades) / len(grades)
print(f'Class Average: {avg_score:.1f}/100')
"
```

#### Extensions and Projects

Suggested follow-up projects for advanced learners:
1. **Real Data Integration**: Use WM-811K dataset (`datasets/wm811k/`)
2. **Deep Learning**: Implement CNN models (see Module 6.2)
3. **Feature Engineering**: Add spatial statistics and pattern descriptors
4. **Model Ensemble**: Combine predictions for improved robustness
5. **Real-time Deployment**: Use FastAPI template (`infrastructure/api/`)
6. **Drift Monitoring**: Track performance degradation (see Module 5.2)

### Related Projects

After completing this project, learners can progress to:
- **yield_regression**: Predict wafer yield from process parameters
- **equipment_drift_monitor**: Time series anomaly detection for equipment health
- **die_defect_segmentation**: Computer vision for defect localization

### Prerequisites

**Required Knowledge**:
- Python basics (variables, functions, loops)
- Basic ML concepts (train/test split, overfitting, metrics)
- Numpy and pandas fundamentals

**Recommended Background**:
- Module 1: Python for Data Analysis
- Module 2: Statistics for Manufacturing
- Module 3: Machine Learning Fundamentals

**Semiconductor Domain**:
- Basic understanding of wafer fabrication process
- Quality control concepts (Type I/II errors)
- Manufacturing cost structure (scrap vs rework vs ship)

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
