# Advanced Equipment Anomaly Detection Project

## Overview

This project implements unsupervised anomaly detection for semiconductor equipment monitoring using state-of-the-art machine learning techniques. The system provides real-time anomaly detection capabilities with comprehensive threshold tuning and detailed interval reporting.

## Features

### Core Algorithms
- **Isolation Forest**: Tree-based anomaly detection optimized for high-dimensional time series
- **Gaussian Mixture Models (GMM)**: Probabilistic approach for complex anomaly patterns

### Time-Series Engineering
- Rolling statistics (5, 15, 60-minute windows)
- First and second order differences
- Percentage change calculations
- Cross-correlations between sensor readings
- Time-based features (hour, day of week, weekend flags)

### Manufacturing-Specific Metrics
- **Detection Rate**: Percentage of true anomalies correctly identified
- **False Alarm Rate**: Percentage of normal operations flagged as anomalous
- **Precision Within Spec (PWS)**: Equipment-specific accuracy metrics
- **Estimated Cost**: Financial impact of false alarms vs missed anomalies
- **ROC AUC**: Area under receiver operating characteristic curve
- **PR AUC**: Area under precision-recall curve

### Advanced Features
- Automated threshold optimization using Youden's J statistic
- Anomaly interval detection and export
- Model persistence with metadata tracking
- Comprehensive JSON-based reporting
- Synthetic data generation for testing and validation

## Installation and Setup

```bash
# Navigate to project directory
cd projects/advanced/anomaly_detection_equipment/

# Ensure dependencies are installed (from repository root)
python env_setup.py --tier basic
source .venv/bin/activate
```

## Usage Examples

### 1. Train an Isolation Forest Model

```bash
python anomaly_detection_pipeline.py train \
    --dataset synthetic_equipment \
    --method isolation_forest \
    --contamination 0.05 \
    --save equipment_anomaly_model.joblib
```

**Output Example:**
```json
{
  "status": "trained",
  "method": "isolation_forest",
  "optimal_threshold": -0.036,
  "metrics": {
    "precision": 0.500,
    "recall": 0.633,
    "f1_score": 0.559,
    "roc_auc": 0.906,
    "detection_rate": 0.633,
    "false_alarm_rate": 0.033,
    "estimated_cost": 12900.0
  }
}
```

### 2. Train a GMM Model

```bash
python anomaly_detection_pipeline.py train \
    --method gmm \
    --n-components 3 \
    --contamination 0.05 \
    --save gmm_equipment_model.joblib
```

### 3. Evaluate Model Performance

```bash
python anomaly_detection_pipeline.py evaluate \
    --model-path equipment_anomaly_model.joblib \
    --dataset synthetic_equipment
```

### 4. Real-time Single Prediction

```bash
python anomaly_detection_pipeline.py predict \
    --model-path equipment_anomaly_model.joblib \
    --input-json '{"temperature":455, "pressure":2.6, "vibration":0.8, "flow":120, "power":1000}'
```

**Output:**
```json
{
  "status": "predicted",
  "predictions": [1],
  "anomaly_scores": [-0.017],
  "anomalies_detected": 0
}
```

### 5. Batch Processing with Interval Export

```bash
python anomaly_detection_pipeline.py predict \
    --model-path equipment_anomaly_model.joblib \
    --input-file equipment_data.csv \
    --export-intervals anomaly_report.json
```

**Exported Report Example:**
```json
{
  "summary": {
    "total_samples": 2000,
    "detected_anomalies": 85,
    "anomaly_rate": 0.043,
    "num_intervals": 12
  },
  "intervals": [
    {
      "start_timestamp": "2024-01-01T08:15:00",
      "end_timestamp": "2024-01-01T08:23:00", 
      "duration_minutes": 8,
      "max_score": 0.045,
      "mean_score": 0.032
    }
  ]
}
```

## Data Format

### Input Requirements
The pipeline expects time-series data with the following core sensors:
- `temperature`: Equipment temperature (°C)
- `pressure`: Process pressure (bar)
- `vibration`: Vibration levels
- `flow`: Flow rate (L/min)
- `power`: Power consumption (W)
- `timestamp`: ISO format timestamp (optional for single predictions)

### Synthetic Data Generation
The pipeline includes realistic synthetic equipment data with:
- Daily temperature cycles
- Correlated sensor relationships
- Multiple anomaly types:
  - Temperature spikes
  - Pressure drops  
  - Vibration anomalies
  - Flow irregularities

## Model Configuration

### Isolation Forest Parameters
- `contamination`: Expected anomaly rate (default: 0.05)
- `n_estimators`: Number of trees (default: 100)
- `max_features`: Feature sampling ratio (default: 1.0)

### GMM Parameters  
- `n_components`: Number of Gaussian components (default: 2)
- `contamination`: Expected anomaly rate (default: 0.05)

## Performance Benchmarks

### Isolation Forest Results
- **ROC AUC**: 0.89-0.91
- **Precision**: 0.45-0.55
- **Recall**: 0.55-0.65
- **Training Time**: ~2-5 seconds (2000 samples)
- **Prediction Time**: ~10ms per sample

### GMM Results
- **ROC AUC**: 0.85-0.90
- **Precision**: 0.20-0.35 (higher recall, lower precision)
- **Recall**: 0.75-0.85
- **Training Time**: ~5-10 seconds (2000 samples)

## Manufacturing Integration

### Cost Analysis
The system provides cost-based evaluation metrics:
- **False Alarm Cost**: $100 per investigation
- **Missed Anomaly Cost**: $1000 per equipment failure
- **Total Cost**: Weighted combination based on operational impact

### Threshold Optimization
Automatic threshold tuning maximizes the Youden Index (Sensitivity + Specificity - 1) for optimal balance between detection rate and false alarms.

### Alert Prioritization
Anomalies are ranked by:
1. Anomaly score magnitude
2. Duration of anomalous behavior
3. Affected sensor combinations
4. Historical failure patterns

## Files and Output

### Model Artifacts
- `*.joblib`: Serialized trained models with preprocessing pipelines
- Model metadata includes training parameters, feature names, and performance metrics

### Export Files
- `*_intervals.json`: Detected anomaly intervals with timestamps and scores
- `*_detailed.csv`: Complete time series with anomaly scores and predictions

### Logs and Monitoring
- All operations produce structured JSON output for integration with monitoring systems
- Model metadata tracks training time, feature engineering, and performance metrics

## Next Steps and Extensions

### Advanced Features
1. **Ensemble Methods**: Combine Isolation Forest and GMM for improved performance
2. **Adaptive Thresholds**: Dynamic threshold adjustment based on operational conditions
3. **Multivariate Analysis**: Cross-equipment anomaly detection
4. **Root Cause Analysis**: Feature importance and anomaly explanation

### Production Deployment
1. **Real-time Streaming**: Integration with Kafka/streaming platforms
2. **Model Monitoring**: Drift detection and automatic retraining
3. **Alert Management**: Integration with existing SCADA/MES systems
4. **Dashboard Integration**: Real-time visualization and reporting

### Domain Extensions
1. **Equipment-Specific Models**: Tool-specific anomaly patterns
2. **Predictive Maintenance**: Time-to-failure estimation
3. **Quality Correlation**: Link anomalies to product quality metrics
4. **Environmental Factors**: Include fab environmental data

## Technical Validation

The pipeline has been validated with:
- ✅ End-to-end training and evaluation workflows
- ✅ Single-record and batch prediction capabilities  
- ✅ Model persistence and loading
- ✅ Comprehensive metrics and reporting
- ✅ Synthetic data generation with realistic patterns
- ✅ Multiple algorithm support (Isolation Forest, GMM)
- ✅ Time-series feature engineering
- ✅ Threshold optimization and ROC analysis
- ✅ Anomaly interval detection and export

## Contact and Support

For questions about implementation or extension of this anomaly detection system, refer to the module documentation in `modules/intermediate/module-5/` (predictive maintenance) and `modules/advanced/module-7/` (advanced detection methods).