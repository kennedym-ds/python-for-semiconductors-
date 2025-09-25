# Anomaly Detection for Equipment Monitoring - Quick Reference

## Command Line Interface

### Training Models
```bash
# Basic training with default parameters
python anomaly_detection_pipeline.py train --data-path data/equipment_logs.csv --output-path models/anomaly_detector.joblib

# Advanced training with custom configuration
python anomaly_detection_pipeline.py train \
  --data-path data/equipment_logs.csv \
  --output-path models/anomaly_detector.joblib \
  --algorithms isolation_forest,one_class_svm,autoencoder \
  --ensemble-method voting \
  --contamination 0.05 \
  --test-size 0.3

# Time series training
python anomaly_detection_pipeline.py train \
  --data-path data/time_series_data.csv \
  --output-path models/ts_anomaly_detector.joblib \
  --time-column timestamp \
  --window-size 100 \
  --algorithms isolation_forest,autoencoder
```

### Real-time Detection
```bash
# Start real-time monitoring
python anomaly_detection_pipeline.py monitor \
  --model-path models/anomaly_detector.joblib \
  --data-stream tcp://localhost:5555 \
  --alert-threshold -0.5 \
  --output-alerts alerts/

# Batch prediction
python anomaly_detection_pipeline.py predict \
  --model-path models/anomaly_detector.joblib \
  --input-data data/new_equipment_data.csv \
  --output-path results/anomaly_predictions.csv
```

### Model Evaluation
```bash
# Evaluate trained model
python anomaly_detection_pipeline.py evaluate \
  --model-path models/anomaly_detector.joblib \
  --test-data data/test_equipment_data.csv \
  --labels-column is_anomaly \
  --metrics precision,recall,f1,roc_auc

# Cross-validation evaluation
python anomaly_detection_pipeline.py cross-validate \
  --data-path data/equipment_logs.csv \
  --algorithms isolation_forest,one_class_svm \
  --cv-folds 5 \
  --contamination 0.05
```

## Python API

### Basic Usage
```python
from anomaly_detection_pipeline import AnomalyDetectionPipeline
import pandas as pd
import numpy as np

# Load equipment data
data = pd.read_csv('equipment_data.csv')
feature_columns = ['temperature', 'pressure', 'flow_rate', 'vibration']
X = data[feature_columns].values

# Initialize pipeline
pipeline = AnomalyDetectionPipeline(
    algorithms=['isolation_forest', 'one_class_svm'],
    ensemble_method='voting',
    contamination=0.05
)

# Train the model
pipeline.fit(X)

# Detect anomalies
predictions = pipeline.predict(X)
scores = pipeline.decision_function(X)

# Save model
pipeline.save('models/equipment_anomaly_detector.joblib')
```

### Advanced Configuration
```python
# Custom algorithm parameters
pipeline = AnomalyDetectionPipeline(
    algorithms=['isolation_forest', 'one_class_svm', 'autoencoder'],
    algorithm_params={
        'isolation_forest': {
            'n_estimators': 200,
            'max_samples': 'auto',
            'contamination': 0.05,
            'random_state': 42
        },
        'one_class_svm': {
            'kernel': 'rbf',
            'gamma': 'scale',
            'nu': 0.05
        },
        'autoencoder': {
            'encoding_dim': 10,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    },
    ensemble_method='weighted',
    ensemble_weights=[0.4, 0.3, 0.3]
)
```

### Time Series Anomaly Detection
```python
from anomaly_detection_pipeline import TimeSeriesAnomalyDetector

# Initialize time series detector
ts_detector = TimeSeriesAnomalyDetector(
    window_size=100,
    algorithms=['isolation_forest', 'autoencoder'],
    seasonal_decomposition=True
)

# Prepare time series data
time_data = pd.read_csv('equipment_timeseries.csv', parse_dates=['timestamp'])
ts_detector.fit(time_data)

# Detect anomalies in new time series
new_data = pd.read_csv('new_timeseries.csv', parse_dates=['timestamp'])
anomaly_periods = ts_detector.detect_anomaly_periods(new_data)
```

### Real-time Streaming Detection
```python
from anomaly_detection_pipeline import StreamingAnomalyDetector

# Initialize streaming detector
streaming_detector = StreamingAnomalyDetector(
    model_path='models/anomaly_detector.joblib',
    buffer_size=1000,
    alert_threshold=-0.5
)

# Process streaming data
def process_data_stream():
    while True:
        # Get new data point (from sensors, database, etc.)
        new_sample = get_next_sensor_reading()
        
        # Process through detector
        result = streaming_detector.process_sample(new_sample)
        
        if result['is_anomaly']:
            print(f"⚠️ ANOMALY DETECTED: Score {result['score']:.3f}")
            send_alert(result)
        
        time.sleep(1)  # 1 second intervals
```

## Algorithm Quick Reference

### 1. Isolation Forest
**Best for**: High-dimensional data, fast processing
```python
from sklearn.ensemble import IsolationForest

detector = IsolationForest(
    n_estimators=100,      # Number of trees
    contamination=0.1,     # Expected anomaly rate
    max_samples='auto',    # Samples per tree
    max_features=1.0,      # Features per tree
    random_state=42
)
```

**Parameters**:
- `n_estimators`: 100-200 for most applications
- `contamination`: 0.01-0.1 (1%-10% anomalies)
- `max_samples`: 'auto' or 256 for large datasets
- `max_features`: 1.0 for full features, 0.5-0.8 for speedup

### 2. One-Class SVM
**Best for**: Complex decision boundaries, small datasets
```python
from sklearn.svm import OneClassSVM

detector = OneClassSVM(
    kernel='rbf',          # Radial basis function
    gamma='scale',         # Kernel coefficient
    nu=0.05,              # Upper bound on outliers
    degree=3,             # For polynomial kernel
    coef0=0.0,            # For sigmoid/polynomial
    tol=1e-3,             # Tolerance for stopping
    shrinking=True,       # Use shrinking heuristic
    cache_size=200        # Kernel cache size (MB)
)
```

**Parameters**:
- `kernel`: 'rbf' (most common), 'linear', 'poly', 'sigmoid'
- `gamma`: 'scale', 'auto', or float (0.001-1.0)
- `nu`: 0.01-0.1 (expected anomaly fraction)

### 3. Autoencoder
**Best for**: Complex patterns, high-dimensional data
```python
import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

**Hyperparameters**:
- `encoding_dim`: 5-20 for most applications
- `learning_rate`: 0.001-0.01
- `batch_size`: 32-128
- `epochs`: 50-200

### 4. Local Outlier Factor (LOF)
**Best for**: Local density-based anomalies
```python
from sklearn.neighbors import LocalOutlierFactor

detector = LocalOutlierFactor(
    n_neighbors=20,        # Number of neighbors
    algorithm='auto',      # Algorithm for neighbor search
    leaf_size=30,         # Leaf size for tree algorithms
    metric='minkowski',   # Distance metric
    p=2,                  # Power for Minkowski metric
    contamination=0.1,    # Expected anomaly rate
    novelty=True          # For prediction on new data
)
```

## Performance Metrics

### Classification Metrics (when labels available)
```python
from sklearn.metrics import classification_report, roc_auc_score

# Basic metrics
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

# ROC-AUC
auc_score = roc_auc_score(y_true, anomaly_scores)

# Comprehensive report
report = classification_report(y_true, predictions, 
                             target_names=['Normal', 'Anomaly'])
```

### Manufacturing-Specific Metrics
```python
# Mean Time Between False Alarms
def calculate_mtbfa(predictions, timestamps):
    false_alarms = np.where((predictions == 1) & (y_true == 0))[0]
    if len(false_alarms) <= 1:
        return float('inf')
    
    time_diffs = np.diff([timestamps[i] for i in false_alarms])
    return np.mean(time_diffs)

# Detection Delay
def calculate_detection_delay(predictions, y_true, timestamps):
    anomaly_starts = np.where(np.diff(np.concatenate(([0], y_true))) == 1)[0]
    detection_delays = []
    
    for start in anomaly_starts:
        detection_idx = np.where(predictions[start:] == 1)[0]
        if len(detection_idx) > 0:
            delay = timestamps[start + detection_idx[0]] - timestamps[start]
            detection_delays.append(delay)
    
    return np.mean(detection_delays) if detection_delays else float('inf')
```

## Threshold Tuning

### Percentile-Based Thresholds
```python
def set_percentile_threshold(anomaly_scores, percentile=95):
    """Set threshold based on score percentile."""
    threshold = np.percentile(anomaly_scores, percentile)
    predictions = (anomaly_scores > threshold).astype(int)
    return threshold, predictions
```

### ROC-Based Optimal Threshold
```python
from sklearn.metrics import roc_curve

def find_optimal_threshold(y_true, scores):
    """Find threshold that maximizes Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]
```

### Precision-Recall Based Threshold
```python
from sklearn.metrics import precision_recall_curve

def find_precision_threshold(y_true, scores, min_precision=0.8):
    """Find threshold that maintains minimum precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid_idx = np.where(precision >= min_precision)[0]
    if len(valid_idx) > 0:
        return thresholds[valid_idx[0]]
    return thresholds[-1]  # Most conservative threshold
```

## Data Preprocessing

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard scaling (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Robust scaling (using median and IQR)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### Handling Missing Values
```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# KNN imputation
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Forward fill for time series
df_filled = df.fillna(method='ffill')
```

### Feature Engineering
```python
# Rolling statistics for time series
def add_rolling_features(df, window=10):
    df['rolling_mean'] = df['value'].rolling(window).mean()
    df['rolling_std'] = df['value'].rolling(window).std()
    df['rolling_min'] = df['value'].rolling(window).min()
    df['rolling_max'] = df['value'].rolling(window).max()
    return df

# Lag features
def add_lag_features(df, lags=[1, 2, 3]):
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    return df

# Difference features
def add_difference_features(df):
    df['diff_1'] = df['value'].diff()
    df['diff_2'] = df['value'].diff(2)
    df['pct_change'] = df['value'].pct_change()
    return df
```

## Model Comparison

### Cross-Validation for Anomaly Detection
```python
def anomaly_detection_cv(X, algorithms, cv_folds=5, contamination=0.1):
    """Cross-validation for anomaly detection models."""
    from sklearn.model_selection import KFold
    
    results = {}
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for alg_name in algorithms:
        scores = []
        
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            
            # Initialize detector
            if alg_name == 'isolation_forest':
                detector = IsolationForest(contamination=contamination)
            elif alg_name == 'one_class_svm':
                detector = OneClassSVM(nu=contamination)
            
            # Train and predict
            detector.fit(X_train)
            test_scores = detector.decision_function(X_test)
            
            # Use average score as performance metric
            scores.append(np.mean(test_scores))
        
        results[alg_name] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
    
    return results
```

### Ensemble Model Comparison
```python
def compare_ensemble_methods(X, y, algorithms):
    """Compare different ensemble methods."""
    from anomaly_detection_pipeline import AnomalyDetectionPipeline
    from sklearn.metrics import f1_score
    
    ensemble_methods = ['voting', 'weighted', 'stacking']
    results = {}
    
    for method in ensemble_methods:
        pipeline = AnomalyDetectionPipeline(
            algorithms=algorithms,
            ensemble_method=method,
            contamination=0.05
        )
        
        # Train and predict
        pipeline.fit(X)
        predictions = pipeline.predict(X)
        
        # Evaluate (if labels available)
        if y is not None:
            f1 = f1_score(y, predictions)
            results[method] = f1
        else:
            # Use internal metrics
            scores = pipeline.decision_function(X)
            results[method] = np.mean(scores)
    
    return results
```

## Troubleshooting Guide

### Common Issues

#### 1. High False Positive Rate
**Symptoms**: Too many false alarms
**Solutions**:
```python
# Increase contamination parameter
detector = IsolationForest(contamination=0.01)  # Reduced from 0.1

# Use more conservative threshold
threshold = np.percentile(scores, 99)  # Top 1% instead of 5%

# Add domain knowledge filters
def filter_false_positives(predictions, scores, context):
    # Example: Filter during maintenance windows
    maintenance_mask = context['is_maintenance']
    predictions[maintenance_mask] = 0
    return predictions
```

#### 2. Missing True Anomalies
**Symptoms**: Low recall, missing actual failures
**Solutions**:
```python
# Increase contamination parameter
detector = IsolationForest(contamination=0.1)  # Increased sensitivity

# Use ensemble methods
pipeline = AnomalyDetectionPipeline(
    algorithms=['isolation_forest', 'one_class_svm', 'lof'],
    ensemble_method='voting'
)

# Lower threshold for more sensitive detection
threshold = np.percentile(scores, 90)  # Top 10% instead of 5%
```

#### 3. Model Degradation Over Time
**Symptoms**: Increasing false positives or missed anomalies
**Solutions**:
```python
# Implement periodic retraining
class AdaptiveAnomalyDetector:
    def __init__(self, retrain_frequency_days=30):
        self.retrain_frequency = retrain_frequency_days
        self.last_training = datetime.now()
        self.performance_history = []
    
    def check_retraining_needed(self):
        days_since_training = (datetime.now() - self.last_training).days
        
        if days_since_training >= self.retrain_frequency:
            return True
        
        # Check performance degradation
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-5:])
            historical_performance = np.mean(self.performance_history[:-5])
            
            if recent_performance < 0.8 * historical_performance:
                return True
        
        return False
```

#### 4. Scalability Issues
**Symptoms**: Slow training/prediction, memory issues
**Solutions**:
```python
# Use sampling for large datasets
from sklearn.utils import resample

if len(X) > 100000:
    X_sample = resample(X, n_samples=50000, random_state=42)
    detector.fit(X_sample)
else:
    detector.fit(X)

# Batch processing for predictions
def batch_predict(detector, X, batch_size=1000):
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_pred = detector.predict(batch)
        predictions.extend(batch_pred)
    return np.array(predictions)
```

## Best Practices Checklist

### Data Quality
- [ ] Remove or impute missing values
- [ ] Handle outliers in training data
- [ ] Ensure consistent data types and formats
- [ ] Validate sensor readings and remove impossible values
- [ ] Check for data leakage from future information

### Model Selection
- [ ] Choose appropriate algorithms for data characteristics
- [ ] Consider ensemble methods for robustness
- [ ] Validate contamination parameter with domain expertise
- [ ] Test multiple threshold selection strategies
- [ ] Compare performance across different time periods

### Deployment
- [ ] Implement model versioning and rollback capability
- [ ] Set up performance monitoring and alerting
- [ ] Create clear escalation procedures for different alert levels
- [ ] Document model assumptions and limitations
- [ ] Train operations staff on interpreting alerts

### Monitoring
- [ ] Track false positive and false negative rates
- [ ] Monitor data drift and concept drift
- [ ] Log all predictions and outcomes for analysis
- [ ] Regular performance reviews with domain experts
- [ ] Automated retraining triggers and procedures