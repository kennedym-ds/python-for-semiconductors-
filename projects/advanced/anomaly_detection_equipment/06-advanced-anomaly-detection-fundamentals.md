# Advanced Anomaly Detection for Equipment Monitoring - Fundamentals

## Overview

Anomaly detection in semiconductor manufacturing is a critical component of predictive maintenance and quality assurance systems. Unlike traditional supervised learning approaches, anomaly detection often operates in an unsupervised manner, identifying patterns that deviate significantly from normal operating conditions without requiring labeled examples of all possible failure modes.

## Theoretical Foundation

### What is Anomaly Detection?

Anomaly detection, also known as outlier detection or novelty detection, is the identification of observations that deviate significantly from the expected pattern in data. In semiconductor manufacturing, these anomalies often represent:

1. **Equipment malfunctions**: Sensor drift, mechanical wear, or component failures
2. **Process deviations**: Temperature excursions, pressure variations, or chemical composition changes  
3. **Environmental factors**: Contamination, vibration, or power fluctuations
4. **Maintenance needs**: Predictive indicators of required service or calibration

### Types of Anomalies

#### 1. Point Anomalies
Individual data points that are anomalous with respect to the rest of the data.

**Example**: A single temperature reading of 500°C when normal operation is 200°C ± 5°C

#### 2. Contextual Anomalies
Data points that are anomalous in a specific context but not otherwise.

**Example**: A pressure reading of 1 Torr during process startup (normal) vs. during steady-state operation (anomalous)

#### 3. Collective Anomalies
A collection of data points that together form an anomalous pattern.

**Example**: A gradual drift in multiple correlated parameters over several hours, indicating equipment degradation

### Mathematical Framework

#### Statistical Approach
Define anomalies based on statistical properties of the data:

```
Anomaly Score = |x - μ| / σ
```

Where:
- `x` is the observed value
- `μ` is the expected mean
- `σ` is the standard deviation

A threshold (typically 2-3 standard deviations) determines anomaly classification.

#### Distance-Based Approach
Anomalies are data points that are far from their k-nearest neighbors:

```
Anomaly Score = distance_to_kth_nearest_neighbor(x)
```

#### Density-Based Approach
Anomalies exist in regions of low data density:

```
Local Outlier Factor (LOF) = Σ(density_ratio(x, neighbor)) / k
```

## Core Algorithms

### 1. Isolation Forest

#### Principle
Isolation Forest detects anomalies by isolating observations through random feature selection and random split values. Anomalies are easier to isolate and thus require fewer splits in the decision tree.

#### Algorithm Details
```python
class IsolationForest:
    def __init__(self, n_estimators=100, contamination=0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.trees = []
    
    def fit(self, X):
        for _ in range(self.n_estimators):
            tree = self._build_isolation_tree(X)
            self.trees.append(tree)
    
    def _build_isolation_tree(self, X, depth=0, max_depth=None):
        if len(X) <= 1 or depth == max_depth:
            return LeafNode(size=len(X))
        
        # Random feature selection
        feature = np.random.choice(X.shape[1])
        split_value = np.random.uniform(X[:, feature].min(), X[:, feature].max())
        
        # Split data
        left_mask = X[:, feature] < split_value
        right_mask = ~left_mask
        
        return InternalNode(
            feature=feature,
            split_value=split_value,
            left=self._build_isolation_tree(X[left_mask], depth+1, max_depth),
            right=self._build_isolation_tree(X[right_mask], depth+1, max_depth)
        )
```

#### Advantages
- Fast training and prediction
- Linear time complexity O(n)
- No need for distance calculations
- Effective for high-dimensional data

#### Disadvantages
- Less effective in very high dimensions (curse of dimensionality)
- Performance depends on contamination parameter

#### Manufacturing Applications
- Real-time equipment monitoring
- Batch process anomaly detection
- Sensor drift identification

### 2. One-Class SVM

#### Principle
One-Class SVM learns a decision boundary around normal data points in feature space, treating anomalies as points that fall outside this boundary.

#### Mathematical Foundation
The optimization problem for One-Class SVM:

```
minimize: (1/2)||w||² + (1/νn)Σξᵢ - ρ
subject to: (w·φ(xᵢ)) ≥ ρ - ξᵢ, ξᵢ ≥ 0
```

Where:
- `w` is the normal vector to the hyperplane
- `φ(x)` is the kernel mapping function
- `ρ` is the offset parameter
- `ν` controls the fraction of outliers
- `ξᵢ` are slack variables

#### Kernel Functions
**RBF Kernel (most common)**:
```
K(x, y) = exp(-γ||x - y||²)
```

**Polynomial Kernel**:
```
K(x, y) = (γ⟨x, y⟩ + r)^d
```

#### Implementation Considerations
```python
from sklearn.svm import OneClassSVM

# Hyperparameter tuning is critical
svm_detector = OneClassSVM(
    kernel='rbf',           # Radial Basis Function
    gamma='scale',          # Kernel coefficient
    nu=0.05,               # Upper bound on outliers
    degree=3               # For polynomial kernel
)
```

#### Advantages
- Theoretically well-founded
- Effective with kernel trick for non-linear boundaries
- Memory efficient (support vectors only)
- Good performance with limited training data

#### Disadvantages
- Sensitive to hyperparameter selection
- Computationally expensive for large datasets
- Requires feature scaling

#### Manufacturing Applications
- Process control limit detection
- Equipment health monitoring
- Quality control in production

### 3. Autoencoder-Based Detection

#### Principle
Autoencoders are neural networks trained to reconstruct their input. Anomalies are identified by high reconstruction error, as the network has not learned to represent anomalous patterns.

#### Architecture
```python
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.Sigmoid()  # Assuming normalized input [0,1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

#### Training Process
```python
def train_autoencoder(model, data_loader, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
    
    return model
```

#### Anomaly Scoring
```python
def compute_reconstruction_error(model, x):
    with torch.no_grad():
        reconstructed = model(x)
        mse = ((x - reconstructed) ** 2).mean(dim=1)
        return mse.numpy()
```

#### Advanced Variants

**Variational Autoencoders (VAE)**:
- Add probabilistic encoding
- Better handling of data distribution
- More robust anomaly detection

**Denoising Autoencoders**:
- Trained with corrupted inputs
- Learn robust representations
- Better generalization to anomalies

#### Advantages
- Can capture complex non-linear relationships
- Unsupervised learning approach
- Scalable to high-dimensional data
- Interpretable reconstruction errors

#### Disadvantages
- Requires hyperparameter tuning
- Training time can be significant
- May require domain expertise for architecture design
- Sensitive to data preprocessing

#### Manufacturing Applications
- Complex multivariate process monitoring
- Image-based defect detection
- Vibration pattern analysis
- Chemical composition monitoring

## Ensemble Methods

### Voting-Based Ensemble
Combine predictions from multiple algorithms:

```python
class VotingEnsemble:
    def __init__(self, detectors):
        self.detectors = detectors
    
    def predict(self, X):
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(X)
            predictions.append(pred)
        
        # Majority voting
        ensemble_pred = np.mean(predictions, axis=0) > 0.5
        return ensemble_pred.astype(int)
```

### Weighted Ensemble
Weight detectors based on their individual performance:

```python
class WeightedEnsemble:
    def __init__(self, detectors, weights):
        self.detectors = detectors
        self.weights = weights / np.sum(weights)  # Normalize
    
    def decision_function(self, X):
        scores = []
        for detector in self.detectors:
            score = detector.decision_function(X)
            scores.append(score)
        
        weighted_score = np.average(scores, axis=0, weights=self.weights)
        return weighted_score
```

### Stacking Ensemble
Use a meta-learner to combine detector outputs:

```python
from sklearn.linear_model import LogisticRegression

class StackingEnsemble:
    def __init__(self, base_detectors, meta_learner=None):
        self.base_detectors = base_detectors
        self.meta_learner = meta_learner or LogisticRegression()
    
    def fit(self, X, y=None):
        # Train base detectors
        for detector in self.base_detectors:
            detector.fit(X)
        
        # Create meta-features
        meta_features = self._create_meta_features(X)
        
        # Train meta-learner (requires labels)
        if y is not None:
            self.meta_learner.fit(meta_features, y)
    
    def _create_meta_features(self, X):
        meta_features = []
        for detector in self.base_detectors:
            scores = detector.decision_function(X)
            meta_features.append(scores)
        return np.column_stack(meta_features)
```

## Performance Evaluation

### Challenges in Anomaly Detection Evaluation

1. **Imbalanced datasets**: Anomalies are typically rare (1-5% of data)
2. **Lack of labeled data**: True anomalies may be unknown
3. **Concept drift**: Definition of "normal" may change over time
4. **Context dependency**: What's anomalous in one situation may be normal in another

### Evaluation Metrics

#### When Labels are Available

**Precision**: 
```
Precision = TP / (TP + FP)
```
Fraction of detected anomalies that are actually anomalous.

**Recall (Sensitivity)**:
```
Recall = TP / (TP + FN)
```
Fraction of actual anomalies that are detected.

**F1-Score**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall.

**ROC-AUC**: Area under the Receiver Operating Characteristic curve
- Measures trade-off between true positive rate and false positive rate
- Values closer to 1.0 indicate better performance

**PR-AUC**: Area under the Precision-Recall curve
- More appropriate for imbalanced datasets
- Focuses on minority class (anomalies) performance

#### Manufacturing-Specific Metrics

**Mean Time Between False Alarms (MTBFA)**:
```
MTBFA = Total_Operating_Time / Number_of_False_Alarms
```

**Detection Delay**:
```
Detection_Delay = Time_of_Detection - Time_of_Actual_Anomaly_Onset
```

**Economic Impact Score**:
```
Economic_Impact = (Cost_of_Missed_Anomalies × FN) + (Cost_of_False_Alarms × FP)
```

#### When Labels are Not Available

**Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters

**Davies-Bouldin Index**: Ratio of within-cluster distances to between-cluster distances

**Reconstruction Error Statistics**: For autoencoder-based methods
- Mean reconstruction error
- Standard deviation of reconstruction error
- Percentage of samples above threshold

### Cross-Validation for Anomaly Detection

Traditional k-fold cross-validation may not be appropriate. Alternative approaches:

**Time Series Cross-Validation**:
```python
def time_series_cv_anomaly_detection(data, n_splits=5):
    split_size = len(data) // n_splits
    
    for i in range(n_splits - 1):
        train_end = (i + 1) * split_size
        test_start = train_end
        test_end = test_start + split_size
        
        train_data = data[:train_end]
        test_data = data[test_start:test_end]
        
        yield train_data, test_data
```

**Contamination-Aware Cross-Validation**:
- Ensure consistent anomaly rates across folds
- Stratify based on time periods or operating conditions

## Real-Time Implementation

### Streaming Anomaly Detection

#### Challenges
1. **Concept drift**: Normal patterns may change over time
2. **Limited memory**: Cannot store all historical data
3. **Low latency requirements**: Real-time decision making
4. **Continuous learning**: Model must adapt to new patterns

#### Sliding Window Approach
```python
class StreamingAnomalyDetector:
    def __init__(self, window_size=1000, retrain_frequency=100):
        self.window_size = window_size
        self.retrain_frequency = retrain_frequency
        self.data_buffer = []
        self.model = None
        self.samples_processed = 0
    
    def process_sample(self, sample):
        # Add to buffer
        self.data_buffer.append(sample)
        
        # Maintain window size
        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)
        
        # Retrain periodically
        if self.samples_processed % self.retrain_frequency == 0:
            self._retrain_model()
        
        # Make prediction
        if self.model is not None:
            prediction = self.model.predict([sample])
            score = self.model.decision_function([sample])
            return prediction[0], score[0]
        
        self.samples_processed += 1
        return 0, 0.0  # Default: not anomalous
    
    def _retrain_model(self):
        if len(self.data_buffer) >= 100:  # Minimum training size
            X = np.array(self.data_buffer)
            self.model = IsolationForest(contamination=0.1)
            self.model.fit(X)
```

#### Incremental Learning
```python
class IncrementalAnomalyDetector:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.statistics = {}
        self.thresholds = {}
    
    def update_statistics(self, sample):
        for i, value in enumerate(sample):
            if i not in self.statistics:
                self.statistics[i] = {'mean': value, 'var': 0, 'n': 1}
            else:
                stats = self.statistics[i]
                stats['n'] += 1
                delta = value - stats['mean']
                stats['mean'] += delta / stats['n']
                stats['var'] += delta * (value - stats['mean'])
    
    def compute_anomaly_score(self, sample):
        scores = []
        for i, value in enumerate(sample):
            if i in self.statistics:
                stats = self.statistics[i]
                if stats['n'] > 1:
                    std = np.sqrt(stats['var'] / (stats['n'] - 1))
                    z_score = abs(value - stats['mean']) / (std + 1e-8)
                    scores.append(z_score)
        
        return np.mean(scores) if scores else 0.0
```

### Performance Optimization

#### Memory Management
- Use circular buffers for data storage
- Implement lazy loading for large datasets
- Regular garbage collection for long-running processes

#### Computational Efficiency
- Vectorized operations using NumPy
- Parallel processing for ensemble methods
- GPU acceleration for deep learning models

#### Network Optimization
- Data compression for network transmission
- Batching of predictions to reduce overhead
- Caching of frequently accessed models

## Production Deployment Considerations

### Infrastructure Requirements

#### Hardware Specifications
**CPU-based Deployment**:
- Minimum: 4 cores, 8GB RAM
- Recommended: 8+ cores, 16GB+ RAM
- Storage: SSD for model loading performance

**GPU-based Deployment** (for deep learning models):
- NVIDIA GPU with CUDA support
- Minimum: 4GB VRAM
- Recommended: 8GB+ VRAM

#### Software Stack
```yaml
# Docker container specification
FROM python:3.9-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ /app/src/
COPY models/ /app/models/

WORKDIR /app
CMD ["python", "src/anomaly_detection_service.py"]
```

### Model Management

#### Version Control
```python
class ModelVersionManager:
    def __init__(self, model_registry_path):
        self.registry_path = model_registry_path
        self.models = {}
    
    def register_model(self, model, version, metadata):
        model_info = {
            'model': model,
            'version': version,
            'timestamp': datetime.now(),
            'metadata': metadata,
            'performance_metrics': {}
        }
        self.models[version] = model_info
    
    def get_best_model(self, metric='f1_score'):
        best_version = None
        best_score = -1
        
        for version, info in self.models.items():
            if metric in info['performance_metrics']:
                score = info['performance_metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_version = version
        
        return self.models[best_version]['model'] if best_version else None
```

#### Model Monitoring
```python
class ModelPerformanceMonitor:
    def __init__(self):
        self.performance_history = []
        self.drift_detector = None
    
    def log_prediction(self, sample, prediction, score, actual=None):
        record = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'score': score,
            'actual': actual,
            'sample_stats': {
                'mean': np.mean(sample),
                'std': np.std(sample),
                'min': np.min(sample),
                'max': np.max(sample)
            }
        }
        self.performance_history.append(record)
    
    def detect_performance_drift(self, window_size=1000):
        if len(self.performance_history) < window_size * 2:
            return False
        
        recent_scores = [r['score'] for r in self.performance_history[-window_size:]]
        historical_scores = [r['score'] for r in self.performance_history[-2*window_size:-window_size]]
        
        # Statistical test for distribution shift
        from scipy.stats import ks_2samp
        statistic, p_value = ks_2samp(historical_scores, recent_scores)
        
        return p_value < 0.05  # Significant drift detected
```

### Integration with Manufacturing Systems

#### MES Integration
```python
class MESIntegration:
    def __init__(self, mes_endpoint):
        self.mes_endpoint = mes_endpoint
        self.alert_queue = []
    
    def send_anomaly_alert(self, equipment_id, anomaly_info):
        alert_payload = {
            'equipment_id': equipment_id,
            'timestamp': datetime.now().isoformat(),
            'severity': self._determine_severity(anomaly_info['score']),
            'description': f"Anomaly detected with score {anomaly_info['score']:.3f}",
            'recommended_action': self._get_recommendation(anomaly_info),
            'raw_data': anomaly_info
        }
        
        # Send to MES
        response = requests.post(
            f"{self.mes_endpoint}/alerts",
            json=alert_payload,
            headers={'Content-Type': 'application/json'}
        )
        
        return response.status_code == 200
    
    def _determine_severity(self, score):
        if score < -1.0:
            return 'CRITICAL'
        elif score < -0.5:
            return 'WARNING'
        else:
            return 'INFO'
    
    def _get_recommendation(self, anomaly_info):
        score = anomaly_info['score']
        if score < -1.0:
            return "Immediate maintenance required"
        elif score < -0.5:
            return "Schedule preventive maintenance"
        else:
            return "Monitor closely"
```

#### SCADA Integration
```python
class SCADADataCollector:
    def __init__(self, scada_tags):
        self.scada_tags = scada_tags
        self.data_buffer = {}
    
    def collect_data_point(self, tag_values):
        timestamp = datetime.now()
        
        # Validate and clean data
        cleaned_data = self._validate_data(tag_values)
        
        # Store in buffer
        self.data_buffer[timestamp] = cleaned_data
        
        return cleaned_data
    
    def _validate_data(self, data):
        # Remove invalid readings
        validated = {}
        for tag, value in data.items():
            if self._is_valid_reading(tag, value):
                validated[tag] = value
            else:
                # Use last known good value or interpolation
                validated[tag] = self._get_fallback_value(tag)
        
        return validated
```

## Advanced Topics

### Concept Drift Handling

#### Types of Concept Drift
1. **Sudden drift**: Abrupt change in data distribution
2. **Gradual drift**: Slow change over time
3. **Recurring drift**: Cyclical changes (e.g., seasonal patterns)
4. **Incremental drift**: Small, continuous changes

#### Drift Detection Methods
```python
class DriftDetector:
    def __init__(self, window_size=1000, sensitivity=0.05):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.reference_window = []
        self.current_window = []
    
    def detect_drift(self, new_data):
        self.current_window.extend(new_data)
        
        if len(self.current_window) > self.window_size:
            # Perform drift test
            drift_detected = self._statistical_test()
            
            if drift_detected:
                # Update reference and trigger retraining
                self.reference_window = self.current_window.copy()
                self.current_window = []
                return True
            
            # Slide the window
            self.current_window = self.current_window[-self.window_size:]
        
        return False
    
    def _statistical_test(self):
        if len(self.reference_window) == 0:
            self.reference_window = self.current_window.copy()
            return False
        
        from scipy.stats import ks_2samp
        statistic, p_value = ks_2samp(self.reference_window, self.current_window)
        return p_value < self.sensitivity
```

### Multi-variate Anomaly Detection

#### Correlation-Based Detection
```python
def detect_correlation_anomalies(data, correlation_threshold=0.8):
    correlation_matrix = np.corrcoef(data.T)
    
    anomalies = []
    for i in range(len(data)):
        sample = data[i]
        
        # Compute correlation with expected relationships
        anomaly_score = 0
        for j in range(len(sample)):
            for k in range(j+1, len(sample)):
                expected_corr = correlation_matrix[j, k]
                actual_corr = np.corrcoef([sample[j]], [sample[k]])[0, 1]
                
                if abs(expected_corr) > correlation_threshold:
                    anomaly_score += abs(expected_corr - actual_corr)
        
        anomalies.append(anomaly_score)
    
    return np.array(anomalies)
```

#### Principal Component Analysis (PCA) for Anomaly Detection
```python
from sklearn.decomposition import PCA

class PCAAnomalyDetector:
    def __init__(self, n_components=0.95):
        self.pca = PCA(n_components=n_components)
        self.threshold = None
    
    def fit(self, X):
        self.pca.fit(X)
        
        # Compute reconstruction errors for training data
        X_transformed = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        reconstruction_errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        # Set threshold based on training data
        self.threshold = np.percentile(reconstruction_errors, 95)
    
    def predict(self, X):
        X_transformed = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        reconstruction_errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        return (reconstruction_errors > self.threshold).astype(int)
```

## Conclusion

Advanced anomaly detection for equipment monitoring represents a critical capability for modern semiconductor manufacturing. The combination of multiple algorithms, ensemble methods, and real-time processing enables robust detection of equipment issues before they impact production quality or cause costly downtime.

Key success factors include:

1. **Algorithm selection**: Choose appropriate methods based on data characteristics and requirements
2. **Ensemble approaches**: Combine multiple detectors for improved robustness
3. **Real-time processing**: Implement streaming detection for immediate response
4. **Integration**: Connect with existing manufacturing systems and workflows
5. **Continuous improvement**: Monitor performance and adapt to changing conditions

The field continues to evolve with advances in deep learning, edge computing, and IoT technologies, promising even more sophisticated anomaly detection capabilities for future manufacturing environments.