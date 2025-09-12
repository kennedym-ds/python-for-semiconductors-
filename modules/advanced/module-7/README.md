# Module 7: Advanced Computer Vision for Semiconductor Manufacturing

## 🎯 Overview

This module covers advanced computer vision techniques specifically tailored for semiconductor manufacturing applications. Building on foundation concepts, it introduces deep learning, specialized image processing, and production-ready computer vision pipelines for quality control and process monitoring.

## 📚 Module Contents

### 7.2: Wafer Map Pattern Recognition

**Focus**: Automated classification of spatial defect patterns on semiconductor wafers

**Core Technologies**:
- Classical computer vision features (radial histograms, GLCM, HOG)
- Deep learning with compact CNNs
- Hybrid classical-DL approaches
- Manufacturing-specific metrics and explainability

**Key Deliverables**:
- Production-ready pattern recognition pipeline
- Synthetic wafer map data generator
- Classical and deep learning model implementations
- Manufacturing cost-aware evaluation metrics
- Explainability tools (SHAP, Grad-CAM)

## 🚀 Quick Start

### Prerequisites
```bash
# Install advanced tier dependencies
python env_setup.py --tier advanced
```

### Training Your First Model

```bash
cd modules/advanced/module-7

# Classical approach (fast, interpretable)
python 7.2-pattern-recognition-pipeline.py train --approach classical --model svm --save classical_model.joblib

# Deep learning approach (higher accuracy)  
python 7.2-pattern-recognition-pipeline.py train --approach deep_learning --model cnn --epochs 10 --save dl_model.joblib

# Evaluate performance
python 7.2-pattern-recognition-pipeline.py evaluate --model-path classical_model.joblib
```

### Making Predictions

```bash
# Predict on a simple center defect pattern
echo '{"wafer_map": [[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]]}' | \
python 7.2-pattern-recognition-pipeline.py predict --model-path classical_model.joblib --input-json -
```

## 📊 Pattern Types Covered

| Pattern | Description | Manufacturing Impact |
|---------|-------------|---------------------|
| **Normal** | Sparse random defects | Baseline yield monitoring |
| **Center** | High defect density at wafer center | Chuck/temperature issues |
| **Edge** | Defects concentrated at wafer perimeter | Edge processing problems |
| **Scratch** | Linear defect traces | Handling/mechanical damage |
| **Ring** | Concentric or radial patterns | Equipment vibration/rotation |

## ⚡ Performance Benchmarks

### Runtime Requirements (Production)
- **Classical Features**: ~15s training on 600 samples
- **CNN Training**: ~30s for 10 epochs on CPU
- **Inference Time**: <100ms per wafer map
- **Memory Usage**: <512MB peak during training

### Accuracy Targets
- **F1-Score**: ≥0.7 for production deployment
- **False Negative Rate**: ≤0.1 for critical patterns
- **PWS (Prediction Within Spec)**: ≥0.8 overall

## 🔬 Technical Features

### Classical Computer Vision Pipeline
```python
# Feature extraction includes:
- Radial defect density histograms (10 bins)
- Angular defect distribution (8 bins)  
- GLCM texture features (5 properties)
- HOG edge descriptors (~1573 features)
- Connected component properties (6 metrics)
```

### Deep Learning Architecture
```python
class CompactCNN:
    # Optimized for wafer map patterns
    - Conv layers: 32→64→128 filters
    - Max pooling for translation invariance
    - Dropout for regularization
    - Focal loss for class imbalance
```

### Manufacturing Integration
- **JSON-only CLI**: Seamless integration with MES systems
- **Cost-aware metrics**: PWS and estimated loss calculations
- **Explainable predictions**: SHAP and Grad-CAM support
- **Deterministic splits**: By wafer-id for realistic validation

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Full test suite (may take 5-10 minutes)
python -m pytest test_pattern_recognition_pipeline.py -v

# Quick smoke tests
python -m pytest test_pattern_recognition_pipeline.py::test_train_classical_svm -v
python -m pytest test_pattern_recognition_pipeline.py::test_manufacturing_metrics -v
```

## 📖 Documentation

- **[7.2-pattern-recognition-fundamentals.md](7.2-pattern-recognition-fundamentals.md)**: Comprehensive theory and manufacturing context
- **[7.2-pattern-recognition-quick-ref.md](7.2-pattern-recognition-quick-ref.md)**: Command reference and troubleshooting
- **[7.2-pattern-recognition.ipynb](7.2-pattern-recognition.ipynb)**: Interactive tutorial notebook *(coming soon)*

## 🛠️ Advanced Usage

### Custom Cost Matrices
```python
# Define pattern-specific misclassification costs
cost_matrix = {
    'Normal': {'Center': 10, 'Edge': 10, 'Scratch': 15, 'Ring': 12},
    'Scratch': {'Normal': 100, 'Center': 30, 'Edge': 25, 'Ring': 20}
    # Higher cost for missing critical defects
}
```

### Model Ensemble
```python
# Combine classical and deep learning predictions
classical_pred = classical_model.predict_proba(wafer_maps)
dl_pred = dl_model.predict_proba(wafer_maps)
ensemble_pred = 0.6 * classical_pred + 0.4 * dl_pred
```

### Production Monitoring
```python
# Track model performance in production
def monitor_model_health(predictions, confidence_scores):
    low_confidence_rate = np.mean(confidence_scores < 0.7)
    if low_confidence_rate > 0.2:
        trigger_model_retraining_alert()
```

## 🔮 Future Extensions

### Planned Enhancements
- **Real WM-811K dataset integration**: Support for Kaggle wafer map dataset
- **Transfer learning**: Pre-trained feature extractors
- **Active learning**: Human-in-the-loop model improvement
- **Temporal analysis**: Pattern evolution tracking across lots
- **Multi-resolution**: Hierarchical pattern analysis

### Research Directions
- **Few-shot learning**: Rapid adaptation to new defect patterns
- **Anomaly detection**: Novel pattern discovery
- **Physics-informed models**: Incorporate process knowledge
- **Federated learning**: Multi-fab model training

## 🤝 Contributing

When extending this module:

1. **Follow the 4-content pattern**: .py, .md, .ipynb, quick-ref.md
2. **Maintain < 45s test runtime**: Use small datasets for CI
3. **Include manufacturing metrics**: PWS, estimated loss
4. **Document explainability**: How predictions can be interpreted
5. **Ensure JSON-only CLI**: For seamless integration

## 📧 Support

For issues specific to Module 7.2:
- Check the [troubleshooting section](7.2-pattern-recognition-quick-ref.md#troubleshooting)
- Review test logs for common failure patterns
- Validate advanced dependencies are properly installed

---

*This module represents a production-ready implementation of wafer map pattern recognition, suitable for deployment in semiconductor manufacturing environments with appropriate validation and monitoring.*