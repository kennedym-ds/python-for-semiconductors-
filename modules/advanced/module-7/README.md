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
=======
## Overview

Module 7 focuses on advanced computer vision techniques for semiconductor manufacturing applications. This module covers object detection, image segmentation, and deep learning approaches for quality control and defect analysis.

## Module 7.1: Advanced Defect Detection

Advanced defect detection using object detection architectures for semiconductor wafer inspection.

### Contents

- **`7.1-advanced-defect-detection.ipynb`** - Interactive learning notebook with hands-on examples
- **`7.1-advanced-defect-detection-fundamentals.md`** - Comprehensive theory and deep-dive documentation  
- **`7.1-advanced-defect-detection-pipeline.py`** - Production-ready CLI pipeline script
- **`7.1-advanced-defect-detection-quick-ref.md`** - Quick reference guide and cheat sheet
- **`test_advanced_detection_pipeline.py`** - Comprehensive test suite

### Key Features

#### Multiple Detection Backends
- **Classical OpenCV**: Fast, CPU-friendly baseline using traditional computer vision
- **YOLO (ultralytics)**: Real-time object detection with good speed/accuracy balance
- **Faster R-CNN (torchvision)**: High-accuracy two-stage detection architecture
- **Graceful Fallbacks**: Automatic fallback priority: ultralytics → torchvision → classical

#### Semiconductor-Specific Features
- **Synthetic Wafer Generation**: Realistic wafer images with scratches, particles, and cracks
- **Manufacturing Metrics**: PWS (Prediction Within Spec), Estimated Loss calculations
- **Cost-Sensitive Evaluation**: Economic impact analysis for production decisions
- **Multi-Scale Detection**: Handles defects of various sizes and orientations

#### Production-Ready Pipeline
- **CLI Interface**: Train, evaluate, predict subcommands with JSON output
- **Model Persistence**: Save/load trained models with metadata
- **Performance Monitoring**: Inference speed benchmarking and throughput analysis
- **Robust Error Handling**: Graceful degradation for missing dependencies

### Quick Start

#### Install Dependencies
```bash
# Basic requirements (always available)
sudo apt install python3-opencv python3-numpy python3-pandas python3-sklearn

# Optional: Advanced deep learning backends
pip install ultralytics  # For YOLO
pip install torch torchvision  # For Faster R-CNN
```

#### Train a Model
```bash
# Classical detection (no training required)
python 7.1-advanced-defect-detection-pipeline.py train \
  --backend classical \
  --dataset synthetic \
  --n-images 50 \
  --save classical_model.joblib

# YOLO detection (if ultralytics available)
python 7.1-advanced-defect-detection-pipeline.py train \
  --backend yolo \
  --dataset synthetic \
  --n-images 100 \
  --epochs 10 \
  --save yolo_model.joblib
```

#### Evaluate Performance
```bash
python 7.1-advanced-defect-detection-pipeline.py evaluate \
  --model-path classical_model.joblib \
  --dataset synthetic \
  --n-images 20 \
  --iou-threshold 0.5
```

#### Make Predictions
```bash
# On specific image
python 7.1-advanced-defect-detection-pipeline.py predict \
  --model-path classical_model.joblib \
  --image-path wafer_image.jpg

# On synthetic data
python 7.1-advanced-defect-detection-pipeline.py predict \
  --model-path classical_model.joblib \
  --dataset synthetic
```

### Performance Characteristics

| Backend | Speed | Accuracy | CPU-Friendly | Training Required | Dependencies |
|---------|-------|----------|--------------|-------------------|--------------|
| Classical | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ❌ | OpenCV only |
| YOLO | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ | ultralytics |
| Faster R-CNN | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ✅ | torchvision |

### Defect Types Supported

1. **Scratches**: Linear defects from mechanical damage
   - Detection: Line detection algorithms, learned orientation invariance
   - Characteristics: Variable length (20-80px), random orientation

2. **Particles**: Foreign matter contamination
   - Detection: Blob detection, shape-robust classification
   - Characteristics: Circular/irregular, radius 3-15px

3. **Cracks**: Fracture patterns in material
   - Detection: Edge detection with morphology, pattern learning
   - Characteristics: Branching structures, 3-8 connected segments

### Evaluation Metrics

#### Standard Detection Metrics
- **Precision**: TP / (TP + FP) - Accuracy of detections
- **Recall**: TP / (TP + FN) - Coverage of actual defects
- **F1-Score**: Harmonic mean of precision and recall
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5

#### Manufacturing-Specific Metrics
- **PWS (Prediction Within Spec)**: Percentage of correctly identified defects
- **Estimated Loss**: Economic impact = (FN × defect_cost) + (FP × false_alarm_cost)
- **Throughput Analysis**: Images per second for production requirements

### Testing

Run the comprehensive test suite:
```bash
python test_advanced_detection_pipeline.py
```

Tests cover:
- Synthetic data generation
- All detection backends (with fallbacks)
- Model persistence
- CLI functionality
- Error handling and edge cases

### Dependencies and Compatibility

#### Required (Always Available)
- Python 3.8+
- OpenCV (cv2)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

#### Optional (Graceful Fallback)
- **ultralytics**: For YOLO detection
- **torch + torchvision**: For Faster R-CNN detection

The system automatically detects available packages and falls back gracefully:
```
YOLO (ultralytics) → Faster R-CNN (torchvision) → Classical (OpenCV)
```

### Production Deployment Guidelines

#### Performance Requirements
- **Real-time**: Use Classical backend (5-20 ms/image)
- **High accuracy**: Use Faster R-CNN backend
- **Balanced**: Use YOLO backend

#### Typical Throughput
- Classical: ~1000 images/second (CPU)
- YOLO: ~100 images/second (GPU)
- Faster R-CNN: ~10 images/second (GPU)

#### Integration Points
1. **Manufacturing Execution Systems (MES)**
2. **Statistical Process Control (SPC)**
3. **Quality Management Systems (QMS)**
4. **Automated Optical Inspection (AOI)**

### Learning Path

1. **Start with Notebook**: Run `7.1-advanced-defect-detection.ipynb` for interactive learning
2. **Read Fundamentals**: Study `7.1-advanced-defect-detection-fundamentals.md` for theory
3. **Practice CLI**: Use `7.1-advanced-defect-detection-pipeline.py` for hands-on experience
4. **Reference Guide**: Keep `7.1-advanced-defect-detection-quick-ref.md` handy
5. **Understand Tests**: Review `test_advanced_detection_pipeline.py` for best practices

### Future Enhancements

Potential areas for extension:
- **Segmentation**: Pixel-level defect segmentation
- **3D Inspection**: Multi-view defect analysis
- **Real-time Streaming**: Live video processing
- **Edge Deployment**: Optimize for IoT devices
- **Active Learning**: Iterative model improvement

### Common Issues and Solutions

#### "Package not available" warnings
**Solution**: Install optional packages or use classical backend

#### Low detection accuracy
**Solutions**: 
- Increase training data (`--n-images`)
- Tune IoU threshold (`--iou-threshold`)
- Adjust classical parameters (`--blur-kernel`, `--threshold-value`)

#### Slow inference
**Solutions**:
- Use classical backend
- Reduce image resolution
- Optimize for target hardware

### Support and Resources

- **Documentation**: Complete theory in fundamentals.md
- **Examples**: Working code in notebook and pipeline
- **Tests**: Comprehensive test coverage
- **CLI Help**: Run `python 7.1-advanced-defect-detection-pipeline.py --help`

## Module Structure

```
module-7/
├── 7.1-advanced-defect-detection.ipynb          # Interactive notebook
├── 7.1-advanced-defect-detection-fundamentals.md # Theory documentation
├── 7.1-advanced-defect-detection-pipeline.py    # Production pipeline
├── 7.1-advanced-defect-detection-quick-ref.md   # Quick reference
├── test_advanced_detection_pipeline.py          # Test suite
└── README.md                                     # This file
```

---

This module provides a complete foundation for implementing advanced defect detection systems in semiconductor manufacturing environments, with emphasis on practical deployment and production readiness.
