# Module 7: Advanced Computer Vision for Semiconductor Manufacturing

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