# Module 6: Deep Learning for Semiconductor Manufacturing

## Overview

This module introduces deep learning techniques for semiconductor manufacturing applications, focusing on computer vision and pattern recognition in wafer maps and defect detection.

## Submodules

### 6.2 CNNs for Defect Detection (Wafer Maps)

**Learning Objectives:**
- Apply convolutional neural networks to wafer map defect classification
- Understand spatial pattern recognition in semiconductor manufacturing
- Implement end-to-end deep learning pipelines with model persistence
- Handle class imbalanced defect data with appropriate techniques
- Integrate explainability methods for production debugging

**Key Files:**
- `6.2-cnn-defect-detection.ipynb` - Interactive learning notebook with hands-on CNN training
- `6.2-cnn-defect-detection-fundamentals.md` - Deep dive into CNN theory and semiconductor applications
- `6.2-cnn-defect-detection-pipeline.py` - Production-ready CLI for training and inference
- `6.2-cnn-defect-detection-quick-ref.md` - Quick reference for commands and troubleshooting
- `test_cnn_defect_pipeline.py` - Automated tests for pipeline functionality

**Datasets:**
- WM-811K wafer map dataset (if available in `datasets/wm811k/`)
- Synthetic wafer pattern generator for consistent learning experience
- Support for various defect patterns: center, edge, ring, scratch, etc.

**Dependencies:**
- Core: numpy, pandas, scikit-learn, matplotlib
- Advanced: torch, torchvision (optional with graceful fallbacks)
- Visualization: seaborn, opencv-python (optional)

**Prerequisites:**
- Module 3: Classification fundamentals
- Basic understanding of image processing concepts
- Familiarity with neural network concepts

## Getting Started

1. Ensure you have the required dependencies installed:
   ```bash
   python env_setup.py --tier advanced
   ```

2. Start with the fundamentals document to understand CNN theory
3. Work through the interactive notebook for hands-on experience
4. Use the pipeline script for production workloads
5. Reference the quick guide for common operations

## Manufacturing Context

Wafer map defect detection is critical for:
- **Yield Optimization**: Early identification of systematic defects
- **Process Control**: Understanding spatial failure patterns
- **Tool Monitoring**: Detecting equipment-related signature patterns
- **Cost Reduction**: Minimizing false alarms and missed defects

The CNN approach provides superior pattern recognition compared to traditional statistical methods, especially for complex spatial signatures.