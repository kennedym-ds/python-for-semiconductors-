# Dataset Usage Guide - Python for Semiconductors

**Last Updated**: September 30, 2025  
**Purpose**: Comprehensive mapping of datasets to learning outcomes with usage examples and troubleshooting

---

## Table of Contents

- [Overview](#overview)
- [Dataset Inventory](#dataset-inventory)
- [Dataset-to-Module Mapping](#dataset-to-module-mapping)
- [Quick Start Guide](#quick-start-guide)
- [Detailed Dataset Guides](#detailed-dataset-guides)
- [Fallback Strategies](#fallback-strategies)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

This learning series uses a carefully curated mix of **real semiconductor datasets** and **synthetic data generators** to provide hands-on experience with production-level ML workflows while maintaining accessibility when real data is unavailable.

### Dataset Philosophy

1. **Real Data First**: When available, use actual semiconductor manufacturing data
2. **Graceful Fallbacks**: All modules work with synthetic data if real data unavailable  
3. **Manufacturing Context**: All datasets reflect realistic semiconductor scenarios
4. **Progressive Complexity**: Start simple (synthetic), advance to real-world challenges

---

## Dataset Inventory

### Real Datasets

| Dataset | Size | Source | Modules | Status |
|---------|------|--------|---------|--------|
| **SECOM** | 1567 × 590 | UCI ML Repo | 1-4 | ✅ Excellent |
| **WM-811K** | 811,457 maps | Kaggle | 6-7 | ✅ Integrated |
| **Steel Plates** | 1941 × 27 | UCI ML Repo | 4 | ⚠️ Ready (unused) |

### Synthetic Datasets

| Generator | Purpose | Modules | Quality |
|-----------|---------|---------|---------|
| **Synthetic Yield** | Process parameters → yield | 3 | ⭐⭐⭐⭐⭐ Excellent |
| **Synthetic Wafer Maps** | Defect pattern generation | 6-7 | ⭐⭐⭐⭐☆ Good |
| **Synthetic Time Series** | Equipment sensor data | 5 | ⭐⭐⭐⭐☆ Good |
| **Synthetic Process Recipes** | Manufacturing parameters | 2, 4 | ⭐⭐⭐⭐⭐ Excellent |

---

## Dataset-to-Module Mapping

### Module 1: Python & NumPy Basics

**Primary Dataset**: Synthetic Wafer Maps

```python
from generate_wafer_maps import generate_sample_maps
wafer_maps = generate_sample_maps(n=100, defect_types=['center', 'edge'])
```

**Learning Outcomes**:
- NumPy array manipulation
- Basic wafer map visualization
- Statistical analysis fundamentals

**Why This Dataset**: Simple, visual, manageable size for beginners

---

### Module 2: Statistical Analysis for Manufacturing

**Primary Dataset**: SECOM  
**Secondary Dataset**: Synthetic Process Recipes

#### SECOM Usage

```python
# Module 2.1: Data Quality Analysis
from datasets.download_semiconductor_datasets import load_secom
X, y = load_secom()

print(f"Features: {X.shape[1]}")  # 590 process parameters
print(f"Samples: {X.shape[0]}")    # 1567 wafer measurements
print(f"Missing values: {X.isnull().sum().sum()}")  # Realistic data quality issues
```

**What Makes SECOM Ideal**:
- ✅ Real semiconductor process data
- ✅ 590 features (high-dimensional challenge)
- ✅ Missing values (data quality practice)
- ✅ Binary classification (pass/fail)
- ✅ Imbalanced classes (realistic scenario)

**Fallback Strategy**:
```python
if not secom_available():
    from datasets.synthetic_generators import generate_process_data
    X, y = generate_process_data(n_samples=1500, n_features=590)
```

#### Synthetic Process Recipes

```python
# Module 2.2: ANOVA & Statistical Testing
from datasets.synthetic_generators import ProcessRecipeGenerator

generator = ProcessRecipeGenerator()
data = generator.generate_recipes(n_recipes=5, samples_per_recipe=50)
```

**Use Cases**:
- Statistical hypothesis testing
- ANOVA (comparing process recipes)
- Distribution analysis

---

### Module 3: Regression & Classification

**Primary Dataset**: SECOM + Synthetic Yield Data

#### Module 3.1: Regression (Yield Prediction)

```python
from datasets.synthetic_generators import generate_yield_process

# Generate realistic yield data
df = generate_yield_process(n=800, seed=42)

# Features: temperature, pressure, flow, time
# Target: yield (continuous 70-100%)
```

**Why Synthetic for Regression**:
- Controllable relationships for learning
- Known ground truth for validation
- Realistic process parameters
- Configurable complexity

**Real Dataset Alternative (SECOM)**:
```python
# Convert SECOM to regression problem
X, y_binary = load_secom()
# Use feature values as regression targets
y_continuous = X['feature_590']  # Continuous target
```

#### Module 3.2: Classification (Defect Detection)

```python
# Use SECOM directly for binary classification
X, y = load_secom()
# y: 0 = pass, 1 = fail
```

**Perfect Match**:
- Real manufacturing pass/fail decisions
- High-dimensional feature space
- Class imbalance (production reality)
- Missing data handling practice

---

### Module 4: Ensemble Methods & Unsupervised Learning

**Primary Dataset**: SECOM  
**Secondary Dataset**: Synthetic Process Data  
**Future Dataset**: Steel Plates (Module 4.3)

#### Module 4.1: Ensemble Methods

```python
# Use SECOM for ensemble comparison
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = load_secom()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Compare RF, XGBoost, LightGBM
```

**SECOM Advantages**:
- Large feature space benefits tree-based models
- Demonstrates feature importance analysis
- Shows ensemble strengths on real data

#### Module 4.2: Unsupervised Learning (Clustering)

```python
# Cluster semiconductor process conditions
from sklearn.cluster import KMeans, DBSCAN

X, _ = load_secom()  # Ignore labels for unsupervised

# Find natural groupings in process data
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(X)
```

**Use Cases**:
- Process regime identification
- Anomaly detection via density-based clustering
- Dimensionality reduction with PCA/t-SNE

#### Module 4.3: Multi-Label Classification (Steel Plates)

**STATUS**: ⚠️ Ready to implement (see Steel Plates section below)

```python
# Load Steel Plates dataset
from datasets.download_semiconductor_datasets import load_steel_plates

X, y_multilabel = load_steel_plates()
# y_multilabel: 7 binary columns (Pastry, Z_Scratch, K_Scatch, etc.)
```

---

### Module 5: Time Series Analysis

**Primary Dataset**: Synthetic Time Series (Equipment Sensors)

```python
from datasets.synthetic_generators import TimeSensorDataGenerator

generator = TimeSensorDataGenerator(
    n_sensors=10,
    n_points=1000,
    anomaly_rate=0.05
)

data = generator.generate()
# Returns: timestamps, sensor readings, anomaly labels
```

**Why Synthetic**:
- Controllable drift and anomalies
- Known temporal dependencies
- Seasonal patterns
- Equipment maintenance events

**Real Alternative**: None currently available  
**Future Enhancement**: Real fab sensor logs (requires data partnership)

---

### Module 6: Deep Learning (CNN)

**Primary Dataset**: WM-811K (Real Wafer Maps) ✅  
**Fallback Dataset**: Synthetic Wafer Maps

#### Module 6.2: CNN Defect Detection

```python
# Best: Use real WM-811K dataset
python 6.2-cnn-defect-detection-pipeline.py train --dataset wm811k --epochs 20

# Alternative: Synthetic wafer maps
python 6.2-cnn-defect-detection-pipeline.py train --dataset synthetic_wafer --epochs 10
```

**WM-811K Advantages**:
- ✅ 811,457 real wafer maps
- ✅ 9 defect patterns (Center, Edge, Donut, Scratch, etc.)
- ✅ Industry-standard benchmark
- ✅ Realistic spatial defect distributions
- ✅ Class imbalance (production reality)

**Integration Status**: ✅ COMPLETE (as of Sept 30, 2025)

**See**: `datasets/WM811K_INTEGRATION_GUIDE.md` for full details

---

### Module 7: Computer Vision (Pattern Recognition)

**Primary Dataset**: WM-811K ✅  
**Fallback Dataset**: Synthetic Wafer Patterns

#### Module 7.2: Classical + Deep Learning Pattern Recognition

```python
# Use WM-811K for feature extraction and classification
python 7.2-pattern-recognition-pipeline.py extract-features --dataset wm811k
python 7.2-pattern-recognition-pipeline.py train --dataset wm811k --model rf
```

**Dataset Features**:
- Radial defect histograms
- Angular defect distributions
- Texture features (GLCM)
- Shape descriptors

**Why WM-811K Excels**:
- Real defect spatial patterns
- Supports classical CV feature extraction
- Enables deep learning comparison
- Demonstrates transfer learning

---

### Module 8: Generative AI

**Primary Dataset**: Synthetic Wafer Maps

```python
# GAN training for wafer defect augmentation
from datasets.synthetic_generators import WaferDefectPatternGenerator

generator = WaferDefectPatternGenerator()
real_maps = generator.generate(n=1000, pattern='center')

# Train GAN to generate more realistic defect patterns
```

**Why Synthetic**:
- Need large training sets for GANs
- Controllable defect characteristics
- No IP concerns with generated data

---

### Module 9: MLOps

**Datasets**: All (demonstrates deployment flexibility)

#### Module 9.1: Model Deployment

- Uses any trained model from Modules 3-7
- Demonstrates deployment with SECOM classifier
- FastAPI REST API examples

#### Module 9.2: Monitoring & Maintenance

```python
# Monitor SECOM model in production
from datasets.download_semiconductor_datasets import load_secom

# Baseline data
X_baseline, y_baseline = load_secom()

# Production data (simulated drift)
X_production = add_process_drift(X_baseline, drift_amount=0.2)

# Detect drift
drift_detected = monitor_data_drift(X_baseline, X_production)
```

**Monitoring Demonstrations**:
- Data drift detection (PSI, KS test)
- Model performance degradation
- Alert generation

---

### Module 10: Project Development

**Datasets**: All (meta-module for project structure)

- Uses existing datasets from Modules 1-9
- Demonstrates best practices across all data types
- Shows data versioning and lineage

---

## Quick Start Guide

### Step 1: Download Real Datasets

```bash
# Navigate to datasets directory
cd datasets

# Download all datasets (SECOM, Steel Plates)
python download_semiconductor_datasets.py --all

# Optional: Download WM-811K (requires Kaggle API)
cd wm811k
kaggle datasets download -d qingyi/wm811k-wafer-map
Expand-Archive wm811k-wafer-map.zip -DestinationPath raw/
```

### Step 2: Verify Data Availability

```python
from datasets.download_semiconductor_datasets import check_dataset_availability

status = check_dataset_availability()
print(status)
# {
#   'secom': True,
#   'wm811k': True,
#   'steel_plates': True,
#   'synthetic_generators': True
# }
```

### Step 3: Test Synthetic Generators

```python
from datasets.synthetic_generators import (
    generate_yield_process,
    TimeSensorDataGenerator,
    WaferDefectPatternGenerator
)

# Test each generator
yield_data = generate_yield_process(n=100)
print(f"✅ Yield data: {yield_data.shape}")

sensor_gen = TimeSensorDataGenerator()
sensor_data = sensor_gen.generate(n_points=500)
print(f"✅ Sensor data: {sensor_data.shape}")

wafer_gen = WaferDefectPatternGenerator()
wafer_maps = wafer_gen.generate(n=50, pattern='center')
print(f"✅ Wafer maps: {wafer_maps.shape}")
```

---

## Detailed Dataset Guides

### SECOM Dataset

**Full Name**: SECOM Process Data  
**Source**: UCI Machine Learning Repository  
**Use Case**: Binary classification (pass/fail wafer prediction)

#### Download

```bash
python datasets/download_semiconductor_datasets.py --dataset secom
```

#### Structure

```
datasets/secom/
├── secom.data         # Feature matrix (1567 × 590)
├── secom_labels.data  # Target labels (1567 × 1, binary)
└── secom.names        # Feature descriptions
```

#### Loading

```python
import pandas as pd
import numpy as np

# Load features
X = pd.read_csv('datasets/secom/secom.data', sep=' ', header=None)

# Load labels
y = pd.read_csv('datasets/secom/secom_labels.data', sep=' ', header=None)
y = y.iloc[:, 0]  # First column contains labels

print(f"Features: {X.shape}")  # (1567, 590)
print(f"Labels: {y.shape}")    # (1567,)
print(f"Class distribution: {y.value_counts()}")
# -1 (pass): 1463
#  1 (fail): 104  -> Highly imbalanced!
```

#### Preprocessing Tips

```python
# Handle missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Feature selection (many features are redundant)
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X_imputed)

# Handle class imbalance
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected, y)
```

#### Best Modules

- ✅ Module 2.1: Data Quality Analysis
- ✅ Module 2.2: Outlier Detection
- ✅ Module 3.2: Classification
- ✅ Module 4.1: Ensemble Methods
- ✅ Module 4.2: Unsupervised Learning

---

### WM-811K Dataset

**Full Name**: WM-811K Wafer Map Dataset  
**Source**: Kaggle  
**Use Case**: Multi-class defect pattern classification

#### Download

**See full guide**: `datasets/WM811K_INTEGRATION_GUIDE.md`

```bash
cd datasets/wm811k
kaggle datasets download -d qingyi/wm811k-wafer-map
Expand-Archive wm811k-wafer-map.zip -DestinationPath raw/
```

#### Quick Usage

```python
from datasets.wm811k_preprocessing import WM811KPreprocessor, process_wm811k_dataset

# One-line preprocessing
data = process_wm811k_dataset(
    Path("datasets/wm811k"),
    target_size=(64, 64),
    augment=False
)

# Access wafer maps and labels
wafer_maps = data.wafer_maps  # Shape: (N, 64, 64)
labels = data.labels          # Defect types
defect_types = data.defect_types  # ['None', 'Center', 'Donut', ...]
```

#### Defect Patterns

| Pattern | Count | Percentage |
|---------|-------|------------|
| None | 440,000 | 54.2% |
| Center | 120,000 | 14.8% |
| Donut | 80,000 | 9.9% |
| Edge-Loc | 60,000 | 7.4% |
| Edge-Ring | 40,000 | 4.9% |
| Loc | 30,000 | 3.7% |
| Random | 20,000 | 2.5% |
| Scratch | 15,000 | 1.8% |
| Near-full | 6,457 | 0.8% |

#### Best Modules

- ✅ Module 6.2: CNN Defect Detection
- ✅ Module 7.2: Pattern Recognition
- ⚠️ Module 8.1: GAN Training (use synthetic due to IP)

---

### Steel Plates Dataset

**Full Name**: Steel Plates Faults  
**Source**: UCI Machine Learning Repository  
**Use Case**: Multi-label classification (7 fault types)

#### Download

```bash
python datasets/download_semiconductor_datasets.py --dataset steel_plates
```

#### Structure

```
datasets/steel-plates/
├── steel_plates_features.csv  # Features (1941 × 27)
├── steel_plates_targets.csv   # Targets (1941 × 7, multi-label)
└── README.md
```

#### Loading

```python
import pandas as pd

# Load features and targets
X = pd.read_csv('datasets/steel-plates/steel_plates_features.csv')
y = pd.read_csv('datasets/steel-plates/steel_plates_targets.csv')

print(f"Features: {X.shape}")  # (1941, 27)
print(f"Targets: {y.shape}")   # (1941, 7)
print(f"Fault types: {y.columns.tolist()}")
# ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
```

#### Multi-Label Characteristics

```python
# Check label statistics
print("Samples per fault type:")
print(y.sum())

# Samples with multiple faults
multi_fault = (y.sum(axis=1) > 1).sum()
print(f"Samples with multiple faults: {multi_fault}")

# Label combinations
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
label_combos = y.apply(lambda row: tuple(row[row == 1].index), axis=1)
print("Most common fault combinations:")
print(label_combos.value_counts().head())
```

#### Best Module

- 🚧 Module 4.3: Multi-Label Classification (TO BE IMPLEMENTED)

#### Multi-Label Metrics

```python
from sklearn.metrics import hamming_loss, jaccard_score, f1_score

# Hamming loss: Fraction of wrong labels
hamming = hamming_loss(y_true, y_pred)

# Jaccard score: Intersection over union
jaccard = jaccard_score(y_true, y_pred, average='samples')

# F1 score (micro, macro, samples)
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
```

---

### Synthetic Generators

#### Yield Process Generator

```python
from datasets.synthetic_generators import generate_yield_process

# Generate process data with realistic relationships
df = generate_yield_process(
    n=800,
    seed=42,
    noise_level=3.0,
    outlier_fraction=0.05
)

# Columns: temperature, pressure, flow, time, target (yield %)
# Relationships:
# - yield ∝ temperature deviation from 450°C
# - yield ∝ pressure × flow (interaction)
# - yield ∝ optimal processing time (60 min)
```

**Configuration Options**:

```python
# Easy problem (strong signal)
df_easy = generate_yield_process(noise_level=1.0, outlier_fraction=0.0)

# Hard problem (weak signal, noisy)
df_hard = generate_yield_process(noise_level=10.0, outlier_fraction=0.15)
```

#### Time Sensor Data Generator

```python
from datasets.synthetic_generators import TimeSensorDataGenerator

generator = TimeSensorDataGenerator(
    n_sensors=10,
    n_points=1000,
    sampling_rate=1.0,  # 1 Hz
    anomaly_rate=0.05,
    drift_rate=0.0001
)

data = generator.generate()
# Returns DataFrame with columns: timestamp, sensor_0, sensor_1, ..., is_anomaly
```

**Features**:
- Temporal dependencies (autoregressive)
- Sensor correlations
- Anomaly injection (point, contextual, collective)
- Gradual drift simulation
- Seasonal patterns

#### Wafer Defect Pattern Generator

```python
from datasets.synthetic_generators import WaferDefectPatternGenerator

generator = WaferDefectPatternGenerator(map_size=64)

# Generate specific defect patterns
center_defects = generator.generate(n=100, pattern='center')
edge_defects = generator.generate(n=100, pattern='edge')
ring_defects = generator.generate(n=100, pattern='ring')
scratch_defects = generator.generate(n=100, pattern='scratch')

# Random mix
mixed_defects = generator.generate(n=500, pattern='random_mix')
```

**Available Patterns**:
- `center`: Central defect cluster
- `edge`: Edge-localized defects
- `ring`: Ring-shaped defect pattern
- `scratch`: Linear scratch defect
- `random`: Scattered random defects
- `random_mix`: Random combination of all patterns

---

## Fallback Strategies

### Automatic Fallback Pattern

All modules implement graceful degradation:

```python
def load_dataset_with_fallback(dataset_name):
    """Load real dataset with automatic fallback to synthetic"""
    try:
        # Attempt to load real dataset
        if dataset_name == "secom":
            return load_secom()
        elif dataset_name == "wm811k":
            return load_wm811k()
        elif dataset_name == "steel_plates":
            return load_steel_plates()
    except (FileNotFoundError, ImportError) as e:
        warnings.warn(f"{dataset_name} not available. Using synthetic data.")
        return generate_synthetic_equivalent(dataset_name)
```

### Dataset Equivalence

| Real Dataset | Synthetic Equivalent | Fidelity |
|--------------|----------------------|----------|
| SECOM | `generate_process_data()` | ⭐⭐⭐☆☆ (60%) |
| WM-811K | `WaferDefectPatternGenerator()` | ⭐⭐⭐⭐☆ (80%) |
| Steel Plates | Not available yet | N/A |

### When to Use Synthetic

✅ **Learning concepts**: Controllable examples for understanding  
✅ **Testing pipelines**: Verify code works before real data  
✅ **Rapid prototyping**: Iterate quickly without data access  
✅ **Reproducible examples**: Consistent results for teaching

❌ **Final evaluation**: Always validate on real data when available  
❌ **Performance benchmarking**: Real data essential for accurate metrics  
❌ **Production deployment**: Train on actual fab data

---

## Troubleshooting

### Issue 1: "Dataset not found"

**Symptoms**:
```python
FileNotFoundError: datasets/secom/secom.data not found
```

**Solutions**:

1. Download the dataset:
```bash
python datasets/download_semiconductor_datasets.py --dataset secom
```

2. Check dataset directory structure:
```bash
ls datasets/secom/
# Should show: secom.data, secom_labels.data, secom.names
```

3. Use absolute paths if relative paths fail:
```python
from pathlib import Path
DATA_DIR = Path(__file__).parent.parent / "datasets" / "secom"
```

---

### Issue 2: "Import Error: synthetic_generators not found"

**Symptoms**:
```python
ModuleNotFoundError: No module named 'synthetic_generators'
```

**Solutions**:

1. Add datasets directory to Python path:
```python
import sys
from pathlib import Path
datasets_path = Path(__file__).parent.parent / "datasets"
sys.path.insert(0, str(datasets_path))

from synthetic_generators import generate_yield_process  # Now works
```

2. Use absolute imports (if in module):
```python
from datasets.synthetic_generators import generate_yield_process
```

---

### Issue 3: "Memory Error loading WM-811K"

**Symptoms**:
```python
MemoryError: Unable to allocate array with shape (811457, 64, 64)
```

**Solutions**:

1. Use smaller subset:
```python
python 6.2-cnn-defect-detection-pipeline.py train --dataset wm811k_small
```

2. Reduce image size:
```python
data = process_wm811k_dataset(data_root, target_size=(32, 32))  # Smaller
```

3. Use synthetic data:
```python
python 6.2-cnn-defect-detection-pipeline.py train --dataset synthetic_wafer
```

---

### Issue 4: "SECOM class imbalance affecting results"

**Symptoms**:
```
Training accuracy: 95%
Test accuracy: 60%
-> Model predicting all negative class!
```

**Solutions**:

```python
# Option 1: Use class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
model = RandomForestClassifier(class_weight='balanced')

# Option 2: Resample training data
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Option 3: Use stratified splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

### Issue 5: "WM-811K preprocessing taking too long"

**Symptoms**:
```
Processing 811,457 wafer maps... (30+ minutes)
```

**Solutions**:

1. Preprocess once, save results:
```python
from datasets.wm811k_preprocessing import process_wm811k_dataset

# Preprocess and save (one-time, 5-15 minutes)
data = process_wm811k_dataset(data_root, target_size=(64, 64))

# Future loads are instant (3-5 seconds)
preprocessor = WM811KPreprocessor(data_root)
data = preprocessor.load_processed_data()  # Fast!
```

2. Use smaller subset during development:
```python
data_small = process_wm811k_dataset(
    data_root,
    target_size=(32, 32),
    max_samples=10000  # Only process first 10K
)
```

---

## Best Practices

### For Students / Learners

1. **Start with synthetic data** to understand concepts
2. **Progress to real data** for practical validation
3. **Compare results** between synthetic and real datasets
4. **Use fallbacks** when real data unavailable
5. **Document which dataset** you used in reports

### For Instructors

1. **Provide both options** in all exercises
2. **Test synthetic fallbacks** to ensure they work
3. **Emphasize differences** between synthetic and real data
4. **Use real data** for final assessments when possible
5. **Create optional advanced exercises** requiring real datasets

### For Practitioners

1. **Always validate on real data** before production
2. **Use synthetic for testing** new algorithms quickly
3. **Benchmark performance** on industry-standard datasets (WM-811K)
4. **Document data lineage** for reproducibility
5. **Version control datasets** with DVC or similar tools

---

## Summary Table

| Module | Primary Dataset | Fallback | Complexity | Real Data Value |
|--------|----------------|----------|------------|-----------------|
| 1 | Synthetic Wafer | N/A | Low | Low |
| 2 | SECOM | Synthetic Process | Medium | High |
| 3 | Synthetic Yield | SECOM | Medium | Medium |
| 4 | SECOM | Synthetic | High | High |
| 5 | Synthetic Time | None | Medium | Medium |
| 6 | WM-811K | Synthetic Wafer | High | Very High |
| 7 | WM-811K | Synthetic Wafer | High | Very High |
| 8 | Synthetic Wafer | N/A | Medium | Low |
| 9 | All | All | Medium | High |
| 10 | All | All | Low | Medium |

---

**Last Updated**: September 30, 2025  
**Maintainer**: Python for Semiconductors Team  
**Version**: 1.0.0

For dataset-specific guides, see:
- `datasets/WM811K_INTEGRATION_GUIDE.md` - WM-811K detailed guide
- `datasets/README.md` - Dataset overview and download instructions
- Individual module READMEs - Module-specific dataset usage
