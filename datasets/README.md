# Datasets Overview

This repository provides organized access to semiconductor manufacturing and materials quality datasets for ML education and experimentation.

## Available Datasets

### 1. SECOM Dataset (`secom/`)
- **Source**: UCI Machine Learning Repository (ID: 179)
- **Task**: Binary classification (Pass/Fail prediction)
- **Size**: 1,567 instances × 591 features
- **Download**: `python datasets/download_semiconductor_datasets.py --dataset secom`
- **Description**: Real semiconductor fab sensor measurements with high dimensionality and missing values
- **Usage**: Data quality (Module 2.1), outlier detection (2.2), classification (3.2), ensemble methods (4.1)

### 2. Steel Plates Faults Dataset (`steel-plates/`)
- **Source**: UCI Machine Learning Repository (ID: 198)  
- **Task**: Multi-class classification (7 fault types)
- **Size**: 1,941 instances × 27 features
- **Download**: `python datasets/download_semiconductor_datasets.py --dataset steel-plates`
- **Description**: Steel manufacturing defect classification with geometric and luminosity features
- **Usage**: Multi-class classification examples, manufacturing quality control

### 3. WM-811K Wafer Maps (`wm811k/`)
- **Source**: [Kaggle WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- **Task**: Wafer defect pattern classification
- **Size**: ~811K wafer maps (~5GB unzipped)
- **Download**: Manual via Kaggle (see instructions: `--dataset wm811k`)
- **Description**: Industrial wafer fabrication defect patterns for computer vision
- **Usage**: Pattern recognition (future modules 6-7), imbalanced learning

## Quick Download Commands

```bash
# Download individual datasets
python datasets/download_semiconductor_datasets.py --dataset secom
python datasets/download_semiconductor_datasets.py --dataset steel-plates

# Get instructions for Kaggle datasets
python datasets/download_semiconductor_datasets.py --dataset wm811k

# Download all supported datasets at once
python datasets/download_semiconductor_datasets.py --dataset all
```

## Directory Structure

```text
datasets/
├── README.md                           # This file
├── download_semiconductor_datasets.py  # Download utility script
├── secom/                             # SECOM dataset folder
│   ├── README.md                      # Dataset description
│   ├── secom.data                     # Feature matrix (1567×590)
│   ├── secom_labels.data              # Target labels
│   └── secom.names                    # Variable descriptions
├── steel-plates/                      # Steel Plates Faults dataset
│   ├── README.md                      # Dataset description
│   ├── steel_plates_features.csv     # Feature matrix (1941×27)
│   ├── steel_plates_targets.csv      # Target labels (7 fault types)
│   └── metadata.txt                   # UCI ML repo metadata
├── wm811k/                           # WM-811K dataset (user download)
│   ├── README.md                     # Download instructions
│   └── README_fetch.txt              # Kaggle setup guide
├── time_series/                      # Time series datasets
│   └── README.md                     # Instructions for sensor data
└── vision_defects/                   # Computer vision datasets
    └── README.md                     # Instructions for defect images
```

## Data Handling Guidelines

- Keep raw data immutable; create processed artifacts under module-specific folders
- Large binary datasets (>50MB) should not be committed; prefer download scripts
- Each dataset includes a dedicated README with usage examples
- Use the `ucimlrepo` package for modern UCI ML repository access when available

## Quick Load Examples

### SECOM Dataset
```python
import pandas as pd
from pathlib import Path

# Load SECOM data from organized subfolder
secom_dir = Path('datasets/secom')
X = pd.read_csv(secom_dir / 'secom.data', sep=' ', header=None)
labels = pd.read_csv(secom_dir / 'secom_labels.data', sep=' ', header=None, names=['label','timestamp'])
```

### Steel Plates Dataset  
```python
import pandas as pd
from pathlib import Path

# Load Steel Plates data (modern CSV format)
steel_dir = Path('datasets/steel-plates')
features = pd.read_csv(steel_dir / 'steel_plates_features.csv')
targets = pd.read_csv(steel_dir / 'steel_plates_targets.csv')
```

## Dataset Status Checklist

- [x] SECOM available with organized structure
- [x] Steel Plates Faults downloadable via UCI ML repo
- [x] WM-811K placeholder with Kaggle instructions
- [ ] Time series sensor data (user provided)
- [ ] Computer vision defect datasets (user provided)

## Requirements

The download script requires basic Python packages (included in basic tier):
- `urllib` (built-in) for direct downloads
- `pandas` for UCI ML repo datasets
- `ucimlrepo` (auto-installed) for modern UCI access

For Kaggle datasets:
```bash
pip install kaggle
# Set up API credentials per Kaggle documentation
```

## Future Enhancements

- Add checksum validation for downloaded files
- Include sample notebooks demonstrating each dataset
- Automated preprocessing scripts for common transformations
- Integration with MLflow for dataset versioning
