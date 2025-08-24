# Setup Guide: Machine Learning for Semiconductor Engineers

## 🚀 Quick Start Guide

This guide will help you set up your development environment for the Machine Learning for Semiconductor Engineers course.

## Prerequisites

- **Python 3.9+** (recommended: Python 3.11)
- **Git** for version control
- **Docker** (optional but recommended for consistency)
- **8GB+ RAM** for ML workloads
- **Modern web browser** for Jupyter Lab

## Setup Options

### Option 1: Docker Setup (Recommended)

Docker provides a consistent environment across all operating systems.

#### Windows (Docker)

```powershell
# Clone the repository
git clone https://github.com/kennedym-ds/python-for-semiconductors-.git
cd python-for-semiconductors-

# Build and run with Docker Compose
docker-compose up --build

# Access Jupyter Lab at http://localhost:8888
```

#### macOS/Linux (Docker)

```bash
# Clone the repository
git clone https://github.com/kennedym-ds/python-for-semiconductors-.git
cd python-for-semiconductors-

# Build and run with Docker Compose
docker-compose up --build

# Access Jupyter Lab at http://localhost:8888
```

### Option 2: Virtual Environment Setup

#### Windows (Virtual Environment)

```powershell
# Clone the repository
git clone https://github.com/kennedym-ds/python-for-semiconductors-.git
cd python-for-semiconductors-

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Lab
jupyter lab
```

#### macOS/Linux (Virtual Environment)

```bash
# Clone the repository
git clone https://github.com/kennedym-ds/python-for-semiconductors-.git
cd python-for-semiconductors-

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Lab
jupyter lab
```

#### Ubuntu/Debian Additional Steps

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip build-essential libhdf5-dev pkg-config

# Then follow the standard Linux setup above
```

### Option 3: Conda Environment Setup

If you prefer Conda:

```bash
# Create conda environment
conda create -n ml-semiconductors python=3.11
conda activate ml-semiconductors

# Install core packages via conda
conda install numpy pandas scipy scikit-learn matplotlib seaborn jupyter

# Install remaining packages via pip
pip install -r requirements.txt

# Start Jupyter Lab
jupyter lab
```

## Environment Verification

Run this verification script to ensure everything is working:

```python
# verification.py
import sys
import importlib

required_packages = [
    'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
    'jupyter', 'torch', 'tensorflow', 'xgboost', 'lightgbm',
    'opencv-python', 'plotly', 'dash', 'streamlit'
]

print(f"Python version: {sys.version}")
print("\nChecking required packages:")

for package in required_packages:
    try:
        module = importlib.import_module(package.replace('-', '_'))
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package}: {version}")
    except ImportError:
        print(f"❌ {package}: Not installed")

print("\n🎉 Environment verification complete!")
```

Save this as `verification.py` and run:

```bash
python verification.py
```

## Directory Structure Overview

After setup, your directory structure should look like:

```text
python-for-semiconductors-/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── verification.py
├── docs/                    # Documentation and guides
├── modules/                 # Learning modules
│   ├── foundation/         # Modules 1-3
│   ├── intermediate/       # Modules 4-5
│   ├── advanced/          # Modules 6-7
│   ├── cutting-edge/      # Modules 8-9
│   └── project-dev/       # Module 10
├── projects/              # Starter and advanced projects
├── datasets/              # Sample datasets
├── resources/             # Reference materials
├── assessments/           # Quizzes and evaluations
├── infrastructure/        # CI/CD and deployment
└── community/            # Forums and mentorship
```

## Troubleshooting

### Common Issues

#### Package Installation Errors

```bash
# For tensorflow/torch installation issues on Windows
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For opencv issues
pip uninstall opencv-python
pip install opencv-python-headless
```

#### Jupyter Kernel Issues

```bash
# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name ml-semiconductors --display-name "ML Semiconductors"
```

#### Memory Issues

- Ensure you have at least 8GB RAM
- Close other applications when running ML workloads
- Consider using cloud platforms (Google Colab, AWS SageMaker) for heavy computations

#### Permission Issues (Linux/macOS)

```bash
# If you encounter permission errors
sudo chown -R $USER:$USER ~/.local/
```

### Getting Help

1. **Check the FAQ**: See [docs/faq.md](faq.md)
2. **Community Forum**: Visit [community/discussions.md](../community/discussions.md)
3. **GitHub Issues**: Report bugs at the repository issues page
4. **Office Hours**: Weekly Q&A sessions (schedule in community section)

## Next Steps

1. **Verify your setup** using the verification script above
2. **Explore Module 1**: Start with [modules/foundation/module-1/](../modules/foundation/module-1/)
3. **Join the community**: Introduce yourself in [community/introductions.md](../community/introductions.md)
4. **Download datasets**: Follow the guide in [datasets/README.md](../datasets/README.md)

## Performance Tips

### Jupyter Lab Optimization

```python
# Add to your Jupyter config for better performance
c.NotebookApp.iopub_data_rate_limit = 1.0e10
c.NotebookApp.iopub_msg_rate_limit = 3000
```

### Memory Management

```python
# Best practices for memory management in notebooks
import gc
import pandas as pd

# Clear variables when done
del large_dataframe
gc.collect()

# Use chunking for large datasets
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    # Process chunk
    pass
```

Ready to start your journey? Head to Module 1 and begin transforming your semiconductor engineering career with ML! 🚀
