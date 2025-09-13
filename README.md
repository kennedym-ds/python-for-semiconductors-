# Machine Learning for Semiconductor Engineers: 20-Week Learning Series

[![CI](https://github.com/kennedym-ds/python-for-semiconductors-/actions/workflows/ci.yml/badge.svg)](https://github.com/kennedym-ds/python-for-semiconductors-/actions/workflows/ci.yml)

## 🚀 Project Overview

Welcome to the most comprehensive **Machine Learning for Semiconductor Engineers** learning pathway available! This 20-week program transforms semiconductor professionals into ML-powered engineers, bridging traditional semiconductor expertise with cutting-edge AI/ML technologies.

### What Makes This Special

- **Industry-Focused**: Real semiconductor datasets (SECOM, WM-811K wafer maps)
- **Hands-On Learning**: 4 content types per module (notebooks, theory, scripts, quick refs)
- **Production-Ready**: Professional code quality with Docker, CI/CD, and testing
- **Complete Pathway**: From Python basics to advanced MLOps deployment

## 📚 Learning Series Structure

### 🔧 Foundation Series (Weeks 1-5)

**Module 1: Python & Data Fundamentals**

- 1.1: Python for Engineers with wafer map data examples
- 1.2: Statistical Foundations with DOE concepts
- **Deliverables**: Wafer yield analysis script, SPC toolkit

**Module 2: Data Quality & Statistical Analysis**

- 2.1: Data Quality using SECOM dataset
- 2.2: Outlier Detection with semiconductor methods
- 2.3: Advanced Statistical Analysis with ANOVA
- **Deliverables**: Data quality framework, outlier detection pipeline

**Module 3: Introduction to Machine Learning**

- 3.1: Regression for Process Engineers
- 3.2: Classification Fundamentals
- **Deliverables**: Process parameter predictor, wafer pass/fail classifier

### ⚙️ Intermediate Series (Weeks 6-8)

**Module 4: Advanced ML Techniques**

- 4.1: Ensemble Methods (XGBoost/LightGBM)
- 4.2: Unsupervised Learning (clustering, PCA)
- **Deliverables**: Advanced yield prediction, anomaly detection system

**Module 5: Time Series & Predictive Maintenance**

- 5.1: Time Series Analysis (ARIMA, Prophet)
- 5.2: Predictive Maintenance systems
- **Deliverables**: Tool drift predictor, equipment health monitoring

### 🧠 Advanced Series (Weeks 9-12)

**Module 6: Deep Learning Foundations**

- 6.1: Neural Networks with PyTorch/TensorFlow
- 6.2: CNNs for Defect Detection
- **Deliverables**: Deep learning optimizer, visual defect classifier

**Module 7: Computer Vision Applications**

- 7.1: Advanced Defect Analysis (YOLO, Faster R-CNN)
- 7.2: Pattern Recognition for wafer maps
- **Deliverables**: Automated optical inspection, pattern classification

### 🚀 Cutting-Edge Series (Weeks 13-16)

**Module 8: Generative AI & Advanced Applications**

- 8.1: GANs for Data Augmentation
- 8.2: LLMs for Manufacturing
- **Deliverables**: Data augmentation pipeline, intelligent report analyzer

**Module 9: MLOps & Deployment**

- 9.1: Model Deployment (APIs, containers)
- 9.2: Monitoring & Maintenance
- **Deliverables**: Deployable model template, MLOps starter kit

### 🎯 Project Development Series (Weeks 17-20)

**Module 10: Production-Ready ML Projects**

- 10.1: Project Architecture & Best Practices
- 10.2: Testing & Quality Assurance
- 10.3: Documentation & Reproducibility
- 10.4: Scaling & Optimization

## 🛠️ Installation & Setup

### Quick Start with Docker

```bash
git clone https://github.com/kennedym-ds/python-for-semiconductors-.git
cd python-for-semiconductors-
docker build -t ml-semiconductors .
docker run -p 8888:8888 ml-semiconductors
```

### Tiered Virtual Environment Setup

This project provides **tiered dependency sets** so you only install what you need:

| Tier | File | Includes | Suitable For |
|------|------|----------|--------------|
| basic | `requirements-basic.txt` | Core Python + statistics + notebooks | Modules 1.1–3.1 |
| intermediate | `requirements-intermediate.txt` | + imbalance, boosting, time series basics | Outliers, Classification, Ensembles |
| advanced | `requirements-advanced.txt` | + deep learning, CV, optimization, MLOps | Vision, Deployment prep |
| full | `requirements-full.txt` | + Prophet, simulation (PySpice), RF, typing tools | All modules & experiments |

Use the automation script (recommended):

```powershell
# Clone repository
git clone https://github.com/kennedym-ds/python-for-semiconductors-.git
cd python-for-semiconductors-

# Create a basic environment
python env_setup.py --tier basic

# Or create intermediate / advanced / full
python env_setup.py --tier intermediate
python env_setup.py --tier advanced
python env_setup.py --tier full

# Recreate from scratch (danger: deletes .venv)
python env_setup.py --tier full --force
```

Manual (not recommended) full install:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1  # (cmd: .venv\Scripts\activate.bat)
pip install -r requirements-full.txt
```

VS Code tasks are available (Terminal > Run Task) under names:
`Env: Setup Basic`, `Env: Setup Intermediate`, `Env: Setup Advanced`, `Env: Setup Full`.

After activation launch Jupyter:

```powershell
python -m ipykernel install --user --name semicon-basic
jupyter lab
```

To verify environment:

```powershell
python -c "import sys, sklearn, numpy; print(sys.executable); print(sklearn.__version__)"
```

## 🎯 Starter Projects (Fully Implemented)

1. **🔍 Yield Prediction Dashboard**
   - Real-time monitoring with predictive analytics
   - Interactive visualizations and alerts

2. **🔬 Defect Pattern Classifier**
   - Automated categorization with root cause analysis
   - Computer vision for wafer map analysis

3. **⚡ Equipment Maintenance Scheduler**
   - Predictive maintenance with optimization
   - Cost-benefit analysis and scheduling

4. **📊 Process Optimization Tool**
   - Multi-parameter optimization with recommendations
   - Statistical process control integration

## 📊 Content Format (Each Module Includes)

1. **📓 Interactive Jupyter Notebook** (.ipynb)
   - Hands-on exercises with real semiconductor data
   - Step-by-step guided learning

2. **📚 Technical Deep-Dive Document** (.md)
   - Theory, mathematics, flowcharts, case studies
   - Industry context and best practices

3. **⚙️ Production-Ready Python Script** (.py)
   - Modular code with CLI interface
   - Professional documentation and testing

4. **📋 Quick Reference Card** (.md)
   - Summary, commands, troubleshooting
   - Cheat sheets and workflow guides

## 🗂️ Repository Structure

```text
python-for-semiconductors-/
├── README.md
├── docs/                    # Documentation and guides
├── modules/                 # All 10 modules with 4 content types each
│   ├── foundation/         # Modules 1-3
│   ├── intermediate/       # Modules 4-5
│   ├── advanced/          # Modules 6-7
│   ├── cutting-edge/      # Modules 8-9
│   └── project-dev/       # Module 10
├── projects/              # 4 starter + 4 advanced projects
│   ├── starter/
│   └── advanced/
├── datasets/              # SECOM, WM-811K, synthetic generators
├── resources/             # Reference materials and tools
├── assessments/           # Quizzes and evaluations
├── infrastructure/        # Docker, CI/CD, deployment
└── community/            # Forums, mentorship, events
```

## 🚀 Getting Started

1. **Choose Your Path**: Start with Foundation if new to Python/ML
2. **Set Up Environment**: Use Docker for consistency or virtual env
3. **Follow Module Order**: Each builds on previous knowledge
4. **Complete Projects**: Apply learning with hands-on projects
5. **Join Community**: Engage in discussions and peer learning

## 🎓 Learning Outcomes

By completion, you'll be able to:

- ✅ Apply ML techniques to semiconductor manufacturing challenges
- ✅ Build production-ready ML systems with proper testing/deployment
- ✅ Analyze semiconductor datasets and extract actionable insights
- ✅ Implement computer vision solutions for defect detection
- ✅ Design predictive maintenance systems for equipment
- ✅ Create automated process optimization tools

## 🤝 Contributing

We welcome contributions! This project maintains high code quality standards with synchronized CI and pre-commit workflows.

### Code Quality Setup

Both CI and local development use identical linting and formatting standards:

- **Black** code formatting with 120-character line length
- **Flake8** linting with complexity limits and critical error detection  
- **Pre-commit hooks** that mirror CI exactly

To set up local development:
```bash
pip install pre-commit
pre-commit install
```

The `.pre-commit-config.yaml` and `.flake8` configurations ensure your local environment matches our CI standards exactly.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Ready to transform your semiconductor engineering career with ML? Let's get started!** 🚀
