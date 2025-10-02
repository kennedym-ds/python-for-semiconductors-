# Machine Learning for Semiconductor Engineers: Complete Learning Series

[![CI](https://github.com/kennedym-ds/python-for-semiconductors-/actions/workflows/ci.yml/badge.svg)](https://github.com/kennedym-ds/python-for-semiconductors-/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Repository Status - v1.0.0 Production Release! 🎉

**Current Status**: Production-ready with comprehensive testing and documentation

- ✅ **Modules 1-11**: All content complete and validated (11 modules, 44 files)
- ✅ **Assessments**: 685 questions across all modules, fully tested and validated
- ✅ **Testing Infrastructure**: 201 comprehensive tests (100% pass rate)
- ✅ **Documentation**: 150+ pages of curated resources (papers, case studies, tools)
- ✅ **CI/CD**: Full automation with coverage reporting and GitHub Actions
- ✅ **Community**: Issue templates and contribution workflows ready

**Key Metrics**:

- **11 modules** with 4 content types each (notebooks, theory, scripts, quick refs)
- **685 assessment questions** validated across all modules
- **201 automated tests** (81 unit tests, 32 assessment tests, 88 notebook tests)
- **15 research papers** from top conferences (2024-2025)
- **15+ tools evaluated** (ML frameworks, cloud platforms, MLOps tools)

See [CHANGELOG.md](CHANGELOG.md) for full details.

---

## 🚀 Project Overview

Welcome to the most comprehensive **Machine Learning for Semiconductor Engineers** learning pathway available! This 22-week program transforms semiconductor professionals into ML-powered engineers, bridging traditional semiconductor expertise with cutting-edge AI/ML technologies.

### What Makes This Special

- **Industry-Focused**: Real semiconductor datasets (SECOM, WM-811K wafer maps)
- **Hands-On Learning**: 4 content types per module (notebooks, theory, scripts, quick refs)
- **Production-Ready**: Professional code quality with Docker, CI/CD, and testing
- **Complete Pathway**: From Python basics to advanced MLOps deployment

---

## 📚 Learning Series Structure

### 🔧 Foundation Series (Weeks 1-5)

#### Module 1: Python & Data Fundamentals

- 1.1: Python for Engineers with wafer map data examples
- 1.2: Statistical Foundations with DOE concepts
- **Deliverables**: Wafer yield analysis script, SPC toolkit

#### Module 2: Data Quality & Statistical Analysis

- 2.1: Data Quality using SECOM dataset
- 2.2: Outlier Detection with semiconductor methods
- 2.3: Advanced Statistical Analysis with ANOVA
- **Deliverables**: Data quality framework, outlier detection pipeline

#### Module 3: Introduction to Machine Learning

- 3.1: Regression for Process Engineers
- 3.2: Classification Fundamentals
- **Deliverables**: Process parameter predictor, wafer pass/fail classifier

### ⚙️ Intermediate Series (Weeks 6-8)

#### Module 4: Advanced ML Techniques

- 4.1: Ensemble Methods (XGBoost/LightGBM)
- 4.2: Unsupervised Learning (clustering, PCA)
- 4.3: Multilabel Classification
- **Deliverables**: Advanced yield prediction, anomaly detection system

#### Module 5: Time Series & Predictive Maintenance

- 5.1: Time Series Analysis (ARIMA, Prophet)
- 5.2: Predictive Maintenance systems
- **Deliverables**: Tool drift predictor, equipment health monitoring

### 🧠 Advanced Series (Weeks 9-14)

#### Module 6: Deep Learning Foundations

- 6.1: Neural Networks with PyTorch/TensorFlow
- 6.2: CNNs for Defect Detection
- **Deliverables**: Deep learning optimizer, visual defect classifier

#### Module 7: Computer Vision Applications

- 7.1: Advanced Defect Analysis (YOLO, Faster R-CNN)
- 7.2: Pattern Recognition for wafer maps
- **Deliverables**: Automated optical inspection, pattern classification

### 🚀 Cutting-Edge Series (Weeks 15-20)

#### Module 8: Generative AI & Advanced Applications

- 8.1: GANs for Data Augmentation
- 8.2: LLMs for Manufacturing Intelligence
- **Deliverables**: Data augmentation pipeline, intelligent report analyzer

#### Module 9: MLOps & Deployment

- 9.1: Model Deployment (APIs, containers)
- 9.2: Monitoring & Maintenance
- 9.3: Real-time Inference Systems
- **Deliverables**: Deployable model template, MLOps starter kit

### 🎯 Project Development Series (Weeks 21-22)

#### Module 10: Production-Ready ML Projects

- 10.1: Project Architecture & Best Practices
- 10.2: Testing & Quality Assurance
- 10.3: Documentation & Reproducibility
- 10.4: Scaling & Optimization
- **Deliverables**: Complete production pipeline, deployment-ready system

#### Module 11: Edge AI for Inline Inspection

- 11.1: TensorFlow Lite & Edge Optimization
- **Deliverables**: Optimized edge model, inline inspection system

---

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
| basic | `requirements-basic.txt` | Core Python + statistics + notebooks | Modules 1-3 |
| intermediate | `requirements-intermediate.txt` | + imbalance, boosting, time series | Modules 4-5 |
| advanced | `requirements-advanced.txt` | + deep learning, CV, optimization, MLOps | Modules 6-9 |
| full | `requirements-full.txt` | + Prophet, simulation, advanced tools | All modules |

Use the automation script (recommended):

```powershell
# Clone repository
git clone https://github.com/kennedym-ds/python-for-semiconductors-.git
cd python-for-semiconductors-

# Setup basic tier (recommended for beginners)
python env_setup.py --tier basic

# Or setup full environment
python env_setup.py --tier full

# Verify installation
python verification.py
```

Manual setup (advanced users):

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies for your tier
pip install -r requirements-basic.txt
```

---

## 📖 How to Use This Repository

### For Learners

1. **Start with Module 1**: Begin with `modules/foundation/module-1/1.1-python-engineers.ipynb`
2. **Follow the 4-content pattern** for each module:
   - 📓 **Notebook** (`.ipynb`): Interactive hands-on exercises
   - 📚 **Theory** (`.md`): Deep technical explanations
   - 🔧 **Pipeline** (`.py`): Production-ready CLI scripts
   - ⚡ **Quick Reference** (`.md`): Commands and troubleshooting
3. **Complete assessments**: Test your knowledge with `assessments/module-X/` questions
4. **Build projects**: Apply learning in `projects/` directory

### For Instructors

- Use `assessments/` for quizzes and evaluations (685 questions available)
- Customize notebooks for your cohort
- Reference `docs/` for supplementary materials
- Leverage VS Code tasks for automated workflows (`.vscode/tasks.json`)

---

## 🎯 Datasets

This repository includes access to industry-standard semiconductor datasets:

- **SECOM**: UCI semiconductor manufacturing process control dataset (1567×590)
- **WM-811K**: Wafer map defect patterns (811,000+ samples)
- **Steel Plates**: Multi-class defect classification
- **Synthetic Generators**: Custom wafer defect and time series data

Download datasets:

```bash
python datasets/download_semiconductor_datasets.py --dataset secom
python datasets/download_semiconductor_datasets.py --dataset wm811k
```

See [datasets/DATASET_USAGE_GUIDE.md](datasets/DATASET_USAGE_GUIDE.md) for detailed documentation.

---

## 🧪 Testing & Quality

### Run Tests

```powershell
# Run all tests (201 tests)
pytest tests/ -v

# Run with coverage report
pytest --cov=modules --cov=assessments --cov-report=html

# Validate assessments
python assessments/validation/validate_all.py
```

### Code Quality

```powershell
# Format code (Black)
black . --check

# Lint code (Flake8)
flake8 .

# Run pre-commit hooks
pre-commit run --all-files
```

---

## 📊 Assessment System

Interactive assessment system with **685 validated questions** across all modules:

```bash
# Launch assessment app
streamlit run assessments/assessment_app.py

# Validate question banks
python assessments/validation/validate_all.py --module 1
```

Features:
- Multiple choice, coding exercises, and conceptual questions
- Instant feedback and explanations
- Progress tracking via SQLite database
- Difficulty progression (easy → medium → hard)

See [assessments/README.md](assessments/README.md) for details.

---

## 🔧 VS Code Integration

This repository includes **22 automated VS Code tasks** for common workflows:

**Access tasks**: Press `Ctrl+Shift+B` or `Terminal > Run Task`

**Popular tasks**:
- `Env: Setup Basic/Intermediate/Advanced/Full` - Environment setup
- `Streamlit: Run Assessment App` - Launch assessment interface
- `Tests: Run All` - Execute full test suite
- `Jupyter: Start Lab` - Launch JupyterLab

See [.vscode/TASKS_README.md](.vscode/TASKS_README.md) for complete list.

---

## 📚 Documentation

### Getting Started
- [Setup Guide](docs/setup-guide.md) - Detailed installation instructions
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Series Announcement](docs/SERIES_ANNOUNCEMENT.md) - Project overview and features

### For Contributors
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute
- [Gap Analysis](docs/REPOSITORY_GAP_ANALYSIS.md) - Current status and roadmap
- [Architecture](docs/architecture/) - System design documentation

### Research & Industry
- [2025 AI Industry Trends](docs/2025-AI-INDUSTRY-TRENDS.md) - Latest AI developments
- [Industry Case Studies](docs/industry-case-studies.md) - Real-world applications
- [Performance Benchmarking](docs/performance-benchmarking.py) - Model comparison tools

---

## 🚀 Projects

### Starter Projects (Fully Implemented)

Located in `projects/starter/`:

1. **Wafer Defect Classifier** - Multi-class defect detection with XGBoost
   - Dataset: WM-811K wafer maps
   - Techniques: Ensemble learning, class imbalance handling
   - Deliverable: Production-ready classifier with evaluation notebook

2. **Yield Prediction System** - Regression pipeline for semiconductor yield
   - Dataset: SECOM manufacturing data
   - Techniques: Feature engineering, ensemble methods
   - Deliverable: CLI tool with model persistence

3. **Equipment Health Monitor** - Time series anomaly detection
   - Dataset: Synthetic equipment sensor data
   - Techniques: ARIMA, Prophet, LSTM
   - Deliverable: Real-time monitoring dashboard

### Advanced Projects (Coming Soon)

Located in `projects/advanced/`:
- Real-time defect detection with YOLO
- LLM-powered manufacturing intelligence
- Edge AI for inline inspection

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

**Ways to contribute**:
- 🐛 Report bugs or issues
- 💡 Suggest new features or modules
- 📝 Improve documentation
- 🧪 Add tests or enhance coverage
- 📊 Share semiconductor datasets (anonymized)
- 🎓 Contribute educational content

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Datasets**: UCI Machine Learning Repository, semiconductor industry partners
- **Frameworks**: scikit-learn, PyTorch, TensorFlow, XGBoost, LightGBM
- **Community**: Contributors and educators in semiconductor ML

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/kennedym-ds/python-for-semiconductors-/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kennedym-ds/python-for-semiconductors-/discussions)
- **Repository**: [python-for-semiconductors-](https://github.com/kennedym-ds/python-for-semiconductors-)

---

**Ready to start your ML journey in semiconductors?** Clone the repository and begin with Module 1! 🚀
