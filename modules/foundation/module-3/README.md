# Module 3: Introduction to Machine Learning for Semiconductor Engineers

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Master regression techniques for semiconductor process parameter prediction
- Implement classification algorithms for wafer pass/fail prediction
- Apply feature engineering methods specific to semiconductor datasets
- Build machine learning pipelines for production manufacturing environments
- Create predictive models for yield optimization and defect classification

## 📚 Module Contents

### 3.1: Regression for Process Engineers

**Content Types:**

- 📓 **Interactive Notebook**: `3.1-regression-analysis.ipynb`
- 📚 **Technical Deep-Dive**: `3.1-regression-fundamentals.md`
- ⚙️ **Production Script**: `3.1-regression-pipeline.py`
- 📋 **Quick Reference**: `3.1-regression-quick-ref.md`

**Topics Covered:**

- Linear and polynomial regression fundamentals
- Feature engineering for semiconductor process data
- Model selection and cross-validation techniques
- Regularization methods (Ridge, Lasso, Elastic Net)
- Process parameter prediction and yield optimization
- Model interpretation and coefficient analysis

### 3.2: Classification Fundamentals

**Content Types:**

- 📓 **Interactive Notebook**: `3.2-classification-analysis.ipynb`
- 📚 **Technical Deep-Dive**: `3.2-classification-fundamentals.md`
- ⚙️ **Production Script**: `3.2-classification-pipeline.py`
- 📋 **Quick Reference**: `3.2-classification-quick-ref.md`

**Topics Covered:**

- Logistic regression and decision trees
- Support Vector Machines (SVM) for manufacturing classification
- Model evaluation metrics for imbalanced datasets
- ROC curves and precision-recall analysis
- Wafer pass/fail prediction and defect classification
- Feature importance and decision boundary analysis

### Classification Pipeline CLI Examples

Train (synthetic events dataset, logistic with min precision constraint):

```bash
python 3.2-classification-pipeline.py train --dataset synthetic_events --model logistic --min-precision 0.9 --save logistic_events.joblib
```

Evaluate saved model:

```bash
python 3.2-classification-pipeline.py evaluate --model-path logistic_events.joblib --dataset synthetic_events
```

Predict with a single JSON record (provide engineered fields if not auto-generated):

```bash
python 3.2-classification-pipeline.py predict --model-path logistic_events.joblib --input-json '{"temperature":455, "pressure":2.6, "flow":118, "time":62, "temp_centered":5, "pressure_sq":6.76, "flow_time_inter":7316, "temp_flow_inter":53690}'
```

Enable SMOTE inside training:

```bash
python 3.2-classification-pipeline.py train --dataset synthetic_events --model rf --use-smote --min-precision 0.9 --save rf_smote_events.joblib
```

Note: As with regression, derived features must currently be provided for prediction. A unified feature engineering layer is a future enhancement.

## 🎯 Module Deliverables

### Core Deliverables

1. **Process Parameter Predictor**: Regression model for predicting critical process outcomes
2. **Wafer Pass/Fail Classifier**: Classification system for automated quality assessment
3. **Feature Engineering Toolkit**: Reusable functions for semiconductor data preprocessing
4. **Model Evaluation Dashboard**: Comprehensive performance assessment tools

### Production Artifacts

- Automated machine learning pipelines with logging and monitoring
- Model serialization and deployment-ready code
- Performance benchmarking and validation frameworks
- Integration patterns for manufacturing execution systems (MES)

## 🔧 Prerequisites

### Required Knowledge (from Previous Modules)

- Python programming fundamentals (Module 1.1)
- Statistical analysis and hypothesis testing (Module 1.2)
- Data quality assessment and outlier detection (Module 2.1-2.2)
- Analysis of Variance (ANOVA) concepts (Module 2.3)

### Technical Skills

- Pandas for data manipulation
- NumPy for numerical computations
- Matplotlib/Seaborn for visualization
- Basic understanding of scientific computing

### Software Requirements

- Python 3.8+ with scikit-learn
- Jupyter Notebook environment
- Standard data science libraries (pandas, numpy, matplotlib, seaborn)

## 📊 Datasets Used

### Primary Dataset: SECOM (Semiconductor Manufacturing)

- **Source**: UCI Machine Learning Repository
- **Size**: 1,567 examples with 590 features
- **Target**: Binary classification (pass/fail)
- **Use Cases**: Feature selection, regression, classification

### Synthetic Process Data

- **Source**: Generated process parameter datasets
- **Variables**: Temperature, pressure, flow rates, concentration levels
- **Use Cases**: Regression modeling, process optimization

### Wafer Map Data

- **Source**: Semiconductor wafer testing results
- **Structure**: Spatial defect patterns and electrical test data
- **Use Cases**: Spatial pattern classification, yield prediction

## ⏱️ Estimated Time

- **Total Module Time**: 8-10 hours
- **3.1 Regression Analysis**: 4-5 hours
- **3.2 Classification**: 4-5 hours
- **Hands-on Practice**: 2-3 hours per submodule
- **Project Work**: 2-3 additional hours

## 🚀 Getting Started

### Quick Start

1. **Environment Setup**: Ensure all dependencies are installed

   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn jupyter
   ```

2. **Interactive Learning**: Start with the regression notebook

   ```bash
   jupyter notebook 3.1-regression-analysis.ipynb
   ```

### Regression Pipeline CLI Examples

Train (synthetic yield dataset, Ridge):

```bash
python 3.1-regression-pipeline.py train --dataset synthetic_yield --model ridge --alpha 1.0 --save ridge_yield.joblib
```

Evaluate saved model:

```bash
python 3.1-regression-pipeline.py evaluate --model-path ridge_yield.joblib --dataset synthetic_yield
```

Predict with a single JSON record:

```bash
python 3.1-regression-pipeline.py predict --model-path ridge_yield.joblib --input-json '{"temperature":455, "pressure":2.6, "flow":118, "time":62, "temp_centered":5, "pressure_sq":6.76, "flow_time_inter":7316, "temp_flow_inter":53690}'
```

Note: Derived features (temp_centered, pressure_sq, flow_time_inter, temp_flow_inter) must be supplied for prediction because the current CLI does not recompute them from base inputs. A future enhancement will add automatic feature engineering for predict mode.

### Learning Path

1. Begin with `3.1-regression-fundamentals.md` for theoretical foundation
2. Work through `3.1-regression-analysis.ipynb` for hands-on practice
3. Explore `3.1-regression-pipeline.py` for production implementation
4. Reference `3.1-regression-quick-ref.md` for formulas and troubleshooting
5. Repeat sequence for Module 3.2 classification content

## 📋 Assessment

### Knowledge Checks

- **Regression Concepts**: Model assumptions, evaluation metrics, overfitting
- **Classification Theory**: Decision boundaries, class imbalance, performance metrics
- **Feature Engineering**: Selection methods, dimensionality reduction, domain knowledge
- **Model Selection**: Cross-validation, hyperparameter tuning, bias-variance tradeoff

### Practical Exercises

- Build a yield prediction model using process parameters
- Create a defect classification system using wafer test data
- Implement feature selection pipeline for high-dimensional semiconductor data
- Develop model evaluation framework with appropriate metrics

### Final Project Options

1. **Process Optimization**: Multi-objective optimization using regression models
2. **Quality Prediction**: Real-time classification system for manufacturing line
3. **Yield Enhancement**: Feature importance analysis for process improvement

## 🔗 Module Resources

### Documentation Links

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Regression Analysis Best Practices](https://scikit-learn.org/stable/modules/linear_model.html)
- [Classification Algorithms Overview](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)

### Industry Applications

- Semiconductor yield modeling and prediction
- Process parameter optimization studies
- Defect classification and root cause analysis
- Quality control and statistical process control integration

### Additional Reading

- "Hands-On Machine Learning" by Aurélien Géron (Chapters 3-4)
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- IEEE papers on semiconductor manufacturing applications

## 🎯 Success Metrics

### Technical Proficiency

- [ ] Can implement and evaluate linear/polynomial regression models
- [ ] Understands feature engineering principles for manufacturing data
- [ ] Can build and assess classification models with appropriate metrics
- [ ] Demonstrates model selection and hyperparameter tuning skills

### Practical Application

- [ ] Builds production-ready machine learning pipelines
- [ ] Integrates domain knowledge into feature engineering
- [ ] Selects appropriate evaluation metrics for manufacturing contexts
- [ ] Creates interpretable models for process engineers

### Problem-Solving Capability

- [ ] Diagnoses and resolves common modeling issues (overfitting, underfitting)
- [ ] Adapts models for different semiconductor manufacturing scenarios
- [ ] Balances model complexity with interpretability requirements
- [ ] Implements robust validation and testing frameworks

## 🔄 Integration with Previous Modules

This module builds upon previous modules by:

- Extending statistical foundations to predictive modeling techniques
- Applying data quality frameworks to machine learning preprocessing
- Using outlier detection methods for model validation and data cleaning
- Integrating ANOVA concepts with feature selection and model interpretation

## 🔮 Preparation for Next Module

This module prepares you for Module 4 by:

- Establishing machine learning workflow foundations for ensemble methods
- Building feature engineering expertise for advanced algorithms
- Creating model evaluation standards for complex ML systems
- Developing production pipeline patterns for advanced deployment scenarios

---

**Previous Module**: [Module 2: Data Quality & Statistical Analysis](../module-2/README.md)  
**Next Module**: [Module 4: Advanced ML Techniques](../module-4/README.md)

**Course Home**: [Main README](../../../README.md)  
**Learning Resources**: [Resources Directory](../../../resources/)
