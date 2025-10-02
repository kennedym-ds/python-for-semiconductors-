# Starter Yield Regression Project

A production-ready baseline regression pipeline for semiconductor yield prediction that demonstrates industry best practices and serves as a foundation for more complex yield optimization projects.

## Project Overview

This project provides a standardized, reproducible regression baseline for predicting semiconductor manufacturing yield. It includes comprehensive CLI interface, semiconductor-specific metrics, and robust testing framework.

### Key Features

- **🎯 Production-Ready CLI**: Standardized train/evaluate/predict commands with JSON outputs
- **📊 Semiconductor Metrics**: Industry-standard metrics including PWS (Prediction Within Spec) and Estimated Loss
- **🔧 Model Persistence**: Save/load functionality with complete metadata
- **🧪 Comprehensive Testing**: Full test coverage with CI-ready test suite
- **♻️ Reproducible Results**: Fixed random seed and deterministic processing
- **⚡ Multiple Algorithms**: Support for Ridge, Lasso, ElasticNet, Linear, and Random Forest models

## Quick Start

### Installation

```bash
# Navigate to the project directory
cd projects/starter/yield_regression/

# Install dependencies (from repository root)
cd ../../../
python env_setup.py --tier basic
source .venv/bin/activate
```

### Basic Usage

```bash
# Train a model
python yield_regression_pipeline.py train --dataset synthetic_yield --model ridge

# Train and save model
python yield_regression_pipeline.py train --dataset synthetic_yield --model ridge --save my_model.joblib

# Evaluate saved model
python yield_regression_pipeline.py evaluate --model-path my_model.joblib --dataset synthetic_yield

# Make predictions
python yield_regression_pipeline.py predict --model-path my_model.joblib \
  --input-json '{"temperature":455, "pressure":2.6, "flow":118, "time":62, "temp_centered":5.0, "pressure_sq":6.76, "flow_time_inter":7316, "temp_flow_inter":53690}'
```

## Pipeline Architecture

### Data Flow

```
Raw Data → Imputation → Scaling → Feature Selection → PCA → Model → Prediction
```

### Processing Steps

1. **Imputation**: Missing values filled using median strategy
2. **Scaling**: StandardScaler normalization for consistent feature ranges
3. **Feature Selection**: SelectKBest with f_regression scoring (optional)
4. **Dimensionality Reduction**: PCA for noise reduction and efficiency
5. **Model Training**: Configurable regression algorithms

### Supported Models

| Model | Use Case | Key Parameters |
|-------|----------|----------------|
| **Ridge** | Baseline with L2 regularization | `alpha` |
| **Lasso** | Feature selection with L1 regularization | `alpha` |
| **ElasticNet** | Balanced L1/L2 regularization | `alpha`, `l1_ratio` |
| **Linear** | Simple linear regression | None |
| **Random Forest** | Non-linear patterns | `n_estimators`, `max_depth` |

## Semiconductor Metrics

### Standard Regression Metrics
- **MAE (Mean Absolute Error)**: Average prediction error magnitude
- **RMSE (Root Mean Square Error)**: Penalizes large errors more heavily
- **R² (Coefficient of Determination)**: Proportion of variance explained

### Manufacturing-Specific Metrics
- **PWS (Prediction Within Spec)**: Percentage of predictions within specification limits (60-100%)
- **Estimated Loss**: Financial impact of prediction errors beyond tolerance threshold

### Metric Calculation Example

```python
from yield_regression_pipeline import YieldRegressionPipeline

# Calculate metrics with custom thresholds
metrics = YieldRegressionPipeline.compute_metrics(
    y_true=actual_yields,
    y_pred=predicted_yields,
    tolerance=2.0,        # Acceptable error tolerance
    spec_low=60,          # Lower specification limit
    spec_high=100,        # Upper specification limit  
    cost_per_unit=1.0     # Cost per unit for loss calculation
)
```

## Data Description

### Synthetic Yield Dataset

The pipeline includes a realistic synthetic dataset simulating semiconductor manufacturing:

**Process Parameters:**
- `temperature`: Process temperature (°C, nominal 450°C ±15°C)
- `pressure`: Chamber pressure (bar, nominal 2.5 ±0.3 bar)
- `flow`: Gas flow rate (sccm, nominal 120 ±10 sccm)
- `time`: Process time (minutes, nominal 60 ±5 min)

**Engineered Features:**
- `temp_centered`: Temperature deviation from mean
- `pressure_sq`: Squared pressure (captures non-linear effects)
- `flow_time_inter`: Flow-time interaction term
- `temp_flow_inter`: Temperature-flow interaction term

**Target Variable:**
- `yield_pct`: Yield percentage (0-100%)

### Yield Model

The synthetic data follows a realistic yield model incorporating:
- Linear temperature effects
- Quadratic pressure effects (optimal pressure window)
- Flow rate and time contributions
- Interaction effects between parameters
- Gaussian noise component

## CLI Reference

### Train Command

```bash
python yield_regression_pipeline.py train [OPTIONS]
```

**Options:**
- `--dataset`: Dataset name (default: synthetic_yield)
- `--model`: Model type (ridge, lasso, elasticnet, linear, rf)
- `--alpha`: Regularization strength (default: 1.0)
- `--l1-ratio`: ElasticNet L1 ratio (default: 0.5)
- `--k-best`: Number of features to select (default: 20)
- `--pca-components`: PCA components (default: 0.95)
- `--no-feature-selection`: Disable feature selection
- `--save`: Path to save trained model

**Output:** JSON with training metrics and metadata

### Evaluate Command

```bash
python yield_regression_pipeline.py evaluate --model-path MODEL_PATH [OPTIONS]
```

**Options:**
- `--model-path`: Path to saved model (required)
- `--dataset`: Dataset name (default: synthetic_yield)

**Output:** JSON with evaluation metrics and model metadata

### Predict Command

```bash
python yield_regression_pipeline.py predict --model-path MODEL_PATH [INPUT]
```

**Input Options:**
- `--input-json`: JSON string with feature values
- `--input-file`: Path to JSON file with feature values

**Output:** JSON with prediction, input echo, and model metadata

## Testing

### Run All Tests

```bash
python -m pytest test_yield_regression_pipeline.py -v
```

### Test Coverage

- **CLI Functionality**: All train/evaluate/predict commands
- **Model Types**: All supported regression algorithms
- **Hyperparameters**: Various parameter combinations
- **Persistence**: Save/load model functionality
- **Metrics**: Semiconductor-specific metric calculations
- **Error Handling**: Invalid inputs and edge cases
- **Reproducibility**: Consistent results with fixed seed

### Example Test Execution

```bash
================================================= test session starts ==================================================
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_train_basic PASSED                     [  8%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_train_different_models PASSED          [ 16%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_train_with_save PASSED                 [ 25%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_train_hyperparameters PASSED           [ 33%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_evaluate PASSED                        [ 41%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_predict_json_input PASSED              [ 50%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_predict_file_input PASSED              [ 58%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_pipeline_class_directly PASSED         [ 66%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_synthetic_data_generation PASSED       [ 75%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_metrics_calculation PASSED             [ 83%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_error_handling PASSED                  [ 91%]
projects/starter/yield_regression/test_yield_regression_pipeline.py::test_reproducibility PASSED                 [100%]

======================================================= 12 passed in 20s =======================================================
```

## Performance Baselines

### Typical Results on Synthetic Data

With default Ridge regression settings:

```json
{
  "metrics": {
    "MAE": 2.38,
    "RMSE": 2.97,
    "R2": 0.15,
    "PWS": 1.0,
    "Estimated_Loss": 714.13
  }
}
```

### Model Comparison

| Model | MAE | RMSE | R² | PWS | Training Time |
|-------|-----|------|----|----|---------------|
| Linear | 2.38 | 2.97 | 0.15 | 1.0 | Fast |
| Ridge | 2.38 | 2.97 | 0.15 | 1.0 | Fast |
| Lasso | 2.41 | 3.01 | 0.13 | 1.0 | Fast |
| ElasticNet | 2.39 | 2.98 | 0.14 | 1.0 | Fast |
| Random Forest | 1.89 | 2.35 | 0.49 | 1.0 | Medium |

## 📚 Educational Materials

### For Students

This project includes comprehensive learning materials for hands-on practice with regression analysis in semiconductor manufacturing.

#### 📘 Interactive Tutorial (`yield_regression_tutorial.ipynb`)

An interactive Jupyter notebook with hands-on exercises covering the complete regression workflow:

**Section 1: Data Generation and Exploration**
- Exercise 1.1 (★ beginner): Generate synthetic yield data with realistic process parameters
- Exercise 1.2 (★★ intermediate): Analyze correlations between process parameters and yield

**Section 2: Model Training and Comparison**
- Exercise 2.1 (★★ intermediate): Train multiple regression models (Linear, Ridge, Lasso, ElasticNet, Random Forest)
- Exercise 2.2 (★★★ advanced): Compare model performance using R², RMSE, and MAE

**Section 3: Manufacturing-Specific Metrics**
- Exercise 3.1 (★★★ advanced): Analyze residuals to validate model assumptions
- Exercise 3.2 (★★★ advanced): Calculate PWS (Prediction Within Spec) and Estimated Loss for manufacturing context

**Section 4: Model Deployment**
- Exercise 4.1 (★★ intermediate): Save production models with complete metadata
- Exercise 4.2 (★★ intermediate): Test CLI interface for deployment scenarios

**Expected Completion Time**: 100 minutes (typical lab session)

```bash
# Run locally
jupyter notebook yield_regression_tutorial.ipynb

# Run in Docker
docker-compose up jupyter
# Then navigate to http://localhost:8888
```

#### 📗 Solution Notebook (`yield_regression_solution.ipynb`)

Complete reference implementations for all tutorial exercises:
- **Exercise 1**: Data generation with statistical summaries and correlation heatmaps
- **Exercise 2**: 5-model comparison demonstrating regularization effects
- **Exercise 3**: Residual analysis with histogram, scatter, and Q-Q plots
- **Exercise 4**: Production deployment with round-trip verification

Includes best practices, debugging tips, and manufacturing interpretation throughout.

#### 🔍 Automated Grading (`evaluate_submission.py`)

Automated evaluation script for student submissions with 100-point scoring rubric:

```bash
# Grade a completed notebook
python evaluate_submission.py --notebook yield_regression_tutorial.ipynb

# Save results to JSON for LMS integration
python evaluate_submission.py --notebook submission.ipynb --output-json grades.json

# Get detailed feedback
python evaluate_submission.py --notebook submission.ipynb --verbose
```

**Scoring Rubric** (100 points total):
- Exercise 1 (Data Exploration): 20 points
  - Data generation: 5 pts
  - Distribution analysis: 5 pts
  - Correlation analysis: 5 pts
  - Visualization quality: 5 pts
- Exercise 2 (Model Training): 30 points
  - Data splitting: 5 pts
  - Multiple model training: 10 pts
  - Performance evaluation: 10 pts
  - Model comparison: 5 pts
- Exercise 3 (Manufacturing Metrics): 25 points
  - Residual analysis: 10 pts
  - PWS calculation: 5 pts
  - Estimated Loss: 5 pts
  - Manufacturing interpretation: 5 pts
- Exercise 4 (Deployment): 15 points
  - Model saving: 5 pts
  - Model loading: 5 pts
  - CLI demonstration: 5 pts
- Code Quality: 10 points
  - Documentation: 3 pts
  - Code style: 3 pts
  - Error handling: 2 pts
  - Best practices: 2 pts

**Features**:
- Automated notebook execution and validation
- Code quality checks (PEP 8, documentation)
- Results validation against expected ranges
- Detailed feedback with actionable suggestions
- JSON output for Learning Management System integration

**Batch Grading Example**:
```bash
# Grade all submissions
for notebook in submissions/*.ipynb; do
    python evaluate_submission.py \
        --notebook "$notebook" \
        --output-json "grades/$(basename $notebook .ipynb)_grade.json"
done
```

#### Common Student Mistakes

1. **Missing Engineered Features**: Forgetting to include `temp_centered`, `pressure_sq`, `flow_time_inter`, and `temp_flow_inter` when making predictions
2. **Metric Interpretation**: Confusing R² (higher is better) with RMSE (lower is better)
3. **Residual Analysis**: Only looking at metrics without visualizing residual plots
4. **PWS Calculation**: Not applying spec limits (60-100%) correctly or using wrong thresholds
5. **scikit-learn Compatibility**: Using deprecated `squared=False` parameter in RMSE calculation (fixed in sklearn 1.7+)

#### Prerequisites

**Required Knowledge**:
- Python basics (functions, loops, conditionals)
- NumPy and Pandas fundamentals
- Basic statistics (mean, variance, correlation)
- Regression concepts (linear models, R², residuals)

**Recommended Background**:
- Module 1: Python for Data Analysis
- Module 2: Statistics for Manufacturing  
- Module 3.1: Regression Analysis Fundamentals

**Semiconductor Domain**:
- Basic understanding of wafer fabrication process
- Process control concepts (spec limits, tolerance)
- Manufacturing metrics (yield, defect density)

### For Instructors

#### Teaching Tips

1. **Start with Synthetic Data**: Use the built-in synthetic generator to build intuition before moving to real datasets
2. **Emphasize Manufacturing Context**: Connect R², RMSE, and MAE to real-world yield optimization scenarios
3. **Show Residual Plots Early**: Don't rely solely on metrics—visualize residuals to check assumptions
4. **Compare Multiple Models**: Demonstrate regularization effects (Ridge vs Lasso vs ElasticNet)
5. **Connect to Real Processes**: Relate temperature, pressure, flow to actual CVD/etch processes

#### Grading Workflow

**Automated Grading**:
```bash
# Single notebook
python evaluate_submission.py --notebook student_submission.ipynb --verbose

# Batch processing
for nb in submissions/*.ipynb; do
    python evaluate_submission.py --notebook "$nb" --output-json "grades/$(basename $nb .ipynb).json"
done
```

**Class Summary Generation**:
```bash
# Generate class statistics
python -c "
import json
from pathlib import Path

grades = [json.load(open(f)) for f in Path('grades').glob('*.json')]
avg_score = sum(g['total_score'] for g in grades) / len(grades)
passing = sum(1 for g in grades if g['total_score'] >= 60)

print(f'Class Average: {avg_score:.1f}/100')
print(f'Passing Rate: {passing}/{len(grades)} ({passing/len(grades)*100:.1f}%)')
print(f'Letter Grade Distribution:')
for grade in ['A', 'B', 'C', 'D', 'F']:
    count = sum(1 for g in grades if g['letter_grade'].startswith(grade))
    print(f'  {grade}: {count} ({count/len(grades)*100:.1f}%)')
"
```

#### Exercise Breakdown

| Exercise | Points | Auto-Gradable | Manual Review | Time Est. |
|----------|--------|---------------|---------------|-----------|
| 1: Data Exploration | 20 | ✅ Yes | Visualization quality | 20 min |
| 2: Model Training | 30 | ✅ Yes | Model selection rationale | 30 min |
| 3: Manufacturing Metrics | 25 | ✅ Yes | Interpretation depth | 30 min |
| 4: Deployment | 15 | ✅ Yes | Production considerations | 20 min |
| Code Quality | 10 | ⚠️ Partial | Comments, organization | (continuous) |
| **TOTAL** | **100** | **~80%** | **~20%** | **100 min** |

#### Time Estimates

- **Exercise 1**: 20 min (data generation + correlation analysis)
- **Exercise 2**: 30 min (train 5 models + comparison)
- **Exercise 3**: 30 min (residual analysis + manufacturing metrics)
- **Exercise 4**: 20 min (deployment + CLI testing)
- **Total**: 100 minutes (fits standard 90-120 min lab session)

#### Assessment Rubric

**90-100 (Exceptional)**:
- All exercises complete with insightful analysis
- Residual plots show proper interpretation
- Manufacturing metrics correctly calculated and explained
- Code is well-documented and follows best practices
- Demonstrates understanding beyond requirements

**75-89 (Proficient)**:
- All exercises complete and correct
- Standard metrics calculated properly
- Residual analysis included
- Code is functional and reasonably documented
- Meets all core requirements

**60-74 (Developing)**:
- Most exercises complete with some errors
- Some metrics missing or incorrect
- Residual analysis incomplete
- Code works but lacks documentation
- Core concepts understood but execution needs improvement

**<60 (Needs Improvement)**:
- Multiple exercises incomplete
- Major errors in metric calculation
- Missing residual analysis
- Code doesn't execute or has critical errors
- Requires additional instruction

#### Extensions and Projects

Suggested follow-up projects for advanced learners:

1. **Real Data Integration**: Use SECOM dataset (`datasets/secom/`) with 590 features
2. **Hyperparameter Optimization**: Implement GridSearchCV for automated tuning
3. **Feature Engineering**: Create domain-specific features (thermal budget, process interactions)
4. **Ensemble Methods**: Combine multiple models for improved predictions
5. **Deployment Integration**: Build FastAPI wrapper (`infrastructure/api/`)
6. **Drift Monitoring**: Track model performance degradation over time (Module 5.2)
7. **Uncertainty Quantification**: Add prediction intervals using quantile regression
8. **Optimization**: Use model for process parameter optimization

#### Common Discussion Points

**Q: Why does Random Forest perform better than Linear models?**
A: The synthetic data includes non-linear effects (pressure squared) and interaction terms that tree-based models capture naturally.

**Q: When should we use Ridge vs Lasso?**
A: Ridge (L2) when all features are potentially useful; Lasso (L1) when you want automatic feature selection with sparse coefficients.

**Q: What does PWS mean in manufacturing?**
A: Prediction Within Spec—the percentage of predictions that fall within acceptable yield range (60-100%). Critical for production decisions.

**Q: Why is Estimated Loss more important than RMSE?**
A: Estimated Loss incorporates tolerance thresholds and cost per unit, making it actionable for manufacturing decisions (e.g., scrapping out-of-spec wafers).

**Q: How do we handle the sklearn deprecation warning?**
A: This project uses `np.sqrt(mean_squared_error())` instead of the deprecated `squared=False` parameter, ensuring compatibility with sklearn 1.7+.

### Related Projects

After completing this project, learners can progress to:
- **wafer_defect_classifier**: Binary classification for defect detection
- **equipment_drift_monitor**: Time series analysis for equipment health
- **die_defect_segmentation**: Computer vision for spatial defect analysis

### Reference Documentation

- **Grading Guide**: See `GRADING_SCRIPT_GUIDE.md` for detailed grading logic and pattern matching
- **Test Suite**: See `TEST_SUITE_EXPANSION_SUMMARY.md` for 37 test cases covering edge cases, manufacturing scenarios, integration, and performance
- **Solution Content**: See `SOLUTION_NOTEBOOK_CONTENT.md` for complete exercise solutions

## Advanced Usage

### Custom Hyperparameter Tuning

```bash
# High regularization for noisy data
python yield_regression_pipeline.py train --model ridge --alpha 10.0

# Aggressive feature selection
python yield_regression_pipeline.py train --k-best 5 --pca-components 0.8

# Disable preprocessing steps
python yield_regression_pipeline.py train --no-feature-selection --pca-components 8
```

### Programmatic Usage

```python
from yield_regression_pipeline import YieldRegressionPipeline, generate_yield_process

# Generate data
df = generate_yield_process(n=1000, seed=42)
X = df.drop(columns=['yield_pct'])
y = df['yield_pct'].values

# Train pipeline
pipeline = YieldRegressionPipeline(model='ridge', alpha=1.0)
pipeline.fit(X, y)

# Evaluate
metrics = pipeline.evaluate(X, y)
print(f"R² Score: {metrics['R2']:.3f}")
print(f"PWS: {metrics['PWS']:.1%}")

# Make predictions
predictions = pipeline.predict(X[:10])
```

## Integration with Module-3

This project builds upon concepts from Module 3 (Regression Analysis) and serves as a practical application:

- **Related Notebook**: `modules/foundation/module-3/3.1-regression-analysis.ipynb`
- **Reference Pipeline**: `modules/foundation/module-3/3.1-regression-pipeline.py`
- **Compatibility**: Fixed scikit-learn version compatibility issues
- **Extended Metrics**: Enhanced semiconductor-specific metrics

## Troubleshooting

### Common Issues

**Q: "TypeError: got an unexpected keyword argument 'squared'"**
A: This project fixes the scikit-learn compatibility issue by using `np.sqrt(mean_squared_error())` instead of the deprecated `squared=False` parameter.

**Q: "RuntimeError: Pipeline not fitted"**
A: Make sure to train the model first or load a pre-trained model before making predictions.

**Q: "KeyError: Missing features in input"**
A: Ensure input data includes all required features including engineered features (temp_centered, pressure_sq, etc.).

**Q: Tests fail with import errors**
A: Run tests from the repository root with activated virtual environment.

### Performance Tips

- Use Random Forest for better accuracy on complex non-linear relationships
- Increase PCA components if overfitting to training data
- Reduce k_best for faster training on high-dimensional data
- Set alpha=0.1 for Ridge/Lasso when working with clean data

## Contributing

This project follows the repository's contribution guidelines:

1. **Make minimal changes** - Focus on bug fixes and clear improvements
2. **Add tests** for any new functionality
3. **Follow existing patterns** - Use established CLI and metric patterns
4. **Update documentation** - Keep README and docstrings current

## License

Part of the Python for Semiconductors learning series. See repository license for usage terms.

---

**🚀 Ready to predict semiconductor yield with confidence!**

This baseline provides a solid foundation for yield optimization projects while demonstrating industry best practices for ML pipelines in semiconductor manufacturing.
