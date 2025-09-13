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