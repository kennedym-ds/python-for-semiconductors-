# Module 5 (Intermediate): Time Series & Predictive Maintenance

This intermediate module advances from ensemble and unsupervised learning (Module 4) into time-series analysis and predictive maintenance for semiconductor manufacturing.

## Module Overview

Module 5 focuses on temporal data analysis and predictive maintenance strategies:

- **5.1 Time Series Analysis** (Future): Forecasting, trend analysis, seasonality detection
- **5.2 Predictive Maintenance**: Equipment health monitoring, failure prediction, maintenance optimization

## 5.2 Predictive Maintenance (Current)

### Artifacts

- `5.2-predictive-maintenance-analysis.ipynb` - Interactive analysis and EDA
- `5.2-predictive-maintenance-fundamentals.md` - Theory and concepts
- `5.2-predictive-maintenance-pipeline.py` - Production CLI pipeline  
- `5.2-predictive-maintenance-quick-ref.md` - Usage reference
- `test_predictive_maintenance_pipeline.py` - Comprehensive test suite

### Key Features

**Use Cases Covered:**
- Remaining Useful Life (RUL) estimation
- Early warning alert systems  
- Tool health scoring and monitoring

**Labeling Strategies:**
- `event_in_next_k_hours` classification (24h, 72h horizons)
- `time_to_event` regression with censoring handling
- Cost-sensitive threshold optimization

**Feature Engineering:**
- Rolling window statistics (mean, std, min/max, range)
- Exponentially weighted moving averages (EWMA)
- Lag features and temporal dependencies
- Trend analysis and degradation indicators

**Models Supported:**
- Tree-based: XGBoost, LightGBM, CatBoost, Random Forest
- Linear baselines: Logistic Regression, Linear Regression
- Proper handling of class imbalance (SMOTE, class weights)

**Manufacturing Metrics:**
- PWS (Prediction Within Specification): tolerance-based accuracy
- Estimated Loss: cost-aware performance measurement
- Standard metrics: ROC AUC, PR AUC, MAE, RMSE, R²

**Temporal Validation:**
- Time-based cross-validation with embargo periods
- Group-based splits (tool-aware validation)
- No future information leakage

### CLI Usage Examples

```bash
# Train classification model for 24h maintenance prediction
python 5.2-predictive-maintenance-pipeline.py train \
    --model xgboost \
    --target event_in_24h \
    --horizon 24 \
    --save maintenance_model.joblib

# Train regression model for time-to-event prediction  
python 5.2-predictive-maintenance-pipeline.py train \
    --task regression \
    --target time_to_event \
    --model rf \
    --save rul_model.joblib

# Evaluate model performance
python 5.2-predictive-maintenance-pipeline.py evaluate \
    --model-path maintenance_model.joblib

# Make predictions
python 5.2-predictive-maintenance-pipeline.py predict \
    --model-path maintenance_model.joblib \
    --input-json '{"sensor_1":0.5, "sensor_2":1.2, "tool_id":"T001"}'
```

### Advanced Features

**Threshold Optimization:**
- Youden's J statistic (sensitivity + specificity - 1)
- Cost-based optimization (custom FP/FN costs)
- Business-driven alert threshold selection

**Model Options:**
- Classification vs regression tasks
- Multiple prediction horizons (24h, 72h, custom)
- Ensemble methods with built-in class balancing
- Robust handling of missing data and sensor drift

**Manufacturing Integration:**
- JSON-only CLI output for MLOps integration
- Model persistence with comprehensive metadata
- Manufacturing-specific performance metrics
- Real-time inference capabilities

## Educational Goals

| Aspect | Predictive Maintenance |
|--------|------------------------|
| Core Skill | Temporal feature engineering & failure prediction |
| Manufacturing Use | Equipment health monitoring, maintenance optimization |
| Metrics Emphasis | PWS, cost-aware loss, temporal validation |
| Challenges | Class imbalance, temporal dependencies, censoring |

## Prerequisites

- Module 3 (Classification/Regression fundamentals)
- Module 4 (Ensemble methods, handling imbalance)
- Understanding of manufacturing processes and maintenance strategies

## Dependencies

Requires `requirements-intermediate.txt`:
- XGBoost, LightGBM, CatBoost for advanced tree ensembles
- imbalanced-learn for SMOTE and class balancing
- Standard ML stack: scikit-learn, pandas, numpy

## Key Learning Outcomes

After completing Module 5.2, you will be able to:

1. **Engineer Time-Series Features**: Create rolling statistics, EWMA, lag features, and trend indicators for equipment sensor data

2. **Handle Temporal Data**: Implement proper time-based validation with embargo periods to prevent data leakage

3. **Build Predictive Models**: Develop both classification (alert systems) and regression (RUL estimation) models for maintenance prediction

4. **Optimize Business Metrics**: Use manufacturing-specific metrics (PWS, Estimated Loss) and cost-sensitive threshold optimization

5. **Deploy Production Pipelines**: Create robust, JSON-based CLI tools suitable for MLOps and manufacturing system integration

6. **Address Class Imbalance**: Apply appropriate techniques for rare event prediction in manufacturing contexts

## Next Module Preview

Module 6 will extend into deep learning approaches for computer vision and advanced pattern recognition in semiconductor manufacturing (defect detection, wafer map analysis).

---

Maintain reproducibility with `RANDOM_SEED = 42` and follow established CLI patterns with JSON output for production integration.