# Module 5 (Intermediate): Time Series Analysis

This intermediate module extends supervised learning (Modules 1–3) and ensemble methods (Module 4) into time series forecasting for semiconductor manufacturing.

## 5.1 Time Series Analysis

Focus on semiconductor time series forecasting using ARIMA/Seasonal ARIMA models for tool drift detection, SPC signal analysis, chamber parameter forecasting, and yield KPI trend modeling.

Artifacts:

- `5.1-time-series-analysis.ipynb`
- `5.1-time-series-fundamentals.md`  
- `5.1-time-series-pipeline.py`
- `5.1-time-series-quick-ref.md`

Core Capabilities:

- Stationarity testing (ADF test, differencing, seasonal decomposition)
- ARIMA/Seasonal ARIMA modeling via statsmodels with optional pmdarima auto-selection
- Time series cross-validation with temporal splits to prevent data leakage
- Manufacturing-specific metrics including prediction intervals and uncertainty quantification
- Forecast reconciliation to engineering constraints (non-negativity, practical bounds)
- Synthetic time series generators for semiconductor process parameters

## Educational Goals

| Aspect | Focus |
|--------|-------|
| Core Skill | Time series modeling, stationarity, seasonality handling |
| Manufacturing Use | Tool drift detection, parameter forecasting, yield trend analysis |
| Metrics Emphasis | MAE, RMSE, MAPE, PWS (Prediction Within Spec), forecast intervals |
| Risks Highlighted | Data leakage, over-differencing, seasonal misspecification |

## CLI Pattern Consistency

All pipelines expose: `train`, `evaluate`, `predict` with JSON outputs for downstream MLOps integration.

## Dataset Requirements

Time series data should be placed in `datasets/time_series/` with proper temporal indexing. The pipeline includes synthetic data generators when real datasets are unavailable.

---
Maintain reproducibility with `RANDOM_SEED = 42` and follow dataset path conventions from project root.