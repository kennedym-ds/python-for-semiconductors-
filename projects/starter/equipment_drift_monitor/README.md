# Equipment Drift Monitor - Time Series Project

A production-ready time series pipeline for semiconductor equipment drift detection and forecasting.

## Project Overview

This project provides a baseline time-series implementation for predictive maintenance in semiconductor manufacturing environments. It focuses on equipment drift detection using sliding window features and anomaly metrics to enable early warning systems for process parameter deviations.

**Project Type**: Time Series  
**Focus**: Equipment drift detection and forecasting  
**Related Modules**: Module 5 (Time Series), Module 10.1 (Project Architecture)

## Features

### Core Functionality
- **Feature Extraction**: Sliding window statistics, rolling means, trend analysis
- **Drift Detection**: Statistical process control and anomaly detection
- **Forecasting**: Multi-step ahead predictions with confidence intervals
- **CLI Interface**: Standardized train/evaluate/predict commands with JSON output

### Manufacturing-Specific Metrics
- **MAE (Mean Absolute Error)**: Standard forecasting accuracy
- **MAPE (Mean Absolute Percentage Error)**: Scale-independent accuracy
- **Anomaly Rate**: Percentage of predictions outside control limits
- **PWS (Prediction Within Spec)**: Manufacturing tolerance compliance
- **Estimated Loss**: Financial impact of prediction errors

### Data Management
- **Synthetic Data Generator**: Realistic equipment drift scenarios
- **Standardized Paths**: Consistent DATA_DIR resolution
- **Model Persistence**: Save/load with comprehensive metadata

## Quick Start

### Installation
```bash
# From repository root
cd projects/starter/equipment_drift_monitor
pip install -r ../../../requirements-basic.txt
pip install statsmodels  # Additional time series dependency
```

### Basic Usage

#### Train a Drift Detection Model
```bash
python equipment_drift_monitor.py train \
    --data synthetic_equipment \
    --window-size 24 \
    --horizon 12 \
    --save drift_model.joblib
```

#### Evaluate Model Performance
```bash
python equipment_drift_monitor.py evaluate \
    --model-path drift_model.joblib \
    --data synthetic_equipment \
    --tolerance 2.0 \
    --cost-per-unit 1.5
```

#### Generate Drift Forecasts
```bash
python equipment_drift_monitor.py predict \
    --model-path drift_model.joblib \
    --horizon 24 \
    --output forecasts.json
```

### JSON Output Example
```json
{
  "status": "predicted",
  "horizon": 24,
  "predictions": {
    "forecasts": [98.2, 98.1, 97.9, ...],
    "forecast_index": ["2023-01-10 01:00:00", "2023-01-10 02:00:00", ...],
    "confidence_intervals": {
      "lower": [96.8, 96.7, 96.5, ...],
      "upper": [99.6, 99.5, 99.3, ...]
    },
    "anomaly_flags": [false, false, true, ...]
  }
}
```

## Technical Architecture

### Data Flow
1. **Data Ingestion**: Load time series data with proper datetime indexing
2. **Feature Engineering**: Extract sliding window features and trend indicators
3. **Model Training**: Fit ARIMA/SARIMA models with drift detection parameters
4. **Anomaly Detection**: Apply statistical process control limits
5. **Forecasting**: Generate multi-step predictions with uncertainty quantification

### Feature Engineering
- **Rolling Statistics**: Mean, std, min, max over configurable windows
- **Trend Analysis**: Linear regression slopes over sliding windows
- **Lag Features**: Previous values at specified intervals
- **Seasonal Decomposition**: Trend, seasonal, and residual components

### Model Architecture
- **Base Model**: ARIMA/SARIMA for time series forecasting
- **Drift Detection**: Statistical process control with configurable thresholds
- **Ensemble Option**: Multiple models for different time horizons
- **Anomaly Scoring**: Deviation from expected patterns

## Configuration

### Model Parameters
```python
{
    "window_size": 24,          # Hours for feature extraction
    "horizon": 12,              # Forecast horizon
    "drift_threshold": 2.0,     # Standard deviations for anomaly detection
    "confidence_level": 0.95,   # Confidence interval level
    "seasonal_period": 168      # Weekly seasonality (hours)
}
```

### Manufacturing Limits
```python
{
    "temperature": {"min": 400, "max": 500, "tolerance": 5},
    "pressure": {"min": 1.0, "max": 5.0, "tolerance": 0.1},
    "flow_rate": {"min": 50, "max": 200, "tolerance": 10}
}
```

## Data Requirements

### Input Format
- **Time Series**: Pandas DataFrame with DatetimeIndex
- **Columns**: Equipment parameters (temperature, pressure, flow_rate, etc.)
- **Frequency**: Regular intervals (hourly, daily, etc.)
- **Duration**: Minimum 200 observations for reliable model fitting

### Synthetic Data
The project includes a synthetic data generator that creates realistic equipment drift scenarios:
- **Gradual Drift**: Linear trends in process parameters
- **Sudden Shifts**: Step changes representing equipment adjustments
- **Seasonal Patterns**: Daily and weekly cycles
- **Random Noise**: Realistic measurement variability
- **Failure Events**: Anomalous periods requiring maintenance

## Testing

### Unit Tests
```bash
python -m pytest test_equipment_drift_monitor.py -v
```

### Integration Tests
```bash
# Test complete CLI workflow
python equipment_drift_monitor.py train --data synthetic_equipment --save test_model.joblib
python equipment_drift_monitor.py evaluate --model-path test_model.joblib
python equipment_drift_monitor.py predict --model-path test_model.joblib --horizon 6
```

### Performance Validation
- **Training Time**: < 60 seconds for 1000 observations
- **Prediction Latency**: < 1 second for 24-hour forecast
- **Memory Usage**: < 100MB for typical semiconductor datasets

## Production Deployment

### Docker Support
```dockerfile
FROM python:3.12-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "equipment_drift_monitor.py", "predict", "--model-path", "model.joblib"]
```

### Monitoring Integration
- **Logging**: Structured JSON logs for production monitoring
- **Metrics**: Prometheus-compatible metrics export
- **Alerts**: Threshold-based alerting for anomaly detection
- **Dashboards**: Grafana dashboard templates included

## Example Use Cases

### 1. CVD Chamber Temperature Monitoring
```bash
python equipment_drift_monitor.py train \
    --data cvd_chamber_data.csv \
    --target temperature \
    --window-size 48 \
    --tolerance 3.0 \
    --save cvd_model.joblib
```

### 2. Etch Tool Pressure Stability
```bash
python equipment_drift_monitor.py evaluate \
    --model-path etch_model.joblib \
    --data etch_pressure_data.csv \
    --cost-per-unit 5.0
```

### 3. Ion Implanter Beam Current Drift
```bash
python equipment_drift_monitor.py predict \
    --model-path implanter_model.joblib \
    --horizon 72 \
    --confidence-level 0.99
```

## Troubleshooting

### Common Issues

**Q: Model training fails with "insufficient data" error**  
A: Ensure at least 200 observations and proper datetime indexing

**Q: Predictions show high uncertainty**  
A: Increase training data or adjust model order parameters

**Q: Anomaly detection too sensitive**  
A: Increase drift_threshold or adjust confidence_level

**Q: CLI returns JSON parsing errors**  
A: Verify Python 3.8+ and all dependencies installed

### Data Quality Checks
- Missing values: Automatic interpolation for gaps < 5% of window
- Outliers: Automatic detection and flagging
- Frequency: Validation of regular time intervals
- Stationarity: ADF test warnings for non-stationary series

## Contributing

1. Follow the standardized CLI pattern from existing modules
2. Maintain semiconductor manufacturing context in examples
3. Include comprehensive tests for new features
4. Update documentation for API changes
5. Validate against existing time series modules

## License

Part of the Python for Semiconductors learning series. See main repository for license terms.

---

**Ready for Production**: This project follows industry best practices for semiconductor manufacturing environments and provides a solid foundation for equipment drift monitoring systems.