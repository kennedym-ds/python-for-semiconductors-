# Infrastructure Monitoring Demo

This directory contains optional demonstration scripts and configurations for setting up MLflow and monitoring infrastructure.

## Contents

### MLflow Configuration
- `mlflow_config.yaml` - Sample MLflow server configuration
- `start_mlflow.sh` - Script to start MLflow tracking server
- `docker-compose.yml` - Docker setup for MLflow with database backend

### Sample Scripts
- `health_check.py` - Script to check monitoring system health
- `alert_simulator.py` - Generate synthetic alerts for testing
- `dashboard_demo.py` - Simple monitoring dashboard using Streamlit

## Quick Start

1. **Start MLflow Server**:
   ```bash
   ./start_mlflow.sh
   ```

2. **Run Health Check**:
   ```bash
   python health_check.py
   ```

3. **Launch Dashboard** (optional):
   ```bash
   streamlit run dashboard_demo.py
   ```

## Production Deployment

For production deployment, consider:
- Using a proper database backend (PostgreSQL, MySQL)
- Setting up proper authentication and authorization
- Implementing secure artifact storage (S3, Azure Blob)
- Configuring load balancing and high availability

See the main module documentation for detailed deployment guidance.