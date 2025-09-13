#!/bin/bash
# MLflow Server Startup Script

echo "🚀 Starting MLflow Tracking Server..."

# Create directories
mkdir -p mlruns
mkdir -p mlflow_artifacts

# Set environment variables
export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="./mlflow_artifacts"
export MLFLOW_HOST="127.0.0.1"
export MLFLOW_PORT="5000"

# Start MLflow server
echo "📊 MLflow Configuration:"
echo "   Backend Store: $MLFLOW_BACKEND_STORE_URI"
echo "   Artifact Root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"
echo "   Host: $MLFLOW_HOST"
echo "   Port: $MLFLOW_PORT"

mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \
    --host $MLFLOW_HOST \
    --port $MLFLOW_PORT \
    --serve-artifacts

echo "✅ MLflow server started at http://$MLFLOW_HOST:$MLFLOW_PORT"
