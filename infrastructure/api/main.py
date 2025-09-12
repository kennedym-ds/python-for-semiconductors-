"""FastAPI Model Serving Application

A production-influenced template for deploying ML models as REST APIs.
Provides health checks, model versioning, input validation, and basic monitoring.

Features:
- Health endpoint for service monitoring
- Model prediction endpoint with input validation
- Model metadata and version information
- Error handling with proper HTTP status codes
- Logging for requests and responses
- Environment-based configuration
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from .schemas import (
    PredictionRequest, 
    PredictionResponse, 
    HealthResponse, 
    ModelInfoResponse,
    ErrorResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_PATH = os.getenv('MODEL_PATH', '/home/runner/work/python-for-semiconductors-/python-for-semiconductors-/temp_models/model.joblib')
METADATA_PATH = os.getenv('METADATA_PATH', '/home/runner/work/python-for-semiconductors-/python-for-semiconductors-/temp_models/metadata.json')
API_VERSION = os.getenv('API_VERSION', 'v1')
SERVICE_NAME = os.getenv('SERVICE_NAME', 'semiconductor-ml-api')

# Global model and metadata storage
model = None
model_metadata = None
service_start_time = time.time()

# Initialize FastAPI app
app = FastAPI(
    title="Semiconductor ML Model API",
    description="REST API for serving trained ML models in semiconductor manufacturing",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)


def load_model():
    """Load model and metadata at startup."""
    global model, model_metadata
    
    try:
        # Try to load from configured path first
        model_path = Path(MODEL_PATH)
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            model = joblib.load(model_path)
        else:
            # Fallback: look for any model in temp_models directory
            temp_models_dir = Path('/home/runner/work/python-for-semiconductors-/python-for-semiconductors-/temp_models')
            if temp_models_dir.exists():
                model_files = list(temp_models_dir.glob('*.joblib'))
                if model_files:
                    model_path = model_files[0]
                    logger.info(f"Loading fallback model from {model_path}")
                    model = joblib.load(model_path)
                else:
                    logger.warning("No model files found, using mock model")
                    model = None
            else:
                logger.warning("temp_models directory not found, using mock model")
                model = None
        
        # Try to load metadata
        metadata_path = Path(METADATA_PATH)
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        else:
            # Default metadata for demo
            model_metadata = {
                'model_name': 'demo_model',
                'version': '1.0.0',
                'model_type': 'regression',
                'created_at': '2024-01-01T00:00:00',
                'input_schema': {
                    'temperature': 'float',
                    'pressure': 'float', 
                    'flow': 'float',
                    'time': 'float'
                },
                'output_schema': {
                    'prediction': 'float'
                }
            }
        
        logger.info(f"Model loaded successfully: {model_metadata.get('model_name', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        model_metadata = {'error': f'Failed to load: {e}'}


def make_prediction(input_data: Dict[str, float]) -> Dict[str, Any]:
    """Make prediction using loaded model or mock prediction."""
    if model is None:
        # Mock prediction for demo purposes
        temp = input_data.get('temperature', 450.0)
        pressure = input_data.get('pressure', 2.5)
        flow = input_data.get('flow', 120.0)
        time_val = input_data.get('time', 60.0)
        
        # Simple mock calculation
        mock_yield = 85.0 + (temp - 450) * 0.1 + (pressure - 2.5) * 2.0 + (flow - 120) * 0.05
        mock_yield = max(0.0, min(100.0, mock_yield))  # Clamp to [0, 100]
        
        return {
            'prediction': round(mock_yield, 2),
            'confidence': 0.85,
            'model_used': 'mock_model'
        }
    
    try:
        # Convert input to DataFrame for model prediction
        df = pd.DataFrame([input_data])
        
        # Make prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(df)
            result = {
                'prediction': float(prediction[0]) if hasattr(prediction, '__getitem__') else float(prediction),
                'model_used': model_metadata.get('model_name', 'loaded_model')
            }
        else:
            # Handle pipeline objects
            prediction = model.predict(df)
            result = {
                'prediction': float(prediction[0]),
                'model_used': model_metadata.get('model_name', 'pipeline_model')
            }
        
        # Add confidence if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)
            result['confidence'] = float(np.max(proba))
        else:
            result['confidence'] = 0.9  # Default confidence for regression
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information."""
    return {
        "service": SERVICE_NAME,
        "version": API_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    uptime = time.time() - service_start_time
    
    status = "healthy"
    if model is None:
        status = "degraded"  # Service running but no model loaded
    
    return HealthResponse(
        status=status,
        uptime_seconds=round(uptime, 2),
        model_loaded=model is not None,
        version=API_VERSION
    )


@app.get("/version", response_model=ModelInfoResponse)
async def get_version():
    """Get model version and metadata."""
    return ModelInfoResponse(
        model_name=model_metadata.get('model_name', 'unknown'),
        version=model_metadata.get('version', '0.0.0'),
        model_type=model_metadata.get('model_type', 'unknown'),
        created_at=model_metadata.get('created_at', ''),
        input_schema=model_metadata.get('input_schema', {}),
        output_schema=model_metadata.get('output_schema', {})
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the loaded model."""
    try:
        # Convert request to dict
        input_data = request.dict()
        
        # Make prediction
        result = make_prediction(input_data)
        
        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result.get('confidence', 0.9),
            model_version=model_metadata.get('version', '1.0.0'),
            metadata={
                'model_used': result.get('model_used', 'unknown'),
                'timestamp': time.time()
            }
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Input validation error: {e}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint."""
    uptime = time.time() - service_start_time
    
    return {
        "uptime_seconds": round(uptime, 2),
        "model_loaded": model is not None,
        "service_name": SERVICE_NAME,
        "api_version": API_VERSION,
        "model_info": {
            "name": model_metadata.get('model_name', 'unknown'),
            "version": model_metadata.get('version', '0.0.0'),
            "type": model_metadata.get('model_type', 'unknown')
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            path=str(request.url.path)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            status_code=500,
            path=str(request.url.path)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)