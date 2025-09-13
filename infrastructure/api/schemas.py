"""Pydantic schemas for the FastAPI model serving application.

Defines request/response models for input validation and API documentation.
Uses Pydantic v2 syntax for semiconductor manufacturing data.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request schema for model predictions."""

    temperature: float = Field(..., description="Process temperature in Celsius", ge=300.0, le=600.0, example=450.0)
    pressure: float = Field(..., description="Process pressure in torr", ge=0.1, le=10.0, example=2.5)
    flow: float = Field(..., description="Gas flow rate in sccm", ge=50.0, le=200.0, example=120.0)
    time: float = Field(..., description="Process time in seconds", ge=10.0, le=120.0, example=60.0)

    class Config:
        json_schema_extra = {"example": {"temperature": 450.0, "pressure": 2.5, "flow": 120.0, "time": 60.0}}


class PredictionResponse(BaseModel):
    """Response schema for model predictions."""

    prediction: float = Field(..., description="Model prediction value")
    confidence: float = Field(..., description="Prediction confidence score", ge=0.0, le=1.0)
    model_version: str = Field(..., description="Version of the model used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional prediction metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 87.5,
                "confidence": 0.92,
                "model_version": "1.0.0",
                "metadata": {"model_used": "regression_model", "timestamp": 1640995200.0},
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service health status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    version: str = Field(..., description="API version")

    class Config:
        json_schema_extra = {
            "example": {"status": "healthy", "uptime_seconds": 3600.0, "model_loaded": True, "version": "v1"}
        }


class ModelInfoResponse(BaseModel):
    """Response schema for model information endpoint."""

    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of model (e.g., regression, classification)")
    created_at: str = Field(..., description="Model creation timestamp")
    input_schema: Dict[str, str] = Field(..., description="Input feature schema")
    output_schema: Dict[str, str] = Field(..., description="Output schema")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "yield_predictor",
                "version": "1.0.0",
                "model_type": "regression",
                "created_at": "2024-01-01T12:00:00Z",
                "input_schema": {"temperature": "float", "pressure": "float", "flow": "float", "time": "float"},
                "output_schema": {"prediction": "float"},
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    path: str = Field(..., description="Request path that caused the error")
    timestamp: Optional[float] = Field(default=None, description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid input: temperature out of range",
                "status_code": 422,
                "path": "/predict",
                "timestamp": 1640995200.0,
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions (future extension)."""

    samples: list[PredictionRequest] = Field(..., description="List of prediction requests", max_items=100)

    class Config:
        json_schema_extra = {
            "example": {
                "samples": [
                    {"temperature": 450.0, "pressure": 2.5, "flow": 120.0, "time": 60.0},
                    {"temperature": 455.0, "pressure": 2.6, "flow": 118.0, "time": 62.0},
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions (future extension)."""

    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of predictions in batch")
    processing_time_seconds: float = Field(..., description="Time taken to process batch")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [{"prediction": 87.5, "confidence": 0.92, "model_version": "1.0.0", "metadata": {}}],
                "batch_size": 1,
                "processing_time_seconds": 0.05,
            }
        }
