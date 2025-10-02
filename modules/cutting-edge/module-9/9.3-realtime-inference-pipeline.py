#!/usr/bin/env python3
"""
Module 9.3: Real-Time Inference Pipeline

Production-ready real-time inference system for semiconductor manufacturing.
Supports streaming data, async endpoints, batch inference, and websockets.

Usage Examples:

    # Start async FastAPI server
    python 9.3-realtime-inference-pipeline.py serve --model models/defect_detector.joblib --port 8000

    # Stream sensor data simulation
    python 9.3-realtime-inference-pipeline.py stream --rate 10 --model models/anomaly_detector.joblib

    # Batch inference on streaming data
    python 9.3-realtime-inference-pipeline.py batch-stream --batch-size 32 --model models/yield_predictor.joblib

    # Test streaming performance
    python 9.3-realtime-inference-pipeline.py benchmark --requests 1000 --concurrent 10

Author: Python for Semiconductors
Date: September 2025
"""

import argparse
import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

import joblib
import numpy as np
import pandas as pd

# Optional imports (graceful degradation)
try:
    from fastapi import FastAPI, WebSocket, HTTPException, Header, BackgroundTasks
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("Warning: FastAPI not available. Install with: pip install fastapi uvicorn")

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class SensorReading:
    """Single sensor reading"""

    timestamp: str
    sensor_id: str
    temperature: float
    pressure: float
    flow_rate: float
    vibration: float

    @classmethod
    def from_dict(cls, data: dict) -> "SensorReading":
        return cls(**data)

    def to_features(self) -> np.ndarray:
        """Convert to feature array for model"""
        return np.array([self.temperature, self.pressure, self.flow_rate, self.vibration])


@dataclass
class PredictionResult:
    """Inference result"""

    timestamp: str
    prediction: float
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


if HAS_FASTAPI:

    class PredictionRequest(BaseModel):
        """REST API prediction request"""

        timestamp: str
        temperature: float
        pressure: float
        flow_rate: float
        vibration: float

    class BatchPredictionRequest(BaseModel):
        """REST API batch prediction request"""

        readings: List[Dict[str, Any]]


# ============================================================================
# Streaming Data Simulator
# ============================================================================


class SensorStreamSimulator:
    """Simulates streaming sensor data from semiconductor equipment"""

    def __init__(self, sensor_id: str = "SENSOR_001", rate_hz: float = 10.0, anomaly_rate: float = 0.05):
        self.sensor_id = sensor_id
        self.rate_hz = rate_hz
        self.interval = 1.0 / rate_hz
        self.anomaly_rate = anomaly_rate

        # Normal operating ranges
        self.temp_mean = 350.0
        self.temp_std = 5.0
        self.pressure_mean = 1013.0
        self.pressure_std = 10.0
        self.flow_mean = 100.0
        self.flow_std = 2.0
        self.vib_mean = 0.5
        self.vib_std = 0.1

    def generate_reading(self) -> SensorReading:
        """Generate single sensor reading"""
        is_anomaly = np.random.random() < self.anomaly_rate

        if is_anomaly:
            # Anomalous reading (out of range)
            temperature = self.temp_mean + np.random.normal(0, self.temp_std * 5)
            pressure = self.pressure_mean + np.random.normal(0, self.pressure_std * 5)
            flow_rate = self.flow_mean + np.random.normal(0, self.flow_std * 5)
            vibration = self.vib_mean + np.random.normal(0, self.vib_std * 10)
        else:
            # Normal reading
            temperature = np.random.normal(self.temp_mean, self.temp_std)
            pressure = np.random.normal(self.pressure_mean, self.pressure_std)
            flow_rate = np.random.normal(self.flow_mean, self.flow_std)
            vibration = np.random.normal(self.vib_mean, self.vib_std)

        return SensorReading(
            timestamp=datetime.now().isoformat(),
            sensor_id=self.sensor_id,
            temperature=temperature,
            pressure=pressure,
            flow_rate=flow_rate,
            vibration=vibration,
        )

    async def stream(self, duration_seconds: Optional[float] = None):
        """Generate async stream of sensor readings"""
        start_time = time.time()
        count = 0

        while True:
            # Check duration limit
            if duration_seconds and (time.time() - start_time) > duration_seconds:
                break

            reading = self.generate_reading()
            yield reading

            count += 1
            await asyncio.sleep(self.interval)

    def stream_sync(self, num_readings: int = 100):
        """Generate synchronous stream for testing"""
        for _ in range(num_readings):
            yield self.generate_reading()


# ============================================================================
# Real-Time Inference Engine
# ============================================================================


class RealtimeInferenceEngine:
    """Real-time inference engine with batching and monitoring"""

    def __init__(
        self, model_path: Path, batch_size: int = 32, max_wait_ms: float = 100.0, enable_monitoring: bool = True
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.enable_monitoring = enable_monitoring

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        # Monitoring
        if enable_monitoring:
            self.latency_monitor = LatencyMonitor()
            self.throughput_monitor = ThroughputMonitor()
            self.error_monitor = ErrorRateMonitor()

        # Batching queue
        self.queue = asyncio.Queue() if HAS_FASTAPI else None

    def predict_single(self, reading: SensorReading) -> PredictionResult:
        """Synchronous single prediction"""
        start_time = time.time()

        try:
            features = reading.to_features()
            prediction = self.model.predict([features])[0]

            # Try to get confidence if model supports predict_proba
            confidence = None
            if hasattr(self.model, "predict_proba"):
                try:
                    proba = self.model.predict_proba([features])[0]
                    confidence = float(np.max(proba))
                except Exception:
                    pass

            latency_ms = (time.time() - start_time) * 1000

            if self.enable_monitoring:
                self.latency_monitor.record_latency(latency_ms)
                self.throughput_monitor.record_request()
                self.error_monitor.record_result(True)

            return PredictionResult(
                timestamp=datetime.now().isoformat(),
                prediction=float(prediction),
                confidence=confidence,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            if self.enable_monitoring:
                self.error_monitor.record_result(False)
            raise

    def predict_batch(self, readings: List[SensorReading]) -> List[PredictionResult]:
        """Synchronous batch prediction"""
        start_time = time.time()

        try:
            features = np.array([r.to_features() for r in readings])
            predictions = self.model.predict(features)

            # Try to get confidences
            confidences = [None] * len(predictions)
            if hasattr(self.model, "predict_proba"):
                try:
                    probas = self.model.predict_proba(features)
                    confidences = [float(np.max(p)) for p in probas]
                except Exception:
                    pass

            total_latency_ms = (time.time() - start_time) * 1000
            avg_latency_ms = total_latency_ms / len(readings)

            if self.enable_monitoring:
                self.latency_monitor.record_latency(avg_latency_ms)
                self.throughput_monitor.record_request()
                self.error_monitor.record_result(True)

            results = []
            for pred, conf in zip(predictions, confidences):
                results.append(
                    PredictionResult(
                        timestamp=datetime.now().isoformat(),
                        prediction=float(pred),
                        confidence=conf,
                        latency_ms=avg_latency_ms,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            if self.enable_monitoring:
                self.error_monitor.record_result(False)
            raise

    async def predict_async(self, reading: SensorReading) -> PredictionResult:
        """Async single prediction (runs in thread pool)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_single, reading)

    def get_metrics(self) -> dict:
        """Get monitoring metrics"""
        if not self.enable_monitoring:
            return {}

        return {
            "latency": self.latency_monitor.get_stats(),
            "throughput_rps": self.throughput_monitor.get_throughput(),
            "error_rate": self.error_monitor.get_error_rate(),
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# Monitoring Components
# ============================================================================


class LatencyMonitor:
    """Monitor prediction latency"""

    def __init__(self, window_size: int = 1000):
        self.latencies = deque(maxlen=window_size)

    def record_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def get_stats(self) -> dict:
        if not self.latencies:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "avg_ms": 0.0, "max_ms": 0.0, "count": 0}

        latencies = list(self.latencies)
        return {
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "avg_ms": float(np.mean(latencies)),
            "max_ms": float(np.max(latencies)),
            "count": len(latencies),
        }


class ThroughputMonitor:
    """Monitor request throughput"""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def record_request(self):
        now = time.time()
        self.timestamps.append(now)

        # Remove old timestamps
        cutoff = now - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def get_throughput(self) -> float:
        """Get requests per second"""
        if len(self.timestamps) < 2:
            return 0.0

        duration = self.timestamps[-1] - self.timestamps[0]
        if duration == 0:
            return 0.0

        return len(self.timestamps) / duration


class ErrorRateMonitor:
    """Monitor error rate"""

    def __init__(self, window_size: int = 1000):
        self.results = deque(maxlen=window_size)

    def record_result(self, success: bool):
        self.results.append(success)

    def get_error_rate(self) -> float:
        if not self.results:
            return 0.0

        errors = sum(1 for r in self.results if not r)
        return errors / len(self.results)


# ============================================================================
# FastAPI Server
# ============================================================================

if HAS_FASTAPI:
    # Global inference engine (initialized on startup)
    inference_engine: Optional[RealtimeInferenceEngine] = None
    sensor_simulator: Optional[SensorStreamSimulator] = None

    app = FastAPI(
        title="Real-Time Inference API",
        description="Production real-time inference for semiconductor manufacturing",
        version="1.0.0",
    )

    @app.get("/")
    def root():
        """API root endpoint"""
        return {
            "service": "Real-Time Inference API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "predict": "/predict",
                "predict-batch": "/predict-batch",
                "stream": "/stream-predictions",
                "websocket": "/ws/predict",
            },
        }

    @app.get("/health")
    def health_check():
        """Health check endpoint"""
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {"status": "healthy", "model_loaded": True, "timestamp": datetime.now().isoformat()}

    @app.get("/metrics")
    def get_metrics():
        """Get monitoring metrics"""
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return inference_engine.get_metrics()

    @app.post("/predict")
    def predict(request: PredictionRequest):
        """Single prediction endpoint"""
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        reading = SensorReading(
            timestamp=request.timestamp,
            sensor_id="API_REQUEST",
            temperature=request.temperature,
            pressure=request.pressure,
            flow_rate=request.flow_rate,
            vibration=request.vibration,
        )

        result = inference_engine.predict_single(reading)
        return result.to_dict()

    @app.post("/predict-batch")
    def predict_batch(request: BatchPredictionRequest):
        """Batch prediction endpoint"""
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        readings = [
            SensorReading(
                timestamp=r.get("timestamp", datetime.now().isoformat()),
                sensor_id=r.get("sensor_id", "API_BATCH"),
                temperature=r["temperature"],
                pressure=r["pressure"],
                flow_rate=r["flow_rate"],
                vibration=r["vibration"],
            )
            for r in request.readings
        ]

        results = inference_engine.predict_batch(readings)
        return {"predictions": [r.to_dict() for r in results], "count": len(results)}

    @app.get("/stream-predictions")
    async def stream_predictions():
        """Server-Sent Events (SSE) streaming endpoint"""
        if inference_engine is None or sensor_simulator is None:
            raise HTTPException(status_code=503, detail="Service not ready")

        async def generate():
            async for reading in sensor_simulator.stream(duration_seconds=60):
                result = await inference_engine.predict_async(reading)

                data = {
                    "reading": {
                        "timestamp": reading.timestamp,
                        "temperature": reading.temperature,
                        "pressure": reading.pressure,
                        "flow_rate": reading.flow_rate,
                        "vibration": reading.vibration,
                    },
                    "prediction": result.to_dict(),
                }

                yield f"data: {json.dumps(data)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.websocket("/ws/predict")
    async def websocket_predict(websocket: WebSocket):
        """WebSocket endpoint for bidirectional streaming"""
        if inference_engine is None:
            await websocket.close(code=1011, reason="Model not loaded")
            return

        await websocket.accept()
        logger.info("WebSocket connection accepted")

        try:
            while True:
                # Receive sensor data from client
                data = await websocket.receive_json()

                # Create reading
                reading = SensorReading(
                    timestamp=data.get("timestamp", datetime.now().isoformat()),
                    sensor_id=data.get("sensor_id", "WS_CLIENT"),
                    temperature=data["temperature"],
                    pressure=data["pressure"],
                    flow_rate=data["flow_rate"],
                    vibration=data["vibration"],
                )

                # Make prediction
                result = await inference_engine.predict_async(reading)

                # Send result back
                await websocket.send_json(result.to_dict())

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close()


# ============================================================================
# CLI Actions
# ============================================================================


def action_serve(args):
    """Start FastAPI server"""
    if not HAS_FASTAPI:
        print("Error: FastAPI not available. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    global inference_engine, sensor_simulator

    # Initialize inference engine
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    inference_engine = RealtimeInferenceEngine(
        model_path=model_path, batch_size=args.batch_size, enable_monitoring=True
    )

    # Initialize sensor simulator for streaming endpoint
    sensor_simulator = SensorStreamSimulator(rate_hz=args.stream_rate)

    print(
        f"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║         Real-Time Inference API Server                   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

Model: {model_path}
Port: {args.port}
Workers: {args.workers}
Batch Size: {args.batch_size}
Stream Rate: {args.stream_rate} Hz

Endpoints:
  - http://localhost:{args.port}/          (API info)
  - http://localhost:{args.port}/health    (Health check)
  - http://localhost:{args.port}/metrics   (Monitoring)
  - http://localhost:{args.port}/predict   (Single prediction)
  - http://localhost:{args.port}/predict-batch (Batch)
  - http://localhost:{args.port}/stream-predictions (SSE)
  - ws://localhost:{args.port}/ws/predict  (WebSocket)

Starting server...
"""
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def action_stream(args):
    """Stream sensor data and make predictions"""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Initialize components
    engine = RealtimeInferenceEngine(model_path, enable_monitoring=True)
    simulator = SensorStreamSimulator(rate_hz=args.rate)

    print(
        f"""
Streaming Inference
═══════════════════════════════════════════════════════════
Model: {model_path}
Rate: {args.rate} Hz
Duration: {args.duration if args.duration else 'Unlimited'}

Press Ctrl+C to stop...
"""
    )

    # Run async stream
    async def run_stream():
        count = 0
        anomalies = 0

        try:
            async for reading in simulator.stream(duration_seconds=args.duration):
                result = await engine.predict_async(reading)
                count += 1

                # Check for anomaly (assuming binary classification)
                if result.prediction > 0.5:
                    anomalies += 1
                    print(f"⚠️  ANOMALY DETECTED at {reading.timestamp}")
                    print(f"   Temperature: {reading.temperature:.2f}°C")
                    print(f"   Pressure: {reading.pressure:.2f} mbar")
                    print(f"   Confidence: {result.confidence:.2%}")
                    print()

                # Print status every 10 readings
                if count % 10 == 0:
                    metrics = engine.get_metrics()
                    print(
                        f"Processed: {count} | Anomalies: {anomalies} | "
                        f"Latency: {metrics['latency']['avg_ms']:.2f}ms | "
                        f"Throughput: {metrics['throughput_rps']:.2f} rps"
                    )

        except KeyboardInterrupt:
            print("\nStopping stream...")

        # Final stats
        print(f"\n{'='*60}")
        print(f"Final Statistics:")
        print(f"  Total readings: {count}")
        print(f"  Anomalies detected: {anomalies} ({anomalies/count*100:.1f}%)")

        metrics = engine.get_metrics()
        print(f"\n  Latency:")
        print(f"    Average: {metrics['latency']['avg_ms']:.2f}ms")
        print(f"    P95: {metrics['latency']['p95_ms']:.2f}ms")
        print(f"    P99: {metrics['latency']['p99_ms']:.2f}ms")
        print(f"\n  Throughput: {metrics['throughput_rps']:.2f} requests/second")
        print(f"  Error rate: {metrics['error_rate']:.2%}")
        print(f"{'='*60}")

    # Run
    asyncio.run(run_stream())


def action_batch_stream(args):
    """Batch inference on streaming data"""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Initialize components
    engine = RealtimeInferenceEngine(model_path, enable_monitoring=True)
    simulator = SensorStreamSimulator(rate_hz=args.rate)

    print(
        f"""
Batch Streaming Inference
═══════════════════════════════════════════════════════════
Model: {model_path}
Batch Size: {args.batch_size}
Rate: {args.rate} Hz
Duration: {args.duration if args.duration else 'Unlimited'}

Press Ctrl+C to stop...
"""
    )

    # Run async stream with batching
    async def run_batch_stream():
        count = 0
        anomalies = 0
        batch = []

        try:
            async for reading in simulator.stream(duration_seconds=args.duration):
                batch.append(reading)

                # Process when batch is full
                if len(batch) >= args.batch_size:
                    results = engine.predict_batch(batch)

                    for reading, result in zip(batch, results):
                        count += 1
                        if result.prediction > 0.5:
                            anomalies += 1

                    # Print batch stats
                    metrics = engine.get_metrics()
                    print(
                        f"Batch {count//args.batch_size} | "
                        f"Processed: {count} | "
                        f"Anomalies: {anomalies} | "
                        f"Latency: {metrics['latency']['avg_ms']:.2f}ms/sample"
                    )

                    batch.clear()

        except KeyboardInterrupt:
            print("\nStopping stream...")

        # Process remaining batch
        if batch:
            results = engine.predict_batch(batch)
            count += len(batch)

        # Final stats
        print(f"\n{'='*60}")
        print(f"Final Statistics:")
        print(f"  Total readings: {count}")
        print(f"  Anomalies detected: {anomalies} ({anomalies/count*100:.1f}%)")

        metrics = engine.get_metrics()
        print(f"\n  Latency (per sample):")
        print(f"    Average: {metrics['latency']['avg_ms']:.2f}ms")
        print(f"    P95: {metrics['latency']['p95_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_rps']:.2f} requests/second")
        print(f"{'='*60}")

    # Run
    asyncio.run(run_batch_stream())


def action_benchmark(args):
    """Benchmark streaming performance"""
    print(
        f"""
Benchmark Configuration
═══════════════════════════════════════════════════════════
Target URL: {args.url}
Total Requests: {args.requests}
Concurrent: {args.concurrent}
"""
    )

    if not HAS_FASTAPI:
        print("Error: FastAPI required for benchmarking")
        sys.exit(1)

    # Simple benchmark using async requests
    import aiohttp

    async def send_request(session, url, reading_data):
        start = time.time()
        async with session.post(f"{url}/predict", json=reading_data) as response:
            await response.json()
            return time.time() - start

    async def run_benchmark():
        simulator = SensorStreamSimulator()
        latencies = []
        errors = 0

        async with aiohttp.ClientSession() as session:
            tasks = []

            for i in range(args.requests):
                reading = simulator.generate_reading()
                reading_data = {
                    "timestamp": reading.timestamp,
                    "temperature": reading.temperature,
                    "pressure": reading.pressure,
                    "flow_rate": reading.flow_rate,
                    "vibration": reading.vibration,
                }

                task = send_request(session, args.url, reading_data)
                tasks.append(task)

                # Control concurrency
                if len(tasks) >= args.concurrent:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for r in results:
                        if isinstance(r, Exception):
                            errors += 1
                        else:
                            latencies.append(r * 1000)  # Convert to ms
                    tasks.clear()

                    # Progress
                    print(f"Progress: {len(latencies) + errors}/{args.requests}", end="\r")

            # Process remaining
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        errors += 1
                    else:
                        latencies.append(r * 1000)

        # Results
        print(f"\n{'='*60}")
        print(f"Benchmark Results:")
        print(f"  Total requests: {args.requests}")
        print(f"  Successful: {len(latencies)}")
        print(f"  Errors: {errors}")
        print(f"\n  Latency:")
        print(f"    Average: {np.mean(latencies):.2f}ms")
        print(f"    Median: {np.median(latencies):.2f}ms")
        print(f"    P95: {np.percentile(latencies, 95):.2f}ms")
        print(f"    P99: {np.percentile(latencies, 99):.2f}ms")
        print(f"    Min: {np.min(latencies):.2f}ms")
        print(f"    Max: {np.max(latencies):.2f}ms")
        print(f"{'='*60}")

    try:
        asyncio.run(run_benchmark())
    except Exception as e:
        print(f"\nBenchmark error: {e}")
        print("Make sure the server is running:")
        print(f"  python {sys.argv[0]} serve --model <model_path>")
        sys.exit(1)


# ============================================================================
# CLI Setup
# ============================================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description="Real-Time Inference Pipeline", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Serve command
    p_serve = subparsers.add_parser("serve", help="Start FastAPI server")
    p_serve.add_argument("--model", type=str, required=True, help="Path to model file")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    p_serve.add_argument("--port", type=int, default=8000, help="Port number")
    p_serve.add_argument("--workers", type=int, default=1, help="Number of workers")
    p_serve.add_argument("--batch-size", type=int, default=32, help="Max batch size")
    p_serve.add_argument("--stream-rate", type=float, default=10.0, help="Stream rate (Hz)")
    p_serve.set_defaults(func=action_serve)

    # Stream command
    p_stream = subparsers.add_parser("stream", help="Stream sensor data with inference")
    p_stream.add_argument("--model", type=str, required=True, help="Path to model file")
    p_stream.add_argument("--rate", type=float, default=10.0, help="Stream rate (Hz)")
    p_stream.add_argument("--duration", type=float, default=None, help="Duration (seconds)")
    p_stream.set_defaults(func=action_stream)

    # Batch stream command
    p_batch = subparsers.add_parser("batch-stream", help="Batch inference on stream")
    p_batch.add_argument("--model", type=str, required=True, help="Path to model file")
    p_batch.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_batch.add_argument("--rate", type=float, default=10.0, help="Stream rate (Hz)")
    p_batch.add_argument("--duration", type=float, default=None, help="Duration (seconds)")
    p_batch.set_defaults(func=action_batch_stream)

    # Benchmark command
    p_bench = subparsers.add_parser("benchmark", help="Benchmark API performance")
    p_bench.add_argument("--url", type=str, default="http://localhost:8000", help="API URL")
    p_bench.add_argument("--requests", type=int, default=1000, help="Total requests")
    p_bench.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    p_bench.set_defaults(func=action_benchmark)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
