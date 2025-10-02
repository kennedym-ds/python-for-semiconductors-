#!/usr/bin/env python3
"""
Tests for Module 9.3 Real-time Inference and Model Serving

This module tests real-time inference capabilities including caching, batching,
latency tracking, and serving patterns for semiconductor ML applications.
"""

import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any
import json

RANDOM_SEED = 42


class SimpleCache:
    """Simple TTL-based cache for testing."""

    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Any:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now() < expiry:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set value in cache with TTL."""
        expiry = datetime.now() + timedelta(seconds=self.ttl_seconds)
        self.cache[key] = (value, expiry)

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()

    def size(self) -> int:
        """Get number of cached items."""
        # Remove expired entries
        now = datetime.now()
        expired_keys = [k for k, (v, exp) in self.cache.items() if exp < now]
        for k in expired_keys:
            del self.cache[k]
        return len(self.cache)


class LatencyTracker:
    """Track inference latencies for performance monitoring."""

    def __init__(self):
        self.latencies: List[float] = []

    def record(self, latency_ms: float):
        """Record a latency measurement."""
        self.latencies.append(latency_ms)

    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {"count": 0}

        latencies = np.array(self.latencies)
        return {
            "count": len(latencies),
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        }

    def clear(self):
        """Clear recorded latencies."""
        self.latencies.clear()


class BatchProcessor:
    """Simple batch processor for inference."""

    def __init__(self, batch_size: int = 32, timeout_ms: int = 100):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.buffer: List[np.ndarray] = []
        self.buffer_start_time = None

    def add_sample(self, sample: np.ndarray) -> bool:
        """Add sample to buffer. Returns True if batch is ready."""
        if not self.buffer:
            self.buffer_start_time = time.time()

        self.buffer.append(sample)

        # Check if batch is ready
        return len(self.buffer) >= self.batch_size or self._is_timeout()

    def _is_timeout(self) -> bool:
        """Check if timeout has been reached."""
        if self.buffer_start_time is None:
            return False
        elapsed_ms = (time.time() - self.buffer_start_time) * 1000
        return elapsed_ms >= self.timeout_ms

    def get_batch(self) -> np.ndarray:
        """Get current batch and clear buffer."""
        if not self.buffer:
            return np.array([])

        batch = np.array(self.buffer)
        self.buffer.clear()
        self.buffer_start_time = None
        return batch


class TestCaching:
    """Test caching mechanisms for inference."""

    @pytest.fixture
    def cache(self):
        """Create a cache instance."""
        return SimpleCache(ttl_seconds=2)

    def test_cache_set_get(self, cache):
        """Test basic cache set and get operations."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        assert cache.get("nonexistent") is None

    def test_cache_expiry(self, cache):
        """Test that cached values expire after TTL."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiry (TTL is 2 seconds)
        time.sleep(2.1)
        assert cache.get("key1") is None

    def test_cache_clear(self, cache):
        """Test cache clear operation."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.size() == 2

        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None

    def test_cache_size(self, cache):
        """Test cache size tracking."""
        assert cache.size() == 0

        cache.set("key1", "value1")
        assert cache.size() == 1

        cache.set("key2", "value2")
        assert cache.size() == 2

    def test_cache_overwrite(self, cache):
        """Test that setting same key overwrites value."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_cache_with_numpy_arrays(self, cache):
        """Test caching numpy arrays."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        cache.set("array1", arr)

        cached_arr = cache.get("array1")
        assert cached_arr is not None
        assert np.array_equal(cached_arr, arr)

    def test_cache_hit_rate_simulation(self, cache):
        """Test cache hit rate with repeated queries."""
        # Simulate repeated inference requests
        keys = ["wafer_001", "wafer_002", "wafer_001", "wafer_003", "wafer_001"]
        predictions = [0.8, 0.6, None, 0.7, None]  # None = cache hit expected

        hits = 0
        misses = 0

        for i, key in enumerate(keys):
            cached = cache.get(key)
            if cached is not None:
                hits += 1
            else:
                misses += 1
                # Simulate prediction
                pred = predictions[i] if predictions[i] is not None else 0.5
                cache.set(key, pred)

        # Should have 2 cache hits (wafer_001 appears 3 times, cached after first)
        assert hits == 2
        assert misses == 3


class TestLatencyTracking:
    """Test latency tracking and monitoring."""

    @pytest.fixture
    def tracker(self):
        """Create a latency tracker."""
        return LatencyTracker()

    def test_record_latency(self, tracker):
        """Test recording latencies."""
        tracker.record(10.5)
        tracker.record(15.2)
        tracker.record(12.8)

        stats = tracker.get_stats()
        assert stats["count"] == 3
        assert stats["mean"] > 0

    def test_empty_tracker(self, tracker):
        """Test stats on empty tracker."""
        stats = tracker.get_stats()
        assert stats["count"] == 0

    def test_latency_percentiles(self, tracker):
        """Test percentile calculations."""
        # Add 100 latencies
        np.random.seed(RANDOM_SEED)
        latencies = np.random.uniform(10, 100, 100)

        for lat in latencies:
            tracker.record(lat)

        stats = tracker.get_stats()
        assert stats["p50"] > stats["min"]
        assert stats["p95"] > stats["p50"]
        assert stats["p99"] > stats["p95"]
        assert stats["p99"] <= stats["max"]

    def test_latency_clear(self, tracker):
        """Test clearing latency history."""
        tracker.record(10.0)
        tracker.record(20.0)
        assert tracker.get_stats()["count"] == 2

        tracker.clear()
        assert tracker.get_stats()["count"] == 0

    def test_latency_statistics_accuracy(self, tracker):
        """Test accuracy of latency statistics."""
        # Add known values
        values = [10, 20, 30, 40, 50]
        for v in values:
            tracker.record(v)

        stats = tracker.get_stats()
        assert stats["mean"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["p50"] == 30.0


class TestBatchProcessing:
    """Test batch processing for inference."""

    @pytest.fixture
    def processor(self):
        """Create a batch processor."""
        return BatchProcessor(batch_size=4, timeout_ms=50)

    def test_batch_full(self, processor):
        """Test batch processing when batch is full."""
        sample = np.array([1, 2, 3])

        # Add samples until batch is full
        assert processor.add_sample(sample) is False
        assert processor.add_sample(sample) is False
        assert processor.add_sample(sample) is False
        assert processor.add_sample(sample) is True  # Batch full

        batch = processor.get_batch()
        assert batch.shape[0] == 4

    def test_batch_timeout(self, processor):
        """Test batch processing with timeout."""
        sample = np.array([1, 2, 3])

        # Add one sample
        processor.add_sample(sample)

        # Wait for timeout
        time.sleep(0.06)  # 60ms > 50ms timeout

        # Next add should trigger timeout
        assert processor.add_sample(sample) is True

    def test_empty_batch(self, processor):
        """Test getting batch when buffer is empty."""
        batch = processor.get_batch()
        assert batch.shape[0] == 0

    def test_batch_clear(self, processor):
        """Test that getting batch clears buffer."""
        sample = np.array([1, 2, 3])

        processor.add_sample(sample)
        processor.add_sample(sample)

        batch = processor.get_batch()
        assert batch.shape[0] == 2

        # Buffer should be empty now
        batch2 = processor.get_batch()
        assert batch2.shape[0] == 0

    def test_batch_shape(self, processor):
        """Test batch shape is correct."""
        samples = [np.array([i, i + 1, i + 2]) for i in range(4)]

        for sample in samples:
            processor.add_sample(sample)

        batch = processor.get_batch()
        assert batch.shape == (4, 3)


class TestInferenceAPI:
    """Test inference API patterns."""

    def test_request_validation(self):
        """Test request payload validation."""

        def validate_request(payload: Dict) -> bool:
            """Validate inference request."""
            required_fields = ["wafer_id", "features"]
            return all(field in payload for field in required_fields)

        # Valid request
        valid = {"wafer_id": "W001", "features": [1.0, 2.0, 3.0]}
        assert validate_request(valid) is True

        # Invalid requests
        invalid1 = {"wafer_id": "W001"}  # Missing features
        assert validate_request(invalid1) is False

        invalid2 = {"features": [1.0, 2.0]}  # Missing wafer_id
        assert validate_request(invalid2) is False

    def test_response_formatting(self):
        """Test response formatting."""

        def format_response(wafer_id: str, prediction: float, latency_ms: float) -> Dict:
            """Format inference response."""
            return {
                "wafer_id": wafer_id,
                "prediction": float(prediction),
                "confidence": float(abs(prediction - 0.5) * 2),  # 0 to 1 scale
                "latency_ms": float(latency_ms),
                "timestamp": datetime.now().isoformat(),
            }

        response = format_response("W001", 0.85, 12.5)

        assert "wafer_id" in response
        assert "prediction" in response
        assert "confidence" in response
        assert "latency_ms" in response
        assert "timestamp" in response
        assert 0 <= response["prediction"] <= 1

    def test_batch_request_processing(self):
        """Test processing batch requests."""
        batch_request = {
            "wafers": [
                {"wafer_id": "W001", "features": [1.0, 2.0, 3.0]},
                {"wafer_id": "W002", "features": [1.5, 2.5, 3.5]},
                {"wafer_id": "W003", "features": [2.0, 3.0, 4.0]},
            ]
        }

        # Extract features for batch inference
        wafer_ids = [w["wafer_id"] for w in batch_request["wafers"]]
        features = np.array([w["features"] for w in batch_request["wafers"]])

        assert len(wafer_ids) == 3
        assert features.shape == (3, 3)

    def test_error_handling(self):
        """Test error handling in inference."""

        def safe_predict(features: np.ndarray) -> Dict:
            """Safe prediction with error handling."""
            try:
                if features.shape[0] == 0:
                    raise ValueError("Empty input")

                # Simulate prediction
                prediction = float(features.mean())

                return {"status": "success", "prediction": prediction}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        # Valid input
        result = safe_predict(np.array([1, 2, 3]))
        assert result["status"] == "success"

        # Invalid input
        result = safe_predict(np.array([]))
        assert result["status"] == "error"


class TestModelVersioning:
    """Test model versioning for A/B testing and rollback."""

    def test_model_registry(self):
        """Test model registry functionality."""
        registry = {
            "v1": {"accuracy": 0.85, "created": "2024-01-01"},
            "v2": {"accuracy": 0.88, "created": "2024-02-01"},
            "v3": {"accuracy": 0.90, "created": "2024-03-01"},
        }

        # Get latest version
        latest = max(registry.keys())
        assert latest == "v3"

        # Get best performing version
        best = max(registry.keys(), key=lambda k: registry[k]["accuracy"])
        assert best == "v3"

    def test_ab_testing_split(self):
        """Test A/B testing traffic split."""
        np.random.seed(RANDOM_SEED)

        def route_to_model(user_id: str, split_ratio: float = 0.5) -> str:
            """Route request to model version based on split ratio."""
            # Use hash for consistent routing
            hash_val = hash(user_id) % 100
            return "model_a" if hash_val < split_ratio * 100 else "model_b"

        # Test routing
        user_ids = [f"user_{i}" for i in range(1000)]
        routes = [route_to_model(uid, 0.5) for uid in user_ids]

        # Should be approximately 50/50 split
        model_a_count = routes.count("model_a")
        model_b_count = routes.count("model_b")

        assert 400 < model_a_count < 600  # Allow some variance
        assert 400 < model_b_count < 600

    def test_model_rollback(self):
        """Test model rollback capability."""
        active_models = {"current": "v3", "previous": "v2", "stable": "v1"}

        # Rollback to previous
        def rollback():
            active_models["current"] = active_models["previous"]

        rollback()
        assert active_models["current"] == "v2"


class TestPerformanceMetrics:
    """Test performance monitoring metrics."""

    def test_throughput_calculation(self):
        """Test throughput (requests per second) calculation."""
        start_time = time.time()

        # Simulate processing requests
        n_requests = 100
        for _ in range(n_requests):
            time.sleep(0.001)  # 1ms per request

        elapsed = time.time() - start_time
        throughput = n_requests / elapsed

        assert throughput > 0
        # Should be roughly 1000 requests/sec (1ms each)
        assert 500 < throughput < 2000  # Allow variance

    def test_resource_utilization_tracking(self):
        """Test resource utilization metrics."""
        metrics = {"cpu_percent": 45.2, "memory_mb": 512, "gpu_percent": 80.0, "requests_per_sec": 150}

        # Validate metrics are in expected ranges
        assert 0 <= metrics["cpu_percent"] <= 100
        assert metrics["memory_mb"] > 0
        assert 0 <= metrics["gpu_percent"] <= 100
        assert metrics["requests_per_sec"] > 0

    def test_error_rate_monitoring(self):
        """Test error rate calculation."""
        total_requests = 1000
        failed_requests = 5

        error_rate = (failed_requests / total_requests) * 100

        assert error_rate == 0.5
        assert error_rate < 1.0  # Should be below 1% SLA


class TestSemiconductorInference:
    """Test semiconductor-specific inference scenarios."""

    def test_wafer_classification_inference(self):
        """Test real-time wafer defect classification."""
        # Simulate wafer features
        wafer_features = np.array(
            [
                [0.5, 0.8, 0.3, 0.9],  # Feature vector for one wafer
            ]
        )

        # Simulate inference
        def predict_defect(features: np.ndarray) -> Dict:
            """Predict defect probability."""
            # Simple mock prediction
            prediction = float(features.mean())
            return {"defect_probability": prediction, "pass": prediction < 0.6}

        result = predict_defect(wafer_features[0])
        assert "defect_probability" in result
        assert "pass" in result
        assert isinstance(result["pass"], bool)

    def test_yield_prediction_inference(self):
        """Test real-time yield prediction."""
        # Process parameters
        process_params = {"temperature": 350, "pressure": 100, "time": 60, "gas_flow": 50}

        def predict_yield(params: Dict) -> float:
            """Predict yield from process parameters."""
            # Mock prediction based on params
            score = (params["temperature"] / 400 + params["pressure"] / 150) / 2
            return min(max(score, 0.0), 1.0)

        yield_pred = predict_yield(process_params)
        assert 0 <= yield_pred <= 1

    def test_anomaly_detection_inference(self):
        """Test real-time equipment anomaly detection."""
        # Equipment sensor readings with clear anomaly (more samples for better statistics)
        sensor_data = np.array([100.0, 100.5, 99.5, 100.2, 99.8, 100.3, 100.1, 99.9, 100.4, 200.0])

        def detect_anomaly(readings: np.ndarray, threshold: float = 2.0) -> List[int]:
            """Detect anomalies using z-score."""
            mean = readings.mean()
            std = readings.std()
            if std == 0:
                return []
            z_scores = np.abs((readings - mean) / std)
            return list(np.where(z_scores > threshold)[0])

        anomalies = detect_anomaly(sensor_data)
        assert len(anomalies) >= 1  # Should detect at least the extreme value
        assert 9 in anomalies  # Last reading should be anomaly

    def test_multi_stage_inference(self):
        """Test multi-stage inference pipeline."""
        # Stage 1: Defect detection
        # Stage 2: Defect classification
        # Stage 3: Root cause analysis

        wafer_image = np.random.rand(128, 128)

        results = {}

        # Stage 1: Detection
        has_defect = wafer_image.mean() > 0.5
        results["has_defect"] = has_defect

        if has_defect:
            # Stage 2: Classification
            defect_type = "scratch" if wafer_image.std() > 0.3 else "particle"
            results["defect_type"] = defect_type

            # Stage 3: Root cause
            if defect_type == "scratch":
                results["likely_cause"] = "mechanical_handling"
            else:
                results["likely_cause"] = "clean_room_contamination"

        assert "has_defect" in results
        if results["has_defect"]:
            assert "defect_type" in results
            assert "likely_cause" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
