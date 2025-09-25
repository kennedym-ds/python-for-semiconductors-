"""
Shared pytest fixtures and configuration for Python for Semiconductors tests.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Generator


@pytest.fixture(scope="session")
def random_seed() -> int:
    """Consistent random seed for reproducible tests."""
    return 42


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def synthetic_regression_data(random_seed: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic regression dataset for testing."""
    np.random.seed(random_seed)
    n_samples, n_features = 1000, 20
    
    # Generate correlated features mimicking semiconductor process parameters
    feature_names = [f"process_param_{i:02d}" for i in range(n_features)]
    X = np.random.randn(n_samples, n_features)
    
    # Add some correlation structure
    X[:, 1] = X[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2
    X[:, 2] = X[:, 0] * 0.6 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1
    
    # Generate target with known relationship
    true_coef = np.random.randn(n_features) * 0.5
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    
    df_X = pd.DataFrame(X, columns=feature_names)
    series_y = pd.Series(y, name="yield")
    
    return df_X, series_y


@pytest.fixture
def synthetic_classification_data(random_seed: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic classification dataset for testing."""
    np.random.seed(random_seed)
    n_samples, n_features = 1500, 15
    
    # Generate features
    feature_names = [f"sensor_{i:02d}" for i in range(n_features)]
    X = np.random.randn(n_samples, n_features)
    
    # Create two distinct clusters for binary classification
    mask = np.arange(n_samples) < n_samples // 2
    X[mask, :3] += 2  # Shift first cluster
    X[~mask, :3] -= 2  # Shift second cluster
    
    # Generate binary target
    y = mask.astype(int)
    
    df_X = pd.DataFrame(X, columns=feature_names)
    series_y = pd.Series(y, name="defect_class")
    
    return df_X, series_y


@pytest.fixture
def sample_pipeline_config() -> Dict[str, Any]:
    """Standard pipeline configuration for testing."""
    return {
        "random_seed": 42,
        "test_size": 0.2,
        "cv_folds": 3,
        "max_samples": 1000,
        "output_format": "json",
        "model_params": {
            "n_estimators": 10,  # Small for fast testing
            "random_state": 42,
            "max_depth": 3
        }
    }


@pytest.fixture
def mock_dataset_paths() -> Dict[str, str]:
    """Mock dataset paths for testing."""
    return {
        "secom": "datasets/secom/secom.data",
        "steel_plates": "datasets/steel-plates/steel_plates_features.csv",
        "wm811k": "datasets/wm811k/LSWMD.pkl"
    }


@pytest.fixture(scope="session")
def performance_thresholds() -> Dict[str, float]:
    """Performance thresholds for regression testing."""
    return {
        "training_time_max": 30.0,  # seconds
        "memory_usage_max": 1000.0,  # MB
        "min_accuracy": 0.6,
        "min_r2_score": 0.3,
        "max_mae": 1.0
    }


@pytest.fixture
def manufacturing_metrics_config() -> Dict[str, Any]:
    """Configuration for semiconductor manufacturing-specific metrics."""
    return {
        "tolerance": 0.1,
        "cost_per_false_positive": 100.0,
        "cost_per_false_negative": 500.0,
        "target_yield": 0.95,
        "process_capability_min": 1.33
    }