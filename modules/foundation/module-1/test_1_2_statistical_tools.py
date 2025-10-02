#!/usr/bin/env python3
"""
Tests for Module 1.2 Statistical Analysis Tools

This module tests statistical analysis functionality for semiconductor data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add module path
MODULE_PATH = Path(__file__).parent
sys.path.insert(0, str(MODULE_PATH))

RANDOM_SEED = 42


class TestStatisticalAnalysis:
    """Test suite for statistical analysis tools."""

    @pytest.fixture
    def sample_process_data(self):
        """Create sample process parameter data."""
        np.random.seed(RANDOM_SEED)
        n_samples = 200

        data = pd.DataFrame(
            {
                "temperature": np.random.normal(350, 5, n_samples),
                "pressure": np.random.normal(100, 3, n_samples),
                "flow_rate": np.random.normal(50, 2, n_samples),
                "thickness": np.random.normal(1000, 50, n_samples),
                "yield": np.random.normal(95, 3, n_samples),
            }
        )

        return data

    def test_hypothesis_testing_ttest(self, sample_process_data):
        """Test t-test for comparing two groups."""
        from scipy import stats

        # Split data into two groups
        group1 = sample_process_data["yield"][:100]
        group2 = sample_process_data["yield"][100:]

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1, group2)

        # Verify results are valid
        assert isinstance(t_stat, (int, float))
        assert isinstance(p_value, (int, float))
        assert 0 <= p_value <= 1

    def test_anova(self, sample_process_data):
        """Test ANOVA for comparing multiple groups."""
        from scipy import stats

        # Create three groups
        sample_process_data["tool"] = np.random.choice(["A", "B", "C"], len(sample_process_data))

        groups = [group["yield"].values for name, group in sample_process_data.groupby("tool")]

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Verify results
        assert isinstance(f_stat, (int, float))
        assert f_stat >= 0
        assert 0 <= p_value <= 1

    def test_correlation_analysis(self, sample_process_data):
        """Test correlation calculation."""
        # Calculate correlation matrix
        corr_matrix = sample_process_data.corr()

        # Verify properties of correlation matrix
        assert corr_matrix.shape == (5, 5)

        # Diagonal should be 1
        assert np.allclose(np.diag(corr_matrix), 1.0)

        # Matrix should be symmetric
        assert np.allclose(corr_matrix, corr_matrix.T)

        # All values should be between -1 and 1
        assert (corr_matrix.abs() <= 1.0).all().all()

    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        from scipy import stats

        # Generate sample data
        np.random.seed(RANDOM_SEED)
        data = np.random.normal(100, 10, 50)

        # Calculate 95% confidence interval
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)

        # Verify interval properties
        assert ci[0] < mean < ci[1]
        assert ci[1] - ci[0] > 0

    def test_doe_factorial_design(self):
        """Test Design of Experiments (DOE) setup."""
        # Simple 2^3 factorial design
        factors = {"temperature": [340, 360], "pressure": [95, 105], "time": [60, 80]}

        # Generate full factorial design
        from itertools import product

        experiments = list(product(*factors.values()))

        # Should have 2^3 = 8 experiments
        assert len(experiments) == 8

        # Verify all combinations are present
        temps = [exp[0] for exp in experiments]
        assert temps.count(340) == 4
        assert temps.count(360) == 4


class TestDescriptiveStatistics:
    """Test descriptive statistics calculations."""

    @pytest.fixture
    def measurement_data(self):
        """Create measurement data with known properties."""
        np.random.seed(RANDOM_SEED)
        return np.random.normal(100, 15, 1000)

    def test_mean_calculation(self, measurement_data):
        """Test mean calculation."""
        mean = np.mean(measurement_data)

        # Should be close to 100 (population mean)
        assert 98 < mean < 102

    def test_std_calculation(self, measurement_data):
        """Test standard deviation calculation."""
        std = np.std(measurement_data, ddof=1)  # Sample std

        # Should be close to 15 (population std)
        assert 14 < std < 16

    def test_percentiles(self, measurement_data):
        """Test percentile calculations."""
        p25 = np.percentile(measurement_data, 25)
        p50 = np.percentile(measurement_data, 50)
        p75 = np.percentile(measurement_data, 75)

        # Verify ordering
        assert p25 < p50 < p75

        # 50th percentile should be close to mean for normal distribution
        mean = np.mean(measurement_data)
        assert abs(p50 - mean) < 2

    def test_process_capability(self):
        """Test process capability (Cp, Cpk) calculation."""
        # Generate process data
        np.random.seed(RANDOM_SEED)
        data = np.random.normal(100, 2, 200)

        # Specification limits
        USL = 110  # Upper specification limit
        LSL = 90  # Lower specification limit

        # Calculate Cp
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        cp = (USL - LSL) / (6 * std)

        # Calculate Cpk
        cpk = min((USL - mean) / (3 * std), (mean - LSL) / (3 * std))

        # Verify calculations
        assert cp > 0
        assert cpk > 0
        assert cpk <= cp  # Cpk is always <= Cp


class TestStatisticalProcessControl:
    """Test SPC (Statistical Process Control) functionality."""

    def test_control_limits(self):
        """Test control limit calculation for control charts."""
        np.random.seed(RANDOM_SEED)

        # Generate in-control process data
        data = np.random.normal(100, 3, 100)

        # Calculate control limits (mean ± 3 sigma)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        ucl = mean + 3 * std  # Upper control limit
        lcl = mean - 3 * std  # Lower control limit

        # Most data should be within control limits
        within_limits = (data >= lcl) & (data <= ucl)
        assert within_limits.sum() > 95  # > 95% should be within limits

    def test_out_of_control_detection(self):
        """Test detection of out-of-control points."""
        np.random.seed(RANDOM_SEED)

        # Create data with an outlier
        data = np.random.normal(100, 3, 100)
        data[50] = 120  # Introduce outlier

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        ucl = mean + 3 * std
        lcl = mean - 3 * std

        # Detect out-of-control points
        out_of_control = (data < lcl) | (data > ucl)

        # Should detect at least one point
        assert out_of_control.sum() >= 1


class TestDataTransformations:
    """Test data transformation techniques."""

    def test_normalization(self):
        """Test min-max normalization."""
        np.random.seed(RANDOM_SEED)
        data = np.random.uniform(50, 150, 100)

        # Min-max normalization to [0, 1]
        normalized = (data - data.min()) / (data.max() - data.min())

        # Verify range
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert np.isclose(normalized.min(), 0, atol=1e-10)
        assert np.isclose(normalized.max(), 1, atol=1e-10)

    def test_standardization(self):
        """Test z-score standardization."""
        np.random.seed(RANDOM_SEED)
        data = np.random.normal(100, 15, 1000)

        # Z-score standardization
        standardized = (data - np.mean(data)) / np.std(data, ddof=1)

        # Verify properties
        assert np.abs(np.mean(standardized)) < 0.01  # Mean close to 0
        assert np.abs(np.std(standardized, ddof=1) - 1.0) < 0.01  # Std close to 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
