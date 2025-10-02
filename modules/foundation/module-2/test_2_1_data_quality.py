#!/usr/bin/env python3
"""
Tests for Module 2.1 Data Quality Framework

This module tests the data quality assessment framework for semiconductor data.
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


class TestDataQualityDimensions:
    """Test data quality dimension assessments."""

    @pytest.fixture
    def sample_secom_data(self):
        """Create sample SECOM-like dataset."""
        np.random.seed(RANDOM_SEED)
        n_samples = 500
        n_features = 50

        # Create features with varying quality issues
        data = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)])

        # Introduce missing values
        data.loc[0:10, "feature_0"] = np.nan

        # Introduce outliers
        data.loc[20:25, "feature_1"] = 10

        # Add constant column (zero variance)
        data["feature_constant"] = 42

        return data

    def test_completeness_assessment(self, sample_secom_data):
        """Test completeness dimension (missing values)."""
        # Calculate completeness per column
        completeness = 1 - (sample_secom_data.isnull().sum() / len(sample_secom_data))

        # Verify calculation
        assert completeness["feature_0"] < 1.0  # Has missing values
        assert completeness["feature_1"] == 1.0  # No missing values

        # Overall completeness
        overall = completeness.mean()
        assert 0 <= overall <= 1

    def test_uniqueness_assessment(self, sample_secom_data):
        """Test uniqueness dimension (duplicate rows)."""
        # Check for duplicate rows
        n_duplicates = sample_secom_data.duplicated().sum()
        uniqueness_score = 1 - (n_duplicates / len(sample_secom_data))

        # Should have high uniqueness (likely no duplicates in random data)
        assert uniqueness_score > 0.9

    def test_consistency_assessment(self, sample_secom_data):
        """Test consistency dimension (data type consistency)."""
        # All columns should be numeric
        numeric_cols = sample_secom_data.select_dtypes(include=[np.number]).columns

        consistency_score = len(numeric_cols) / len(sample_secom_data.columns)

        # Should be 100% numeric
        assert consistency_score == 1.0

    def test_validity_assessment(self, sample_secom_data):
        """Test validity dimension (values within acceptable ranges)."""
        # For normal distribution, check if values are within reasonable range
        # Using ±4 sigma as validity threshold

        invalid_count = 0
        for col in sample_secom_data.columns:
            if col != "feature_constant":
                mean = sample_secom_data[col].mean()
                std = sample_secom_data[col].std()

                # Count values outside ±4 sigma
                invalid = ((sample_secom_data[col] < mean - 4 * std) | (sample_secom_data[col] > mean + 4 * std)).sum()
                invalid_count += invalid

        # Should have very few invalid values for normal data
        total_values = len(sample_secom_data) * (len(sample_secom_data.columns) - 1)
        validity_score = 1 - (invalid_count / total_values)

        assert validity_score > 0.95

    def test_timeliness_assessment(self):
        """Test timeliness dimension (data freshness)."""
        # Create data with timestamps
        dates = pd.date_range("2024-01-01", periods=100, freq="H")
        data = pd.DataFrame({"timestamp": dates, "value": np.random.randn(100)})

        # Check data age
        current_time = pd.Timestamp("2024-01-05")
        data_age_hours = (current_time - data["timestamp"]).dt.total_seconds() / 3600

        # Older data = lower timeliness score
        timeliness_scores = 1 / (1 + data_age_hours / 24)  # Decay over days

        assert (timeliness_scores >= 0).all()
        assert (timeliness_scores <= 1).all()


class TestDataQualityMetrics:
    """Test specific data quality metrics."""

    @pytest.fixture
    def quality_test_data(self):
        """Create data for quality testing."""
        np.random.seed(RANDOM_SEED)
        return pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": [1, 2, np.nan, 4, 5] * 20,  # 20% missing
                "feature_3": [1] * 100,  # Constant
                "feature_4": list(range(100)),  # Unique values
            }
        )

    def test_missing_value_rate(self, quality_test_data):
        """Test missing value rate calculation."""
        missing_rates = quality_test_data.isnull().sum() / len(quality_test_data)

        assert missing_rates["feature_1"] == 0.0
        assert missing_rates["feature_2"] == 0.2  # 20% missing
        assert missing_rates["feature_3"] == 0.0

    def test_variance_analysis(self, quality_test_data):
        """Test variance for detecting constant columns."""
        variances = quality_test_data.var()

        # Feature 3 should have zero variance
        assert np.isclose(variances["feature_3"], 0)

        # Others should have non-zero variance
        assert variances["feature_1"] > 0
        assert variances["feature_4"] > 0

    def test_cardinality_check(self, quality_test_data):
        """Test unique value counting (cardinality)."""
        cardinalities = quality_test_data.nunique()

        assert cardinalities["feature_3"] == 1  # Constant
        assert cardinalities["feature_4"] == 100  # All unique

    def test_outlier_detection_iqr(self):
        """Test outlier detection using IQR method."""
        np.random.seed(RANDOM_SEED)

        # Create data with outliers
        data = np.random.randn(100)
        data[0] = 10  # Outlier
        data[1] = -10  # Outlier

        # IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (data < lower_bound) | (data > upper_bound)

        # Should detect the outliers
        assert outliers.sum() >= 2


class TestDataQualityReport:
    """Test data quality report generation."""

    def test_comprehensive_quality_assessment(self):
        """Test complete quality assessment workflow."""
        np.random.seed(RANDOM_SEED)

        # Create sample dataset
        data = pd.DataFrame(
            {
                "temp": np.random.normal(350, 5, 100),
                "pressure": np.random.normal(100, 3, 100),
                "yield": np.random.normal(95, 2, 100),
            }
        )

        # Add some quality issues
        data.loc[0:5, "temp"] = np.nan  # Missing values
        data.loc[10, "pressure"] = 200  # Outlier

        # Generate quality report
        report = {
            "completeness": (1 - data.isnull().sum() / len(data)).mean(),
            "n_outliers": 0,  # Would be calculated with outlier detection
            "n_duplicates": data.duplicated().sum(),
            "constant_columns": (data.var() == 0).sum(),
            "total_rows": len(data),
            "total_columns": len(data.columns),
        }

        # Verify report structure
        assert "completeness" in report
        assert "n_outliers" in report
        assert "n_duplicates" in report
        assert report["total_rows"] == 100
        assert report["total_columns"] == 3

    def test_quality_score_calculation(self):
        """Test overall quality score calculation."""
        # Define quality dimensions with weights
        dimensions = {"completeness": 0.95, "uniqueness": 1.0, "consistency": 1.0, "validity": 0.98, "accuracy": 0.90}

        weights = {"completeness": 0.25, "uniqueness": 0.15, "consistency": 0.20, "validity": 0.25, "accuracy": 0.15}

        # Calculate weighted quality score
        quality_score = sum(dimensions[k] * weights[k] for k in dimensions.keys())

        assert 0 <= quality_score <= 1
        assert quality_score > 0.9  # Should be high quality


class TestDataCleaning:
    """Test data cleaning operations."""

    def test_remove_constant_columns(self):
        """Test removal of zero-variance columns."""
        data = pd.DataFrame(
            {"var_col": np.random.randn(100), "const_col": [42] * 100, "var_col2": np.random.randn(100)}
        )

        # Identify constant columns
        constant_cols = data.columns[data.var() == 0].tolist()

        # Remove them
        cleaned_data = data.drop(columns=constant_cols)

        assert "const_col" not in cleaned_data.columns
        assert "var_col" in cleaned_data.columns
        assert len(cleaned_data.columns) == 2

    def test_handle_missing_values(self):
        """Test missing value handling strategies."""
        data = pd.DataFrame({"feature": [1, 2, np.nan, 4, np.nan, 6, 7, 8, np.nan, 10]})

        # Strategy 1: Drop rows
        dropped = data.dropna()
        assert len(dropped) == 7

        # Strategy 2: Fill with mean
        filled_mean = data.fillna(data.mean())
        assert filled_mean.isnull().sum().sum() == 0

        # Strategy 3: Forward fill
        filled_ffill = data.fillna(method="ffill")
        assert filled_ffill.loc[2, "feature"] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
