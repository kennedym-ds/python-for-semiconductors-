#!/usr/bin/env python3
"""
Tests for Module 2.3 Advanced Statistical Analysis

This module tests advanced statistical analysis methods for semiconductor data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats

# Add module path
MODULE_PATH = Path(__file__).parent
sys.path.insert(0, str(MODULE_PATH))

RANDOM_SEED = 42


class TestANOVA:
    """Test Analysis of Variance (ANOVA) functionality."""

    @pytest.fixture
    def tool_comparison_data(self):
        """Create data for comparing multiple tools."""
        np.random.seed(RANDOM_SEED)

        # Simulate yield data from 3 different tools
        tool_a = np.random.normal(95, 2, 50)
        tool_b = np.random.normal(96, 2, 50)  # Slightly higher
        tool_c = np.random.normal(94, 2, 50)  # Slightly lower

        data = pd.DataFrame(
            {"tool": ["A"] * 50 + ["B"] * 50 + ["C"] * 50, "yield": np.concatenate([tool_a, tool_b, tool_c])}
        )

        return data

    def test_one_way_anova(self, tool_comparison_data):
        """Test one-way ANOVA for comparing tools."""
        # Separate data by tool
        groups = [group["yield"].values for name, group in tool_comparison_data.groupby("tool")]

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Verify statistics
        assert isinstance(f_stat, (int, float))
        assert f_stat >= 0
        assert 0 <= p_value <= 1

    def test_two_way_anova_setup(self):
        """Test two-way ANOVA data setup."""
        np.random.seed(RANDOM_SEED)

        # Create data with two factors: tool and recipe
        tools = ["A", "B", "C"]
        recipes = ["R1", "R2"]

        data = []
        for tool in tools:
            for recipe in recipes:
                yields = np.random.normal(95, 2, 20)
                for y in yields:
                    data.append({"tool": tool, "recipe": recipe, "yield": y})

        df = pd.DataFrame(data)

        # Verify structure
        assert len(df) == 120  # 3 tools × 2 recipes × 20 samples
        assert df["tool"].nunique() == 3
        assert df["recipe"].nunique() == 2


class TestRegression:
    """Test regression analysis functionality."""

    @pytest.fixture
    def process_parameters(self):
        """Create process parameter data with relationships."""
        np.random.seed(RANDOM_SEED)
        n = 100

        # Temperature affects yield
        temp = np.random.uniform(340, 360, n)
        yield_val = 70 + 0.1 * temp + np.random.normal(0, 2, n)

        data = pd.DataFrame({"temperature": temp, "yield": yield_val})

        return data

    def test_linear_regression(self, process_parameters):
        """Test linear regression fitting."""
        from scipy.stats import linregress

        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = linregress(
            process_parameters["temperature"], process_parameters["yield"]
        )

        # Verify results
        assert isinstance(slope, (int, float))
        assert isinstance(intercept, (int, float))
        assert -1 <= r_value <= 1
        assert 0 <= p_value <= 1

    def test_multiple_regression_setup(self):
        """Test multiple regression data setup."""
        np.random.seed(RANDOM_SEED)
        n = 100

        # Multiple parameters affecting yield
        temp = np.random.uniform(340, 360, n)
        pressure = np.random.uniform(95, 105, n)
        time = np.random.uniform(60, 80, n)

        # Yield as function of all parameters
        yield_val = 50 + 0.1 * temp + 0.2 * pressure + 0.05 * time + np.random.normal(0, 2, n)

        data = pd.DataFrame({"temperature": temp, "pressure": pressure, "time": time, "yield": yield_val})

        # Verify setup
        assert len(data) == 100
        assert len(data.columns) == 4


class TestChiSquare:
    """Test Chi-Square tests for categorical data."""

    def test_chi_square_independence(self):
        """Test chi-square test for independence."""
        # Create contingency table
        # Tool vs Defect Type
        contingency_table = np.array(
            [
                [30, 20, 10],  # Tool A defect counts
                [25, 25, 15],  # Tool B defect counts
                [20, 30, 20],  # Tool C defect counts
            ]
        )

        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Verify results
        assert isinstance(chi2, (int, float))
        assert chi2 >= 0
        assert 0 <= p_value <= 1
        assert dof == 4  # (3-1) * (3-1)
        assert expected.shape == contingency_table.shape

    def test_goodness_of_fit(self):
        """Test chi-square goodness of fit test."""
        # Observed frequencies
        observed = np.array([45, 55, 50, 40, 60])

        # Expected frequencies (uniform distribution)
        expected = np.array([50, 50, 50, 50, 50])

        # Perform test
        chi2, p_value = stats.chisquare(observed, expected)

        # Verify
        assert isinstance(chi2, (int, float))
        assert chi2 >= 0
        assert 0 <= p_value <= 1


class TestNonParametric:
    """Test non-parametric statistical tests."""

    def test_mann_whitney_u(self):
        """Test Mann-Whitney U test (non-parametric alternative to t-test)."""
        np.random.seed(RANDOM_SEED)

        # Two samples (not necessarily normal)
        sample1 = np.random.exponential(2, 50)
        sample2 = np.random.exponential(2.5, 50)

        # Perform test
        statistic, p_value = stats.mannwhitneyu(sample1, sample2)

        # Verify
        assert isinstance(statistic, (int, float))
        assert 0 <= p_value <= 1

    def test_kruskal_wallis(self):
        """Test Kruskal-Wallis H test (non-parametric ANOVA)."""
        np.random.seed(RANDOM_SEED)

        # Three groups
        group1 = np.random.exponential(2, 30)
        group2 = np.random.exponential(2.5, 30)
        group3 = np.random.exponential(3, 30)

        # Perform test
        statistic, p_value = stats.kruskal(group1, group2, group3)

        # Verify
        assert isinstance(statistic, (int, float))
        assert 0 <= p_value <= 1


class TestCorrelationAnalysis:
    """Test correlation analysis methods."""

    @pytest.fixture
    def multivariate_data(self):
        """Create multivariate correlated data."""
        np.random.seed(RANDOM_SEED)

        # Create correlated variables
        temp = np.random.normal(350, 5, 100)
        pressure = 50 + 0.5 * temp + np.random.normal(0, 2, 100)
        yield_val = 60 + 0.05 * temp + 0.1 * pressure + np.random.normal(0, 1, 100)

        return pd.DataFrame({"temperature": temp, "pressure": pressure, "yield": yield_val})

    def test_pearson_correlation(self, multivariate_data):
        """Test Pearson correlation coefficient."""
        # Calculate correlation
        corr, p_value = stats.pearsonr(multivariate_data["temperature"], multivariate_data["pressure"])

        # Verify
        assert -1 <= corr <= 1
        assert 0 <= p_value <= 1

    def test_spearman_correlation(self, multivariate_data):
        """Test Spearman rank correlation (non-parametric)."""
        # Calculate correlation
        corr, p_value = stats.spearmanr(multivariate_data["temperature"], multivariate_data["yield"])

        # Verify
        assert -1 <= corr <= 1
        assert 0 <= p_value <= 1

    def test_correlation_matrix(self, multivariate_data):
        """Test full correlation matrix calculation."""
        # Calculate correlation matrix
        corr_matrix = multivariate_data.corr()

        # Verify properties
        assert corr_matrix.shape == (3, 3)

        # Diagonal should be 1
        assert np.allclose(np.diag(corr_matrix), 1.0)

        # Symmetric
        assert np.allclose(corr_matrix, corr_matrix.T)


class TestPowerAnalysis:
    """Test statistical power analysis."""

    def test_sample_size_calculation(self):
        """Test sample size estimation for desired power."""
        from scipy.stats import norm

        # Parameters
        alpha = 0.05  # Significance level
        power = 0.80  # Desired power
        effect_size = 0.5  # Cohen's d

        # Z-scores
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        # Sample size per group (simplified formula)
        n = ((z_alpha + z_beta) / effect_size) ** 2 * 2

        # Verify reasonable sample size
        assert n > 0
        assert n < 1000  # Should be reasonable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
