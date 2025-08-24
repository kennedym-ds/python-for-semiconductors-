#!/usr/bin/env python3
"""
Module 1.2: Statistical Analysis Tools for Semiconductor Manufacturing

This module provides production-ready statistical analysis tools specifically designed
for semiconductor manufacturing processes. It includes comprehensive implementations
of statistical methods commonly used in semiconductor fabs.

Author: Python for Semiconductors Course
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, lognorm, weibull_min, poisson
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class SpecificationLimits:
    """Data class for specification limits"""
    lsl: float  # Lower specification limit
    usl: float  # Upper specification limit
    target: Optional[float] = None  # Target value
    
    def __post_init__(self):
        if self.lsl >= self.usl:
            raise ValueError("Lower specification limit must be less than upper specification limit")
        if self.target is None:
            self.target = (self.lsl + self.usl) / 2
        if not (self.lsl <= self.target <= self.usl):
            logger.warning(f"Target value {self.target} is outside specification limits [{self.lsl}, {self.usl}]")

@dataclass
class StatisticalResult:
    """Base class for statistical analysis results"""
    test_name: str
    statistic: float
    p_value: float
    confidence_level: float
    interpretation: str
    raw_data: Optional[Dict[str, Any]] = None

class StatisticalAnalyzer(ABC):
    """Abstract base class for statistical analyzers"""
    
    def __init__(self, confidence_level: float = 0.95):
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    @abstractmethod
    def analyze(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Perform statistical analysis"""
        pass
    
    def validate_data(self, data: Union[List, np.ndarray]) -> np.ndarray:
        """Validate and clean input data"""
        data = np.asarray(data, dtype=float)
        
        if data.size == 0:
            raise ValueError("Data array cannot be empty")
        
        # Check for and handle missing values
        valid_mask = np.isfinite(data)
        if not np.all(valid_mask):
            invalid_count = np.sum(~valid_mask)
            logger.warning(f"Removing {invalid_count} invalid values (NaN, inf) from data")
            data = data[valid_mask]
            
        if data.size == 0:
            raise ValueError("No valid data remaining after cleaning")
            
        return data

class ProcessCapabilityAnalyzer(StatisticalAnalyzer):
    """
    Process capability analysis for semiconductor manufacturing.
    
    Provides comprehensive capability analysis including Cp, Cpk, Pp, Ppk indices
    and statistical confidence intervals.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        super().__init__(confidence_level)
        
    def analyze(self, data: np.ndarray, specs: SpecificationLimits) -> Dict[str, Any]:
        """
        Perform complete process capability analysis
        
        Parameters:
        -----------
        data : array-like
            Process measurement data
        specs : SpecificationLimits
            Specification limits object
            
        Returns:
        --------
        dict : Comprehensive capability analysis results
        """
        data = self.validate_data(data)
        n = len(data)
        
        if n < 30:
            logger.warning(f"Sample size ({n}) is small. Capability estimates may be unreliable.")
        
        # Basic statistics
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        
        # Process capability indices
        cp = (specs.usl - specs.lsl) / (6 * std_dev)
        cpu = (specs.usl - mean) / (3 * std_dev)
        cpl = (mean - specs.lsl) / (3 * std_dev)
        cpk = min(cpu, cpl)
        
        # Process performance indices (using sample statistics)
        pp = cp  # For this implementation, assuming stable process
        ppk = cpk
        
        # Confidence intervals for Cpk
        cpk_ci = self._calculate_cpk_confidence_interval(cpk, n)
        
        # Process centering
        process_center = mean
        spec_center = specs.target
        centering_ratio = abs(process_center - spec_center) / ((specs.usl - specs.lsl) / 2)
        is_centered = centering_ratio < 0.25  # Industry threshold
        
        # Yield prediction
        predicted_yield = self._calculate_predicted_yield(data, specs)
        
        # Capability classification
        capability_class = self._classify_capability(cpk)
        
        results = {
            'sample_statistics': {
                'n': n,
                'mean': mean,
                'std_dev': std_dev,
                'min': np.min(data),
                'max': np.max(data)
            },
            'specifications': {
                'lsl': specs.lsl,
                'usl': specs.usl,
                'target': specs.target,
                'tolerance': specs.usl - specs.lsl
            },
            'capability_indices': {
                'cp': cp,
                'cpk': cpk,
                'cpu': cpu,
                'cpl': cpl,
                'pp': pp,
                'ppk': ppk
            },
            'confidence_intervals': {
                'cpk_lower': cpk_ci[0],
                'cpk_upper': cpk_ci[1],
                'confidence_level': self.confidence_level
            },
            'process_assessment': {
                'is_centered': is_centered,
                'centering_ratio': centering_ratio,
                'predicted_yield_percent': predicted_yield,
                'capability_class': capability_class
            }
        }
        
        return results
    
    def _calculate_cpk_confidence_interval(self, cpk: float, n: int) -> Tuple[float, float]:
        """Calculate confidence interval for Cpk"""
        if cpk <= 0:
            return (0, 0)
            
        # Approximation for large samples
        se_cpk = cpk * np.sqrt((1/(9*n*cpk**2)) + (1/(2*(n-1))))
        t_critical = stats.t.ppf(1 - self.alpha/2, n - 1)
        
        ci_lower = max(0, cpk - t_critical * se_cpk)
        ci_upper = cpk + t_critical * se_cpk
        
        return (ci_lower, ci_upper)
    
    def _calculate_predicted_yield(self, data: np.ndarray, specs: SpecificationLimits) -> float:
        """Calculate predicted yield based on normal distribution assumption"""
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        
        yield_fraction = (stats.norm.cdf(specs.usl, mean, std_dev) - 
                         stats.norm.cdf(specs.lsl, mean, std_dev))
        
        return yield_fraction * 100
    
    def _classify_capability(self, cpk: float) -> str:
        """Classify process capability based on Cpk value"""
        if cpk >= 1.67:
            return "World Class (Cpk ≥ 1.67)"
        elif cpk >= 1.33:
            return "Adequate (1.33 ≤ Cpk < 1.67)"
        elif cpk >= 1.0:
            return "Marginal (1.0 ≤ Cpk < 1.33)"
        elif cpk >= 0.67:
            return "Poor (0.67 ≤ Cpk < 1.0)"
        else:
            return "Unacceptable (Cpk < 0.67)"
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted capability analysis report"""
        report = []
        report.append("=" * 60)
        report.append("PROCESS CAPABILITY ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Sample statistics
        stats_data = results['sample_statistics']
        report.append(f"\nSample Statistics:")
        report.append(f"  Sample Size (n): {stats_data['n']}")
        report.append(f"  Mean: {stats_data['mean']:.4f}")
        report.append(f"  Std Dev: {stats_data['std_dev']:.4f}")
        report.append(f"  Range: [{stats_data['min']:.4f}, {stats_data['max']:.4f}]")
        
        # Specifications
        specs_data = results['specifications']
        report.append(f"\nSpecifications:")
        report.append(f"  LSL: {specs_data['lsl']:.4f}")
        report.append(f"  Target: {specs_data['target']:.4f}")
        report.append(f"  USL: {specs_data['usl']:.4f}")
        report.append(f"  Tolerance: {specs_data['tolerance']:.4f}")
        
        # Capability indices
        cap_data = results['capability_indices']
        report.append(f"\nCapability Indices:")
        report.append(f"  Cp:  {cap_data['cp']:.4f}")
        report.append(f"  Cpk: {cap_data['cpk']:.4f}")
        report.append(f"  Cpu: {cap_data['cpu']:.4f}")
        report.append(f"  Cpl: {cap_data['cpl']:.4f}")
        
        # Confidence interval
        ci_data = results['confidence_intervals']
        report.append(f"\nCpk Confidence Interval ({ci_data['confidence_level']*100:.0f}%):")
        report.append(f"  [{ci_data['cpk_lower']:.4f}, {ci_data['cpk_upper']:.4f}]")
        
        # Assessment
        assess_data = results['process_assessment']
        report.append(f"\nProcess Assessment:")
        report.append(f"  Classification: {assess_data['capability_class']}")
        report.append(f"  Process Centered: {'Yes' if assess_data['is_centered'] else 'No'}")
        report.append(f"  Predicted Yield: {assess_data['predicted_yield_percent']:.2f}%")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

class ControlChartAnalyzer(StatisticalAnalyzer):
    """
    Statistical Process Control chart analysis.
    
    Implements Shewhart X-bar and R charts with Western Electric rules
    for out-of-control detection.
    """
    
    def __init__(self, subgroup_size: int, confidence_level: float = 0.95):
        super().__init__(confidence_level)
        self.subgroup_size = subgroup_size
        self.constants = self._get_control_chart_constants(subgroup_size)
        
    def _get_control_chart_constants(self, n: int) -> Dict[str, float]:
        """Get control chart constants for given subgroup size"""
        constants_table = {
            2: {'A2': 1.880, 'D3': 0, 'D4': 3.267, 'd2': 1.128},
            3: {'A2': 1.023, 'D3': 0, 'D4': 2.574, 'd2': 1.693},
            4: {'A2': 0.729, 'D3': 0, 'D4': 2.282, 'd2': 2.059},
            5: {'A2': 0.577, 'D3': 0, 'D4': 2.114, 'd2': 2.326},
            6: {'A2': 0.483, 'D3': 0, 'D4': 2.004, 'd2': 2.534},
            7: {'A2': 0.419, 'D3': 0.076, 'D4': 1.924, 'd2': 2.704},
            8: {'A2': 0.373, 'D3': 0.136, 'D4': 1.864, 'd2': 2.847},
            9: {'A2': 0.337, 'D3': 0.184, 'D4': 1.816, 'd2': 2.970},
            10: {'A2': 0.308, 'D3': 0.223, 'D4': 1.777, 'd2': 3.078}
        }
        
        if n not in constants_table:
            raise ValueError(f"Control chart constants not available for subgroup size {n}")
        
        return constants_table[n]
    
    def calculate_control_limits(self, subgroups: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Calculate control limits from Phase I data
        
        Parameters:
        -----------
        subgroups : list of arrays
            List of subgroup data arrays
            
        Returns:
        --------
        dict : Control limits for X-bar and R charts
        """
        if len(subgroups) < 20:
            logger.warning("Fewer than 20 subgroups provided. Control limits may be unreliable.")
        
        # Calculate subgroup statistics
        subgroup_means = [np.mean(subgroup) for subgroup in subgroups]
        subgroup_ranges = [np.max(subgroup) - np.min(subgroup) for subgroup in subgroups]
        
        # Grand averages
        xbar_bar = np.mean(subgroup_means)
        r_bar = np.mean(subgroup_ranges)
        
        # X-bar chart limits
        xbar_ucl = xbar_bar + self.constants['A2'] * r_bar
        xbar_lcl = xbar_bar - self.constants['A2'] * r_bar
        
        # R chart limits
        r_ucl = self.constants['D4'] * r_bar
        r_lcl = self.constants['D3'] * r_bar
        
        limits = {
            'xbar': {
                'center': xbar_bar,
                'ucl': xbar_ucl,
                'lcl': xbar_lcl
            },
            'r': {
                'center': r_bar,
                'ucl': r_ucl,
                'lcl': r_lcl
            }
        }
        
        return limits
    
    def detect_out_of_control_signals(self, values: np.ndarray, limits: Dict[str, float]) -> List[Tuple[int, str]]:
        """
        Detect out-of-control signals using Western Electric rules
        
        Parameters:
        -----------
        values : array-like
            Control chart values (means or ranges)
        limits : dict
            Control limits with keys 'center', 'ucl', 'lcl'
            
        Returns:
        --------
        list : List of (index, rule_description) tuples for signals
        """
        center = limits['center']
        ucl = limits['ucl']
        lcl = limits['lcl']
        
        # Calculate zone boundaries
        zone_a_upper = center + 2 * (ucl - center) / 3
        zone_a_lower = center - 2 * (center - lcl) / 3
        zone_b_upper = center + (ucl - center) / 3
        zone_b_lower = center - (center - lcl) / 3
        
        signals = []
        
        for i, value in enumerate(values):
            # Rule 1: Single point beyond control limits
            if value > ucl or value < lcl:
                signals.append((i, "Rule 1: Point beyond control limits"))
            
            # Rule 2: 9 consecutive points on same side of center line
            if i >= 8:
                last_9 = values[i-8:i+1]
                if np.all(last_9 > center) or np.all(last_9 < center):
                    signals.append((i, "Rule 2: 9 consecutive points on one side"))
            
            # Rule 3: 6 consecutive points steadily increasing or decreasing
            if i >= 5:
                last_6 = values[i-5:i+1]
                diffs = np.diff(last_6)
                if np.all(diffs > 0) or np.all(diffs < 0):
                    signals.append((i, "Rule 3: 6 consecutive trending points"))
            
            # Rule 4: 14 consecutive points alternating up and down
            if i >= 13:
                last_14 = values[i-13:i+1]
                diffs = np.diff(last_14)
                alternating = True
                for j in range(len(diffs) - 1):
                    if diffs[j] * diffs[j+1] >= 0:
                        alternating = False
                        break
                if alternating:
                    signals.append((i, "Rule 4: 14 consecutive alternating points"))
            
            # Rule 5: 2 out of 3 consecutive points in Zone A or beyond
            if i >= 2:
                last_3 = values[i-2:i+1]
                zone_a_count = np.sum((last_3 > zone_a_upper) | (last_3 < zone_a_lower))
                if zone_a_count >= 2:
                    signals.append((i, "Rule 5: 2 of 3 points in Zone A"))
            
            # Rule 6: 4 out of 5 consecutive points in Zone B or beyond
            if i >= 4:
                last_5 = values[i-4:i+1]
                zone_b_count = np.sum((last_5 > zone_b_upper) | (last_5 < zone_b_lower))
                if zone_b_count >= 4:
                    signals.append((i, "Rule 6: 4 of 5 points in Zone B or beyond"))
        
        return signals
    
    def analyze(self, phase1_subgroups: List[np.ndarray], 
                phase2_subgroups: List[np.ndarray]) -> Dict[str, Any]:
        """
        Complete control chart analysis with Phase I and Phase II data
        
        Parameters:
        -----------
        phase1_subgroups : list
            Phase I subgroups for establishing control limits
        phase2_subgroups : list
            Phase II subgroups for monitoring
            
        Returns:
        --------
        dict : Complete control chart analysis results
        """
        # Calculate control limits from Phase I data
        limits = self.calculate_control_limits(phase1_subgroups)
        
        # Calculate Phase II statistics
        phase2_means = [np.mean(subgroup) for subgroup in phase2_subgroups]
        phase2_ranges = [np.max(subgroup) - np.min(subgroup) for subgroup in phase2_subgroups]
        
        # Detect signals
        xbar_signals = self.detect_out_of_control_signals(phase2_means, limits['xbar'])
        r_signals = self.detect_out_of_control_signals(phase2_ranges, limits['r'])
        
        results = {
            'control_limits': limits,
            'phase2_statistics': {
                'subgroup_means': phase2_means,
                'subgroup_ranges': phase2_ranges
            },
            'signals': {
                'xbar_signals': xbar_signals,
                'r_signals': r_signals,
                'total_signals': len(xbar_signals) + len(r_signals)
            },
            'process_stability': {
                'xbar_in_control': len(xbar_signals) == 0,
                'r_in_control': len(r_signals) == 0,
                'overall_in_control': len(xbar_signals) == 0 and len(r_signals) == 0
            }
        }
        
        return results

class HypothesisTestAnalyzer(StatisticalAnalyzer):
    """
    Hypothesis testing for semiconductor process analysis.
    
    Provides comprehensive hypothesis testing capabilities including
    t-tests, ANOVA, and non-parametric alternatives.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        super().__init__(confidence_level)
    
    def one_sample_ttest(self, data: np.ndarray, null_value: float, 
                        alternative: str = 'two-sided') -> StatisticalResult:
        """
        Perform one-sample t-test
        
        Parameters:
        -----------
        data : array-like
            Sample data
        null_value : float
            Null hypothesis value
        alternative : str
            'two-sided', 'greater', or 'less'
            
        Returns:
        --------
        StatisticalResult : Test results and interpretation
        """
        data = self.validate_data(data)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(data, null_value, alternative=alternative)
        
        # Effect size (Cohen's d)
        effect_size = (np.mean(data) - null_value) / np.std(data, ddof=1)
        
        # Confidence interval for mean
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        t_critical = stats.t.ppf(1 - self.alpha/2, n - 1)
        ci_lower = mean - t_critical * se
        ci_upper = mean + t_critical * se
        
        # Interpretation
        significant = p_value < self.alpha
        interpretation = f"{'Reject' if significant else 'Fail to reject'} null hypothesis (μ = {null_value})"
        
        return StatisticalResult(
            test_name="One-Sample t-test",
            statistic=t_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            raw_data={
                'sample_mean': mean,
                'sample_std': np.std(data, ddof=1),
                'sample_size': n,
                'effect_size': effect_size,
                'confidence_interval': (ci_lower, ci_upper),
                'alternative': alternative
            }
        )
    
    def two_sample_ttest(self, group1: np.ndarray, group2: np.ndarray,
                        equal_var: bool = True, alternative: str = 'two-sided') -> StatisticalResult:
        """
        Perform two-sample t-test
        
        Parameters:
        -----------
        group1, group2 : array-like
            Sample data for two groups
        equal_var : bool
            Assume equal variances (Welch's t-test if False)
        alternative : str
            'two-sided', 'greater', or 'less'
            
        Returns:
        --------
        StatisticalResult : Test results and interpretation
        """
        group1 = self.validate_data(group1)
        group2 = self.validate_data(group2)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var, alternative=alternative)
        
        # Effect size (Cohen's d)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        if equal_var:
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                 (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                (len(group1) + len(group2) - 2))
        else:
            pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
        
        effect_size = (mean1 - mean2) / pooled_std
        
        # Interpretation
        significant = p_value < self.alpha
        interpretation = f"{'Significant' if significant else 'No significant'} difference between groups"
        
        return StatisticalResult(
            test_name=f"Two-Sample t-test ({'Equal' if equal_var else 'Unequal'} variance)",
            statistic=t_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            raw_data={
                'group1_mean': mean1,
                'group2_mean': mean2,
                'group1_std': np.std(group1, ddof=1),
                'group2_std': np.std(group2, ddof=1),
                'effect_size': effect_size,
                'alternative': alternative
            }
        )
    
    def anova_test(self, *groups) -> StatisticalResult:
        """
        Perform one-way ANOVA
        
        Parameters:
        -----------
        *groups : variable number of array-like
            Sample data for multiple groups
            
        Returns:
        --------
        StatisticalResult : ANOVA results and interpretation
        """
        # Validate all groups
        validated_groups = [self.validate_data(group) for group in groups]
        
        if len(validated_groups) < 2:
            raise ValueError("At least 2 groups required for ANOVA")
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*validated_groups)
        
        # Calculate effect size (eta-squared)
        group_means = [np.mean(group) for group in validated_groups]
        group_sizes = [len(group) for group in validated_groups]
        overall_mean = np.mean(np.concatenate(validated_groups))
        
        ss_between = sum(n * (mean - overall_mean)**2 for n, mean in zip(group_sizes, group_means))
        ss_total = sum((x - overall_mean)**2 for group in validated_groups for x in group)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Interpretation
        significant = p_value < self.alpha
        interpretation = f"{'Significant' if significant else 'No significant'} difference between groups"
        
        return StatisticalResult(
            test_name="One-Way ANOVA",
            statistic=f_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            raw_data={
                'group_means': group_means,
                'group_sizes': group_sizes,
                'eta_squared': eta_squared,
                'degrees_freedom': (len(validated_groups) - 1, sum(group_sizes) - len(validated_groups))
            }
        )

class DistributionAnalyzer(StatisticalAnalyzer):
    """
    Distribution analysis and goodness-of-fit testing.
    
    Provides tools for identifying the best distribution for semiconductor data
    and performing comprehensive distribution analysis.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        super().__init__(confidence_level)
        self.available_distributions = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'weibull': stats.weibull_min,
            'gamma': stats.gamma,
            'beta': stats.beta,
            'exponential': stats.expon
        }
    
    def normality_test(self, data: np.ndarray) -> Dict[str, StatisticalResult]:
        """
        Comprehensive normality testing using multiple tests
        
        Parameters:
        -----------
        data : array-like
            Sample data to test for normality
            
        Returns:
        --------
        dict : Results from multiple normality tests
        """
        data = self.validate_data(data)
        n = len(data)
        
        results = {}
        
        # Shapiro-Wilk test (best for small samples)
        if n <= 5000:  # Shapiro-Wilk limitation
            w_stat, w_p = stats.shapiro(data)
            results['shapiro_wilk'] = StatisticalResult(
                test_name="Shapiro-Wilk Test",
                statistic=w_stat,
                p_value=w_p,
                confidence_level=self.confidence_level,
                interpretation=f"Data {'appears normal' if w_p > self.alpha else 'does not appear normal'}"
            )
        
        # Kolmogorov-Smirnov test
        mean, std = np.mean(data), np.std(data, ddof=1)
        ks_stat, ks_p = stats.kstest(data, lambda x: stats.norm.cdf(x, mean, std))
        results['kolmogorov_smirnov'] = StatisticalResult(
            test_name="Kolmogorov-Smirnov Test",
            statistic=ks_stat,
            p_value=ks_p,
            confidence_level=self.confidence_level,
            interpretation=f"Data {'appears normal' if ks_p > self.alpha else 'does not appear normal'}"
        )
        
        # Anderson-Darling test
        ad_stat, ad_critical, ad_significance = stats.anderson(data, dist='norm')
        # Find the significance level corresponding to alpha
        alpha_percent = self.alpha * 100
        ad_p_approx = None
        for i, sig_level in enumerate(ad_significance):
            if alpha_percent <= sig_level:
                ad_p_approx = sig_level / 100
                break
        
        if ad_p_approx is None:
            ad_interpretation = "Insufficient evidence to determine normality"
        else:
            ad_interpretation = f"Data {'appears normal' if ad_stat < ad_critical[i] else 'does not appear normal'}"
        
        results['anderson_darling'] = StatisticalResult(
            test_name="Anderson-Darling Test",
            statistic=ad_stat,
            p_value=ad_p_approx or np.nan,
            confidence_level=self.confidence_level,
            interpretation=ad_interpretation
        )
        
        return results
    
    def fit_distribution(self, data: np.ndarray, distribution_name: str) -> Dict[str, Any]:
        """
        Fit a specific distribution to data and assess goodness of fit
        
        Parameters:
        -----------
        data : array-like
            Sample data
        distribution_name : str
            Name of distribution to fit
            
        Returns:
        --------
        dict : Distribution fitting results
        """
        data = self.validate_data(data)
        
        if distribution_name not in self.available_distributions:
            raise ValueError(f"Distribution '{distribution_name}' not available. "
                           f"Choose from: {list(self.available_distributions.keys())}")
        
        dist = self.available_distributions[distribution_name]
        
        # Fit distribution
        try:
            if distribution_name == 'lognormal':
                # For lognormal, fix location parameter to 0
                params = dist.fit(data, floc=0)
            else:
                params = dist.fit(data)
        except Exception as e:
            raise RuntimeError(f"Failed to fit {distribution_name} distribution: {e}")
        
        # Goodness of fit test
        ks_stat, ks_p = stats.kstest(data, lambda x: dist.cdf(x, *params))
        
        # Calculate AIC and BIC
        log_likelihood = np.sum(dist.logpdf(data, *params))
        k = len(params)
        n = len(data)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        # Distribution statistics
        try:
            dist_mean = dist.mean(*params)
            dist_var = dist.var(*params)
            dist_std = np.sqrt(dist_var)
        except:
            dist_mean = dist_var = dist_std = np.nan
        
        results = {
            'distribution': distribution_name,
            'parameters': params,
            'goodness_of_fit': {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'aic': aic,
                'bic': bic
            },
            'distribution_statistics': {
                'mean': dist_mean,
                'variance': dist_var,
                'std_dev': dist_std
            },
            'sample_statistics': {
                'mean': np.mean(data),
                'std_dev': np.std(data, ddof=1),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
        }
        
        return results
    
    def compare_distributions(self, data: np.ndarray, 
                            distributions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Compare multiple distributions and rank by goodness of fit
        
        Parameters:
        -----------
        data : array-like
            Sample data
        distributions : list, optional
            List of distribution names to compare. If None, uses all available.
            
        Returns:
        --------
        list : Distribution comparison results sorted by AIC
        """
        data = self.validate_data(data)
        
        if distributions is None:
            distributions = list(self.available_distributions.keys())
        
        results = []
        
        for dist_name in distributions:
            try:
                fit_result = self.fit_distribution(data, dist_name)
                results.append(fit_result)
            except Exception as e:
                logger.warning(f"Failed to fit {dist_name}: {e}")
                continue
        
        # Sort by AIC (lower is better)
        results.sort(key=lambda x: x['goodness_of_fit']['aic'])
        
        return results

class YieldAnalyzer:
    """
    Semiconductor yield analysis and prediction tools.
    
    Provides comprehensive yield analysis capabilities including
    Monte Carlo simulation and yield prediction models.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def calculate_current_yield(self, data: np.ndarray, specs: SpecificationLimits) -> Dict[str, float]:
        """
        Calculate current yield from measurement data
        
        Parameters:
        -----------
        data : array-like
            Measurement data
        specs : SpecificationLimits
            Specification limits
            
        Returns:
        --------
        dict : Yield analysis results
        """
        data = np.asarray(data)
        
        # Count passing units
        passing = np.sum((data >= specs.lsl) & (data <= specs.usl))
        total = len(data)
        yield_percent = (passing / total) * 100
        
        # Calculate defect rates
        below_lsl = np.sum(data < specs.lsl)
        above_usl = np.sum(data > specs.usl)
        
        results = {
            'total_units': total,
            'passing_units': passing,
            'failing_units': total - passing,
            'yield_percent': yield_percent,
            'defect_rate_ppm': (total - passing) / total * 1e6,
            'below_lsl_count': below_lsl,
            'above_usl_count': above_usl,
            'below_lsl_rate': below_lsl / total * 100,
            'above_usl_rate': above_usl / total * 100
        }
        
        return results
    
    def monte_carlo_yield_simulation(self, mean: float, std_dev: float, specs: SpecificationLimits,
                                   n_simulations: int = 100000) -> Dict[str, Any]:
        """
        Monte Carlo simulation for yield prediction
        
        Parameters:
        -----------
        mean : float
            Process mean
        std_dev : float
            Process standard deviation
        specs : SpecificationLimits
            Specification limits
        n_simulations : int
            Number of simulation iterations
            
        Returns:
        --------
        dict : Simulation results and statistics
        """
        # Generate simulated measurements
        simulated_data = np.random.normal(mean, std_dev, n_simulations)
        
        # Calculate yield
        passing = np.sum((simulated_data >= specs.lsl) & (simulated_data <= specs.usl))
        yield_percent = (passing / n_simulations) * 100
        
        # Calculate confidence interval for yield
        # Using Wilson score interval for proportion
        p = passing / n_simulations
        z = stats.norm.ppf(0.975)  # 95% confidence
        
        denominator = 1 + z**2 / n_simulations
        center = (p + z**2 / (2 * n_simulations)) / denominator
        half_width = z * np.sqrt(p * (1 - p) / n_simulations + z**2 / (4 * n_simulations**2)) / denominator
        
        ci_lower = max(0, (center - half_width) * 100)
        ci_upper = min(100, (center + half_width) * 100)
        
        # Defect analysis
        below_lsl = np.sum(simulated_data < specs.lsl)
        above_usl = np.sum(simulated_data > specs.usl)
        
        results = {
            'simulation_parameters': {
                'mean': mean,
                'std_dev': std_dev,
                'n_simulations': n_simulations
            },
            'yield_results': {
                'predicted_yield_percent': yield_percent,
                'confidence_interval_95': (ci_lower, ci_upper),
                'defect_rate_ppm': (n_simulations - passing) / n_simulations * 1e6
            },
            'defect_breakdown': {
                'below_lsl_count': below_lsl,
                'above_usl_count': above_usl,
                'below_lsl_ppm': below_lsl / n_simulations * 1e6,
                'above_usl_ppm': above_usl / n_simulations * 1e6
            },
            'theoretical_yield': {
                'normal_approximation': (stats.norm.cdf(specs.usl, mean, std_dev) - 
                                       stats.norm.cdf(specs.lsl, mean, std_dev)) * 100
            }
        }
        
        return results
    
    def yield_sensitivity_analysis(self, base_mean: float, base_std: float, specs: SpecificationLimits,
                                 mean_range: Tuple[float, float], std_range: Tuple[float, float],
                                 n_points: int = 10) -> Dict[str, Any]:
        """
        Analyze yield sensitivity to process parameter changes
        
        Parameters:
        -----------
        base_mean : float
            Baseline process mean
        base_std : float
            Baseline process standard deviation
        specs : SpecificationLimits
            Specification limits
        mean_range : tuple
            (min_mean, max_mean) for sensitivity analysis
        std_range : tuple
            (min_std, max_std) for sensitivity analysis
        n_points : int
            Number of points in each parameter range
            
        Returns:
        --------
        dict : Sensitivity analysis results
        """
        mean_values = np.linspace(mean_range[0], mean_range[1], n_points)
        std_values = np.linspace(std_range[0], std_range[1], n_points)
        
        # Mean sensitivity (fixed std)
        mean_sensitivity = []
        for mean_val in mean_values:
            yield_val = (stats.norm.cdf(specs.usl, mean_val, base_std) - 
                        stats.norm.cdf(specs.lsl, mean_val, base_std)) * 100
            mean_sensitivity.append(yield_val)
        
        # Std sensitivity (fixed mean)
        std_sensitivity = []
        for std_val in std_values:
            yield_val = (stats.norm.cdf(specs.usl, base_mean, std_val) - 
                        stats.norm.cdf(specs.lsl, base_mean, std_val)) * 100
            std_sensitivity.append(yield_val)
        
        # Calculate sensitivity coefficients (yield change per unit parameter change)
        mean_sensitivity_coeff = np.gradient(mean_sensitivity, mean_values)
        std_sensitivity_coeff = np.gradient(std_sensitivity, std_values)
        
        results = {
            'baseline': {
                'mean': base_mean,
                'std_dev': base_std,
                'yield_percent': (stats.norm.cdf(specs.usl, base_mean, base_std) - 
                                stats.norm.cdf(specs.lsl, base_mean, base_std)) * 100
            },
            'mean_sensitivity': {
                'mean_values': mean_values,
                'yield_values': mean_sensitivity,
                'sensitivity_coefficients': mean_sensitivity_coeff
            },
            'std_sensitivity': {
                'std_values': std_values,
                'yield_values': std_sensitivity,
                'sensitivity_coefficients': std_sensitivity_coeff
            }
        }
        
        return results

# Utility functions
def load_semiconductor_data(file_path: str) -> pd.DataFrame:
    """
    Load semiconductor measurement data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file containing measurement data
        
    Returns:
    --------
    DataFrame : Loaded measurement data
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        logger.info(f"Data shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise

def generate_sample_data(n_samples: int = 1000, parameter_type: str = 'threshold_voltage',
                        mean: float = 650, std_dev: float = 20, 
                        random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate sample semiconductor measurement data for testing
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    parameter_type : str
        Type of parameter ('threshold_voltage', 'leakage_current', 'critical_dimension')
    mean : float
        Mean value
    std_dev : float
        Standard deviation
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    array : Generated sample data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if parameter_type == 'leakage_current':
        # Log-normal distribution for leakage current
        sigma = 0.5  # Shape parameter
        scale = mean  # Scale parameter
        data = np.random.lognormal(np.log(scale), sigma, n_samples)
    elif parameter_type == 'critical_dimension':
        # Normal distribution with occasional outliers
        data = np.random.normal(mean, std_dev, n_samples)
        # Add 1% outliers
        n_outliers = int(0.01 * n_samples)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        data[outlier_indices] += np.random.normal(0, 3*std_dev, n_outliers)
    else:
        # Normal distribution (default for threshold voltage)
        data = np.random.normal(mean, std_dev, n_samples)
    
    return data

def create_comprehensive_report(results_dict: Dict[str, Any], title: str = "Statistical Analysis Report") -> str:
    """
    Create a comprehensive analysis report from multiple analysis results
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing various analysis results
    title : str
        Report title
        
    Returns:
    --------
    str : Formatted report
    """
    report = []
    report.append("=" * 80)
    report.append(title.center(80))
    report.append("=" * 80)
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    for analysis_name, results in results_dict.items():
        report.append("-" * 60)
        report.append(f"{analysis_name.replace('_', ' ').title()}")
        report.append("-" * 60)
        
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    if 'percent' in key.lower() or 'ppm' in key.lower():
                        report.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        report.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
                elif isinstance(value, str):
                    report.append(f"{key.replace('_', ' ').title()}: {value}")
                elif isinstance(value, (list, tuple)) and len(value) == 2:
                    report.append(f"{key.replace('_', ' ').title()}: [{value[0]:.4f}, {value[1]:.4f}]")
        
        report.append("")
    
    report.append("=" * 80)
    return "\n".join(report)

# Example usage and demonstration
def demonstrate_statistical_tools():
    """Demonstrate the usage of statistical analysis tools"""
    print("=== Statistical Analysis Tools Demonstration ===\n")
    
    # Generate sample data
    np.random.seed(42)
    threshold_data = generate_sample_data(200, 'threshold_voltage', mean=650, std_dev=15)
    specs = SpecificationLimits(lsl=610, usl=690, target=650)
    
    # 1. Process Capability Analysis
    print("1. Process Capability Analysis")
    print("-" * 40)
    capability_analyzer = ProcessCapabilityAnalyzer()
    cap_results = capability_analyzer.analyze(threshold_data, specs)
    print(capability_analyzer.generate_report(cap_results))
    
    # 2. Control Chart Analysis
    print("\n2. Control Chart Analysis")
    print("-" * 40)
    
    # Generate subgroup data
    phase1_subgroups = []
    for i in range(25):
        subgroup = np.random.normal(650, 15, 5)
        phase1_subgroups.append(subgroup)
    
    phase2_subgroups = []
    for i in range(20):
        if i < 10:
            subgroup = np.random.normal(650, 15, 5)  # In control
        else:
            subgroup = np.random.normal(660, 15, 5)  # Process shift
        phase2_subgroups.append(subgroup)
    
    control_analyzer = ControlChartAnalyzer(subgroup_size=5)
    control_results = control_analyzer.analyze(phase1_subgroups, phase2_subgroups)
    
    print(f"X-bar Control Limits: {control_results['control_limits']['xbar']}")
    print(f"R Control Limits: {control_results['control_limits']['r']}")
    print(f"Total Signals Detected: {control_results['signals']['total_signals']}")
    print(f"Process Stable: {control_results['process_stability']['overall_in_control']}")
    
    # 3. Hypothesis Testing
    print("\n3. Hypothesis Testing")
    print("-" * 40)
    hypothesis_analyzer = HypothesisTestAnalyzer()
    
    # One-sample t-test
    ttest_result = hypothesis_analyzer.one_sample_ttest(threshold_data, 650)
    print(f"One-sample t-test: {ttest_result.interpretation}")
    print(f"p-value: {ttest_result.p_value:.4f}")
    
    # 4. Distribution Analysis
    print("\n4. Distribution Analysis")
    print("-" * 40)
    dist_analyzer = DistributionAnalyzer()
    
    # Normality tests
    normality_results = dist_analyzer.normality_test(threshold_data)
    for test_name, result in normality_results.items():
        print(f"{result.test_name}: {result.interpretation} (p = {result.p_value:.4f})")
    
    # 5. Yield Analysis
    print("\n5. Yield Analysis")
    print("-" * 40)
    yield_analyzer = YieldAnalyzer(random_seed=42)
    
    # Current yield
    current_yield = yield_analyzer.calculate_current_yield(threshold_data, specs)
    print(f"Current Yield: {current_yield['yield_percent']:.2f}%")
    print(f"Defect Rate: {current_yield['defect_rate_ppm']:.0f} ppm")
    
    # Monte Carlo simulation
    mc_results = yield_analyzer.monte_carlo_yield_simulation(
        mean=np.mean(threshold_data), 
        std_dev=np.std(threshold_data, ddof=1), 
        specs=specs,
        n_simulations=50000
    )
    print(f"Predicted Yield: {mc_results['yield_results']['predicted_yield_percent']:.2f}%")
    print(f"95% CI: {mc_results['yield_results']['confidence_interval_95']}")

if __name__ == "__main__":
    # Run demonstration
    demonstrate_statistical_tools()
