#!/usr/bin/env python3
"""
Data Quality Framework for Semiconductor Manufacturing

This production-ready script provides comprehensive data quality assessment
capabilities for semiconductor manufacturing datasets. It implements industry
best practices and standards for evaluating data quality across six key
dimensions: completeness, accuracy, consistency, validity, uniqueness, and timeliness.

Usage:
    python 2.1-data-quality-framework.py --input data.csv --output report.html
    python 2.1-data-quality-framework.py --config config.yaml --batch

Author: Machine Learning for Semiconductor Engineers
Date: 2025
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_quality.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


@dataclass
class QualityThresholds:
    """Configuration class for data quality thresholds."""

    completeness_critical: float = 95.0
    completeness_acceptable: float = 90.0
    outlier_rate_warning: float = 5.0
    outlier_rate_critical: float = 10.0
    duplicate_rate_warning: float = 1.0
    correlation_threshold: float = 0.95
    scale_ratio_warning: float = 100.0
    missing_column_threshold: float = 50.0


@dataclass
class QualityReport:
    """Data structure for quality assessment results."""

    dataset_info: Dict
    completeness: Dict
    accuracy: Dict
    consistency: Dict
    validity: Dict
    uniqueness: Dict
    timeliness: Dict
    overall_score: float
    recommendations: List[Dict]
    generated_at: str


class DataQualityFramework:
    """
    Comprehensive data quality assessment framework for semiconductor manufacturing.

    This class provides methods to assess data quality across six dimensions
    and generate actionable recommendations for improvement.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data quality framework.

        Args:
            config: Optional configuration dictionary with thresholds and settings
        """
        self.config = config or {}
        self.thresholds = QualityThresholds(**self.config.get("thresholds", {}))
        self.df = None
        self.target = None
        self.report = None

        logger.info("Data Quality Framework initialized")

    def load_data(self, filepath: Union[str, Path], target_column: Optional[str] = None) -> None:
        """
        Load dataset from file.

        Args:
            filepath: Path to the data file (CSV, Excel, or Parquet)
            target_column: Optional target column name
        """
        filepath = Path(filepath)

        try:
            if filepath.suffix.lower() == ".csv":
                self.df = pd.read_csv(filepath)
            elif filepath.suffix.lower() in [".xlsx", ".xls"]:
                self.df = pd.read_excel(filepath)
            elif filepath.suffix.lower() == ".parquet":
                self.df = pd.read_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")

            if target_column and target_column in self.df.columns:
                self.target = self.df[target_column]
                self.df = self.df.drop(columns=[target_column])

            logger.info(f"Data loaded successfully: {self.df.shape}")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def assess_completeness(self) -> Dict:
        """
        Assess data completeness across the dataset.

        Returns:
            Dictionary containing completeness metrics
        """
        logger.info("Assessing data completeness...")

        missing_data = self.df.isnull()
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = missing_data.sum().sum()

        completeness = {
            "overall_completeness_pct": ((total_cells - missing_cells) / total_cells) * 100,
            "total_missing_values": int(missing_cells),
            "columns_with_missing": int((missing_data.sum() > 0).sum()),
            "rows_with_missing": int((missing_data.sum(axis=1) > 0).sum()),
            "complete_rows_pct": ((missing_data.sum(axis=1) == 0).sum() / len(self.df)) * 100,
            "column_completeness": ((1 - missing_data.sum() / len(self.df)) * 100).to_dict(),
            "high_missing_columns": [],
        }

        # Identify columns with high missing rates
        missing_pct_by_col = (missing_data.sum() / len(self.df)) * 100
        high_missing = missing_pct_by_col[missing_pct_by_col > self.thresholds.missing_column_threshold]
        completeness["high_missing_columns"] = high_missing.to_dict()

        return completeness

    def assess_accuracy(self) -> Dict:
        """
        Assess data accuracy using statistical methods.

        Returns:
            Dictionary containing accuracy metrics
        """
        logger.info("Assessing data accuracy...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        accuracy = {
            "outlier_analysis": {},
            "distribution_analysis": {},
            "statistical_anomalies": {},
        }

        # Outlier detection using multiple methods
        total_outliers_iqr = 0
        total_outliers_zscore = 0
        column_outliers = {}

        for col in numeric_cols:
            if self.df[col].notna().sum() > 0:
                data = self.df[col].dropna()

                # IQR method
                Q1, Q3 = data.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers_iqr = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()

                # Z-score method
                z_scores = np.abs(stats.zscore(data))
                outliers_zscore = (z_scores > 3).sum()

                total_outliers_iqr += outliers_iqr
                total_outliers_zscore += outliers_zscore

                column_outliers[col] = {
                    "iqr_outliers": int(outliers_iqr),
                    "zscore_outliers": int(outliers_zscore),
                    "outlier_rate_pct": (outliers_iqr / len(data)) * 100,
                }

        total_numeric_values = self.df[numeric_cols].notna().sum().sum()

        accuracy["outlier_analysis"] = {
            "total_outliers_iqr": int(total_outliers_iqr),
            "total_outliers_zscore": int(total_outliers_zscore),
            "outlier_rate_iqr_pct": (
                (total_outliers_iqr / total_numeric_values) * 100 if total_numeric_values > 0 else 0
            ),
            "outlier_rate_zscore_pct": (
                (total_outliers_zscore / total_numeric_values) * 100 if total_numeric_values > 0 else 0
            ),
            "columns_with_outliers": len([col for col, stats in column_outliers.items() if stats["iqr_outliers"] > 0]),
            "column_details": column_outliers,
        }

        # Statistical anomalies
        accuracy["statistical_anomalies"] = {
            "infinite_values": int(np.isinf(self.df[numeric_cols]).sum().sum()),
            "negative_values": int((self.df[numeric_cols] < 0).sum().sum()),
            "zero_values": int((self.df[numeric_cols] == 0).sum().sum()),
            "constant_columns": int((self.df[numeric_cols].nunique() == 1).sum()),
        }

        return accuracy

    def assess_consistency(self) -> Dict:
        """
        Assess data consistency across the dataset.

        Returns:
            Dictionary containing consistency metrics
        """
        logger.info("Assessing data consistency...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        consistency = {
            "data_type_analysis": {},
            "scale_analysis": {},
            "correlation_analysis": {},
        }

        # Data type analysis
        dtype_counts = self.df.dtypes.value_counts()
        consistency["data_type_analysis"] = {
            "type_diversity": len(dtype_counts),
            "dominant_type": str(dtype_counts.index[0]),
            "type_distribution": {str(dtype): int(count) for dtype, count in dtype_counts.items()},
        }

        # Scale analysis for numeric columns
        if len(numeric_cols) > 0:
            ranges = {}
            scales = {}

            for col in numeric_cols:
                if self.df[col].notna().sum() > 0:
                    data = self.df[col].dropna()
                    ranges[col] = float(data.max() - data.min())
                    scales[col] = float(data.std())

            if ranges:
                range_values = list(ranges.values())
                scale_values = list(scales.values())

                consistency["scale_analysis"] = {
                    "min_range": float(min(range_values)),
                    "max_range": float(max(range_values)),
                    "range_ratio": (
                        float(max(range_values) / min(range_values)) if min(range_values) > 0 else float("inf")
                    ),
                    "min_scale": float(min(scale_values)),
                    "max_scale": float(max(scale_values)),
                    "scale_ratio": (
                        float(max(scale_values) / min(scale_values)) if min(scale_values) > 0 else float("inf")
                    ),
                    "columns_needing_scaling": [
                        col for col, range_val in ranges.items() if range_val > np.percentile(range_values, 90)
                    ],
                }

        # Correlation analysis
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr().abs()
            # Remove self-correlations
            np.fill_diagonal(corr_matrix.values, 0)

            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > self.thresholds.correlation_threshold:
                        high_corr_pairs.append(
                            {
                                "column1": corr_matrix.columns[i],
                                "column2": corr_matrix.columns[j],
                                "correlation": float(corr_val),
                            }
                        )

            consistency["correlation_analysis"] = {
                "high_correlation_pairs": high_corr_pairs,
                "max_correlation": float(corr_matrix.max().max()),
                "avg_correlation": float(corr_matrix.mean().mean()),
            }

        return consistency

    def assess_validity(self) -> Dict:
        """
        Assess data validity against expected formats and rules.

        Returns:
            Dictionary containing validity metrics
        """
        logger.info("Assessing data validity...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        validity = {"format_validity": {}, "range_validity": {}, "domain_rules": {}}

        # Format validity
        validity["format_validity"] = {
            "finite_values_pct": (
                float((np.isfinite(self.df[numeric_cols]).sum().sum() / self.df[numeric_cols].size) * 100)
                if len(numeric_cols) > 0
                else 100
            ),
            "non_null_pct": (
                float((self.df[numeric_cols].notna().sum().sum() / self.df[numeric_cols].size) * 100)
                if len(numeric_cols) > 0
                else 100
            ),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(self.df.select_dtypes(include=["object"]).columns),
        }

        # Domain-specific rules (semiconductor manufacturing)
        validity["domain_rules"] = {
            "sensor_naming_convention": self._check_sensor_naming(),
            "reasonable_value_ranges": self._check_value_ranges(),
            "physical_constraints": self._check_physical_constraints(),
        }

        return validity

    def assess_uniqueness(self) -> Dict:
        """
        Assess data uniqueness and identify duplicates.

        Returns:
            Dictionary containing uniqueness metrics
        """
        logger.info("Assessing data uniqueness...")

        uniqueness = {
            "duplicate_analysis": {},
            "column_uniqueness": {},
            "identifier_analysis": {},
        }

        # Duplicate row analysis
        duplicate_count = self.df.duplicated().sum()
        uniqueness["duplicate_analysis"] = {
            "duplicate_rows": int(duplicate_count),
            "duplicate_rate_pct": float((duplicate_count / len(self.df)) * 100),
            "unique_rows": int(len(self.df) - duplicate_count),
        }

        # Column uniqueness analysis
        column_uniqueness = {}
        low_uniqueness_cols = []

        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            total_count = self.df[col].notna().sum()
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0

            column_uniqueness[col] = {
                "unique_values": int(unique_count),
                "uniqueness_ratio": float(uniqueness_ratio),
            }

            if uniqueness_ratio < 0.1:
                low_uniqueness_cols.append(col)

        uniqueness["column_uniqueness"] = column_uniqueness
        uniqueness["low_uniqueness_columns"] = low_uniqueness_cols

        return uniqueness

    def assess_timeliness(self) -> Dict:
        """
        Assess data timeliness (placeholder for timestamp analysis).

        Returns:
            Dictionary containing timeliness metrics
        """
        logger.info("Assessing data timeliness...")

        # Look for timestamp columns
        timestamp_cols = []
        for col in self.df.columns:
            if "time" in col.lower() or "date" in col.lower() or self.df[col].dtype == "datetime64[ns]":
                timestamp_cols.append(col)

        timeliness = {
            "timestamp_columns": timestamp_cols,
            "has_timestamp_data": len(timestamp_cols) > 0,
            "data_freshness": "Not applicable - no timestamp analysis",
            "temporal_coverage": "Not applicable - no timestamp analysis",
        }

        if timestamp_cols:
            # Basic temporal analysis for first timestamp column
            ts_col = timestamp_cols[0]
            try:
                ts_data = pd.to_datetime(self.df[ts_col], errors="coerce")
                timeliness.update(
                    {
                        "earliest_timestamp": str(ts_data.min()),
                        "latest_timestamp": str(ts_data.max()),
                        "temporal_span_days": float((ts_data.max() - ts_data.min()).days),
                        "timestamp_completeness_pct": float((ts_data.notna().sum() / len(ts_data)) * 100),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not analyze timestamp column {ts_col}: {e}")

        return timeliness

    def _check_sensor_naming(self) -> bool:
        """Check if sensor columns follow expected naming convention."""
        sensor_cols = [col for col in self.df.columns if col.lower().startswith("sensor")]
        return len(sensor_cols) > len(self.df.columns) * 0.5

    def _check_value_ranges(self) -> Dict:
        """Check if values are within reasonable ranges for semiconductor data."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        range_check = {"reasonable_ranges": 0, "total_columns": len(numeric_cols)}

        for col in numeric_cols:
            if self.df[col].notna().sum() > 0:
                data = self.df[col].dropna()
                # Simple heuristic: values should not be extremely large or small
                if -1e6 <= data.min() and data.max() <= 1e6:
                    range_check["reasonable_ranges"] += 1

        range_check["reasonable_range_pct"] = (
            (range_check["reasonable_ranges"] / len(numeric_cols)) * 100 if len(numeric_cols) > 0 else 100
        )
        return range_check

    def _check_physical_constraints(self) -> Dict:
        """Check basic physical constraints."""
        constraints = {"passed_checks": 0, "total_checks": 0}

        # Example checks (would be domain-specific in practice)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if "temp" in col.lower():
                constraints["total_checks"] += 1
                # Temperature should be above absolute zero (in reasonable units)
                if self.df[col].min() > -500:  # Accommodating different units
                    constraints["passed_checks"] += 1

        return constraints

    def generate_recommendations(self, report_data: Dict) -> List[Dict]:
        """
        Generate actionable recommendations based on quality assessment.

        Args:
            report_data: Complete quality assessment report

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Completeness recommendations
        completeness = report_data["completeness"]
        if completeness["overall_completeness_pct"] < self.thresholds.completeness_critical:
            recommendations.append(
                {
                    "category": "completeness",
                    "priority": "HIGH",
                    "issue": "Low Overall Data Completeness",
                    "description": f"Only {completeness['overall_completeness_pct']:.1f}% of data is complete",
                    "action": "Investigate data collection processes and implement missing data strategies",
                    "impact": "Critical for reliable analysis and modeling",
                }
            )

        if completeness["high_missing_columns"]:
            recommendations.append(
                {
                    "category": "completeness",
                    "priority": "HIGH",
                    "issue": "Columns with High Missing Rates",
                    "description": f"{len(completeness['high_missing_columns'])} columns have >50% missing data",
                    "action": "Review sensor functionality and consider removing or imputing these columns",
                    "impact": "Reduces dataset utility and model performance",
                }
            )

        # Accuracy recommendations
        accuracy = report_data["accuracy"]
        outlier_rate = accuracy["outlier_analysis"]["outlier_rate_iqr_pct"]
        if outlier_rate > self.thresholds.outlier_rate_critical:
            recommendations.append(
                {
                    "category": "accuracy",
                    "priority": "HIGH",
                    "issue": "High Outlier Rate",
                    "description": f"{outlier_rate:.1f}% of values are potential outliers",
                    "action": "Investigate process anomalies and implement outlier handling procedures",
                    "impact": "May indicate process instability or measurement errors",
                }
            )

        # Consistency recommendations
        consistency = report_data["consistency"]
        if (
            "scale_analysis" in consistency
            and consistency["scale_analysis"]["scale_ratio"] > self.thresholds.scale_ratio_warning
        ):
            recommendations.append(
                {
                    "category": "consistency",
                    "priority": "MEDIUM",
                    "issue": "Inconsistent Feature Scales",
                    "description": f"Feature scales vary by factor of {consistency['scale_analysis']['scale_ratio']:.0f}",
                    "action": "Implement feature scaling (standardization or normalization)",
                    "impact": "Important for machine learning algorithms and analysis",
                }
            )

        # Uniqueness recommendations
        uniqueness = report_data["uniqueness"]
        if uniqueness["duplicate_analysis"]["duplicate_rate_pct"] > self.thresholds.duplicate_rate_warning:
            recommendations.append(
                {
                    "category": "uniqueness",
                    "priority": "MEDIUM",
                    "issue": "Duplicate Records Present",
                    "description": f"{uniqueness['duplicate_analysis']['duplicate_rate_pct']:.2f}% of records are duplicates",
                    "action": "Investigate data collection process and remove duplicates",
                    "impact": "May bias analysis results and model training",
                }
            )

        return recommendations

    def calculate_overall_score(self, report_data: Dict) -> float:
        """
        Calculate overall data quality score.

        Args:
            report_data: Complete quality assessment report

        Returns:
            Overall quality score (0-100)
        """
        scores = {
            "completeness": report_data["completeness"]["overall_completeness_pct"],
            "accuracy": 100 - min(report_data["accuracy"]["outlier_analysis"]["outlier_rate_iqr_pct"], 100),
            "validity": report_data["validity"]["format_validity"]["finite_values_pct"],
            "uniqueness": 100 - report_data["uniqueness"]["duplicate_analysis"]["duplicate_rate_pct"],
        }

        # Weight the scores (can be customized)
        weights = {
            "completeness": 0.3,
            "accuracy": 0.3,
            "validity": 0.2,
            "uniqueness": 0.2,
        }

        weighted_score = sum(scores[dim] * weights[dim] for dim in scores.keys())
        return min(max(weighted_score, 0), 100)  # Ensure score is between 0-100

    def assess_all_dimensions(self) -> QualityReport:
        """
        Perform comprehensive data quality assessment across all dimensions.

        Returns:
            QualityReport object with complete assessment results
        """
        logger.info("Starting comprehensive data quality assessment...")

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Dataset basic information
        dataset_info = {
            "shape": self.df.shape,
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / 1024**2),
            "column_count": len(self.df.columns),
            "row_count": len(self.df),
            "numeric_columns": len(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(self.df.select_dtypes(include=["object"]).columns),
        }

        # Assess all dimensions
        completeness = self.assess_completeness()
        accuracy = self.assess_accuracy()
        consistency = self.assess_consistency()
        validity = self.assess_validity()
        uniqueness = self.assess_uniqueness()
        timeliness = self.assess_timeliness()

        # Combine results
        report_data = {
            "dataset_info": dataset_info,
            "completeness": completeness,
            "accuracy": accuracy,
            "consistency": consistency,
            "validity": validity,
            "uniqueness": uniqueness,
            "timeliness": timeliness,
        }

        # Calculate overall score and generate recommendations
        overall_score = self.calculate_overall_score(report_data)
        recommendations = self.generate_recommendations(report_data)

        # Create final report
        self.report = QualityReport(
            dataset_info=dataset_info,
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            validity=validity,
            uniqueness=uniqueness,
            timeliness=timeliness,
            overall_score=overall_score,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat(),
        )

        logger.info(f"Quality assessment completed. Overall score: {overall_score:.1f}")
        return self.report

    def create_visualizations(self, save_path: Optional[str] = None) -> Dict:
        """
        Create comprehensive data quality visualizations.

        Args:
            save_path: Optional path to save visualizations

        Returns:
            Dictionary containing figure objects
        """
        if self.report is None:
            raise ValueError("No quality report available. Run assess_all_dimensions() first.")

        logger.info("Creating data quality visualizations...")

        figures = {}

        # 1. Quality Scores Overview
        scores = {
            "Completeness": self.report.completeness["overall_completeness_pct"],
            "Accuracy": 100 - min(self.report.accuracy["outlier_analysis"]["outlier_rate_iqr_pct"], 100),
            "Validity": self.report.validity["format_validity"]["finite_values_pct"],
            "Uniqueness": 100 - self.report.uniqueness["duplicate_analysis"]["duplicate_rate_pct"],
        }

        fig_scores = go.Figure(
            data=[
                go.Bar(
                    x=list(scores.values()),
                    y=list(scores.keys()),
                    orientation="h",
                    marker_color=["green" if v >= 80 else "orange" if v >= 60 else "red" for v in scores.values()],
                    text=[f"{v:.1f}%" for v in scores.values()],
                    textposition="auto",
                )
            ]
        )
        fig_scores.update_layout(
            title="Data Quality Scores by Dimension",
            xaxis_title="Quality Score (%)",
            xaxis_range=[0, 100],
            height=400,
        )
        figures["quality_scores"] = fig_scores

        # 2. Missing Data Analysis
        missing_data = self.df.isnull().sum().sort_values(ascending=False).head(20)
        missing_pct = (missing_data / len(self.df)) * 100

        fig_missing = go.Figure(
            data=[
                go.Bar(
                    x=missing_pct.values,
                    y=missing_pct.index,
                    orientation="h",
                    marker_color="lightcoral",
                )
            ]
        )
        fig_missing.update_layout(
            title="Missing Data by Column (Top 20)",
            xaxis_title="Missing Percentage (%)",
            height=600,
        )
        figures["missing_data"] = fig_missing

        # 3. Outlier Analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:10]
        outlier_counts = []

        for col in numeric_cols:
            if self.df[col].notna().sum() > 0:
                Q1, Q3 = self.df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = ((self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)).sum()
                outlier_counts.append(outliers)
            else:
                outlier_counts.append(0)

        fig_outliers = go.Figure(data=[go.Bar(x=list(numeric_cols), y=outlier_counts, marker_color="orange")])
        fig_outliers.update_layout(
            title="Outlier Count by Column (First 10 Numeric)",
            xaxis_title="Column",
            yaxis_title="Number of Outliers",
            height=400,
        )
        figures["outliers"] = fig_outliers

        # Save figures if path provided
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(exist_ok=True)

            for name, fig in figures.items():
                fig.write_html(save_dir / f"{name}.html")
                fig.write_image(save_dir / f"{name}.png")

        return figures

    def export_report(self, filepath: Union[str, Path], format: str = "json") -> None:
        """
        Export quality report to file.

        Args:
            filepath: Output file path
            format: Export format ('json', 'yaml', or 'html')
        """
        if self.report is None:
            raise ValueError("No quality report available. Run assess_all_dimensions() first.")

        filepath = Path(filepath)

        if format.lower() == "json":
            with open(filepath, "w") as f:
                json.dump(asdict(self.report), f, indent=2, default=str)

        elif format.lower() == "yaml":
            with open(filepath, "w") as f:
                yaml.dump(asdict(self.report), f, indent=2, default_flow_style=False)

        elif format.lower() == "html":
            self._generate_html_report(filepath)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Report exported to {filepath}")

    def _generate_html_report(self, filepath: Path) -> None:
        """Generate comprehensive HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007acc; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .high {{ color: red; }}
                .medium {{ color: orange; }}
                .low {{ color: green; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Assessment Report</h1>
                <p>Generated: {self.report.generated_at}</p>
                <p class="score">Overall Quality Score: {self.report.overall_score:.1f}/100</p>
            </div>

            <div class="section">
                <h2>Dataset Information</h2>
                <div class="metric">Shape: {self.report.dataset_info['shape']}</div>
                <div class="metric">Memory Usage: {self.report.dataset_info['memory_usage_mb']:.2f} MB</div>
                <div class="metric">Numeric Columns: {self.report.dataset_info['numeric_columns']}</div>
            </div>

            <div class="section">
                <h2>Quality Dimensions</h2>
                <div class="metric">Completeness: {self.report.completeness['overall_completeness_pct']:.1f}%</div>
                <div class="metric">Missing Values: {self.report.completeness['total_missing_values']:,}</div>
                <div class="metric">Outlier Rate: {self.report.accuracy['outlier_analysis']['outlier_rate_iqr_pct']:.2f}%</div>
                <div class="metric">Duplicate Rate: {self.report.uniqueness['duplicate_analysis']['duplicate_rate_pct']:.2f}%</div>
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <table>
                    <tr><th>Priority</th><th>Category</th><th>Issue</th><th>Action</th></tr>
        """

        for rec in self.report.recommendations:
            priority_class = rec["priority"].lower()
            html_content += f"""
                    <tr>
                        <td class="{priority_class}">{rec['priority']}</td>
                        <td>{rec['category']}</td>
                        <td>{rec['issue']}</td>
                        <td>{rec['action']}</td>
                    </tr>
            """

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        with open(filepath, "w") as f:
            f.write(html_content)


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Data Quality Framework for Semiconductor Manufacturing")

    parser.add_argument("--input", "-i", required=True, help="Input data file path")
    parser.add_argument("--output", "-o", default="quality_report", help="Output file prefix")
    parser.add_argument(
        "--format",
        "-f",
        default="json",
        choices=["json", "yaml", "html"],
        help="Output format",
    )
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--target", "-t", help="Target column name")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualizations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            if args.config.endswith(".yaml") or args.config.endswith(".yml"):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

    # Initialize framework and run assessment
    try:
        dq = DataQualityFramework(config)
        dq.load_data(args.input, args.target)
        report = dq.assess_all_dimensions()

        # Export report
        output_file = f"{args.output}.{args.format}"
        dq.export_report(output_file, args.format)

        # Generate visualizations if requested
        if args.visualize:
            viz_dir = f"{args.output}_visualizations"
            figures = dq.create_visualizations(viz_dir)
            logger.info(f"Visualizations saved to {viz_dir}")

        # Print summary
        print(f"\n✅ Data Quality Assessment Complete!")
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Report saved to: {output_file}")
        print(f"Recommendations: {len(report.recommendations)}")

        # Print high-priority recommendations
        high_priority = [r for r in report.recommendations if r["priority"] == "HIGH"]
        if high_priority:
            print(f"\n🚨 High Priority Issues ({len(high_priority)}):")
            for rec in high_priority:
                print(f"  • {rec['issue']}: {rec['action']}")

    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
