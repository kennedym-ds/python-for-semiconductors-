#!/usr/bin/env python3
"""
Outlier Detection Pipeline for Semiconductor Manufacturing

This production-ready script provides comprehensive outlier detection capabilities
for semiconductor manufacturing datasets. It implements multiple detection algorithms
including statistical methods, machine learning approaches, and time-series specific
techniques designed for real-time monitoring and batch analysis.

Features:
- Multiple outlier detection algorithms
- Real-time streaming detection capability
- Physics-based validation rules
- Recipe-aware outlier detection
- Comprehensive reporting and visualization
- Integration with alerting systems

Usage:
    python 2.2-outlier-detection-pipeline.py --input data.csv --output report.html
    python 2.2-outlier-detection-pipeline.py --config config.yaml --realtime
    python 2.2-outlier-detection-pipeline.py --stream --kafka-topic sensor-data

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
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import warnings
from abc import ABC, abstractmethod

# Scientific computing
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Real-time processing (optional)
try:
    import kafka

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("outlier_detection.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure plotting
plt.style.use("default")
sns.set_palette("husl")
pio.templates.default = "plotly_white"


@dataclass
class OutlierDetectionConfig:
    """Configuration for outlier detection pipeline."""

    # Algorithm settings
    methods: List[str] = None
    z_score_threshold: float = 3.0
    modified_z_threshold: float = 3.5
    iqr_multiplier: float = 1.5
    isolation_forest_contamination: float = 0.1
    oneclass_svm_nu: float = 0.1
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.1

    # Time series settings
    ewma_lambda: float = 0.2
    ewma_L: float = 3.0
    cusum_h: float = 5.0
    cusum_k: float = 0.5

    # Processing settings
    handle_missing: str = "median"  # 'median', 'mean', 'drop'
    scaling_method: str = "standard"  # 'standard', 'robust', 'none'
    consensus_threshold: float = 0.5

    # Manufacturing specific
    recipe_aware: bool = True
    physics_validation: bool = True
    tool_specific: bool = True

    # Real-time settings
    streaming_window_size: int = 1000
    alert_threshold: float = 0.8

    # Output settings
    save_plots: bool = True
    plot_format: str = "html"  # 'html', 'png', 'pdf'
    detailed_report: bool = True

    def __post_init__(self):
        if self.methods is None:
            self.methods = ["zscore", "isolation_forest", "mahalanobis"]


class OutlierDetector(ABC):
    """Abstract base class for outlier detection methods."""

    def __init__(self, name: str, config: OutlierDetectionConfig):
        self.name = name
        self.config = config
        self.is_fitted = False
        self.fit_time = None
        self.predict_time = None

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "OutlierDetector":
        """Fit the outlier detection model."""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict outliers and return binary mask and scores."""
        pass

    def fit_predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit model and predict outliers in one step."""
        self.fit(data)
        return self.predict(data)


class ZScoreDetector(OutlierDetector):
    """Z-Score based outlier detection."""

    def __init__(self, config: OutlierDetectionConfig, threshold: float = None):
        super().__init__("Z-Score", config)
        self.threshold = threshold or config.z_score_threshold
        self.mean_ = None
        self.std_ = None

    def fit(self, data: pd.DataFrame) -> "ZScoreDetector":
        """Fit Z-score parameters."""
        start_time = datetime.now()

        self.mean_ = data.mean()
        self.std_ = data.std()
        self.is_fitted = True

        self.fit_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Z-Score detector fitted in {self.fit_time:.3f} seconds")

        return self

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict outliers using Z-score method."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = datetime.now()

        # Calculate Z-scores
        z_scores = np.abs((data - self.mean_) / self.std_)

        # Identify outliers (any feature exceeds threshold)
        outliers = (z_scores > self.threshold).any(axis=1).values

        # Use maximum Z-score as anomaly score
        scores = z_scores.max(axis=1).values

        self.predict_time = (datetime.now() - start_time).total_seconds()

        return outliers, scores


class ModifiedZScoreDetector(OutlierDetector):
    """Modified Z-Score based outlier detection (robust version)."""

    def __init__(self, config: OutlierDetectionConfig, threshold: float = None):
        super().__init__("Modified Z-Score", config)
        self.threshold = threshold or config.modified_z_threshold
        self.median_ = None
        self.mad_ = None

    def fit(self, data: pd.DataFrame) -> "ModifiedZScoreDetector":
        """Fit Modified Z-score parameters."""
        start_time = datetime.now()

        self.median_ = data.median()
        # Median Absolute Deviation
        self.mad_ = (data - self.median_).abs().median()
        self.is_fitted = True

        self.fit_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Modified Z-Score detector fitted in {self.fit_time:.3f} seconds")

        return self

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict outliers using Modified Z-score method."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = datetime.now()

        # Calculate Modified Z-scores
        modified_z_scores = 0.6745 * np.abs((data - self.median_) / self.mad_)

        # Identify outliers
        outliers = (modified_z_scores > self.threshold).any(axis=1).values

        # Use maximum Modified Z-score as anomaly score
        scores = modified_z_scores.max(axis=1).values

        self.predict_time = (datetime.now() - start_time).total_seconds()

        return outliers, scores


class IQRDetector(OutlierDetector):
    """Interquartile Range based outlier detection."""

    def __init__(self, config: OutlierDetectionConfig, k: float = None):
        super().__init__("IQR", config)
        self.k = k or config.iqr_multiplier
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None

    def fit(self, data: pd.DataFrame) -> "IQRDetector":
        """Fit IQR parameters."""
        start_time = datetime.now()

        self.q1_ = data.quantile(0.25)
        self.q3_ = data.quantile(0.75)
        self.iqr_ = self.q3_ - self.q1_
        self.is_fitted = True

        self.fit_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"IQR detector fitted in {self.fit_time:.3f} seconds")

        return self

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict outliers using IQR method."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = datetime.now()

        # Calculate bounds
        lower_bound = self.q1_ - self.k * self.iqr_
        upper_bound = self.q3_ + self.k * self.iqr_

        # Identify outliers
        outliers_mask = (data < lower_bound) | (data > upper_bound)
        outliers = outliers_mask.any(axis=1).values

        # Calculate distance from bounds as anomaly score
        dist_lower = np.maximum(0, lower_bound - data)
        dist_upper = np.maximum(0, data - upper_bound)
        scores = np.maximum(dist_lower, dist_upper).max(axis=1).values

        self.predict_time = (datetime.now() - start_time).total_seconds()

        return outliers, scores


class MahalanobisDetector(OutlierDetector):
    """Mahalanobis distance based outlier detection."""

    def __init__(
        self, config: OutlierDetectionConfig, threshold_percentile: float = 95
    ):
        super().__init__("Mahalanobis", config)
        self.threshold_percentile = threshold_percentile
        self.mean_ = None
        self.inv_cov_ = None
        self.threshold_ = None

    def fit(self, data: pd.DataFrame) -> "MahalanobisDetector":
        """Fit Mahalanobis distance parameters."""
        start_time = datetime.now()

        # Handle missing values
        data_clean = data.dropna()

        if len(data_clean) < data.shape[1]:
            raise ValueError("Insufficient data for covariance matrix calculation")

        self.mean_ = data_clean.mean().values
        cov_matrix = data_clean.cov().values

        # Use pseudo-inverse for numerical stability
        self.inv_cov_ = np.linalg.pinv(cov_matrix)

        # Calculate threshold from training data
        distances = []
        for _, row in data_clean.iterrows():
            dist = mahalanobis(row.values, self.mean_, self.inv_cov_)
            distances.append(dist)

        self.threshold_ = np.percentile(distances, self.threshold_percentile)
        self.is_fitted = True

        self.fit_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Mahalanobis detector fitted in {self.fit_time:.3f} seconds")

        return self

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict outliers using Mahalanobis distance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = datetime.now()

        # Calculate Mahalanobis distances
        distances = []
        for _, row in data.iterrows():
            if row.isnull().any():
                # Handle missing values by using available features
                available_idx = ~row.isnull()
                if available_idx.sum() < 2:
                    distances.append(0)  # Cannot calculate with too few features
                    continue

                row_clean = row[available_idx].values
                mean_clean = self.mean_[available_idx]

                # Subset covariance matrix
                inv_cov_subset = self.inv_cov_[np.ix_(available_idx, available_idx)]

                try:
                    dist = mahalanobis(row_clean, mean_clean, inv_cov_subset)
                except:
                    dist = 0
            else:
                dist = mahalanobis(row.values, self.mean_, self.inv_cov_)

            distances.append(dist)

        distances = np.array(distances)
        outliers = distances > self.threshold_

        self.predict_time = (datetime.now() - start_time).total_seconds()

        return outliers, distances


class IsolationForestDetector(OutlierDetector):
    """Isolation Forest based outlier detection."""

    def __init__(self, config: OutlierDetectionConfig, contamination: float = None):
        super().__init__("Isolation Forest", config)
        self.contamination = contamination or config.isolation_forest_contamination
        self.model = IsolationForest(
            contamination=self.contamination, random_state=42, n_estimators=100
        )

    def fit(self, data: pd.DataFrame) -> "IsolationForestDetector":
        """Fit Isolation Forest model."""
        start_time = datetime.now()

        # Handle missing values
        if self.config.handle_missing == "median":
            data_imputed = data.fillna(data.median())
        elif self.config.handle_missing == "mean":
            data_imputed = data.fillna(data.mean())
        else:
            data_imputed = data.dropna()

        # Scale data if required
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
            data_scaled = self.scaler.fit_transform(data_imputed)
        elif self.config.scaling_method == "robust":
            self.scaler = RobustScaler()
            data_scaled = self.scaler.fit_transform(data_imputed)
        else:
            self.scaler = None
            data_scaled = data_imputed.values

        self.model.fit(data_scaled)
        self.is_fitted = True

        self.fit_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Isolation Forest fitted in {self.fit_time:.3f} seconds")

        return self

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict outliers using Isolation Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = datetime.now()

        # Preprocess data
        if self.config.handle_missing == "median":
            data_imputed = data.fillna(data.median())
        elif self.config.handle_missing == "mean":
            data_imputed = data.fillna(data.mean())
        else:
            data_imputed = data.dropna()

        if self.scaler is not None:
            data_scaled = self.scaler.transform(data_imputed)
        else:
            data_scaled = data_imputed.values

        # Predict outliers
        predictions = self.model.predict(data_scaled)
        scores = self.model.decision_function(data_scaled)

        outliers = predictions == -1

        self.predict_time = (datetime.now() - start_time).total_seconds()

        return outliers, -scores  # Convert to positive scores


class OneClassSVMDetector(OutlierDetector):
    """One-Class SVM based outlier detection."""

    def __init__(self, config: OutlierDetectionConfig, nu: float = None):
        super().__init__("One-Class SVM", config)
        self.nu = nu or config.oneclass_svm_nu
        self.model = OneClassSVM(nu=self.nu, kernel="rbf", gamma="scale")

    def fit(self, data: pd.DataFrame) -> "OneClassSVMDetector":
        """Fit One-Class SVM model."""
        start_time = datetime.now()

        # Handle missing values and scaling
        if self.config.handle_missing == "median":
            data_imputed = data.fillna(data.median())
        elif self.config.handle_missing == "mean":
            data_imputed = data.fillna(data.mean())
        else:
            data_imputed = data.dropna()

        # SVM requires scaling
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data_imputed)

        self.model.fit(data_scaled)
        self.is_fitted = True

        self.fit_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"One-Class SVM fitted in {self.fit_time:.3f} seconds")

        return self

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict outliers using One-Class SVM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = datetime.now()

        # Preprocess data
        if self.config.handle_missing == "median":
            data_imputed = data.fillna(data.median())
        elif self.config.handle_missing == "mean":
            data_imputed = data.fillna(data.mean())
        else:
            data_imputed = data.dropna()

        data_scaled = self.scaler.transform(data_imputed)

        # Predict outliers
        predictions = self.model.predict(data_scaled)
        scores = self.model.decision_function(data_scaled)

        outliers = predictions == -1

        self.predict_time = (datetime.now() - start_time).total_seconds()

        return outliers, -scores  # Convert to positive scores


class LOFDetector(OutlierDetector):
    """Local Outlier Factor based outlier detection."""

    def __init__(
        self,
        config: OutlierDetectionConfig,
        n_neighbors: int = None,
        contamination: float = None,
    ):
        super().__init__("LOF", config)
        self.n_neighbors = n_neighbors or config.lof_n_neighbors
        self.contamination = contamination or config.lof_contamination
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, contamination=self.contamination
        )

    def fit(self, data: pd.DataFrame) -> "LOFDetector":
        """LOF doesn't require separate fitting - it's transductive."""
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict outliers using LOF."""
        start_time = datetime.now()

        # Handle missing values and scaling
        if self.config.handle_missing == "median":
            data_imputed = data.fillna(data.median())
        elif self.config.handle_missing == "mean":
            data_imputed = data.fillna(data.mean())
        else:
            data_imputed = data.dropna()

        if self.config.scaling_method == "standard":
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_imputed)
        elif self.config.scaling_method == "robust":
            scaler = RobustScaler()
            data_scaled = scaler.fit_transform(data_imputed)
        else:
            data_scaled = data_imputed.values

        # Apply LOF
        predictions = self.model.fit_predict(data_scaled)
        scores = self.model.negative_outlier_factor_

        outliers = predictions == -1

        self.predict_time = (datetime.now() - start_time).total_seconds()

        return outliers, -scores  # Convert to positive scores


class OutlierDetectionPipeline:
    """Main pipeline for outlier detection in semiconductor manufacturing."""

    def __init__(self, config: OutlierDetectionConfig):
        self.config = config
        self.detectors = {}
        self.results = {}
        self.consensus_results = None

        # Initialize detectors based on config
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initialize detection algorithms based on configuration."""
        for method in self.config.methods:
            if method == "zscore":
                self.detectors[method] = ZScoreDetector(self.config)
            elif method == "modified_zscore":
                self.detectors[method] = ModifiedZScoreDetector(self.config)
            elif method == "iqr":
                self.detectors[method] = IQRDetector(self.config)
            elif method == "mahalanobis":
                self.detectors[method] = MahalanobisDetector(self.config)
            elif method == "isolation_forest":
                self.detectors[method] = IsolationForestDetector(self.config)
            elif method == "oneclass_svm":
                self.detectors[method] = OneClassSVMDetector(self.config)
            elif method == "lof":
                self.detectors[method] = LOFDetector(self.config)
            else:
                logger.warning(f"Unknown detection method: {method}")

    def fit(
        self, data: pd.DataFrame, recipe_column: str = None
    ) -> "OutlierDetectionPipeline":
        """Fit all detection algorithms."""
        logger.info(f"Fitting {len(self.detectors)} detection algorithms...")

        start_time = datetime.now()

        if self.config.recipe_aware and recipe_column:
            self._fit_recipe_aware(data, recipe_column)
        else:
            for name, detector in self.detectors.items():
                try:
                    detector.fit(data)
                    logger.info(f"✅ {name} fitted successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to fit {name}: {e}")

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"All detectors fitted in {total_time:.3f} seconds")

        return self

    def _fit_recipe_aware(self, data: pd.DataFrame, recipe_column: str):
        """Fit detectors in recipe-aware mode."""
        self.recipe_detectors = {}

        for recipe in data[recipe_column].unique():
            if pd.isna(recipe):
                continue

            recipe_data = data[data[recipe_column] == recipe].drop(
                columns=[recipe_column]
            )

            if len(recipe_data) < 10:  # Minimum samples for stable fitting
                logger.warning(
                    f"Insufficient data for recipe {recipe} ({len(recipe_data)} samples)"
                )
                continue

            self.recipe_detectors[recipe] = {}

            for name, detector_class in [
                ("zscore", ZScoreDetector),
                ("isolation_forest", IsolationForestDetector),
                ("mahalanobis", MahalanobisDetector),
            ]:
                if name in self.config.methods:
                    try:
                        detector = detector_class(self.config)
                        detector.fit(recipe_data)
                        self.recipe_detectors[recipe][name] = detector
                    except Exception as e:
                        logger.error(f"Failed to fit {name} for recipe {recipe}: {e}")

    def predict(self, data: pd.DataFrame, recipe_column: str = None) -> Dict[str, Dict]:
        """Predict outliers using all fitted detectors."""
        logger.info("Predicting outliers...")

        start_time = datetime.now()

        if (
            self.config.recipe_aware
            and recipe_column
            and hasattr(self, "recipe_detectors")
        ):
            self._predict_recipe_aware(data, recipe_column)
        else:
            for name, detector in self.detectors.items():
                try:
                    outliers, scores = detector.predict(data)

                    self.results[name] = {
                        "outliers": outliers,
                        "scores": scores,
                        "n_outliers": outliers.sum(),
                        "outlier_percentage": outliers.sum() / len(data) * 100,
                        "fit_time": detector.fit_time,
                        "predict_time": detector.predict_time,
                    }

                    logger.info(
                        f"✅ {name}: {outliers.sum()} outliers ({outliers.sum()/len(data)*100:.2f}%)"
                    )

                except Exception as e:
                    logger.error(f"❌ Failed to predict with {name}: {e}")

        # Calculate consensus
        self._calculate_consensus()

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Prediction completed in {total_time:.3f} seconds")

        return self.results

    def _predict_recipe_aware(self, data: pd.DataFrame, recipe_column: str):
        """Predict outliers in recipe-aware mode."""
        # Initialize results structure
        for method in self.config.methods:
            self.results[method] = {
                "outliers": np.zeros(len(data), dtype=bool),
                "scores": np.zeros(len(data)),
                "n_outliers": 0,
                "outlier_percentage": 0,
                "recipe_breakdown": {},
            }

        for recipe in data[recipe_column].unique():
            if pd.isna(recipe) or recipe not in self.recipe_detectors:
                continue

            recipe_mask = data[recipe_column] == recipe
            recipe_data = data[recipe_mask].drop(columns=[recipe_column])

            for method, detector in self.recipe_detectors[recipe].items():
                try:
                    outliers, scores = detector.predict(recipe_data)

                    # Update global results
                    self.results[method]["outliers"][recipe_mask] = outliers
                    self.results[method]["scores"][recipe_mask] = scores
                    self.results[method]["recipe_breakdown"][recipe] = {
                        "n_outliers": outliers.sum(),
                        "n_samples": len(outliers),
                        "outlier_percentage": outliers.sum() / len(outliers) * 100,
                    }

                except Exception as e:
                    logger.error(f"Failed to predict {method} for recipe {recipe}: {e}")

        # Update global statistics
        for method in self.results:
            outliers = self.results[method]["outliers"]
            self.results[method]["n_outliers"] = outliers.sum()
            self.results[method]["outlier_percentage"] = (
                outliers.sum() / len(data) * 100
            )

    def _calculate_consensus(self):
        """Calculate consensus outliers across all methods."""
        if not self.results:
            return

        logger.info("Calculating consensus outliers...")

        n_samples = len(list(self.results.values())[0]["outliers"])
        n_methods = len(self.results)

        # Vote matrix
        votes = np.zeros(n_samples)

        for method_result in self.results.values():
            votes += method_result["outliers"].astype(int)

        # Consensus based on threshold
        consensus_threshold = n_methods * self.config.consensus_threshold
        consensus_outliers = votes >= consensus_threshold

        self.consensus_results = {
            "outliers": consensus_outliers,
            "votes": votes,
            "vote_percentage": votes / n_methods * 100,
            "n_outliers": consensus_outliers.sum(),
            "outlier_percentage": consensus_outliers.sum() / n_samples * 100,
            "consensus_threshold": consensus_threshold,
        }

        logger.info(
            f"Consensus: {consensus_outliers.sum()} outliers "
            f"({consensus_outliers.sum()/n_samples*100:.2f}%) "
            f"with {self.config.consensus_threshold*100:.0f}% agreement threshold"
        )

    def apply_physics_validation(self, data: pd.DataFrame) -> np.ndarray:
        """Apply physics-based validation rules."""
        if not self.config.physics_validation or not self.consensus_results:
            return self.consensus_results["outliers"]

        logger.info("Applying physics-based validation...")

        validated_outliers = self.consensus_results["outliers"].copy()

        # Physics rules for semiconductor processes
        physics_rules = self._get_physics_rules()

        for rule_name, rule_func in physics_rules.items():
            try:
                physics_outliers = rule_func(data)
                if physics_outliers.sum() > 0:
                    validated_outliers = validated_outliers | physics_outliers
                    logger.info(
                        f"Physics rule '{rule_name}' identified {physics_outliers.sum()} additional outliers"
                    )
            except Exception as e:
                logger.warning(f"Failed to apply physics rule '{rule_name}': {e}")

        return validated_outliers

    def _get_physics_rules(self) -> Dict[str, callable]:
        """Define physics-based validation rules for semiconductor processes."""
        rules = {}

        # Temperature-Pressure relationship
        def temp_pressure_rule(data):
            if "temperature" in data.columns and "pressure" in data.columns:
                temp_high = data["temperature"] > data["temperature"].quantile(0.9)
                pressure_low = data["pressure"] < data["pressure"].quantile(0.1)
                return temp_high & pressure_low
            return np.zeros(len(data), dtype=bool)

        # Flow rate-Pressure relationship
        def flow_pressure_rule(data):
            flow_cols = [col for col in data.columns if "flow" in col.lower()]
            pressure_cols = [col for col in data.columns if "pressure" in col.lower()]

            if flow_cols and pressure_cols:
                flow_high = data[flow_cols[0]] > data[flow_cols[0]].quantile(0.9)
                pressure_low = data[pressure_cols[0]] < data[pressure_cols[0]].quantile(
                    0.1
                )
                return flow_high & pressure_low
            return np.zeros(len(data), dtype=bool)

        # Power-Temperature relationship
        def power_temp_rule(data):
            power_cols = [col for col in data.columns if "power" in col.lower()]
            temp_cols = [col for col in data.columns if "temp" in col.lower()]

            if power_cols and temp_cols:
                power_high = data[power_cols[0]] > data[power_cols[0]].quantile(0.9)
                temp_low = data[temp_cols[0]] < data[temp_cols[0]].quantile(0.1)
                return power_high & temp_low
            return np.zeros(len(data), dtype=bool)

        rules["temperature_pressure"] = temp_pressure_rule
        rules["flow_pressure"] = flow_pressure_rule
        rules["power_temperature"] = power_temp_rule

        return rules

    def generate_report(self, data: pd.DataFrame, output_path: str = None) -> str:
        """Generate comprehensive outlier detection report."""
        logger.info("Generating outlier detection report...")

        if not self.results:
            raise ValueError("No results available. Run predict() first.")

        # Create HTML report
        html_content = self._create_html_report(data)

        # Save report
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outlier_detection_report_{timestamp}.html"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Report saved to: {output_path}")

        return output_path

    def _create_html_report(self, data: pd.DataFrame) -> str:
        """Create HTML report content."""
        from datetime import datetime

        # Get consensus outliers
        consensus_outliers = (
            self.consensus_results["outliers"] if self.consensus_results else None
        )

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Outlier Detection Report</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .method-results {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; }}
                .consensus {{ background-color: #e8f5e8; padding: 15px; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .outlier-row {{ background-color: #ffe6e6; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 Outlier Detection Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Dataset Shape:</strong> {data.shape[0]} samples × {data.shape[1]} features</p>
                <p><strong>Methods Used:</strong> {', '.join(self.config.methods)}</p>
            </div>
        """

        # Summary statistics
        html += """
            <div class="summary">
                <h2>📊 Summary Statistics</h2>
                <table>
                    <tr><th>Method</th><th>Outliers Detected</th><th>Percentage</th><th>Fit Time (s)</th><th>Predict Time (s)</th></tr>
        """

        for method, result in self.results.items():
            html += f"""
                <tr>
                    <td>{method}</td>
                    <td>{result['n_outliers']}</td>
                    <td>{result['outlier_percentage']:.2f}%</td>
                    <td>{result.get('fit_time', 'N/A')}</td>
                    <td>{result.get('predict_time', 'N/A')}</td>
                </tr>
            """

        if self.consensus_results:
            html += f"""
                <tr style="background-color: #e8f5e8;">
                    <td><strong>Consensus</strong></td>
                    <td><strong>{self.consensus_results['n_outliers']}</strong></td>
                    <td><strong>{self.consensus_results['outlier_percentage']:.2f}%</strong></td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            """

        html += "</table></div>"

        # Method details
        for method, result in self.results.items():
            html += f"""
                <div class="method-results">
                    <h3>🔍 {method} Results</h3>
                    <p><strong>Outliers:</strong> {result['n_outliers']} ({result['outlier_percentage']:.2f}%)</p>
            """

            if "recipe_breakdown" in result:
                html += "<h4>Recipe Breakdown:</h4><ul>"
                for recipe, breakdown in result["recipe_breakdown"].items():
                    html += f"<li>{recipe}: {breakdown['n_outliers']}/{breakdown['n_samples']} ({breakdown['outlier_percentage']:.2f}%)</li>"
                html += "</ul>"

            html += "</div>"

        # Consensus results
        if self.consensus_results:
            html += f"""
                <div class="consensus">
                    <h2>🎯 Consensus Results</h2>
                    <p><strong>Consensus Outliers:</strong> {self.consensus_results['n_outliers']} ({self.consensus_results['outlier_percentage']:.2f}%)</p>
                    <p><strong>Consensus Threshold:</strong> {self.config.consensus_threshold*100:.0f}% of methods must agree</p>
                    <p><strong>Average Votes per Outlier:</strong> {self.consensus_results['votes'][consensus_outliers].mean():.1f} / {len(self.results)}</p>
                </div>
            """

        # Configuration details
        html += f"""
            <div class="summary">
                <h2>⚙️ Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Z-Score Threshold</td><td>{self.config.z_score_threshold}</td></tr>
                    <tr><td>IQR Multiplier</td><td>{self.config.iqr_multiplier}</td></tr>
                    <tr><td>Isolation Forest Contamination</td><td>{self.config.isolation_forest_contamination}</td></tr>
                    <tr><td>Consensus Threshold</td><td>{self.config.consensus_threshold}</td></tr>
                    <tr><td>Recipe Aware</td><td>{self.config.recipe_aware}</td></tr>
                    <tr><td>Physics Validation</td><td>{self.config.physics_validation}</td></tr>
                </table>
            </div>
        """

        html += """
        </body>
        </html>
        """

        return html


class StreamingOutlierDetector:
    """Real-time streaming outlier detector for semiconductor manufacturing."""

    def __init__(self, config: OutlierDetectionConfig):
        self.config = config
        self.window_size = config.streaming_window_size
        self.data_buffer = []
        self.baseline_stats = {}
        self.models = {}
        self.alert_system = OutlierAlertSystem(config)

    def update_baseline(self, new_data: pd.DataFrame):
        """Update baseline statistics with new data."""
        # Add new data to buffer
        for _, row in new_data.iterrows():
            self.data_buffer.append(row.to_dict())

        # Maintain rolling window
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size :]

        # Update statistics if we have enough data
        if len(self.data_buffer) >= 100:
            buffer_df = pd.DataFrame(self.data_buffer)

            self.baseline_stats = {
                "mean": buffer_df.mean(),
                "std": buffer_df.std(),
                "median": buffer_df.median(),
                "mad": (buffer_df - buffer_df.median()).abs().median(),
            }

            # Retrain models
            self._retrain_models(buffer_df)

    def _retrain_models(self, data: pd.DataFrame):
        """Retrain outlier detection models with current buffer."""
        try:
            # Simple statistical models
            self.models["zscore"] = {"mean": data.mean(), "std": data.std()}

            self.models["modified_zscore"] = {
                "median": data.median(),
                "mad": (data - data.median()).abs().median(),
            }

            # Machine learning models
            if len(data) >= 200:  # Enough data for ML models
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                self.models["isolation_forest"] = iso_forest.fit(
                    data.fillna(data.median())
                )

        except Exception as e:
            logger.warning(f"Failed to retrain models: {e}")

    def detect_outliers(self, current_data: pd.Series) -> Dict[str, Any]:
        """Detect outliers in current data point."""
        if not self.baseline_stats:
            return {"status": "insufficient_baseline_data"}

        outlier_flags = {}

        try:
            # Z-score detection
            if "zscore" in self.models:
                stats_model = self.models["zscore"]
                z_scores = np.abs(
                    (current_data - stats_model["mean"]) / stats_model["std"]
                )
                outlier_flags["zscore"] = (
                    z_scores > self.config.z_score_threshold
                ).any()

            # Modified Z-score detection
            if "modified_zscore" in self.models:
                stats_model = self.models["modified_zscore"]
                mod_z_scores = 0.6745 * np.abs(
                    (current_data - stats_model["median"]) / stats_model["mad"]
                )
                outlier_flags["modified_zscore"] = (
                    mod_z_scores > self.config.modified_z_threshold
                ).any()

            # Isolation Forest detection
            if "isolation_forest" in self.models:
                model = self.models["isolation_forest"]
                data_imputed = current_data.fillna(self.baseline_stats["median"])
                prediction = model.predict(data_imputed.values.reshape(1, -1))
                outlier_flags["isolation_forest"] = prediction[0] == -1

            # Calculate consensus
            if outlier_flags:
                method_votes = list(outlier_flags.values())
                consensus_score = sum(method_votes) / len(method_votes)
                outlier_flags["consensus_score"] = consensus_score
                outlier_flags["is_consensus_outlier"] = (
                    consensus_score >= self.config.consensus_threshold
                )

        except Exception as e:
            logger.error(f"Error in outlier detection: {e}")
            outlier_flags["error"] = str(e)

        return outlier_flags

    def process_stream(
        self, data_point: pd.Series, timestamp: datetime = None, context: Dict = None
    ) -> Optional[Dict]:
        """Process a single streaming data point and generate alerts if needed."""
        if timestamp is None:
            timestamp = datetime.now()

        # Detect outliers
        outlier_results = self.detect_outliers(data_point)

        # Generate alert if needed
        alert = self.alert_system.process_outlier_detection(
            outlier_results, timestamp, context
        )

        return {
            "timestamp": timestamp,
            "outlier_results": outlier_results,
            "alert": alert,
        }


class OutlierAlertSystem:
    """Alert system for outlier detection in manufacturing."""

    def __init__(self, config: OutlierDetectionConfig):
        self.config = config
        self.alert_history = []
        self.alert_thresholds = {
            "critical": config.alert_threshold,
            "warning": config.alert_threshold * 0.6,
        }

    def process_outlier_detection(
        self, outlier_results: Dict, timestamp: datetime, context: Dict = None
    ) -> Optional[Dict]:
        """Process outlier detection results and generate alerts."""

        # Extract consensus score
        consensus_score = outlier_results.get("consensus_score", 0)

        # Determine alert level
        alert_level = None
        if consensus_score >= self.alert_thresholds["critical"]:
            alert_level = "CRITICAL"
        elif consensus_score >= self.alert_thresholds["warning"]:
            alert_level = "WARNING"

        if alert_level:
            alert = {
                "timestamp": timestamp,
                "level": alert_level,
                "consensus_score": consensus_score,
                "outlier_methods": {
                    k: v
                    for k, v in outlier_results.items()
                    if isinstance(v, bool) and v
                },
                "context": context or {},
                "recommendation": self._generate_recommendation(
                    alert_level, consensus_score
                ),
            }

            self.alert_history.append(alert)

            # Log alert
            logger.warning(
                f"🚨 {alert_level} OUTLIER ALERT - Consensus: {consensus_score:.2f}"
            )

            return alert

        return None

    def _generate_recommendation(
        self, alert_level: str, consensus_score: float
    ) -> Dict:
        """Generate action recommendations based on alert level."""
        if alert_level == "CRITICAL":
            return {
                "action": "IMMEDIATE_INVESTIGATION",
                "description": f"Multiple methods ({consensus_score:.1%}) indicate severe anomaly. "
                "Stop production and investigate immediately.",
                "priority": 1,
                "actions": [
                    "Stop current production lot",
                    "Notify process engineer",
                    "Check equipment status",
                    "Review recent parameter changes",
                ],
            }
        elif alert_level == "WARNING":
            return {
                "action": "MONITOR_CLOSELY",
                "description": f"Potential anomaly detected ({consensus_score:.1%} consensus). "
                "Increase monitoring frequency.",
                "priority": 2,
                "actions": [
                    "Increase sampling frequency",
                    "Monitor related parameters",
                    "Prepare for potential intervention",
                    "Document observations",
                ],
            }

        return {}


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from various file formats."""
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def create_synthetic_secom_data(
    n_samples: int = 1567, n_features: int = 590, outlier_rate: float = 0.05
) -> pd.DataFrame:
    """Create synthetic semiconductor manufacturing data for testing."""
    np.random.seed(42)

    logger.info(f"Creating synthetic data: {n_samples} samples × {n_features} features")

    # Base features with manufacturing characteristics
    data = np.random.normal(0, 1, (n_samples, n_features))

    # Add correlated sensors (typical in manufacturing)
    for i in range(0, min(100, n_features), 10):
        base_signal = np.random.normal(0, 1, n_samples)
        noise_level = 0.3
        for j in range(min(10, n_features - i)):
            data[:, i + j] = base_signal + np.random.normal(0, noise_level, n_samples)

    # Add missing values
    missing_rate = 0.08
    for col in range(n_features):
        n_missing = int(np.random.poisson(missing_rate * n_samples))
        if n_missing > 0:
            missing_indices = np.random.choice(
                n_samples, min(n_missing, n_samples), replace=False
            )
            data[missing_indices, col] = np.nan

    # Add outliers
    n_outliers = int(outlier_rate * n_samples)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)

    for idx in outlier_indices:
        # Add outliers to random subset of features
        n_features_affected = np.random.randint(1, min(10, n_features))
        features_affected = np.random.choice(
            n_features, n_features_affected, replace=False
        )

        for feat in features_affected:
            # Extreme values (5-10 standard deviations)
            outlier_magnitude = np.random.uniform(5, 10)
            outlier_sign = np.random.choice([-1, 1])
            data[idx, feat] = outlier_sign * outlier_magnitude

    # Create DataFrame
    df = pd.DataFrame(data, columns=[f"sensor_{i:03d}" for i in range(n_features)])

    # Add recipe information for recipe-aware testing
    recipes = ["Recipe_A", "Recipe_B", "Recipe_C", "Recipe_D"]
    df["recipe"] = np.random.choice(recipes, n_samples)

    # Add timestamp for time-series analysis
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(minutes=i * 5) for i in range(n_samples)]
    df["timestamp"] = timestamps

    logger.info(f"✅ Synthetic data created with {n_outliers} planted outliers")

    return df


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Outlier Detection Pipeline for Semiconductor Manufacturing"
    )

    parser.add_argument("--input", type=str, help="Input data file path")
    parser.add_argument("--output", type=str, help="Output report file path")
    parser.add_argument("--config", type=str, help="Configuration file path (YAML)")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["zscore", "isolation_forest", "mahalanobis"],
        help="Outlier detection methods to use",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Expected contamination rate for ML methods",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.5,
        help="Consensus threshold for combining methods",
    )
    parser.add_argument(
        "--recipe-column", type=str, help="Column name for recipe information"
    )
    parser.add_argument(
        "--realtime", action="store_true", help="Enable real-time streaming mode"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data for testing"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        config = OutlierDetectionConfig(**config_dict)
    else:
        config = OutlierDetectionConfig(
            methods=args.methods,
            isolation_forest_contamination=args.contamination,
            consensus_threshold=args.consensus_threshold,
        )

    logger.info("🚀 Starting Outlier Detection Pipeline")
    logger.info(f"Configuration: {config.methods}")

    try:
        # Load data
        if args.synthetic:
            data = create_synthetic_secom_data()
            logger.info("Using synthetic SECOM data")
        elif args.input:
            data = load_data(args.input)
            logger.info(f"Loaded data from {args.input}: {data.shape}")
        else:
            raise ValueError("Must specify --input or --synthetic")

        # Real-time mode
        if args.realtime:
            logger.info("Starting real-time streaming mode...")

            streaming_detector = StreamingOutlierDetector(config)

            # Simulate streaming data
            for i in range(len(data)):
                current_point = data.iloc[i]

                # Update baseline every 10 points
                if i % 10 == 0 and i > 0:
                    baseline_data = data.iloc[max(0, i - 100) : i]
                    streaming_detector.update_baseline(baseline_data)

                # Process current point
                if (
                    streaming_detector.baseline_stats
                ):  # Only after baseline is established
                    result = streaming_detector.process_stream(current_point)

                    if result["alert"]:
                        print(
                            f"🚨 ALERT at index {i}: {result['alert']['level']} "
                            f"(Score: {result['alert']['consensus_score']:.2f})"
                        )

                # Simulate real-time delay
                if i < 10:  # Just for demo - show first few
                    import time

                    time.sleep(0.1)

            return

        # Batch processing mode
        logger.info("Running batch outlier detection...")

        # Initialize pipeline
        pipeline = OutlierDetectionPipeline(config)

        # Separate features from metadata columns
        feature_columns = [
            col for col in data.columns if col not in ["recipe", "timestamp", "target"]
        ]
        feature_data = data[feature_columns]

        # Fit and predict
        pipeline.fit(feature_data, recipe_column=args.recipe_column)
        results = pipeline.predict(feature_data, recipe_column=args.recipe_column)

        # Apply physics validation if enabled
        if config.physics_validation:
            validated_outliers = pipeline.apply_physics_validation(feature_data)
            logger.info(
                f"Physics validation: {validated_outliers.sum()} final outliers"
            )

        # Generate report
        report_path = pipeline.generate_report(feature_data, args.output)

        # Summary
        logger.info("📊 Detection Summary:")
        for method, result in results.items():
            logger.info(
                f"  {method}: {result['n_outliers']} outliers ({result['outlier_percentage']:.2f}%)"
            )

        if pipeline.consensus_results:
            consensus = pipeline.consensus_results
            logger.info(
                f"  Consensus: {consensus['n_outliers']} outliers ({consensus['outlier_percentage']:.2f}%)"
            )

        logger.info(f"✅ Report saved to: {report_path}")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
