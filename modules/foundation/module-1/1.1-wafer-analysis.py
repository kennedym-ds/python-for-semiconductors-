#!/usr/bin/env python3
"""
Wafer Analysis Production Script

This script provides production-ready wafer analysis capabilities for semiconductor
manufacturing, including yield calculations, pattern detection, and statistical analysis.

Usage:
    python 1.1-wafer-analysis.py --input data.csv --output report.html
    python 1.1-wafer-analysis.py --wafer-map wafer_001.txt --format text

Author: ML for Semiconductors Course
Date: 2024
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("wafer_analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class WaferMetrics:
    """Data structure for wafer analysis metrics."""

    wafer_id: str
    total_die: int
    good_die: int
    yield_percentage: float
    center_yield: float
    edge_yield: float
    quadrant_yields: Dict[str, float]
    failure_patterns: List[str]
    timestamp: str


class WaferAnalyzer:
    """Production-grade wafer analysis tool."""

    def __init__(self, edge_width: int = 5):
        """
        Initialize wafer analyzer.

        Args:
            edge_width: Width of edge region for edge/center analysis
        """
        self.edge_width = edge_width
        self.analysis_history = []
        logger.info("WaferAnalyzer initialized")

    def load_wafer_data(
        self, file_path: Union[str, Path], format_type: str = "auto"
    ) -> np.ndarray:
        """
        Load wafer map data from various file formats.

        Args:
            file_path: Path to wafer data file
            format_type: File format ('csv', 'txt', 'npy', 'auto')

        Returns:
            2D numpy array of wafer map data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect format if needed
        if format_type == "auto":
            format_type = file_path.suffix.lower().lstrip(".")

        logger.info(f"Loading wafer data from {file_path} (format: {format_type})")

        try:
            if format_type in ["csv"]:
                data = pd.read_csv(file_path, header=None).values
            elif format_type in ["txt"]:
                data = np.loadtxt(file_path)
            elif format_type in ["npy"]:
                data = np.load(file_path)
            else:
                # Try to load as CSV by default
                data = pd.read_csv(file_path, header=None).values

            # Validate data
            if data.ndim != 2:
                raise ValueError("Wafer data must be 2-dimensional")

            if not np.all(np.isin(data, [0, 1])):
                logger.warning("Data contains values other than 0 and 1")

            logger.info(f"Loaded wafer data: {data.shape} ({np.sum(data)} good die)")
            return data.astype(int)

        except Exception as e:
            logger.error(f"Failed to load wafer data: {e}")
            raise

    def calculate_basic_metrics(
        self, wafer_data: np.ndarray, wafer_id: str = "Unknown"
    ) -> WaferMetrics:
        """
        Calculate basic wafer metrics.

        Args:
            wafer_data: 2D array of pass/fail data (1=pass, 0=fail)
            wafer_id: Identifier for the wafer

        Returns:
            WaferMetrics object with analysis results
        """
        logger.info(f"Calculating metrics for wafer {wafer_id}")

        # Basic yield calculations
        total_die = wafer_data.size
        good_die = np.sum(wafer_data)
        yield_percentage = (good_die / total_die) * 100

        # Edge vs center analysis
        center_yield = self._calculate_center_yield(wafer_data)
        edge_yield = self._calculate_edge_yield(wafer_data)

        # Quadrant analysis
        quadrant_yields = self._calculate_quadrant_yields(wafer_data)

        # Pattern detection
        failure_patterns = self._detect_failure_patterns(wafer_data)

        metrics = WaferMetrics(
            wafer_id=wafer_id,
            total_die=total_die,
            good_die=good_die,
            yield_percentage=yield_percentage,
            center_yield=center_yield,
            edge_yield=edge_yield,
            quadrant_yields=quadrant_yields,
            failure_patterns=failure_patterns,
            timestamp=datetime.now().isoformat(),
        )

        self.analysis_history.append(metrics)
        logger.info(f"Analysis complete for {wafer_id}: {yield_percentage:.2f}% yield")

        return metrics

    def _calculate_center_yield(self, wafer_data: np.ndarray) -> float:
        """Calculate yield in center region of wafer."""
        h, w = wafer_data.shape
        center_h, center_w = h // 2, w // 2

        # Define center region (inner 40% of wafer)
        size = min(h, w) // 5
        center_region = wafer_data[
            center_h - size : center_h + size, center_w - size : center_w + size
        ]

        if center_region.size == 0:
            return 0.0

        return (np.sum(center_region) / center_region.size) * 100

    def _calculate_edge_yield(self, wafer_data: np.ndarray) -> float:
        """Calculate yield in edge region of wafer."""
        h, w = wafer_data.shape

        # Create edge mask
        edge_mask = np.zeros_like(wafer_data, dtype=bool)
        edge_mask[: self.edge_width, :] = True
        edge_mask[-self.edge_width :, :] = True
        edge_mask[:, : self.edge_width] = True
        edge_mask[:, -self.edge_width :] = True

        edge_data = wafer_data[edge_mask]

        if edge_data.size == 0:
            return 0.0

        return (np.sum(edge_data) / edge_data.size) * 100

    def _calculate_quadrant_yields(self, wafer_data: np.ndarray) -> Dict[str, float]:
        """Calculate yield for each quadrant."""
        h, w = wafer_data.shape

        quadrants = {
            "Q1": wafer_data[: h // 2, : w // 2],  # Top-left
            "Q2": wafer_data[: h // 2, w // 2 :],  # Top-right
            "Q3": wafer_data[h // 2 :, : w // 2],  # Bottom-left
            "Q4": wafer_data[h // 2 :, w // 2 :],  # Bottom-right
        }

        yields = {}
        for quad_name, quad_data in quadrants.items():
            if quad_data.size > 0:
                yields[quad_name] = (np.sum(quad_data) / quad_data.size) * 100
            else:
                yields[quad_name] = 0.0

        return yields

    def _detect_failure_patterns(self, wafer_data: np.ndarray) -> List[str]:
        """Detect common failure patterns in wafer data."""
        patterns = []

        # Ring pattern detection (simplified)
        if self._detect_ring_pattern(wafer_data):
            patterns.append("ring_pattern")

        # Edge effects
        center_yield = self._calculate_center_yield(wafer_data)
        edge_yield = self._calculate_edge_yield(wafer_data)

        if center_yield - edge_yield > 10:
            patterns.append("edge_loss")
        elif edge_yield - center_yield > 10:
            patterns.append("center_loss")

        # Quadrant imbalance
        quad_yields = self._calculate_quadrant_yields(wafer_data)
        yield_values = list(quad_yields.values())
        if max(yield_values) - min(yield_values) > 15:
            patterns.append("quadrant_imbalance")

        # Overall yield assessment
        overall_yield = (np.sum(wafer_data) / wafer_data.size) * 100
        if overall_yield < 70:
            patterns.append("low_yield")
        elif overall_yield > 95:
            patterns.append("excellent_yield")

        return patterns if patterns else ["normal"]

    def _detect_ring_pattern(self, wafer_data: np.ndarray) -> bool:
        """Simple ring pattern detection."""
        h, w = wafer_data.shape
        center_h, center_w = h // 2, w // 2

        # Calculate radial yields
        radial_yields = []
        max_radius = min(h, w) // 2

        for radius in range(5, max_radius, 5):
            mask = self._create_ring_mask(
                wafer_data.shape, center_h, center_w, radius - 2, radius + 2
            )
            if np.any(mask):
                ring_yield = np.mean(wafer_data[mask])
                radial_yields.append(ring_yield)

        if len(radial_yields) < 3:
            return False

        # Look for significant dips in radial yield profile
        for i in range(1, len(radial_yields) - 1):
            if (
                radial_yields[i - 1] - radial_yields[i] > 0.2
                and radial_yields[i + 1] - radial_yields[i] > 0.2
            ):
                return True

        return False

    def _create_ring_mask(
        self,
        shape: Tuple[int, int],
        center_h: int,
        center_w: int,
        inner_radius: int,
        outer_radius: int,
    ) -> np.ndarray:
        """Create a ring mask for pattern analysis."""
        h, w = shape
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

        return (distances >= inner_radius) & (distances <= outer_radius)

    def generate_visualization(
        self,
        wafer_data: np.ndarray,
        metrics: WaferMetrics,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Generate comprehensive wafer visualization.

        Args:
            wafer_data: 2D wafer map array
            metrics: Analysis metrics
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Wafer Analysis: {metrics.wafer_id} (Yield: {metrics.yield_percentage:.2f}%)",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Main wafer map
        im1 = axes[0, 0].imshow(wafer_data, cmap="RdYlGn", aspect="equal")
        axes[0, 0].set_title("Wafer Map (Green=Pass, Red=Fail)")
        axes[0, 0].set_xlabel("Die X Position")
        axes[0, 0].set_ylabel("Die Y Position")
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. Radial yield profile
        self._plot_radial_yield(wafer_data, axes[0, 1])

        # 3. Quadrant comparison
        self._plot_quadrant_comparison(metrics.quadrant_yields, axes[0, 2])

        # 4. Edge vs center comparison
        self._plot_edge_center_comparison(metrics, axes[1, 0])

        # 5. Failure pattern summary
        self._plot_failure_patterns(metrics.failure_patterns, axes[1, 1])

        # 6. Statistics summary
        self._plot_statistics_summary(metrics, axes[1, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization saved to {save_path}")

        plt.show()

    def _plot_radial_yield(self, wafer_data: np.ndarray, ax) -> None:
        """Plot radial yield profile."""
        h, w = wafer_data.shape
        center_h, center_w = h // 2, w // 2
        max_radius = min(h, w) // 2

        radii = []
        yields = []

        for radius in range(0, max_radius, 3):
            if radius == 0:
                mask = (np.arange(h)[:, None] - center_h) ** 2 + (
                    np.arange(w) - center_w
                ) ** 2 <= 4
            else:
                mask = self._create_ring_mask(
                    wafer_data.shape, center_h, center_w, radius - 2, radius + 2
                )

            if np.any(mask):
                radii.append(radius)
                yields.append(np.mean(wafer_data[mask]) * 100)

        ax.plot(radii, yields, "bo-", linewidth=2, markersize=6)
        ax.set_title("Radial Yield Profile")
        ax.set_xlabel("Distance from Center")
        ax.set_ylabel("Yield (%)")
        ax.grid(True, alpha=0.3)

    def _plot_quadrant_comparison(self, quadrant_yields: Dict[str, float], ax) -> None:
        """Plot quadrant yield comparison."""
        quadrants = list(quadrant_yields.keys())
        yields = list(quadrant_yields.values())

        colors = ["skyblue", "lightcoral", "lightgreen", "lightyellow"]
        bars = ax.bar(quadrants, yields, color=colors)

        ax.set_title("Quadrant Yield Comparison")
        ax.set_ylabel("Yield (%)")
        ax.set_ylim(0, 100)

        # Add value labels on bars
        for bar, yield_val in zip(bars, yields):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{yield_val:.1f}%",
                ha="center",
                va="bottom",
            )

    def _plot_edge_center_comparison(self, metrics: WaferMetrics, ax) -> None:
        """Plot edge vs center yield comparison."""
        categories = ["Edge", "Center", "Overall"]
        yields = [metrics.edge_yield, metrics.center_yield, metrics.yield_percentage]

        bars = ax.bar(categories, yields, color=["coral", "lightblue", "lightgreen"])
        ax.set_title("Edge vs Center Yield")
        ax.set_ylabel("Yield (%)")
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, yield_val in zip(bars, yields):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{yield_val:.1f}%",
                ha="center",
                va="bottom",
            )

    def _plot_failure_patterns(self, patterns: List[str], ax) -> None:
        """Plot identified failure patterns."""
        pattern_names = {
            "normal": "Normal",
            "edge_loss": "Edge Loss",
            "center_loss": "Center Loss",
            "ring_pattern": "Ring Pattern",
            "quadrant_imbalance": "Quadrant Imbalance",
            "low_yield": "Low Yield",
            "excellent_yield": "Excellent Yield",
        }

        display_patterns = [pattern_names.get(p, p) for p in patterns]

        if len(display_patterns) > 6:
            display_patterns = display_patterns[:6]

        y_pos = np.arange(len(display_patterns))

        if display_patterns:
            bars = ax.barh(
                y_pos,
                [1] * len(display_patterns),
                color=[
                    (
                        "red"
                        if "loss" in p.lower() or "low" in p.lower()
                        else (
                            "green"
                            if "excellent" in p.lower() or "normal" in p.lower()
                            else "orange"
                        )
                    )
                    for p in display_patterns
                ],
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(display_patterns)
            ax.set_xlabel("Detected")
            ax.set_title("Failure Patterns")
            ax.set_xlim(0, 1.2)

            # Remove x-axis ticks
            ax.set_xticks([])
        else:
            ax.text(
                0.5,
                0.5,
                "No patterns\ndetected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Failure Patterns")

    def _plot_statistics_summary(self, metrics: WaferMetrics, ax) -> None:
        """Plot statistics summary."""
        stats_text = f"""
Wafer ID: {metrics.wafer_id}
Total Die: {metrics.total_die:,}
Good Die: {metrics.good_die:,}
Yield: {metrics.yield_percentage:.2f}%

Center Yield: {metrics.center_yield:.2f}%
Edge Yield: {metrics.edge_yield:.2f}%

Analysis Time: {metrics.timestamp.split('T')[1][:8]}
        """.strip()

        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Statistics Summary")

    def export_results(
        self, metrics: WaferMetrics, output_path: str, format_type: str = "json"
    ) -> None:
        """
        Export analysis results to file.

        Args:
            metrics: Analysis metrics to export
            output_path: Output file path
            format_type: Export format ('json', 'csv', 'txt')
        """
        output_path = Path(output_path)

        try:
            if format_type.lower() == "json":
                data = {
                    "wafer_id": metrics.wafer_id,
                    "total_die": metrics.total_die,
                    "good_die": metrics.good_die,
                    "yield_percentage": metrics.yield_percentage,
                    "center_yield": metrics.center_yield,
                    "edge_yield": metrics.edge_yield,
                    "quadrant_yields": metrics.quadrant_yields,
                    "failure_patterns": metrics.failure_patterns,
                    "timestamp": metrics.timestamp,
                }

                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2)

            elif format_type.lower() == "csv":
                # Create a flattened CSV format
                data = {
                    "wafer_id": [metrics.wafer_id],
                    "total_die": [metrics.total_die],
                    "good_die": [metrics.good_die],
                    "yield_percentage": [metrics.yield_percentage],
                    "center_yield": [metrics.center_yield],
                    "edge_yield": [metrics.edge_yield],
                    "q1_yield": [metrics.quadrant_yields["Q1"]],
                    "q2_yield": [metrics.quadrant_yields["Q2"]],
                    "q3_yield": [metrics.quadrant_yields["Q3"]],
                    "q4_yield": [metrics.quadrant_yields["Q4"]],
                    "failure_patterns": [", ".join(metrics.failure_patterns)],
                    "timestamp": [metrics.timestamp],
                }

                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)

            elif format_type.lower() == "txt":
                with open(output_path, "w") as f:
                    f.write(f"Wafer Analysis Report\n")
                    f.write(f"=====================\n\n")
                    f.write(f"Wafer ID: {metrics.wafer_id}\n")
                    f.write(f"Analysis Time: {metrics.timestamp}\n\n")
                    f.write(f"Overall Statistics:\n")
                    f.write(f"  Total Die: {metrics.total_die:,}\n")
                    f.write(f"  Good Die: {metrics.good_die:,}\n")
                    f.write(f"  Yield: {metrics.yield_percentage:.2f}%\n\n")
                    f.write(f"Regional Analysis:\n")
                    f.write(f"  Center Yield: {metrics.center_yield:.2f}%\n")
                    f.write(f"  Edge Yield: {metrics.edge_yield:.2f}%\n\n")
                    f.write(f"Quadrant Analysis:\n")
                    for quad, yield_val in metrics.quadrant_yields.items():
                        f.write(f"  {quad}: {yield_val:.2f}%\n")
                    f.write(
                        f"\nFailure Patterns: {', '.join(metrics.failure_patterns)}\n"
                    )

            logger.info(f"Results exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise


def create_sample_wafer_data(size: int = 50, yield_target: float = 0.85) -> np.ndarray:
    """Create sample wafer data for testing."""
    np.random.seed(42)

    # Base random data
    wafer_data = np.random.choice(
        [0, 1], size=(size, size), p=[1 - yield_target, yield_target]
    )

    # Add some edge effects
    edge_width = 3
    wafer_data[:edge_width, :] = np.random.choice(
        [0, 1], size=(edge_width, size), p=[0.3, 0.7]
    )
    wafer_data[-edge_width:, :] = np.random.choice(
        [0, 1], size=(edge_width, size), p=[0.3, 0.7]
    )
    wafer_data[:, :edge_width] = np.random.choice(
        [0, 1], size=(size, edge_width), p=[0.3, 0.7]
    )
    wafer_data[:, -edge_width:] = np.random.choice(
        [0, 1], size=(size, edge_width), p=[0.3, 0.7]
    )

    return wafer_data


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Wafer Analysis Tool")

    parser.add_argument("--input", "-i", type=str, help="Input wafer data file")
    parser.add_argument(
        "--wafer-id", "-w", type=str, default="Unknown", help="Wafer identifier"
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="auto",
        choices=["auto", "csv", "txt", "npy"],
        help="Input file format",
    )
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "csv", "txt"],
        help="Output format",
    )
    parser.add_argument("--plot", "-p", type=str, help="Save plot to file")
    parser.add_argument(
        "--sample", "-s", action="store_true", help="Use sample data for demonstration"
    )
    parser.add_argument(
        "--edge-width", type=int, default=5, help="Edge region width for analysis"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = WaferAnalyzer(edge_width=args.edge_width)

    try:
        # Load or create wafer data
        if args.sample:
            logger.info("Using sample wafer data")
            wafer_data = create_sample_wafer_data()
            wafer_id = "SAMPLE_WAFER_001"
        elif args.input:
            wafer_data = analyzer.load_wafer_data(args.input, args.format)
            wafer_id = args.wafer_id
        else:
            logger.error("Must specify --input file or use --sample")
            return 1

        # Perform analysis
        metrics = analyzer.calculate_basic_metrics(wafer_data, wafer_id)

        # Print results
        print(f"\n{'='*50}")
        print(f"WAFER ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"Wafer ID: {metrics.wafer_id}")
        print(f"Total Die: {metrics.total_die:,}")
        print(f"Good Die: {metrics.good_die:,}")
        print(f"Overall Yield: {metrics.yield_percentage:.2f}%")
        print(f"Center Yield: {metrics.center_yield:.2f}%")
        print(f"Edge Yield: {metrics.edge_yield:.2f}%")
        print(f"\nQuadrant Yields:")
        for quad, yield_val in metrics.quadrant_yields.items():
            print(f"  {quad}: {yield_val:.2f}%")
        print(f"\nFailure Patterns: {', '.join(metrics.failure_patterns)}")
        print(f"{'='*50}\n")

        # Generate visualization
        if args.plot or not args.output:
            analyzer.generate_visualization(wafer_data, metrics, args.plot)

        # Export results
        if args.output:
            analyzer.export_results(metrics, args.output, args.output_format)
            print(f"Results exported to: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
