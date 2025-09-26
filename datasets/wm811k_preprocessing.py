"""WM-811K Wafer Map Dataset preprocessing and visualization utilities.

This module provides comprehensive preprocessing, visualization, and analysis
tools for the WM-811K wafer map dataset. It handles data loading, cleaning,
augmentation, and visualization of wafer defect patterns.

Features:
- Automated preprocessing of raw WM-811K data
- Wafer map visualization with defect pattern highlighting  
- Data augmentation for imbalanced classes
- Defect pattern analysis and statistics
- Integration with ML pipelines
"""

from __future__ import annotations

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WaferMapData:
    """Container for wafer map data and metadata."""
    wafer_maps: np.ndarray  # Shape: (n_samples, height, width)
    labels: np.ndarray      # Defect pattern labels
    metadata: Dict[str, Any]
    defect_types: List[str]
    
    @property
    def n_samples(self) -> int:
        return len(self.wafer_maps)
    
    @property
    def map_shape(self) -> Tuple[int, int]:
        return self.wafer_maps.shape[1:3]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of defect classes."""
        return dict(Counter(self.labels))


class WM811KPreprocessor:
    """Comprehensive preprocessing for WM-811K wafer map dataset."""
    
    # Standard defect pattern types in WM-811K dataset
    DEFECT_TYPES = [
        'None',        # No defect
        'Center',      # Central defect cluster  
        'Donut',       # Ring-shaped defect
        'Edge-Loc',    # Edge localized defect
        'Edge-Ring',   # Ring near edge
        'Loc',         # Localized defect cluster
        'Random',      # Scattered random defects
        'Scratch',     # Linear scratch defect
        'Near-full'    # Nearly full wafer defect
    ]
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.raw_data_path = self.data_root / "raw"
        self.processed_data_path = self.data_root / "data"
        
    def load_raw_data(self) -> Optional[WaferMapData]:
        """Load raw WM-811K data from downloaded files."""
        
        if not self.raw_data_path.exists():
            logger.error(f"Raw data directory not found: {self.raw_data_path}")
            return None
        
        # Look for common WM-811K file patterns
        pickle_files = list(self.raw_data_path.rglob("*.pkl"))
        csv_files = list(self.raw_data_path.rglob("*.csv"))
        
        if pickle_files:
            return self._load_from_pickle(pickle_files[0])
        elif csv_files:
            return self._load_from_csv(csv_files)
        else:
            logger.error("No recognized WM-811K data files found")
            return None
    
    def _load_from_pickle(self, pickle_path: Path) -> WaferMapData:
        """Load WM-811K data from pickle file format."""
        try:
            import pickle
            
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract wafer maps and labels based on common WM-811K structure
            if isinstance(data, dict):
                wafer_maps = data.get('waferMap', data.get('maps', None))
                labels = data.get('failureType', data.get('labels', None))
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                wafer_maps, labels = data[0], data[1]
            else:
                raise ValueError("Unrecognized pickle data structure")
            
            # Convert to numpy arrays
            wafer_maps = np.array(wafer_maps)
            if isinstance(labels[0], str):
                # String labels - keep as is
                labels = np.array(labels)
            else:
                # Numeric labels - map to defect types
                labels = np.array([self.DEFECT_TYPES[int(l)] if int(l) < len(self.DEFECT_TYPES) 
                                 else 'Unknown' for l in labels])
            
            metadata = {
                'source': 'pickle',
                'file_path': str(pickle_path),
                'original_shape': wafer_maps.shape,
                'data_type': str(wafer_maps.dtype)
            }
            
            logger.info(f"Loaded {len(wafer_maps)} wafer maps from pickle file")
            
            return WaferMapData(
                wafer_maps=wafer_maps,
                labels=labels,
                metadata=metadata,
                defect_types=self.DEFECT_TYPES
            )
            
        except Exception as e:
            logger.error(f"Failed to load pickle data: {e}")
            return None
    
    def _load_from_csv(self, csv_files: List[Path]) -> WaferMapData:
        """Load WM-811K data from CSV files."""
        # This is a placeholder for CSV loading logic
        # The actual implementation would depend on the specific CSV format
        logger.warning("CSV loading not yet implemented for WM-811K")
        return None
    
    def preprocess_data(self, data: WaferMapData, 
                       target_size: Optional[Tuple[int, int]] = None,
                       normalize: bool = True,
                       augment: bool = False) -> WaferMapData:
        """Preprocess wafer map data for ML training."""
        
        wafer_maps = data.wafer_maps.copy()
        
        # Resize if target size specified
        if target_size:
            wafer_maps = self._resize_wafer_maps(wafer_maps, target_size)
        
        # Normalize pixel values
        if normalize:
            wafer_maps = self._normalize_wafer_maps(wafer_maps)
        
        # Apply data augmentation
        if augment:
            wafer_maps, labels = self._augment_data(wafer_maps, data.labels)
        else:
            labels = data.labels.copy()
        
        # Update metadata
        metadata = data.metadata.copy()
        metadata.update({
            'preprocessed': True,
            'target_size': target_size,
            'normalized': normalize,
            'augmented': augment,
            'final_shape': wafer_maps.shape
        })
        
        return WaferMapData(
            wafer_maps=wafer_maps,
            labels=labels,
            metadata=metadata,
            defect_types=data.defect_types
        )
    
    def _resize_wafer_maps(self, wafer_maps: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize wafer maps to target size."""
        try:
            from scipy.ndimage import zoom
            
            current_shape = wafer_maps.shape[1:3]
            zoom_factors = (target_size[0] / current_shape[0], target_size[1] / current_shape[1])
            
            resized_maps = []
            for wafer_map in wafer_maps:
                resized_map = zoom(wafer_map, zoom_factors, order=1)  # Bilinear interpolation
                resized_maps.append(resized_map)
            
            return np.array(resized_maps)
            
        except ImportError:
            logger.warning("scipy not available, skipping resize")
            return wafer_maps
    
    def _normalize_wafer_maps(self, wafer_maps: np.ndarray) -> np.ndarray:
        """Normalize wafer map pixel values."""
        # Convert to float and normalize to [0, 1]
        wafer_maps = wafer_maps.astype(np.float32)
        
        # Handle different value ranges
        min_val = wafer_maps.min()
        max_val = wafer_maps.max()
        
        if max_val > min_val:
            wafer_maps = (wafer_maps - min_val) / (max_val - min_val)
        
        return wafer_maps
    
    def _augment_data(self, wafer_maps: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to balance classes."""
        
        # Count samples per class
        class_counts = Counter(labels)
        max_count = max(class_counts.values())
        
        augmented_maps = []
        augmented_labels = []
        
        for class_label, count in class_counts.items():
            class_indices = np.where(labels == class_label)[0]
            class_maps = wafer_maps[class_indices]
            
            # Add original samples
            augmented_maps.extend(class_maps)
            augmented_labels.extend([class_label] * len(class_maps))
            
            # Augment underrepresented classes
            if count < max_count:
                needed_samples = max_count - count
                
                for _ in range(needed_samples):
                    # Randomly select a sample from this class
                    idx = np.random.randint(0, len(class_maps))
                    base_map = class_maps[idx]
                    
                    # Apply random augmentation
                    augmented_map = self._apply_augmentation(base_map)
                    
                    augmented_maps.append(augmented_map)
                    augmented_labels.append(class_label)
        
        return np.array(augmented_maps), np.array(augmented_labels)
    
    def _apply_augmentation(self, wafer_map: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a single wafer map."""
        
        augmented_map = wafer_map.copy()
        
        # Random rotation (90, 180, 270 degrees)
        if np.random.random() < 0.7:
            k = np.random.randint(1, 4)
            augmented_map = np.rot90(augmented_map, k)
        
        # Random horizontal flip
        if np.random.random() < 0.5:
            augmented_map = np.fliplr(augmented_map)
        
        # Random vertical flip
        if np.random.random() < 0.5:
            augmented_map = np.flipud(augmented_map)
        
        # Add small amount of noise
        if np.random.random() < 0.3:
            noise_std = 0.01 * augmented_map.std()
            noise = np.random.normal(0, noise_std, augmented_map.shape)
            augmented_map += noise
            augmented_map = np.clip(augmented_map, 0, 1)
        
        return augmented_map
    
    def save_processed_data(self, data: WaferMapData) -> None:
        """Save processed data to files."""
        
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Save wafer maps as compressed numpy array
        maps_path = self.processed_data_path / "wafer_maps.npz"
        np.savez_compressed(
            maps_path,
            wafer_maps=data.wafer_maps,
            labels=data.labels
        )
        
        # Save metadata
        metadata_path = self.processed_data_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(data.metadata, f, indent=2, default=str)
        
        # Save class distribution
        class_dist = data.get_class_distribution()
        dist_path = self.processed_data_path / "class_distribution.json"
        with open(dist_path, 'w') as f:
            json.dump(class_dist, f, indent=2)
        
        logger.info(f"Saved processed data to {self.processed_data_path}")
    
    def load_processed_data(self) -> Optional[WaferMapData]:
        """Load previously processed data."""
        
        maps_path = self.processed_data_path / "wafer_maps.npz" 
        metadata_path = self.processed_data_path / "metadata.json"
        
        if not (maps_path.exists() and metadata_path.exists()):
            return None
        
        try:
            # Load wafer maps and labels
            data = np.load(maps_path)
            wafer_maps = data['wafer_maps']
            labels = data['labels']
            
            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            return WaferMapData(
                wafer_maps=wafer_maps,
                labels=labels,
                metadata=metadata,
                defect_types=self.DEFECT_TYPES
            )
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            return None


class WaferMapVisualizer:
    """Visualization utilities for wafer map data."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_wafer_map(self, wafer_map: np.ndarray, 
                      title: str = "Wafer Map",
                      defect_type: str = None,
                      ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot a single wafer map."""
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot wafer map
        im = ax.imshow(wafer_map, cmap='RdYlBu_r', interpolation='nearest')
        
        # Add circle to show wafer boundary
        center = np.array(wafer_map.shape) / 2
        radius = min(wafer_map.shape) / 2 - 1
        circle = plt.Circle(center, radius, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        
        # Formatting
        ax.set_title(f"{title}" + (f" ({defect_type})" if defect_type else ""))
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Die Status")
        
        return ax
    
    def plot_defect_samples(self, data: WaferMapData, 
                           samples_per_class: int = 3,
                           save_path: Optional[Path] = None) -> None:
        """Plot sample wafer maps for each defect type."""
        
        defect_types = list(data.get_class_distribution().keys())
        n_types = len(defect_types)
        
        fig, axes = plt.subplots(n_types, samples_per_class, 
                               figsize=(samples_per_class * 4, n_types * 4))
        
        if n_types == 1:
            axes = axes.reshape(1, -1)
        
        for i, defect_type in enumerate(defect_types):
            # Get samples of this defect type
            type_indices = np.where(data.labels == defect_type)[0]
            sample_indices = np.random.choice(type_indices, 
                                            min(samples_per_class, len(type_indices)), 
                                            replace=False)
            
            for j, sample_idx in enumerate(sample_indices):
                wafer_map = data.wafer_maps[sample_idx]
                ax = axes[i, j] if n_types > 1 else axes[j]
                
                self.plot_wafer_map(wafer_map, 
                                  title=f"Sample {j+1}",
                                  defect_type=defect_type,
                                  ax=ax)
            
            # Hide unused subplots
            for j in range(len(sample_indices), samples_per_class):
                ax = axes[i, j] if n_types > 1 else axes[j]
                ax.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved defect samples plot to {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, data: WaferMapData,
                              save_path: Optional[Path] = None) -> None:
        """Plot distribution of defect classes."""
        
        class_dist = data.get_class_distribution()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Bar plot
        classes = list(class_dist.keys())
        counts = list(class_dist.values())
        
        bars = ax1.bar(classes, counts, color='skyblue', alpha=0.7)
        ax1.set_title("Defect Class Distribution")
        ax1.set_xlabel("Defect Type")
        ax1.set_ylabel("Number of Samples")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.annotate(f'{count}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Defect Class Proportions")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved class distribution plot to {save_path}")
        
        plt.show()
    
    def analyze_defect_patterns(self, data: WaferMapData) -> Dict[str, Any]:
        """Analyze defect patterns and return statistics."""
        
        analysis = {
            'total_samples': data.n_samples,
            'map_shape': data.map_shape,
            'class_distribution': data.get_class_distribution(),
            'class_imbalance_ratio': None,
            'defect_statistics': {}
        }
        
        # Calculate class imbalance ratio
        counts = list(analysis['class_distribution'].values())
        if len(counts) > 1:
            analysis['class_imbalance_ratio'] = max(counts) / min(counts)
        
        # Analyze each defect type
        for defect_type in data.defect_types:
            type_indices = np.where(data.labels == defect_type)[0]
            
            if len(type_indices) > 0:
                type_maps = data.wafer_maps[type_indices]
                
                # Calculate statistics
                defect_stats = {
                    'count': len(type_indices),
                    'mean_defect_density': float(np.mean([np.sum(m == 0) / m.size for m in type_maps])),
                    'std_defect_density': float(np.std([np.sum(m == 0) / m.size for m in type_maps])),
                    'avg_good_dies': float(np.mean([np.sum(m == 1) for m in type_maps])),
                }
                
                analysis['defect_statistics'][defect_type] = defect_stats
        
        return analysis


def process_wm811k_dataset(data_root: Path, 
                          target_size: Tuple[int, int] = (64, 64),
                          augment: bool = True,
                          visualize: bool = True) -> Optional[WaferMapData]:
    """Complete WM-811K dataset processing pipeline."""
    
    logger.info("Starting WM-811K dataset processing...")
    
    # Initialize preprocessor
    preprocessor = WM811KPreprocessor(data_root)
    
    # Try to load processed data first
    processed_data = preprocessor.load_processed_data()
    
    if processed_data is not None:
        logger.info("Found existing processed data")
        return processed_data
    
    # Load raw data
    raw_data = preprocessor.load_raw_data()
    if raw_data is None:
        logger.error("Failed to load raw WM-811K data")
        return None
    
    logger.info(f"Loaded raw data: {raw_data.n_samples} samples, shape: {raw_data.map_shape}")
    
    # Preprocess data
    processed_data = preprocessor.preprocess_data(
        raw_data, 
        target_size=target_size,
        normalize=True,
        augment=augment
    )
    
    logger.info(f"Processed data: {processed_data.n_samples} samples, shape: {processed_data.map_shape}")
    
    # Save processed data
    preprocessor.save_processed_data(processed_data)
    
    # Generate visualizations
    if visualize:
        visualizer = WaferMapVisualizer()
        
        # Plot class distribution
        visualizer.plot_class_distribution(
            processed_data,
            save_path=data_root / "class_distribution.png"
        )
        
        # Plot sample defects
        visualizer.plot_defect_samples(
            processed_data,
            save_path=data_root / "defect_samples.png"
        )
        
        # Generate analysis report
        analysis = visualizer.analyze_defect_patterns(processed_data)
        
        with open(data_root / "analysis_report.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info("Generated visualization outputs")
    
    return processed_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process WM-811K wafer map dataset")
    parser.add_argument("--data-root", required=True, 
                       help="Root directory containing WM-811K data")
    parser.add_argument("--target-size", nargs=2, type=int, default=[64, 64],
                       help="Target size for wafer maps (height width)")
    parser.add_argument("--no-augment", action="store_true",
                       help="Skip data augmentation")
    parser.add_argument("--no-visualize", action="store_true", 
                       help="Skip visualization generation")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    target_size = tuple(args.target_size)
    
    processed_data = process_wm811k_dataset(
        data_root=data_root,
        target_size=target_size,
        augment=not args.no_augment,
        visualize=not args.no_visualize
    )
    
    if processed_data:
        logger.info("WM-811K dataset processing completed successfully")
    else:
        logger.error("WM-811K dataset processing failed")