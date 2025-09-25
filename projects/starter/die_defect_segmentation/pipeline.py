"""Production Die Defect Segmentation Pipeline - Starter Project

Provides a CLI to train, evaluate, and predict using lightweight segmentation models
for semiconductor die defect detection. Supports both synthetic data generation and
real die image processing with fallback models for environments without deep learning.

Features:
- Lightweight U-Net architecture optimized for die segmentation
- sklearn fallback models (Random Forest pixel classifier)
- Synthetic die data generator with configurable defect patterns
- Segmentation metrics: mIoU, pixel accuracy, defect coverage
- Visualization helpers for training and evaluation
- Manufacturing-specific metrics and integration patterns
- CPU-first implementation with deterministic behavior
- JSON output for all operations

Example usage:
    python pipeline.py train --dataset synthetic --model lightweight_unet --epochs 10 --save model.joblib
    
    python pipeline.py evaluate --model-path model.joblib --dataset synthetic --visualize
    
    python pipeline.py predict --model-path model.joblib --input die_image.npy --format json
"""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction import image
import joblib
import matplotlib.pyplot as plt
from scipy import ndimage

# Optional dependencies with graceful fallbacks
HAS_TORCH = False
HAS_CV2 = False
HAS_PIL = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    print("PyTorch not available. Using sklearn fallback models.", file=sys.stderr)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    pass

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    pass

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if HAS_TORCH:
    torch.manual_seed(RANDOM_SEED)

# Constants
DEFAULT_IMAGE_SIZE = 128
DEFECT_TYPES = ['clean', 'scratch', 'contamination', 'edge_defect', 'crack']
DEFECT_THRESHOLD = 0.05  # 5% defective area threshold


# ---------------- Synthetic Data Generator ---------------- #

def generate_synthetic_die(
    size: int = DEFAULT_IMAGE_SIZE,
    defect_type: str = 'clean',
    severity: float = 0.3,
    background_pattern: str = 'grid',
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic die image with defect patterns
    
    Args:
        size: Image dimensions (size x size)
        defect_type: One of 'clean', 'scratch', 'contamination', 'edge_defect', 'crack'
        severity: Defect severity (0.0 to 1.0)
        background_pattern: Background pattern ('grid', 'uniform', 'textured')
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (die_image, defect_mask) as numpy arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate background pattern
    if background_pattern == 'grid':
        die_image = _generate_grid_pattern(size)
    elif background_pattern == 'textured':
        die_image = _generate_textured_pattern(size)
    else:  # uniform
        die_image = np.ones((size, size)) * 0.5
    
    # Initialize defect mask
    defect_mask = np.zeros((size, size), dtype=np.uint8)
    
    # Add defect patterns
    if defect_type == 'clean':
        pass  # No defects
    elif defect_type == 'scratch':
        die_image, defect_mask = _add_scratch_defect(die_image, defect_mask, severity)
    elif defect_type == 'contamination':
        die_image, defect_mask = _add_contamination_defect(die_image, defect_mask, severity)
    elif defect_type == 'edge_defect':
        die_image, defect_mask = _add_edge_defect(die_image, defect_mask, severity)
    elif defect_type == 'crack':
        die_image, defect_mask = _add_crack_defect(die_image, defect_mask, severity)
    
    # Add noise for realism
    noise = np.random.normal(0, 0.05, (size, size))
    die_image = np.clip(die_image + noise, 0, 1)
    
    return die_image.astype(np.float32), defect_mask


def _generate_grid_pattern(size: int) -> np.ndarray:
    """Generate grid background pattern"""
    image = np.ones((size, size)) * 0.7
    grid_spacing = size // 16
    
    # Add grid lines
    for i in range(0, size, grid_spacing):
        image[i, :] = 0.5
        image[:, i] = 0.5
    
    return image


def _generate_textured_pattern(size: int) -> np.ndarray:
    """Generate textured background pattern"""
    # Create perlin-like noise using multiple frequencies
    image = np.zeros((size, size))
    for freq in [8, 16, 32]:
        noise = np.random.random((size // freq + 1, size // freq + 1))
        noise_resized = np.kron(noise, np.ones((freq, freq)))[:size, :size]
        image += noise_resized / freq
    
    return np.clip(image / 2 + 0.5, 0, 1)


def _add_scratch_defect(image: np.ndarray, mask: np.ndarray, severity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Add scratch defect pattern"""
    size = image.shape[0]
    
    # Random scratch parameters
    start_x = np.random.randint(0, max(1, size // 4))
    start_y = np.random.randint(size // 4, max(size // 4 + 1, 3 * size // 4))
    end_x = np.random.randint(max(start_x + 1, 3 * size // 4), size)
    end_y = np.random.randint(size // 4, max(size // 4 + 1, 3 * size // 4))
    
    width = max(1, int(severity * 10))
    
    # Draw scratch line
    num_points = max(1, abs(end_x - start_x))
    y_coords = np.linspace(start_y, end_y, num_points).astype(int)
    x_coords = np.linspace(start_x, end_x, num_points).astype(int)
    
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            y_adj = np.clip(y_coords + dy, 0, size - 1)
            x_adj = np.clip(x_coords + dx, 0, size - 1)
            image[y_adj, x_adj] = 0.2  # Dark scratch
            mask[y_adj, x_adj] = 1
    
    return image, mask


def _add_contamination_defect(image: np.ndarray, mask: np.ndarray, severity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Add contamination defect pattern"""
    size = image.shape[0]
    num_spots = max(1, int(severity * 10))
    
    for _ in range(num_spots):
        center_x = np.random.randint(size // 4, max(size // 4 + 1, 3 * size // 4))
        center_y = np.random.randint(size // 4, max(size // 4 + 1, 3 * size // 4))
        radius = np.random.randint(3, max(4, int(severity * 20)))
        
        # Create circular contamination
        y, x = np.ogrid[:size, :size]
        circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        
        image[circle_mask] = 0.9  # Bright contamination
        mask[circle_mask] = 1
    
    return image, mask


def _add_edge_defect(image: np.ndarray, mask: np.ndarray, severity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Add edge defect pattern"""
    size = image.shape[0]
    edge_width = max(1, int(severity * size // 10))
    
    # Random edge selection
    edge = np.random.choice(['top', 'bottom', 'left', 'right'])
    
    if edge == 'top':
        image[:edge_width, :] = 0.1
        mask[:edge_width, :] = 1
    elif edge == 'bottom':
        image[-edge_width:, :] = 0.1
        mask[-edge_width:, :] = 1
    elif edge == 'left':
        image[:, :edge_width] = 0.1
        mask[:, :edge_width] = 1
    else:  # right
        image[:, -edge_width:] = 0.1
        mask[:, -edge_width:] = 1
    
    return image, mask


def _add_crack_defect(image: np.ndarray, mask: np.ndarray, severity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Add crack defect pattern"""
    size = image.shape[0]
    
    # Create branching crack pattern
    center_x, center_y = size // 2, size // 2
    num_branches = max(1, int(severity * 5))
    
    for branch in range(num_branches):
        angle = np.random.uniform(0, 2 * np.pi)
        length = max(1, int(severity * size // 3))
        
        for step in range(length):
            x = int(center_x + step * np.cos(angle))
            y = int(center_y + step * np.sin(angle))
            
            if 0 <= x < size and 0 <= y < size:
                # Add some randomness to crack path
                x += np.random.randint(-1, 2)
                y += np.random.randint(-1, 2)
                x = np.clip(x, 0, size - 1)
                y = np.clip(y, 0, size - 1)
                
                image[y, x] = 0.0  # Black crack
                mask[y, x] = 1
    
    return image, mask


def generate_synthetic_dataset(
    n_samples: int = 1000,
    size: int = DEFAULT_IMAGE_SIZE,
    defect_distribution: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate complete synthetic dataset
    
    Args:
        n_samples: Number of samples to generate
        size: Image size
        defect_distribution: Distribution of defect types
        
    Returns:
        Tuple of (images, masks) arrays
    """
    if defect_distribution is None:
        defect_distribution = {
            'clean': 0.4,
            'scratch': 0.2,
            'contamination': 0.2,
            'edge_defect': 0.1,
            'crack': 0.1
        }
    
    images = np.zeros((n_samples, size, size), dtype=np.float32)
    masks = np.zeros((n_samples, size, size), dtype=np.uint8)
    
    defect_types = list(defect_distribution.keys())
    defect_probs = list(defect_distribution.values())
    
    for i in range(n_samples):
        defect_type = np.random.choice(defect_types, p=defect_probs)
        severity = np.random.uniform(0.1, 0.8)
        
        image, mask = generate_synthetic_die(
            size=size,
            defect_type=defect_type,
            severity=severity,
            seed=RANDOM_SEED + i
        )
        
        images[i] = image
        masks[i] = mask
    
    return images, masks


# ---------------- Segmentation Metrics ---------------- #

def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute Intersection over Union"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    
    if union == 0:
        return 1.0  # Perfect score for both empty masks
    
    return intersection / union


def compute_pixel_accuracy(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute pixel-wise accuracy"""
    return np.mean(pred_mask == true_mask)


def compute_defect_coverage(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute percentage of true defects detected"""
    if true_mask.sum() == 0:
        return 1.0 if pred_mask.sum() == 0 else 0.0
    
    return np.logical_and(pred_mask, true_mask).sum() / true_mask.sum()


def compute_false_positive_rate(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute false positive rate"""
    true_negatives = (true_mask == 0).sum()
    if true_negatives == 0:
        return 0.0
    
    false_positives = np.logical_and(pred_mask == 1, true_mask == 0).sum()
    return false_positives / true_negatives


def compute_segmentation_metrics(pred_masks: np.ndarray, true_masks: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive segmentation metrics"""
    n_samples = pred_masks.shape[0]
    
    ious = []
    pixel_accs = []
    defect_coverages = []
    false_pos_rates = []
    
    for i in range(n_samples):
        pred = (pred_masks[i] > 0.5).astype(np.uint8)
        true = true_masks[i].astype(np.uint8)
        
        ious.append(compute_iou(pred, true))
        pixel_accs.append(compute_pixel_accuracy(pred, true))
        defect_coverages.append(compute_defect_coverage(pred, true))
        false_pos_rates.append(compute_false_positive_rate(pred, true))
    
    return {
        'mIoU': np.mean(ious),
        'pixel_accuracy': np.mean(pixel_accs),
        'defect_coverage': np.mean(defect_coverages),
        'false_positive_rate': np.mean(false_pos_rates),
        'defect_detection_rate': np.mean([1.0 if iou > 0.1 else 0.0 for iou in ious])
    }


# ---------------- Lightweight U-Net Model ---------------- #

if HAS_TORCH:
    class LightweightUNet(nn.Module):
        """Simplified U-Net for die defect segmentation"""
        
        def __init__(self, in_channels: int = 1, out_channels: int = 1):
            super().__init__()
            
            # Encoder
            self.enc1 = self._conv_block(in_channels, 32)
            self.enc2 = self._conv_block(32, 64)
            self.enc3 = self._conv_block(64, 128)
            self.enc4 = self._conv_block(128, 256)
            
            # Decoder
            self.dec4 = self._upconv_block(256, 128)
            self.dec3 = self._upconv_block(256, 64)  # 128 + 128 from skip
            self.dec2 = self._upconv_block(128, 32)   # 64 + 64 from skip
            self.dec1 = self._upconv_block(64, 16)    # 32 + 32 from skip
            
            # Final layer
            self.final = nn.Conv2d(16, out_channels, kernel_size=1)
            
        def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        def _upconv_block(self, in_ch: int, out_ch: int) -> nn.Module:
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            # Encoder with skip connections
            e1 = self.enc1(x)
            e2 = self.enc2(F.max_pool2d(e1, 2))
            e3 = self.enc3(F.max_pool2d(e2, 2))
            e4 = self.enc4(F.max_pool2d(e3, 2))
            
            # Decoder with skip connections
            d4 = self.dec4(e4)
            d3 = self.dec3(torch.cat([d4, e3], dim=1))
            d2 = self.dec2(torch.cat([d3, e2], dim=1))
            d1 = self.dec1(torch.cat([d2, e1], dim=1))
            
            return torch.sigmoid(self.final(d1))


# ---------------- sklearn Fallback Models ---------------- #

def extract_pixel_features(image: np.ndarray, patch_size: int = 5) -> np.ndarray:
    """Extract features for each pixel"""
    features = []
    
    # Add pixel intensity
    features.append(image.flatten())
    
    # Add local statistics
    for stat_func in [np.mean, np.std, np.min, np.max]:
        stat_values = ndimage.generic_filter(image, stat_func, size=patch_size)
        features.append(stat_values.flatten())
    
    # Add gradient features
    grad_x = ndimage.sobel(image, axis=1)
    grad_y = ndimage.sobel(image, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    features.extend([
        grad_x.flatten(),
        grad_y.flatten(),
        gradient_magnitude.flatten()
    ])
    
    return np.column_stack(features)


# ---------------- Pipeline Implementation ---------------- #

@dataclass
class DieSegmentationPipeline:
    """Die defect segmentation pipeline with multiple model backends"""
    
    model_type: str = 'lightweight_unet'
    image_size: int = DEFAULT_IMAGE_SIZE
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 10
    device: str = 'cpu'
    
    # Fallback model parameters
    fallback_model: str = 'random_forest'
    
    def __post_init__(self):
        self.model = None
        self.scaler = None
        self.trained = False
        
        # Determine available backend
        if self.model_type == 'lightweight_unet' and not HAS_TORCH:
            warnings.warn("PyTorch not available. Switching to fallback model.", file=sys.stderr)
            self.model_type = 'fallback'
    
    def fit(self, images: np.ndarray, masks: np.ndarray) -> 'DieSegmentationPipeline':
        """Train the segmentation model"""
        
        if self.model_type == 'lightweight_unet' and HAS_TORCH:
            self._fit_unet(images, masks)
        else:
            self._fit_fallback(images, masks)
        
        self.trained = True
        return self
    
    def _fit_unet(self, images: np.ndarray, masks: np.ndarray):
        """Train PyTorch U-Net model"""
        # Prepare data
        X_train, X_val, y_train, y_val = train_test_split(
            images, masks, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dim
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val).unsqueeze(1)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create model
        self.model = LightweightUNet()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        train_loader = DataLoader(
            list(zip(X_train, y_train)), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % max(1, self.epochs // 10) == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {epoch_loss/len(train_loader):.4f}", file=sys.stderr)
    
    def _fit_fallback(self, images: np.ndarray, masks: np.ndarray):
        """Train sklearn fallback model"""
        print("Training sklearn fallback model...", file=sys.stderr)
        
        # Extract features for all pixels
        all_features = []
        all_labels = []
        
        # Use subset for faster training
        n_samples = min(100, len(images))
        sample_indices = np.random.choice(len(images), n_samples, replace=False)
        
        for idx in sample_indices:
            features = extract_pixel_features(images[idx])
            labels = masks[idx].flatten()
            
            # Subsample pixels for memory efficiency
            pixel_indices = np.random.choice(len(features), min(1000, len(features)), replace=False)
            all_features.append(features[pixel_indices])
            all_labels.append(labels[pixel_indices])
        
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if self.fallback_model == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=10, 
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        else:  # SVM
            self.model = SVC(
                kernel='rbf', 
                probability=True, 
                random_state=RANDOM_SEED
            )
        
        self.model.fit(X_scaled, y)
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Generate segmentation predictions"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.model_type == 'lightweight_unet' and HAS_TORCH:
            return self._predict_unet(images)
        else:
            return self._predict_fallback(images)
    
    def _predict_unet(self, images: np.ndarray) -> np.ndarray:
        """Predict using PyTorch U-Net"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for image in images:
                x = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
                pred = self.model(x)
                predictions.append(pred.squeeze().numpy())
        
        return np.array(predictions)
    
    def _predict_fallback(self, images: np.ndarray) -> np.ndarray:
        """Predict using sklearn fallback"""
        predictions = []
        
        for image in images:
            features = extract_pixel_features(image)
            features_scaled = self.scaler.transform(features)
            
            # Predict probabilities
            pred_probs = self.model.predict_proba(features_scaled)[:, 1]
            pred_mask = pred_probs.reshape(image.shape)
            predictions.append(pred_mask)
        
        return np.array(predictions)
    
    def evaluate(self, images: np.ndarray, masks: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(images)
        metrics = compute_segmentation_metrics(predictions, masks)
        
        # Add manufacturing-specific metrics
        defect_areas = [np.sum(mask) / mask.size for mask in masks]
        pred_areas = [np.sum(pred > 0.5) / pred.size for pred in predictions]
        
        metrics.update({
            'avg_defect_area_true': np.mean(defect_areas),
            'avg_defect_area_pred': np.mean(pred_areas),
            'area_estimation_error': np.mean(np.abs(np.array(pred_areas) - np.array(defect_areas)))
        })
        
        return metrics
    
    def save(self, path: Path) -> None:
        """Save model to disk"""
        save_data = {
            'model_type': self.model_type,
            'image_size': self.image_size,
            'trained': self.trained,
            'fallback_model': self.fallback_model
        }
        
        if self.model_type == 'lightweight_unet' and HAS_TORCH:
            save_data['model_state'] = self.model.state_dict()
        else:
            save_data['model'] = self.model
            save_data['scaler'] = self.scaler
        
        joblib.dump(save_data, path)
    
    @staticmethod
    def load(path: Path) -> 'DieSegmentationPipeline':
        """Load model from disk"""
        data = joblib.load(path)
        
        pipeline = DieSegmentationPipeline(
            model_type=data['model_type'],
            image_size=data['image_size'],
            fallback_model=data.get('fallback_model', 'random_forest')
        )
        
        if data['model_type'] == 'lightweight_unet' and HAS_TORCH:
            pipeline.model = LightweightUNet()
            pipeline.model.load_state_dict(data['model_state'])
        else:
            pipeline.model = data['model']
            pipeline.scaler = data['scaler']
        
        pipeline.trained = data['trained']
        return pipeline


# ---------------- Visualization Helpers ---------------- #

def visualize_segmentation_results(
    images: np.ndarray,
    true_masks: np.ndarray,
    pred_masks: np.ndarray,
    indices: Optional[List[int]] = None,
    save_path: Optional[Path] = None
):
    """Visualize segmentation results"""
    if indices is None:
        indices = list(range(min(4, len(images))))
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Original image
        axes[i, 0].imshow(images[idx], cmap='gray')
        axes[i, 0].set_title(f'Die {idx}: Original')
        axes[i, 0].axis('off')
        
        # True mask
        axes[i, 1].imshow(true_masks[idx], cmap='Reds', alpha=0.7)
        axes[i, 1].imshow(images[idx], cmap='gray', alpha=0.3)
        axes[i, 1].set_title(f'Die {idx}: Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        pred_binary = (pred_masks[idx] > 0.5).astype(np.uint8)
        axes[i, 2].imshow(pred_binary, cmap='Reds', alpha=0.7)
        axes[i, 2].imshow(images[idx], cmap='gray', alpha=0.3)
        
        # Calculate IoU for this sample
        iou = compute_iou(pred_binary, true_masks[idx])
        axes[i, 2].set_title(f'Die {idx}: Prediction (IoU: {iou:.3f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ---------------- CLI Implementation ---------------- #

def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Die Defect Segmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py train --dataset synthetic --epochs 10 --save model.joblib
  python pipeline.py evaluate --model-path model.joblib --dataset synthetic
  python pipeline.py predict --model-path model.joblib --input die.npy
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train segmentation model')
    train_parser.add_argument('--dataset', choices=['synthetic', 'custom'], default='synthetic',
                             help='Dataset type to use')
    train_parser.add_argument('--data-dir', type=Path,
                             help='Directory containing images/ and masks/ subdirs')
    train_parser.add_argument('--model', choices=['lightweight_unet', 'fallback'], 
                             default='lightweight_unet', help='Model architecture')
    train_parser.add_argument('--fallback-model', choices=['random_forest', 'svm'],
                             default='random_forest', help='Fallback model type')
    train_parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE,
                             help='Image size for training')
    train_parser.add_argument('--n-samples', type=int, default=1000,
                             help='Number of synthetic samples')
    train_parser.add_argument('--save', type=Path, help='Path to save trained model')
    train_parser.set_defaults(func=action_train)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=Path, required=True,
                            help='Path to trained model')
    eval_parser.add_argument('--dataset', choices=['synthetic', 'custom'], default='synthetic',
                            help='Dataset type to use')
    eval_parser.add_argument('--data-dir', type=Path,
                            help='Directory containing test data')
    eval_parser.add_argument('--n-samples', type=int, default=200,
                            help='Number of test samples')
    eval_parser.add_argument('--visualize', action='store_true',
                            help='Generate visualization plots')
    eval_parser.add_argument('--output-dir', type=Path, default=Path('.'),
                            help='Output directory for visualizations')
    eval_parser.set_defaults(func=action_evaluate)
    
    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Make predictions')
    pred_parser.add_argument('--model-path', type=Path, required=True,
                            help='Path to trained model')
    pred_parser.add_argument('--input', type=Path,
                            help='Input image file (.npy)')
    pred_parser.add_argument('--input-dir', type=Path,
                            help='Directory containing input images')
    pred_parser.add_argument('--output-dir', type=Path,
                            help='Output directory for predictions')
    pred_parser.add_argument('--format', choices=['json', 'npy'], default='json',
                            help='Output format')
    pred_parser.set_defaults(func=action_predict)
    
    return parser


def action_train(args) -> Dict[str, Any]:
    """Execute training command"""
    # Generate or load data
    if args.dataset == 'synthetic':
        print(f"Generating {args.n_samples} synthetic die images...", file=sys.stderr)
        images, masks = generate_synthetic_dataset(
            n_samples=args.n_samples,
            size=args.image_size
        )
    else:
        if not args.data_dir:
            raise ValueError("--data-dir required for custom dataset")
        # Load custom data (implementation would depend on format)
        raise NotImplementedError("Custom dataset loading not yet implemented")
    
    # Create and train pipeline
    pipeline = DieSegmentationPipeline(
        model_type=args.model,
        image_size=args.image_size,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        fallback_model=args.fallback_model
    )
    
    print(f"Training {args.model} model...", file=sys.stderr)
    start_time = time.time()
    pipeline.fit(images, masks)
    train_time = time.time() - start_time
    
    # Evaluate on training data
    train_metrics = pipeline.evaluate(images[:100], masks[:100])  # Subset for speed
    
    # Save model if requested
    if args.save:
        pipeline.save(args.save)
        print(f"Model saved to {args.save}", file=sys.stderr)
    
    result = {
        'status': 'trained',
        'model_type': args.model,
        'metrics': train_metrics,
        'metadata': {
            'model_type': args.model,
            'image_size': args.image_size,
            'n_samples': args.n_samples,
            'training_params': {
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'fallback_model': args.fallback_model
            },
            'train_time_seconds': train_time,
            'pytorch_available': HAS_TORCH,
            'random_seed': RANDOM_SEED
        }
    }
    
    return result


def action_evaluate(args) -> Dict[str, Any]:
    """Execute evaluation command"""
    # Load model
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    pipeline = DieSegmentationPipeline.load(args.model_path)
    
    # Generate or load test data
    if args.dataset == 'synthetic':
        print(f"Generating {args.n_samples} test samples...", file=sys.stderr)
        images, masks = generate_synthetic_dataset(
            n_samples=args.n_samples,
            size=pipeline.image_size
        )
    else:
        if not args.data_dir:
            raise ValueError("--data-dir required for custom dataset")
        raise NotImplementedError("Custom dataset loading not yet implemented")
    
    # Evaluate model
    print("Evaluating model...", file=sys.stderr)
    start_time = time.time()
    metrics = pipeline.evaluate(images, masks)
    eval_time = time.time() - start_time
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...", file=sys.stderr)
        predictions = pipeline.predict(images[:8])  # First 8 samples
        
        viz_path = args.output_dir / 'segmentation_results.png'
        args.output_dir.mkdir(exist_ok=True)
        
        visualize_segmentation_results(
            images[:8], masks[:8], predictions,
            save_path=viz_path
        )
        print(f"Visualizations saved to {viz_path}", file=sys.stderr)
    
    result = {
        'status': 'evaluated',
        'metrics': metrics,
        'metadata': {
            'model_type': pipeline.model_type,
            'n_test_samples': args.n_samples,
            'eval_time_seconds': eval_time,
            'model_path': str(args.model_path)
        }
    }
    
    return result


def action_predict(args) -> Dict[str, Any]:
    """Execute prediction command"""
    # Load model
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    pipeline = DieSegmentationPipeline.load(args.model_path)
    
    # Load input data
    if args.input:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        image = np.load(args.input)
        if image.ndim == 3:  # Handle RGB by converting to grayscale
            image = np.mean(image, axis=2)
        
        images = np.array([image])
        input_files = [args.input.name]
        
    elif args.input_dir:
        if not args.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        
        input_files = list(args.input_dir.glob('*.npy'))
        if not input_files:
            raise ValueError(f"No .npy files found in {args.input_dir}")
        
        images = []
        for file_path in input_files:
            image = np.load(file_path)
            if image.ndim == 3:
                image = np.mean(image, axis=2)
            images.append(image)
        images = np.array(images)
        input_files = [f.name for f in input_files]
    else:
        raise ValueError("Either --input or --input-dir must be provided")
    
    # Make predictions
    print(f"Making predictions on {len(images)} images...", file=sys.stderr)
    start_time = time.time()
    predictions = pipeline.predict(images)
    pred_time = time.time() - start_time
    
    # Process outputs
    results = []
    for i, (pred, filename) in enumerate(zip(predictions, input_files)):
        pred_binary = (pred > 0.5).astype(np.uint8)
        defect_area = np.sum(pred_binary) / pred_binary.size
        
        result_item = {
            'filename': filename,
            'defect_detected': bool(defect_area > DEFECT_THRESHOLD),
            'defect_area_percentage': float(defect_area * 100),
            'max_confidence': float(np.max(pred)),
            'mean_confidence': float(np.mean(pred))
        }
        
        # Save prediction if output directory specified
        if args.output_dir:
            args.output_dir.mkdir(exist_ok=True)
            
            if args.format == 'npy':
                pred_path = args.output_dir / f"{Path(filename).stem}_prediction.npy"
                np.save(pred_path, pred)
                result_item['prediction_path'] = str(pred_path)
        
        results.append(result_item)
    
    return {
        'status': 'predicted',
        'predictions': results,
        'metadata': {
            'model_type': pipeline.model_type,
            'n_images': len(images),
            'prediction_time_seconds': pred_time,
            'avg_time_per_image': pred_time / len(images),
            'defect_threshold': DEFECT_THRESHOLD
        }
    }


def main():
    """Main CLI entry point"""
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        result = args.func(args)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        error_result = {
            'status': 'error',
            'error': str(e),
            'command': args.command
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())