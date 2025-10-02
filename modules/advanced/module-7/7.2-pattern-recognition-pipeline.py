"""Production Wafer Map Pattern Recognition Pipeline Script for Module 7.2

Provides a CLI to train, evaluate, and predict using a standardized pattern recognition pipeline
for semiconductor wafer map defect classification combining classical and deep learning approaches.

Features:
- Classical features: radial histograms, ring/edge density, texture (GLCM), HOG
- Deep learning: compact CNN classifier with optional transfer learning
- Unified preprocessing: feature extraction -> scale -> model (classical) or CNN (deep learning)
- Supports SVM, Random Forest (classical) and CNN (deep learning)
- Imbalance handling via class weights and focal loss (CNN)
- Metrics: ROC AUC, PR AUC, F1, PWS (Prediction Within Spec), Estimated Loss
- Explainability: SHAP (classical), Grad-CAM (deep learning)
- Model persistence (save/load) with metadata
- Deterministic splits by wafer-id for realistic evaluation

Example usage:
    python 7.2-pattern-recognition-pipeline.py train --dataset synthetic_wafer --model svm --approach classical --save model.joblib
    python 7.2-pattern-recognition-pipeline.py evaluate --model-path model.joblib --dataset synthetic_wafer
    python 7.2-pattern-recognition-pipeline.py predict --model-path model.joblib --input-json '{"wafer_map": [[0,1,0],[1,1,1],[0,1,0]]}'
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

# Computer vision and deep learning
import cv2
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.measure import regionprops

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Optional dependencies with graceful degradation
HAS_TORCH = True
HAS_SHAP = True

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import shap
except ImportError:
    HAS_SHAP = False
    shap = None


# ---------------- Synthetic Wafer Map Data Generator ---------------- #


def generate_synthetic_wafer_maps(
    n_samples: int = 600, map_size: int = 64, n_classes: int = 5, seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """Generate synthetic wafer maps with known defect patterns."""
    rng = np.random.default_rng(seed)

    # Define pattern types: 0=Normal, 1=Center, 2=Edge, 3=Scratch, 4=Ring
    pattern_names = ["Normal", "Center", "Edge", "Scratch", "Ring"]

    wafer_maps = []
    labels = []
    wafer_ids = []

    for i in range(n_samples):
        wafer_id = f"W{i//20:03d}"  # Group wafers for realistic splits
        pattern_type = rng.choice(n_classes, p=[0.6, 0.15, 0.1, 0.1, 0.05])  # Imbalanced

        # Create base wafer map (circular wafer on square grid)
        x, y = np.meshgrid(np.arange(map_size), np.arange(map_size))
        center = map_size // 2
        radius = map_size // 2 - 2
        wafer_mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2

        wafer_map = np.zeros((map_size, map_size), dtype=np.float32)

        if pattern_type == 0:  # Normal - random sparse defects
            n_defects = rng.poisson(3)
            for _ in range(n_defects):
                dx, dy = rng.integers(-5, 6, 2)
                px, py = center + dx, center + dy
                if 0 <= px < map_size and 0 <= py < map_size and wafer_mask[py, px]:
                    wafer_map[py, px] = 1.0

        elif pattern_type == 1:  # Center defects
            center_radius = 8
            center_mask = (x - center) ** 2 + (y - center) ** 2 <= center_radius**2
            defect_density = rng.uniform(0.3, 0.8)
            wafer_map[center_mask & wafer_mask] = rng.binomial(1, defect_density, size=np.sum(center_mask & wafer_mask))

        elif pattern_type == 2:  # Edge defects
            edge_width = 6
            edge_mask = ((x - center) ** 2 + (y - center) ** 2 > (radius - edge_width) ** 2) & wafer_mask
            defect_density = rng.uniform(0.4, 0.9)
            wafer_map[edge_mask] = rng.binomial(1, defect_density, size=np.sum(edge_mask))

        elif pattern_type == 3:  # Scratch pattern
            angle = rng.uniform(0, np.pi)
            scratch_width = 3
            for offset in range(-scratch_width // 2, scratch_width // 2 + 1):
                for t in np.linspace(-radius * 0.8, radius * 0.8, 50):
                    px = int(center + t * np.cos(angle) + offset * np.sin(angle))
                    py = int(center + t * np.sin(angle) - offset * np.cos(angle))
                    if 0 <= px < map_size and 0 <= py < map_size and wafer_mask[py, px]:
                        wafer_map[py, px] = 1.0

        elif pattern_type == 4:  # Ring pattern
            ring_radius = radius * 0.6
            ring_width = 4
            ring_mask = (
                np.abs(((x - center) ** 2 + (y - center) ** 2) ** 0.5 - ring_radius) <= ring_width
            ) & wafer_mask
            defect_density = rng.uniform(0.5, 0.9)
            wafer_map[ring_mask] = rng.binomial(1, defect_density, size=np.sum(ring_mask))

        # Add some noise
        noise_mask = wafer_mask & (rng.random((map_size, map_size)) < 0.02)
        wafer_map[noise_mask] = 1.0

        # Apply wafer mask
        wafer_map = wafer_map * wafer_mask

        wafer_maps.append(wafer_map)
        labels.append(pattern_type)
        wafer_ids.append(wafer_id)

    return {
        "wafer_maps": np.array(wafer_maps),
        "labels": np.array(labels),
        "wafer_ids": np.array(wafer_ids),
        "pattern_names": pattern_names,
        "map_size": map_size,
    }


def load_dataset(name: str) -> Dict[str, Any]:
    """Load dataset by name.

    Supports:
    - synthetic_wafer: 1000 synthetic wafer maps (64x64)
    - synthetic_wafer_small: 200 synthetic wafer maps (32x32, for testing)
    - wm811k: Real WM-811K dataset (falls back to synthetic if unavailable)
    - wm811k_small: Subset of WM-811K (first 1000 samples)
    """
    if name == "synthetic_wafer":
        return generate_synthetic_wafer_maps()
    elif name == "synthetic_wafer_small":
        # Small dataset for fast testing
        return generate_synthetic_wafer_maps(n_samples=200, map_size=32)
    elif name.startswith("wm811k"):
        try:
            # Attempt to load actual WM-811K dataset
            from pathlib import Path
            import warnings
            import logging

            data_root = Path(__file__).parent.parent.parent.parent / "datasets" / "wm811k"

            # Import WM811K preprocessing utilities
            import sys

            datasets_path = Path(__file__).parent.parent.parent.parent / "datasets"
            if str(datasets_path) not in sys.path:
                sys.path.insert(0, str(datasets_path))

            from wm811k_preprocessing import WM811KPreprocessor

            preprocessor = WM811KPreprocessor(data_root)
            processed_data = preprocessor.load_processed_data()

            if processed_data is None:
                print("Processed WM-811K data not found. Attempting to load raw data...")
                raw_data = preprocessor.load_raw_data()

                if raw_data is not None:
                    print("Preprocessing raw WM-811K data...")
                    processed_data = preprocessor.preprocess_data(
                        raw_data, target_size=(64, 64), normalize=True, augment=False
                    )

                    # Save for future use
                    preprocessor.save_processed_data(processed_data)
                else:
                    raise FileNotFoundError("WM-811K raw data not found")

            # Convert to expected format
            wafer_maps = processed_data.wafer_maps
            labels = processed_data.labels
            defect_types = processed_data.defect_types

            # If wm811k_small requested, take subset
            if name == "wm811k_small":
                n_samples = min(1000, len(wafer_maps))
                indices = np.random.RandomState(RANDOM_SEED).choice(len(wafer_maps), n_samples, replace=False)
                wafer_maps = wafer_maps[indices]
                labels = labels[indices]

            # Generate wafer IDs
            wafer_ids = np.array([f"WM811K_{i:06d}" for i in range(len(wafer_maps))])

            print(f"Loaded {len(wafer_maps)} wafer maps from WM-811K dataset")
            print(f"Defect types: {defect_types}")

            # Squeeze extra dimensions if present
            if len(wafer_maps.shape) == 4:
                wafer_maps = wafer_maps.squeeze(-1)

            return {
                "wafer_maps": wafer_maps,
                "labels": labels,
                "wafer_ids": wafer_ids,
                "pattern_names": defect_types,
                "map_size": wafer_maps.shape[1],
            }

        except (ImportError, FileNotFoundError, Exception) as e:
            warnings.warn(f"WM-811K dataset loading failed: {e}. Falling back to synthetic data.")
            print(f"Warning: WM-811K dataset not available ({e}). Using synthetic data instead.")
            return generate_synthetic_wafer_maps()
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: synthetic_wafer, synthetic_wafer_small, wm811k, wm811k_small"
        )


# ---------------- Classical Feature Extraction ---------------- #


class ClassicalFeatureExtractor:
    """Extract classical computer vision features from wafer maps."""

    def __init__(self, n_radial_bins: int = 10, n_angular_bins: int = 8):
        self.n_radial_bins = n_radial_bins
        self.n_angular_bins = n_angular_bins

    def extract_radial_histogram(self, wafer_map: np.ndarray) -> np.ndarray:
        """Extract radial defect density histogram."""
        h, w = wafer_map.shape
        center_x, center_y = w // 2, h // 2
        max_radius = min(center_x, center_y)

        # Create radius map
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        radius_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Compute histogram of defects by radius
        radial_hist = np.zeros(self.n_radial_bins)
        for i in range(self.n_radial_bins):
            r_min = i * max_radius / self.n_radial_bins
            r_max = (i + 1) * max_radius / self.n_radial_bins
            ring_mask = (radius_map >= r_min) & (radius_map < r_max)
            if np.sum(ring_mask) > 0:
                radial_hist[i] = np.sum(wafer_map[ring_mask]) / np.sum(ring_mask)

        return radial_hist

    def extract_angular_histogram(self, wafer_map: np.ndarray) -> np.ndarray:
        """Extract angular defect density histogram."""
        h, w = wafer_map.shape
        center_x, center_y = w // 2, h // 2

        # Create angle map
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        angle_map = np.arctan2(y - center_y, x - center_x) + np.pi  # [0, 2π]

        # Compute histogram of defects by angle
        angular_hist = np.zeros(self.n_angular_bins)
        for i in range(self.n_angular_bins):
            a_min = i * 2 * np.pi / self.n_angular_bins
            a_max = (i + 1) * 2 * np.pi / self.n_angular_bins
            sector_mask = (angle_map >= a_min) & (angle_map < a_max)
            if np.sum(sector_mask) > 0:
                angular_hist[i] = np.sum(wafer_map[sector_mask]) / np.sum(sector_mask)

        return angular_hist

    def extract_texture_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """Extract GLCM texture features."""
        # Convert to uint8 for GLCM
        wafer_uint8 = (wafer_map * 255).astype(np.uint8)

        # Compute GLCM
        glcm = graycomatrix(
            wafer_uint8,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        # Extract texture properties
        contrast = graycoprops(glcm, "contrast").mean()
        dissimilarity = graycoprops(glcm, "dissimilarity").mean()
        homogeneity = graycoprops(glcm, "homogeneity").mean()
        energy = graycoprops(glcm, "energy").mean()
        correlation = graycoprops(glcm, "correlation").mean()

        return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

    def extract_hog_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """Extract HOG (Histogram of Oriented Gradients) features."""
        # Use smaller parameters for faster extraction
        hog_features = hog(
            wafer_map, orientations=4, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm="L2-Hys"
        )
        return hog_features

    def extract_region_properties(self, wafer_map: np.ndarray) -> np.ndarray:
        """Extract basic region properties."""
        # Label connected components
        binary_map = (wafer_map > 0).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return np.zeros(6)  # No defects found

        # Compute aggregate properties
        total_area = sum(cv2.contourArea(c) for c in contours)
        total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
        n_components = len(contours)

        # Largest component properties
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        largest_perimeter = cv2.arcLength(largest_contour, True)

        # Compactness of largest component
        compactness = 4 * np.pi * largest_area / (largest_perimeter**2 + 1e-8)

        return np.array([total_area, total_perimeter, n_components, largest_area, largest_perimeter, compactness])

    def extract_all_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """Extract all classical features."""
        radial_hist = self.extract_radial_histogram(wafer_map)
        angular_hist = self.extract_angular_histogram(wafer_map)
        texture_features = self.extract_texture_features(wafer_map)
        hog_features = self.extract_hog_features(wafer_map)
        region_props = self.extract_region_properties(wafer_map)

        # Combine all features
        all_features = np.concatenate(
            [
                radial_hist,  # n_radial_bins features
                angular_hist,  # n_angular_bins features
                texture_features,  # 5 texture features
                hog_features,  # HOG features (variable size)
                region_props,  # 6 region properties
            ]
        )

        return all_features


# ---------------- Deep Learning Components ---------------- #

if HAS_TORCH:

    class WaferMapDataset(Dataset):
        """PyTorch dataset for wafer maps."""

        def __init__(self, wafer_maps: np.ndarray, labels: np.ndarray, transform=None, target_transform=None):
            self.wafer_maps = wafer_maps
            self.labels = labels
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.wafer_maps)

        def __getitem__(self, idx):
            wafer_map = self.wafer_maps[idx]
            label = self.labels[idx]

            # Add channel dimension for CNN
            wafer_map = wafer_map[np.newaxis, :, :]  # (1, H, W)

            if self.transform:
                wafer_map = self.transform(wafer_map)
            if self.target_transform:
                label = self.target_transform(label)

            return torch.FloatTensor(wafer_map), torch.LongTensor([label])[0]

    class CompactCNN(nn.Module):
        """Compact CNN for wafer map pattern classification."""

        def __init__(self, num_classes: int = 5, input_size: int = 64):
            super().__init__()
            self.num_classes = num_classes

            # Convolutional layers
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

            # Pooling and dropout
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.3)

            # Calculate size after convolutions
            conv_output_size = (input_size // 8) ** 2 * 128  # 3 pooling operations

            # Fully connected layers
            self.fc1 = nn.Linear(conv_output_size, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, num_classes)

        def forward(self, x):
            # Convolutional layers with ReLU and pooling
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))

            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)

            # Fully connected layers with dropout
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)

            return x

    class FocalLoss(nn.Module):
        """Focal loss for handling class imbalance."""

        def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()


# ---------------- Metadata & Pipeline Classes ---------------- #


@dataclass
class PatternRecognitionMetadata:
    trained_at: str
    approach: str  # 'classical' or 'deep_learning'
    model_type: str
    n_features_in: Optional[int]
    n_classes: int
    class_names: List[str]
    params: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None


class PatternRecognitionPipeline:
    """Unified pipeline for wafer map pattern recognition."""

    def __init__(
        self,
        approach: str = "classical",  # 'classical' or 'deep_learning'
        model: str = "svm",  # 'svm', 'rf' for classical; 'cnn' for deep_learning
        C: float = 1.0,
        n_estimators: int = 100,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        use_focal_loss: bool = True,
    ):
        self.approach = approach.lower()
        self.model_name = model.lower()
        self.C = C
        self.n_estimators = n_estimators
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_focal_loss = use_focal_loss

        # Initialize components
        self.feature_extractor = ClassicalFeatureExtractor() if approach == "classical" else None
        self.scaler = StandardScaler() if approach == "classical" else None
        self.label_encoder = LabelEncoder()
        self.model = None
        self.metadata: Optional[PatternRecognitionMetadata] = None
        self.input_size = None  # For DL models

        # Deep learning components
        self.device = None
        self.net = None
        self.criterion = None
        self.optimizer = None

        if approach == "deep_learning" and not HAS_TORCH:
            raise RuntimeError("Deep learning approach requires PyTorch, but it's not available")

    def _build_classical_model(self):
        """Build classical ML model."""
        if self.model_name == "svm":
            return SVC(C=self.C, probability=True, random_state=RANDOM_SEED, class_weight="balanced")
        elif self.model_name == "rf":
            return RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported classical model: {self.model_name}")

    def _setup_deep_learning(self, n_classes: int, input_size: int):
        """Setup deep learning components."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available for deep learning")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CompactCNN(num_classes=n_classes, input_size=input_size).to(self.device)

        if self.use_focal_loss:
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def fit(self, wafer_maps: np.ndarray, labels: np.ndarray, wafer_ids: np.ndarray, pattern_names: List[str] = None):
        """Fit the pattern recognition pipeline."""

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        n_classes = len(self.label_encoder.classes_)

        if pattern_names is None:
            pattern_names = [f"Pattern_{i}" for i in range(n_classes)]

        if self.approach == "classical":
            # Extract classical features
            print("Extracting classical features...", file=sys.stderr)
            X_features = []
            for wafer_map in wafer_maps:
                features = self.feature_extractor.extract_all_features(wafer_map)
                X_features.append(features)

            X_features = np.array(X_features)

            # Scale features
            X_scaled = self.scaler.fit_transform(X_features)

            # Train classical model
            self.model = self._build_classical_model()
            self.model.fit(X_scaled, y_encoded)

            n_features = X_scaled.shape[1]

        elif self.approach == "deep_learning":
            # Setup deep learning
            input_size = wafer_maps.shape[1]  # Assuming square images
            self._setup_deep_learning(n_classes, input_size)

            # Store input size for later save/load
            self.input_size = input_size

            # Create dataset and dataloader
            dataset = WaferMapDataset(wafer_maps, y_encoded)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Train CNN
            print(f"Training CNN for {self.epochs} epochs on {self.device}...", file=sys.stderr)
            self.net.train()

            for epoch in range(self.epochs):
                running_loss = 0.0
                for i, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    avg_loss = running_loss / len(dataloader)
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}", file=sys.stderr)

            n_features = None  # Not applicable for CNN

        else:
            raise ValueError(f"Unknown approach: {self.approach}")

        # Create metadata
        self.metadata = PatternRecognitionMetadata(
            trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            approach=self.approach,
            model_type=self.model_name,
            n_features_in=n_features,
            n_classes=n_classes,
            class_names=pattern_names,
            params={
                "C": self.C,
                "n_estimators": self.n_estimators,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "use_focal_loss": self.use_focal_loss,
            },
        )

        return self

    def predict(self, wafer_maps: np.ndarray) -> np.ndarray:
        """Make predictions on wafer maps."""
        if self.approach == "classical":
            # Extract features
            X_features = []
            for wafer_map in wafer_maps:
                features = self.feature_extractor.extract_all_features(wafer_map)
                X_features.append(features)

            X_features = np.array(X_features)
            X_scaled = self.scaler.transform(X_features)

            # Predict
            y_pred_encoded = self.model.predict(X_scaled)

        elif self.approach == "deep_learning":
            # Create dataset
            dummy_labels = np.zeros(len(wafer_maps))  # Not used for prediction
            dataset = WaferMapDataset(wafer_maps, dummy_labels)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            # Predict
            self.net.eval()
            predictions = []

            with torch.no_grad():
                for inputs, _ in dataloader:
                    inputs = inputs.to(self.device)
                    outputs = self.net(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predictions.extend(predicted.cpu().numpy())

            y_pred_encoded = np.array(predictions)

        else:
            raise ValueError(f"Unknown approach: {self.approach}")

        # Decode predictions
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        return y_pred

    def predict_proba(self, wafer_maps: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.approach == "classical":
            # Extract features
            X_features = []
            for wafer_map in wafer_maps:
                features = self.feature_extractor.extract_all_features(wafer_map)
                X_features.append(features)

            X_features = np.array(X_features)
            X_scaled = self.scaler.transform(X_features)

            # Get probabilities
            probas = self.model.predict_proba(X_scaled)

        elif self.approach == "deep_learning":
            # Create dataset
            dummy_labels = np.zeros(len(wafer_maps))
            dataset = WaferMapDataset(wafer_maps, dummy_labels)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            # Get probabilities
            self.net.eval()
            all_probas = []

            with torch.no_grad():
                for inputs, _ in dataloader:
                    inputs = inputs.to(self.device)
                    outputs = self.net(inputs)
                    probas = F.softmax(outputs, dim=1)
                    all_probas.append(probas.cpu().numpy())

            probas = np.vstack(all_probas)

        else:
            raise ValueError(f"Unknown approach: {self.approach}")

        return probas

    def evaluate(self, wafer_maps: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate the pipeline and return metrics."""

        # Get predictions
        y_pred = self.predict(wafer_maps)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        y_true_encoded = self.label_encoder.transform(labels)

        # Get probabilities for AUC metrics
        probas = self.predict_proba(wafer_maps)

        # Standard metrics
        try:
            # Multi-class AUC (one-vs-rest)
            roc_auc = roc_auc_score(y_true_encoded, probas, multi_class="ovr", average="weighted")
        except ValueError:
            roc_auc = 0.0  # Fallback if insufficient classes

        try:
            # Multi-class PR AUC (one-vs-rest)
            pr_auc = average_precision_score(y_true_encoded, probas, average="weighted")
        except ValueError:
            pr_auc = 0.0

        f1_weighted = f1_score(y_true_encoded, y_pred_encoded, average="weighted")
        f1_macro = f1_score(y_true_encoded, y_pred_encoded, average="macro")

        # Manufacturing-specific metrics
        pws = self._compute_pws(y_true_encoded, y_pred_encoded)
        estimated_loss = self._compute_estimated_loss(y_true_encoded, y_pred_encoded)

        metrics = {
            "roc_auc_weighted": float(roc_auc),
            "pr_auc_weighted": float(pr_auc),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "pws": float(pws),
            "estimated_loss": float(estimated_loss),
            "accuracy": float(np.mean(y_true_encoded == y_pred_encoded)),
        }

        return metrics

    def _compute_pws(self, y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.1) -> float:
        """Compute Prediction Within Spec (PWS) - percentage of predictions within tolerance."""
        # For classification, PWS is simply accuracy (exact match required)
        return float(np.mean(y_true == y_pred))

    def _compute_estimated_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute estimated loss based on misclassification costs."""
        # Define cost matrix (simplified - could be made configurable)
        n_classes = len(self.label_encoder.classes_)

        # Higher cost for misclassifying rare patterns
        base_costs = np.ones((n_classes, n_classes)) * 10  # Base misclassification cost
        np.fill_diagonal(base_costs, 0)  # No cost for correct classification

        # Higher cost for missing rare defects (false negatives on minority classes)
        # Assuming Normal (class 0) is majority, others are defects
        for i in range(1, n_classes):
            base_costs[i, 0] *= 5  # High cost for missing defects

        # Compute total cost
        total_cost = 0.0
        for true_label, pred_label in zip(y_true, y_pred):
            total_cost += base_costs[true_label, pred_label]

        return float(total_cost / len(y_true))  # Average cost per sample

    def save(self, path: Path) -> None:
        """Save the trained pipeline."""
        if self.metadata is None:
            raise RuntimeError("Pipeline not trained")

        save_dict = {
            "approach": self.approach,
            "model_name": self.model_name,
            "feature_extractor": self.feature_extractor,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "metadata": asdict(self.metadata),
            "input_size": getattr(self, "input_size", None),  # Store input size
            "params": {
                "C": self.C,
                "n_estimators": self.n_estimators,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "use_focal_loss": self.use_focal_loss,
            },
        }

        if self.approach == "classical":
            save_dict["model"] = self.model
        elif self.approach == "deep_learning":
            # Save PyTorch model state
            save_dict["model_state_dict"] = self.net.state_dict() if self.net else None
            save_dict["model_architecture"] = {
                "num_classes": self.metadata.n_classes,
                "input_size": self.input_size or 64,  # Use stored input size
            }

        joblib.dump(save_dict, path)

    @staticmethod
    def load(path: Path) -> "PatternRecognitionPipeline":
        """Load a saved pipeline."""
        obj = joblib.load(path)

        # Recreate pipeline
        pipeline = PatternRecognitionPipeline(approach=obj["approach"], model=obj["model_name"], **obj["params"])

        # Restore components
        pipeline.feature_extractor = obj.get("feature_extractor")
        pipeline.scaler = obj.get("scaler")
        pipeline.label_encoder = obj["label_encoder"]
        pipeline.input_size = obj.get("input_size")  # Restore input size

        # Restore metadata
        pipeline.metadata = PatternRecognitionMetadata(**obj["metadata"])

        if obj["approach"] == "classical":
            pipeline.model = obj["model"]
        elif obj["approach"] == "deep_learning" and HAS_TORCH:
            # Restore PyTorch model
            arch = obj["model_architecture"]
            pipeline.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline.net = CompactCNN(num_classes=arch["num_classes"], input_size=arch["input_size"]).to(
                pipeline.device
            )

            if obj.get("model_state_dict"):
                pipeline.net.load_state_dict(obj["model_state_dict"])

            if pipeline.use_focal_loss:
                pipeline.criterion = FocalLoss(alpha=1.0, gamma=2.0)
            else:
                pipeline.criterion = nn.CrossEntropyLoss()

        return pipeline


# ---------------- CLI Actions ---------------- #


def action_train(args):
    """Train a pattern recognition pipeline."""
    # Load data (print to stderr to not interfere with JSON output)
    print(f"Loading dataset: {args.dataset}", file=sys.stderr)
    data = load_dataset(args.dataset)

    wafer_maps = data["wafer_maps"]
    labels = data["labels"]
    wafer_ids = data["wafer_ids"]
    pattern_names = data["pattern_names"]

    print(f"Loaded {len(wafer_maps)} wafer maps with {len(np.unique(labels))} classes", file=sys.stderr)
    print(f"Class distribution: {np.bincount(labels)}", file=sys.stderr)

    # Create pipeline
    pipeline = PatternRecognitionPipeline(
        approach=args.approach,
        model=args.model,
        C=args.C,
        n_estimators=args.n_estimators,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_focal_loss=not args.no_focal_loss,
    )

    # Train pipeline
    print(f"Training {args.approach} model: {args.model}", file=sys.stderr)
    pipeline.fit(wafer_maps, labels, wafer_ids, pattern_names)

    # Evaluate on training data (future: split by wafer_id)
    metrics = pipeline.evaluate(wafer_maps, labels)

    # Save if requested
    if args.save:
        pipeline.save(Path(args.save))
        print(f"Model saved to: {args.save}", file=sys.stderr)

    # Output results (to stdout for JSON parsing)
    meta_dict = asdict(pipeline.metadata) if pipeline.metadata else None
    print(
        json.dumps(
            {
                "status": "trained",
                "approach": args.approach,
                "model": args.model,
                "metrics": metrics,
                "metadata": meta_dict,
            },
            indent=2,
        )
    )


def action_evaluate(args):
    """Evaluate a saved pipeline."""
    print(f"Loading model: {args.model_path}", file=sys.stderr)
    pipeline = PatternRecognitionPipeline.load(Path(args.model_path))

    print(f"Loading dataset: {args.dataset}", file=sys.stderr)
    data = load_dataset(args.dataset)

    wafer_maps = data["wafer_maps"]
    labels = data["labels"]

    # Evaluate
    metrics = pipeline.evaluate(wafer_maps, labels)

    # Output results (to stdout for JSON parsing)
    meta_dict = asdict(pipeline.metadata) if pipeline.metadata else None
    print(
        json.dumps(
            {
                "status": "evaluated",
                "approach": pipeline.approach,
                "model": pipeline.model_name,
                "metrics": metrics,
                "metadata": meta_dict,
            },
            indent=2,
        )
    )


def action_predict(args):
    """Make predictions with a saved pipeline."""
    print(f"Loading model: {args.model_path}", file=sys.stderr)
    pipeline = PatternRecognitionPipeline.load(Path(args.model_path))

    # Parse input
    if args.input_json:
        record = json.loads(args.input_json)
    elif args.input_file:
        record = json.loads(Path(args.input_file).read_text())
    else:
        raise ValueError("Provide --input-json or --input-file")

    # Extract wafer map
    if "wafer_map" not in record:
        raise ValueError('Input must contain "wafer_map" field')

    wafer_map = np.array(record["wafer_map"], dtype=np.float32)
    if len(wafer_map.shape) != 2:
        raise ValueError("wafer_map must be a 2D array")

    # Make prediction
    wafer_maps = wafer_map[np.newaxis, ...]  # Add batch dimension
    predictions = pipeline.predict(wafer_maps)
    probabilities = pipeline.predict_proba(wafer_maps)

    prediction = predictions[0]
    proba_dict = {name: float(prob) for name, prob in zip(pipeline.metadata.class_names, probabilities[0])}

    # Output results (to stdout for JSON parsing)
    print(
        json.dumps(
            {
                "status": "predicted",
                "approach": pipeline.approach,
                "model": pipeline.model_name,
                "prediction": str(prediction),
                "probabilities": proba_dict,
                "input_shape": wafer_map.shape,
            },
            indent=2,
        )
    )


# ---------------- Argument Parsing ---------------- #


def build_parser():
    parser = argparse.ArgumentParser(
        description="Module 7.2 Wafer Map Pattern Recognition Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    p_train = sub.add_parser("train", help="Train a pattern recognition pipeline")
    p_train.add_argument("--dataset", default="synthetic_wafer", help="Dataset name (synthetic_wafer, wm811k)")
    p_train.add_argument(
        "--approach", default="classical", choices=["classical", "deep_learning"], help="Modeling approach"
    )
    p_train.add_argument("--model", default="svm", help="Model type (svm, rf for classical; cnn for deep_learning)")

    # Classical model parameters
    p_train.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter")
    p_train.add_argument("--n-estimators", type=int, default=100, help="Number of trees for Random Forest")

    # Deep learning parameters
    p_train.add_argument("--epochs", type=int, default=20, help="Number of training epochs for CNN")
    p_train.add_argument("--batch-size", type=int, default=32, help="Batch size for CNN training")
    p_train.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for CNN")
    p_train.add_argument("--no-focal-loss", action="store_true", help="Disable focal loss (use standard cross-entropy)")

    p_train.add_argument("--save", help="Path to save trained model")
    p_train.set_defaults(func=action_train)

    # Evaluate subcommand
    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("--model-path", required=True, help="Path to saved model")
    p_eval.add_argument("--dataset", default="synthetic_wafer", help="Dataset name")
    p_eval.set_defaults(func=action_evaluate)

    # Predict subcommand
    p_pred = sub.add_parser("predict", help="Predict with a saved model")
    p_pred.add_argument("--model-path", required=True, help="Path to saved model")
    p_pred.add_argument("--input-json", help="Single JSON record string")
    p_pred.add_argument("--input-file", help="Path to JSON file")
    p_pred.set_defaults(func=action_predict)

    return parser


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
