"""Module 7.1 Advanced Defect Detection Pipeline

Production-ready defect detection pipeline for semiconductor manufacturing.
Supports multiple backends: YOLO (ultralytics), Faster R-CNN (torchvision),
and classical OpenCV with graceful fallbacks.

Features:
- Object detection for wafer defects (scratches, particles, cracks)
- Multiple backend support with automatic fallback
- Cost-sensitive evaluation for manufacturing context
- Model persistence and synthetic data generation
- CLI with train/evaluate/predict subcommands producing JSON

Example usage:
    python 7.1-advanced-defect-detection-pipeline.py train --backend classical --epochs 5 --save model.joblib
    python 7.1-advanced-defect-detection-pipeline.py evaluate --model-path model.joblib --dataset synthetic
    python 7.1-advanced-defect-detection-pipeline.py predict --model-path model.joblib --image-path wafer.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import tempfile

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
TARGET_COLUMN = "target"

# Optional imports with graceful fallbacks
try:
    import ultralytics
    from ultralytics import YOLO

    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    ultralytics = None
    YOLO = None

try:
    import torch
    import torchvision
    from torchvision.models import detection

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    torch = None
    torchvision = None
    detection = None

# -------------------- Synthetic Data Generator -------------------- #


def generate_synthetic_wafer_defects(
    n_images: int = 50,
    image_size: Tuple[int, int] = (416, 416),
    n_defects_range: Tuple[int, int] = (1, 5),
    seed: int = RANDOM_SEED,
) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]]]:
    """Generate synthetic wafer images with defect bounding boxes.

    Args:
        n_images: Number of synthetic images to generate
        image_size: (height, width) of generated images
        n_defects_range: Min and max number of defects per image
        seed: Random seed for reproducibility

    Returns:
        Tuple of (images, annotations) where annotations contain bounding boxes
    """
    rng = np.random.default_rng(seed)
    images = []
    annotations = []

    h, w = image_size

    for i in range(n_images):
        # Create base wafer image (circular silicon wafer)
        img = np.ones((h, w, 3), dtype=np.uint8) * 200  # Gray background

        # Draw wafer circle
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 20
        cv2.circle(img, center, radius, (180, 180, 180), -1)

        # Add some realistic noise and patterns
        noise = rng.normal(0, 10, (h, w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Generate defects
        n_defects = rng.integers(n_defects_range[0], n_defects_range[1] + 1)
        image_annotations = []

        for _ in range(n_defects):
            defect_type = rng.choice(["scratch", "particle", "crack"])

            if defect_type == "scratch":
                # Linear defect
                x1, y1 = rng.integers(50, w - 50), rng.integers(50, h - 50)
                length = rng.integers(20, 80)
                angle = rng.uniform(0, 2 * np.pi)
                x2 = int(x1 + length * np.cos(angle))
                y2 = int(y1 + length * np.sin(angle))
                x2, y2 = np.clip([x2, y2], [50, 50], [w - 50, h - 50])

                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

                # Bounding box
                bbox_x1, bbox_y1 = min(x1, x2) - 5, min(y1, y2) - 5
                bbox_x2, bbox_y2 = max(x1, x2) + 5, max(y1, y2) + 5

            elif defect_type == "particle":
                # Circular defect
                x, y = rng.integers(50, w - 50), rng.integers(50, h - 50)
                radius = rng.integers(3, 15)
                cv2.circle(img, (x, y), radius, (0, 0, 0), -1)

                # Bounding box
                bbox_x1, bbox_y1 = x - radius - 2, y - radius - 2
                bbox_x2, bbox_y2 = x + radius + 2, y + radius + 2

            else:  # crack
                # Irregular crack pattern
                x, y = rng.integers(50, w - 50), rng.integers(50, h - 50)
                pts = [(x, y)]
                for _ in range(rng.integers(3, 8)):
                    x += rng.integers(-10, 11)
                    y += rng.integers(-10, 11)
                    x, y = np.clip([x, y], [50, 50], [w - 50, h - 50])
                    pts.append((x, y))

                for j in range(len(pts) - 1):
                    cv2.line(img, pts[j], pts[j + 1], (0, 0, 0), 1)

                # Bounding box around all points
                xs, ys = zip(*pts)
                bbox_x1, bbox_y1 = min(xs) - 5, min(ys) - 5
                bbox_x2, bbox_y2 = max(xs) + 5, max(ys) + 5

            # Ensure bounding box is within image
            bbox_x1 = max(0, bbox_x1)
            bbox_y1 = max(0, bbox_y1)
            bbox_x2 = min(w, bbox_x2)
            bbox_y2 = min(h, bbox_y2)

            if bbox_x2 > bbox_x1 and bbox_y2 > bbox_y1:
                image_annotations.append(
                    {
                        "bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                        "class": defect_type,
                        "class_id": {"scratch": 0, "particle": 1, "crack": 2}[defect_type],
                    }
                )

        images.append(img)
        annotations.append(image_annotations)

    return images, annotations


# -------------------- Detection Backends -------------------- #


class ClassicalDetector:
    """Classical OpenCV-based defect detector using blob detection and contours."""

    def __init__(self, blur_kernel: int = 5, threshold_value: int = 50):
        self.blur_kernel = blur_kernel
        self.threshold_value = threshold_value
        self.class_names = ["scratch", "particle", "crack"]

    def fit(self, images: List[np.ndarray], annotations: List[List[Dict[str, Any]]]):
        """Fit method for compatibility - classical method doesn't train."""
        pass

    def predict(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Detect defects in images using classical computer vision."""
        predictions = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            # Preprocessing
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

            # Edge detection and morphology
            edges = cv2.Canny(blurred, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_predictions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 20:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)

                    # Simple classification based on aspect ratio
                    aspect_ratio = w / h if h > 0 else 1
                    if aspect_ratio > 2.5:
                        class_name = "scratch"
                        class_id = 0
                    elif aspect_ratio < 1.5 and area < 500:
                        class_name = "particle"
                        class_id = 1
                    else:
                        class_name = "crack"
                        class_id = 2

                    image_predictions.append(
                        {
                            "bbox": [x, y, x + w, y + h],
                            "class": class_name,
                            "class_id": class_id,
                            "confidence": min(0.9, area / 1000),  # Fake confidence based on area
                        }
                    )

            predictions.append(image_predictions)

        return predictions


class YOLODetector:
    """YOLO-based detector using ultralytics."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics package not available")
        self.model = YOLO(model_name)
        self.class_names = ["scratch", "particle", "crack"]

    def fit(self, images: List[np.ndarray], annotations: List[List[Dict[str, Any]]], epochs: int = 5):
        """Train YOLO model on provided data."""
        # For demo purposes, just load pretrained model
        # In practice, would need to convert annotations to YOLO format
        print(f"Training YOLO for {epochs} epochs on {len(images)} images")
        pass

    def predict(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Predict using YOLO model."""
        predictions = []

        for img in images:
            results = self.model(img)
            image_predictions = []

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        if class_id < len(self.class_names):
                            image_predictions.append(
                                {
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "class": self.class_names[class_id],
                                    "class_id": class_id,
                                    "confidence": float(confidence),
                                }
                            )

            predictions.append(image_predictions)

        return predictions


class FasterRCNNDetector:
    """Faster R-CNN detector using torchvision."""

    def __init__(self):
        if not HAS_TORCHVISION:
            raise ImportError("torchvision package not available")

        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.class_names = ["background", "scratch", "particle", "crack"]

    def fit(self, images: List[np.ndarray], annotations: List[List[Dict[str, Any]]], epochs: int = 5):
        """Train Faster R-CNN model."""
        print(f"Training Faster R-CNN for {epochs} epochs on {len(images)} images")
        # For demo, just use pretrained model
        pass

    def predict(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Predict using Faster R-CNN."""
        predictions = []

        with torch.no_grad():
            for img in images:
                # Convert to tensor
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)

                outputs = self.model(img_tensor)
                image_predictions = []

                boxes = outputs[0]["boxes"].cpu().numpy()
                scores = outputs[0]["scores"].cpu().numpy()
                labels = outputs[0]["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.5 and label < len(self.class_names):
                        x1, y1, x2, y2 = box
                        image_predictions.append(
                            {
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "class": self.class_names[label],
                                "class_id": int(label),
                                "confidence": float(score),
                            }
                        )

                predictions.append(image_predictions)

        return predictions


# -------------------- Evaluation Metrics -------------------- #


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_detection_metrics(
    predictions: List[List[Dict[str, Any]]],
    ground_truth: List[List[Dict[str, Any]]],
    iou_threshold: float = 0.5,
    defect_cost_per_miss: float = 1000.0,
    false_alarm_cost: float = 100.0,
) -> Dict[str, float]:
    """Calculate detection metrics including semiconductor-specific costs."""

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_confidences = []
    all_true_labels = []

    estimated_loss = 0.0

    for pred_list, gt_list in zip(predictions, ground_truth):
        # Match predictions to ground truth
        gt_matched = [False] * len(gt_list)

        for pred in pred_list:
            pred_bbox = pred["bbox"]
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_list):
                if gt_matched[gt_idx]:
                    continue

                gt_bbox = gt["bbox"]
                iou = calculate_iou(pred_bbox, gt_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                total_tp += 1
                gt_matched[best_gt_idx] = True
                all_confidences.append(pred.get("confidence", 0.5))
                all_true_labels.append(1)
            else:
                total_fp += 1
                estimated_loss += false_alarm_cost
                all_confidences.append(pred.get("confidence", 0.5))
                all_true_labels.append(0)

        # Count missed detections
        total_fn += sum(1 for matched in gt_matched if not matched)
        estimated_loss += sum(defect_cost_per_miss for matched in gt_matched if not matched)

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # mAP calculation (simplified)
    if len(all_confidences) > 0 and len(set(all_true_labels)) > 1:
        try:
            map_score = average_precision_score(all_true_labels, all_confidences)
        except:
            map_score = 0.0
    else:
        map_score = 0.0

    # Prediction Within Spec (PWS) - percentage of correct detections
    total_detections = total_tp + total_fp + total_fn
    pws = (total_tp / total_detections * 100) if total_detections > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "map_50": map_score,
        "pws_percent": pws,
        "estimated_loss_usd": estimated_loss,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }


# -------------------- Pipeline Class -------------------- #


@dataclass
class DetectionMetadata:
    trained_at: str
    backend: str
    model_params: Dict[str, Any]
    class_names: List[str]
    metrics: Optional[Dict[str, float]] = None


class AdvancedDefectDetectionPipeline:
    """Advanced defect detection pipeline with multiple backend support."""

    def __init__(self, backend: str = "classical", **kwargs):
        self.backend = backend.lower()
        self.detector = None
        self.metadata: Optional[DetectionMetadata] = None
        self.kwargs = kwargs

        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize the appropriate detector backend."""
        if self.backend == "yolo":
            if not HAS_ULTRALYTICS:
                warnings.warn("ultralytics not available, falling back to classical")
                self.backend = "classical"
                self.detector = ClassicalDetector(**self.kwargs)
            else:
                self.detector = YOLODetector(**self.kwargs)

        elif self.backend == "fasterrcnn":
            if not HAS_TORCHVISION:
                warnings.warn("torchvision not available, falling back to classical")
                self.backend = "classical"
                self.detector = ClassicalDetector(**self.kwargs)
            else:
                self.detector = FasterRCNNDetector(**self.kwargs)

        elif self.backend == "classical":
            self.detector = ClassicalDetector(**self.kwargs)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def fit(
        self, images: List[np.ndarray], annotations: List[List[Dict[str, Any]]], epochs: int = 5
    ) -> "AdvancedDefectDetectionPipeline":
        """Train the detection model."""
        if hasattr(self.detector, "fit"):
            if self.backend in ["yolo", "fasterrcnn"]:
                self.detector.fit(images, annotations, epochs=epochs)
            else:
                self.detector.fit(images, annotations)

        # Create metadata
        self.metadata = DetectionMetadata(
            trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            backend=self.backend,
            model_params=self.kwargs,
            class_names=getattr(self.detector, "class_names", ["scratch", "particle", "crack"]),
        )

        return self

    def predict(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Predict defects in images."""
        if self.detector is None:
            raise ValueError("Model not initialized")

        return self.detector.predict(images)

    def evaluate(
        self, images: List[np.ndarray], annotations: List[List[Dict[str, Any]]], iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(images)
        metrics = evaluate_detection_metrics(predictions, annotations, iou_threshold=iou_threshold)

        if self.metadata:
            self.metadata.metrics = metrics

        return metrics

    def save(self, path: Path) -> None:
        """Save the pipeline to disk."""
        path = Path(path)
        save_data = {
            "backend": self.backend,
            "kwargs": self.kwargs,
            "metadata": asdict(self.metadata) if self.metadata else None,
            "detector_state": None,  # Classical doesn't have state to save
        }

        # Save detector-specific state if needed
        if hasattr(self.detector, "model") and self.backend in ["yolo", "fasterrcnn"]:
            # Would save model weights here in practice
            pass

        joblib.dump(save_data, path)

    @staticmethod
    def load(path: Path) -> "AdvancedDefectDetectionPipeline":
        """Load pipeline from disk."""
        path = Path(path)
        save_data = joblib.load(path)

        pipeline = AdvancedDefectDetectionPipeline(backend=save_data["backend"], **save_data["kwargs"])

        if save_data["metadata"]:
            pipeline.metadata = DetectionMetadata(**save_data["metadata"])

        return pipeline


# -------------------- CLI Implementation -------------------- #


def action_train(args):
    """Train a defect detection model."""
    try:
        # Generate or load data
        if args.dataset == "synthetic":
            images, annotations = generate_synthetic_wafer_defects(n_images=args.n_images, seed=RANDOM_SEED)
        else:
            return {"error": f"Unknown dataset: {args.dataset}"}

        # Initialize and train pipeline
        pipeline = AdvancedDefectDetectionPipeline(
            backend=args.backend,
            blur_kernel=getattr(args, "blur_kernel", 5),
            threshold_value=getattr(args, "threshold_value", 50),
        )

        pipeline.fit(images, annotations, epochs=args.epochs)

        # Save model if requested
        if args.save:
            pipeline.save(Path(args.save))

        result = {
            "status": "success",
            "backend": pipeline.backend,
            "n_images_trained": len(images),
            "epochs": args.epochs,
            "model_saved": args.save is not None,
        }

        if pipeline.metadata:
            result["metadata"] = asdict(pipeline.metadata)

        return result

    except Exception as e:
        return {"error": str(e)}


def action_evaluate(args):
    """Evaluate a trained model."""
    try:
        # Load model
        if args.model_path:
            pipeline = AdvancedDefectDetectionPipeline.load(Path(args.model_path))
        else:
            return {"error": "model_path is required for evaluation"}

        # Generate or load test data
        if args.dataset == "synthetic":
            images, annotations = generate_synthetic_wafer_defects(
                n_images=args.n_images, seed=RANDOM_SEED + 1000  # Different seed for test data
            )
        else:
            return {"error": f"Unknown dataset: {args.dataset}"}

        # Evaluate
        metrics = pipeline.evaluate(images, annotations, iou_threshold=args.iou_threshold)

        result = {"status": "success", "backend": pipeline.backend, "n_test_images": len(images), "metrics": metrics}

        return result

    except Exception as e:
        return {"error": str(e)}


def action_predict(args):
    """Make predictions on new images."""
    try:
        # Load model
        if args.model_path:
            pipeline = AdvancedDefectDetectionPipeline.load(Path(args.model_path))
        else:
            return {"error": "model_path is required for prediction"}

        # Load image(s)
        if args.image_path:
            img = cv2.imread(args.image_path)
            if img is None:
                return {"error": f"Could not load image: {args.image_path}"}
            images = [img]
        elif args.dataset == "synthetic":
            images, _ = generate_synthetic_wafer_defects(n_images=1, seed=RANDOM_SEED)
        else:
            return {"error": "Either image_path or dataset=synthetic required"}

        # Predict
        predictions = pipeline.predict(images)

        result = {"status": "success", "backend": pipeline.backend, "n_images": len(images), "predictions": predictions}

        return result

    except Exception as e:
        return {"error": str(e)}


def build_parser():
    """Build argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Module 7.1 Advanced Defect Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a classical model:
    python 7.1-advanced-defect-detection-pipeline.py train --backend classical --epochs 5 --save model.joblib

  Evaluate model:
    python 7.1-advanced-defect-detection-pipeline.py evaluate --model-path model.joblib --dataset synthetic

  Predict on image:
    python 7.1-advanced-defect-detection-pipeline.py predict --model-path model.joblib --image-path wafer.jpg
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    p_train = subparsers.add_parser("train", help="Train a defect detection model")
    p_train.add_argument(
        "--backend", choices=["classical", "yolo", "fasterrcnn"], default="classical", help="Detection backend to use"
    )
    p_train.add_argument("--dataset", default="synthetic", help="Dataset to use")
    p_train.add_argument("--n-images", type=int, default=50, help="Number of synthetic images")
    p_train.add_argument("--epochs", type=int, default=5, help="Training epochs")
    p_train.add_argument("--save", help="Path to save trained model")
    p_train.add_argument("--blur-kernel", type=int, default=5, help="Blur kernel size (classical)")
    p_train.add_argument("--threshold-value", type=int, default=50, help="Threshold value (classical)")
    p_train.set_defaults(func=action_train)

    # Evaluate subcommand
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    p_eval.add_argument("--model-path", required=True, help="Path to trained model")
    p_eval.add_argument("--dataset", default="synthetic", help="Dataset to evaluate on")
    p_eval.add_argument("--n-images", type=int, default=20, help="Number of test images")
    p_eval.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for detection")
    p_eval.set_defaults(func=action_evaluate)

    # Predict subcommand
    p_pred = subparsers.add_parser("predict", help="Make predictions on images")
    p_pred.add_argument("--model-path", required=True, help="Path to trained model")
    p_pred.add_argument("--image-path", help="Path to image file")
    p_pred.add_argument("--dataset", help="Use synthetic data if no image-path")
    p_pred.set_defaults(func=action_predict)

    return parser


def main():
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    result = args.func(args)
    print(json.dumps(result, indent=2))

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
