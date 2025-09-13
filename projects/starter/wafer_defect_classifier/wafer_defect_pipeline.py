"""Production Wafer Defect Classification Pipeline Script

Provides a CLI to train, evaluate, and predict using a standardized 
classification pipeline for semiconductor wafer defect detection using 
classical machine learning approaches.

Features:
- Unified preprocessing: impute -> scale -> (optional) sampler -> model
- Supports logistic, linear SVM, tree, random forest, gradient boosting
- Imbalance handling via class weights or (optional) SMOTE
- Manufacturing-specific metrics: ROC AUC, PR AUC, PWS, Estimated Loss
- Standard metrics: precision/recall/F1 at chosen threshold
- Threshold optimization to satisfy precision or recall constraints
- Model persistence (save/load) with metadata (including threshold)
- Synthetic wafer defect pattern generation for learning

Example usage:
    python wafer_defect_pipeline.py train --dataset synthetic_wafer \\
        --model logistic --min-precision 0.85 --save wafer_model.joblib
    python wafer_defect_pipeline.py evaluate \\
        --model-path wafer_model.joblib --dataset synthetic_wafer
    python wafer_defect_pipeline.py predict \\
        --model-path wafer_model.joblib \\
        --input-json '{"center_density":0.12, "edge_density":0.05}'
"""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
import joblib

# Constants
RANDOM_SEED = 42
TARGET_COLUMN = "defect"

# Optional SMOTE import
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMB_AVAILABLE = True
except ImportError:
    IMB_AVAILABLE = False

# Dataset loading: use relative path from project location
DATA_DIR = Path("../../../datasets").resolve()


def generate_synthetic_wafer_defects(
    n_samples: int = 1000,
    map_size: int = 64,
    defect_rate: float = 0.15,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic wafer defect data with realistic patterns."""
    np.random.seed(seed)

    records = []
    center = map_size // 2

    for i in range(n_samples):
        # Generate wafer map with circular valid area
        y, x = np.ogrid[:map_size, :map_size]
        valid_mask = (x - center) ** 2 + (y - center) ** 2 <= (map_size / 2.2) ** 2

        # Determine if this wafer has defects
        has_defect = np.random.random() < defect_rate

        if has_defect:
            # Choose defect pattern
            pattern = np.random.choice(["center", "edge", "scratch", "random"])

            if pattern == "center":
                # Center cluster defect
                defect_radius = np.random.uniform(3, 8)
                defect_mask = (x - center) ** 2 + (y - center) ** 2 <= defect_radius**2

            elif pattern == "edge":
                # Edge ring defect
                outer_radius = map_size / 2.2
                inner_radius = outer_radius * 0.8
                defect_mask = (
                    (x - center) ** 2 + (y - center) ** 2 >= inner_radius**2
                ) & ((x - center) ** 2 + (y - center) ** 2 <= outer_radius**2)

            elif pattern == "scratch":
                # Linear scratch defect
                scratch_angle = np.random.uniform(0, np.pi)
                scratch_width = np.random.randint(1, 3)
                scratch_length = np.random.randint(map_size // 3, map_size // 2)

                # Create scratch mask (simplified)
                defect_mask = np.zeros((map_size, map_size), dtype=bool)
                start_x = center - int(scratch_length * np.cos(scratch_angle) / 2)
                end_x = center + int(scratch_length * np.cos(scratch_angle) / 2)
                start_y = center - int(scratch_length * np.sin(scratch_angle) / 2)
                end_y = center + int(scratch_length * np.sin(scratch_angle) / 2)

                for t in np.linspace(0, 1, scratch_length):
                    px = int(start_x + t * (end_x - start_x))
                    py = int(start_y + t * (end_y - start_y))
                    for dx in range(-scratch_width, scratch_width + 1):
                        for dy in range(-scratch_width, scratch_width + 1):
                            if 0 <= px + dx < map_size and 0 <= py + dy < map_size:
                                defect_mask[py + dy, px + dx] = True

            else:  # random
                # Random scattered defects
                defect_mask = np.random.random((map_size, map_size)) < 0.05
        else:
            # No defects
            defect_mask = np.zeros((map_size, map_size), dtype=bool)

        # Apply defect pattern to valid wafer area
        final_defect_mask = defect_mask & valid_mask

        # Calculate features
        total_area = np.sum(valid_mask)
        defect_area = np.sum(final_defect_mask)
        defect_area_ratio = defect_area / total_area if total_area > 0 else 0

        # Center and edge density
        center_radius = map_size / 6
        center_mask = (x - center) ** 2 + (y - center) ** 2 <= center_radius**2
        center_defect_count = np.sum(final_defect_mask & center_mask & valid_mask)
        center_area = np.sum(center_mask & valid_mask)
        center_density = center_defect_count / center_area if center_area > 0 else 0

        edge_outer = (map_size / 2.2) ** 2
        edge_inner = (map_size / 3) ** 2
        edge_mask = ((x - center) ** 2 + (y - center) ** 2 >= edge_inner) & (
            (x - center) ** 2 + (y - center) ** 2 <= edge_outer
        )
        edge_defect_count = np.sum(final_defect_mask & edge_mask & valid_mask)
        edge_area = np.sum(edge_mask & valid_mask)
        edge_density = edge_defect_count / edge_area if edge_area > 0 else 0

        # Additional features
        if defect_area > 0:
            # Defect clustering metric (simplified)
            defect_coords = np.where(final_defect_mask)
            if len(defect_coords[0]) > 1:
                defect_spread = np.std(defect_coords[0]) + np.std(defect_coords[1])
            else:
                defect_spread = 0
        else:
            defect_spread = 0

        # Add noise to features
        center_density += np.random.normal(0, 0.01)
        edge_density += np.random.normal(0, 0.01)
        defect_area_ratio += np.random.normal(0, 0.005)
        defect_spread += np.random.normal(0, 0.5)

        records.append(
            {
                "wafer_id": f"W{i:04d}",
                "center_density": max(0, center_density),
                "edge_density": max(0, edge_density),
                "defect_area_ratio": max(0, min(1, defect_area_ratio)),
                "defect_spread": max(0, defect_spread),
                "total_pixels": int(total_area),
                "defect_pixels": int(defect_area),
                "defect": int(has_defect),
            }
        )

    return pd.DataFrame(records)


def load_dataset(name: str) -> pd.DataFrame:
    """Load wafer defect dataset by name."""
    if name == "synthetic_wafer":
        return generate_synthetic_wafer_defects()
    elif name.startswith("synthetic_wafer_"):
        # Parse parameters from name like "synthetic_wafer_500_0.2"
        parts = name.split("_")
        n_samples = int(parts[2]) if len(parts) > 2 else 1000
        defect_rate = float(parts[3]) if len(parts) > 3 else 0.15
        return generate_synthetic_wafer_defects(
            n_samples=n_samples, defect_rate=defect_rate
        )
    else:
        # Try to load from datasets directory
        dataset_path = DATA_DIR / name
        if dataset_path.exists():
            # Try common file extensions
            for ext in [".csv", ".data", ".txt"]:
                file_path = dataset_path / f"{name}{ext}"
                if file_path.exists():
                    return pd.read_csv(file_path)
        raise ValueError(
            f"Unknown dataset '{name}'. "
            "Supported: synthetic_wafer, synthetic_wafer_<n>_<rate>"
        )


# Metadata & Pipeline Classes


@dataclass
class WaferDefectMetadata:
    trained_at: str
    model_type: str
    n_features_in: int
    sampler: Optional[str]
    params: Dict[str, Any]
    threshold: float
    metrics: Optional[Dict[str, float]] = None


class WaferDefectPipeline:
    """Wafer defect classification pipeline with manufacturing-specific metrics."""

    def __init__(
        self,
        model: str = "logistic",
        use_smote: bool = False,
        smote_k_neighbors: int = 5,
        class_weight_mode: str | None = "balanced",
        min_precision: Optional[float] = None,
        min_recall: Optional[float] = None,
        C: float = 1.0,
        max_depth: int = 6,
        n_estimators: int = 300,
    ) -> None:
        self.model_name = model.lower()
        self.use_smote = use_smote and IMB_AVAILABLE
        self.smote_k_neighbors = smote_k_neighbors

        # Validate class_weight_mode
        if class_weight_mode not in ("balanced", "balanced_subsample", None):
            raise ValueError(
                "class_weight_mode must be 'balanced', 'balanced_subsample', or None"
            )
        self.class_weight_mode = class_weight_mode
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.C = C
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Optional[WaferDefectMetadata] = None
        self.fitted_threshold: float = 0.5

    def _build_estimator(self):
        """Build the ML estimator based on model name."""
        if self.model_name == "logistic":
            return LogisticRegression(
                max_iter=1000,
                class_weight=self.class_weight_mode,
                C=self.C,
                random_state=RANDOM_SEED,
            )
        elif self.model_name == "linear_svm":
            return LinearSVC(
                class_weight=self.class_weight_mode, C=self.C, random_state=RANDOM_SEED
            )
        elif self.model_name == "tree":
            return DecisionTreeClassifier(
                max_depth=self.max_depth,
                class_weight=self.class_weight_mode,
                random_state=RANDOM_SEED,
            )
        elif self.model_name == "rf":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight=self.class_weight_mode,
                random_state=RANDOM_SEED,
            )
        elif self.model_name == "gb":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=RANDOM_SEED,
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WaferDefectPipeline":
        """Fit the pipeline to training data."""
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]

        # Add SMOTE if requested and available
        if self.use_smote:
            smote = SMOTE(k_neighbors=self.smote_k_neighbors, random_state=RANDOM_SEED)
            steps.append(("smote", smote))

        steps.append(("model", self._build_estimator()))

        # Use imbalanced-learn pipeline if SMOTE is used
        PipelineClass = ImbPipeline if self.use_smote else Pipeline
        self.pipeline = PipelineClass(steps)

        # Fit pipeline
        self.pipeline.fit(X, y)

        # Optimize threshold if precision/recall constraints are given
        self._optimize_threshold(X, y)

        # Store metadata
        self.metadata = WaferDefectMetadata(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            model_type=type(self.pipeline.named_steps["model"]).__name__,
            n_features_in=X.shape[1],
            sampler="SMOTE" if self.use_smote else None,
            params={
                "model": self.model_name,
                "use_smote": self.use_smote,
                "class_weight": self.class_weight_mode,
                "C": self.C,
                "max_depth": self.max_depth,
                "n_estimators": self.n_estimators,
            },
            threshold=self.fitted_threshold,
        )
        return self

    def _optimize_threshold(self, X: pd.DataFrame, y: np.ndarray):
        """Optimize classification threshold based on precision/recall constraints."""
        if not (self.min_precision or self.min_recall):
            return

        probs = self.predict_proba(X)[:, 1]
        prec, rec, thr_full = precision_recall_curve(y, probs)

        # Find feasible thresholds
        candidate_idx = []
        for i, (p, r) in enumerate(zip(prec, rec)):
            if self.min_precision and p < self.min_precision:
                continue
            if self.min_recall and r < self.min_recall:
                continue
            candidate_idx.append(i)

        if candidate_idx:
            # Choose threshold giving highest F1 among feasible
            def f1_at(i_: int) -> float:
                return float((2 * prec[i_] * rec[i_]) / (prec[i_] + rec[i_] + 1e-9))

            best_i = max(candidate_idx, key=lambda idx: f1_at(idx))
            self.fitted_threshold = float(thr_full[best_i])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted")
        model = self.pipeline.named_steps["model"]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(self.pipeline[:-1].transform(X))
        # decision function fallback for LinearSVC
        decision = model.decision_function(self.pipeline[:-1].transform(X))
        probs = 1 / (1 + np.exp(-decision))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions using the fitted threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.fitted_threshold).astype(int)

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics including
        manufacturing-specific ones."""
        base_metrics = {
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "pr_auc": float(average_precision_score(y_true, probs)),
            "mcc": float(matthews_corrcoef(y_true, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
        }

        # Manufacturing-specific metrics
        # PWS (Prediction Within Spec) - percentage of predictions matching true labels
        pws = float(np.mean(preds == y_true)) * 100

        # Estimated Loss - simplified cost model
        # Assume: False Negative costs 10x more than False Positive
        fp_cost = 1.0  # Cost of false alarm
        fn_cost = 10.0  # Cost of missing a defect

        fp_count = np.sum((preds == 1) & (y_true == 0))
        fn_count = np.sum((preds == 0) & (y_true == 1))
        estimated_loss = float(fp_count * fp_cost + fn_count * fn_cost)

        # Add manufacturing metrics
        base_metrics.update(
            {
                "pws": pws,
                "estimated_loss": estimated_loss,
                "false_positive_count": int(fp_count),
                "false_negative_count": int(fn_count),
            }
        )

        return base_metrics

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model and return comprehensive metrics."""
        probs = self.predict_proba(X)[:, 1]
        preds = (probs >= self.fitted_threshold).astype(int)
        metrics = self.compute_metrics(y, probs, preds)
        if self.metadata:
            self.metadata.metrics = metrics
        return metrics

    def save(self, path: Path):
        """Save the fitted pipeline and metadata."""
        if self.pipeline is None or self.metadata is None:
            raise RuntimeError("Nothing to save")
        joblib.dump(
            {"pipeline": self.pipeline, "metadata": asdict(self.metadata)}, path
        )

    @staticmethod
    def load(path: Path) -> "WaferDefectPipeline":
        """Load a saved pipeline."""
        obj = joblib.load(path)
        inst = WaferDefectPipeline(model=obj["metadata"]["params"]["model"])
        inst.pipeline = obj["pipeline"]
        inst.metadata = WaferDefectMetadata(**obj["metadata"])
        inst.fitted_threshold = inst.metadata.threshold
        return inst


# CLI Actions


def action_train(args):
    """Handle train command."""
    try:
        df = load_dataset(args.dataset)
        y = df[TARGET_COLUMN].to_numpy(dtype=float)
        X = df.drop(columns=[TARGET_COLUMN, "wafer_id"], errors="ignore")

        pipe = WaferDefectPipeline(
            model=args.model,
            use_smote=args.use_smote,
            smote_k_neighbors=args.smote_k_neighbors,
            class_weight_mode=None if args.no_class_weight else "balanced",
            min_precision=args.min_precision,
            min_recall=args.min_recall,
            C=args.C,
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
        )
        pipe.fit(X, y)
        metrics = pipe.evaluate(X, y)  # train metrics

        if args.save:
            pipe.save(Path(args.save))

        meta_dict = asdict(pipe.metadata) if pipe.metadata else None
        print(
            json.dumps(
                {"status": "trained", "metrics": metrics, "metadata": meta_dict},
                indent=2,
            )
        )
    except Exception as e:
        error_result = {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def action_evaluate(args):
    """Handle evaluate command."""
    try:
        pipe = WaferDefectPipeline.load(Path(args.model_path))
        df = load_dataset(args.dataset)
        y = df[TARGET_COLUMN].to_numpy(dtype=float)
        X = df.drop(columns=[TARGET_COLUMN, "wafer_id"], errors="ignore")
        metrics = pipe.evaluate(X, y)
        meta_dict = asdict(pipe.metadata) if pipe.metadata else None
        print(
            json.dumps(
                {"status": "evaluated", "metrics": metrics, "metadata": meta_dict},
                indent=2,
            )
        )
    except Exception as e:
        error_result = {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def action_predict(args):
    """Handle predict command."""
    try:
        pipe = WaferDefectPipeline.load(Path(args.model_path))

        if args.input_json:
            record = json.loads(args.input_json)
        elif args.input_file:
            record = json.loads(Path(args.input_file).read_text())
        else:
            raise ValueError("Provide --input-json or --input-file")

        df = pd.DataFrame([record])
        probs = pipe.predict_proba(df)[0, 1]
        pred = int(probs >= pipe.fitted_threshold)

        print(
            json.dumps(
                {
                    "prediction": pred,
                    "probability": float(probs),
                    "threshold": pipe.fitted_threshold,
                    "input": record,
                },
                indent=2,
            )
        )
    except Exception as e:
        error_result = {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def build_parser():
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Wafer Defect Classification Production Pipeline CLI"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Train subcommand
    p_train = sub.add_parser(
        "train", help="Train a wafer defect classification pipeline"
    )
    p_train.add_argument("--dataset", default="synthetic_wafer")
    p_train.add_argument(
        "--model",
        default="logistic",
        choices=["logistic", "linear_svm", "tree", "rf", "gb"],
    )
    p_train.add_argument(
        "--use-smote",
        action="store_true",
        help="Apply SMOTE oversampling (requires imbalanced-learn)",
    )
    p_train.add_argument("--smote-k-neighbors", type=int, default=5)
    p_train.add_argument(
        "--no-class-weight", action="store_true", help="Disable class_weight balancing"
    )
    p_train.add_argument(
        "--min-precision",
        type=float,
        help="Minimum precision constraint for threshold selection",
    )
    p_train.add_argument(
        "--min-recall",
        type=float,
        help="Minimum recall constraint for threshold selection",
    )
    p_train.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic/linear SVM",
    )
    p_train.add_argument("--max-depth", type=int, default=6)
    p_train.add_argument("--n-estimators", type=int, default=300)
    p_train.add_argument("--save", help="Path to save trained model")
    p_train.set_defaults(func=action_train)

    # Evaluate subcommand
    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument("--dataset", default="synthetic_wafer")
    p_eval.set_defaults(func=action_evaluate)

    # Predict subcommand
    p_pred = sub.add_parser("predict", help="Predict with a saved model")
    p_pred.add_argument("--model-path", required=True)
    p_pred.add_argument("--input-json")
    p_pred.add_argument("--input-file")
    p_pred.set_defaults(func=action_predict)

    return p


def main(argv: Optional[List[str]] = None):
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
