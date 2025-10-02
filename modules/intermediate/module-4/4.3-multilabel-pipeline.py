"""Module 4.3 Multi-Label Classification Pipeline

Multi-label classification for manufacturing defects using Steel Plates dataset.

Implements Binary Relevance, Classifier Chains, and native multi-output estimators.
CLI Subcommands (train/evaluate/predict) produce JSON for automation.

Usage Examples:

    # Train Binary Relevance model on Steel Plates
    python 4.3-multilabel-pipeline.py train \\
        --method binary_relevance \\
        --model rf \\
        --model-out models/steel_br.joblib

    # Train Classifier Chains
    python 4.3-multilabel-pipeline.py train \\
        --method classifier_chains \\
        --model rf \\
        --model-out models/steel_cc.joblib

    # Evaluate on Steel Plates test set
    python 4.3-multilabel-pipeline.py evaluate \\
        --model-path models/steel_br.joblib \\
        --dataset steel_plates

    # Predict single instance
    python 4.3-multilabel-pipeline.py predict \\
        --model-path models/steel_br.joblib \\
        --input-json '{"X_Min": 42, "X_Max": 308, ...}'

    # Analyze label distribution
    python 4.3-multilabel-pipeline.py analyze \\
        --dataset steel_plates
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Steel Plates fault labels (7 binary indicators)
STEEL_LABELS = [
    "Pastry",
    "Z_Scratch",
    "K_Scratch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]


# ========== Data Loading ========== #


def load_steel_plates_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load Steel Plates Faults dataset.

    Returns:
        X: Feature matrix (1941, 27)
        y: Binary label matrix (1941, 7)
        label_names: List of 7 fault type names
    """
    data_root = Path(__file__).parent.parent.parent.parent / "datasets" / "steel-plates"

    if not data_root.exists():
        raise FileNotFoundError(f"Steel Plates dataset not found at {data_root}")

    features_path = data_root / "steel_plates_features.csv"
    targets_path = data_root / "steel_plates_targets.csv"

    if not features_path.exists() or not targets_path.exists():
        raise FileNotFoundError(
            f"Steel Plates CSV files not found. Expected:\n" f"  {features_path}\n" f"  {targets_path}"
        )

    X = pd.read_csv(features_path)
    y = pd.read_csv(targets_path)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Feature/target size mismatch: {X.shape[0]} vs {y.shape[0]}")

    return X, y, STEEL_LABELS


def generate_multilabel_synthetic(
    n_samples: int = 500, n_labels: int = 5, seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Generate synthetic multi-label classification data.

    Args:
        n_samples: Number of instances
        n_labels: Number of binary labels
        seed: Random seed

    Returns:
        X: Feature matrix (n_samples, 10)
        y: Binary label matrix (n_samples, n_labels)
        label_names: List of label names
    """
    rng = np.random.default_rng(seed)

    # Generate features
    X = pd.DataFrame(rng.normal(0, 1, (n_samples, 10)), columns=[f"feature_{i}" for i in range(10)])

    # Generate labels with correlations
    y_data = np.zeros((n_samples, n_labels), dtype=int)
    for i in range(n_samples):
        # Base probability from features
        base_prob = 1 / (1 + np.exp(-X.iloc[i, 0]))  # Sigmoid of first feature

        for j in range(n_labels):
            # Each label has different feature sensitivity
            prob = base_prob + 0.1 * X.iloc[i, j % 10]
            prob = np.clip(prob, 0.1, 0.9)

            # Add label correlation (label j depends on label j-1)
            if j > 0 and y_data[i, j - 1] == 1:
                prob *= 1.5  # Increase probability if previous label active

            y_data[i, j] = 1 if rng.random() < prob else 0

        # Ensure at least one label per instance
        if y_data[i].sum() == 0:
            y_data[i, rng.integers(0, n_labels)] = 1

    y = pd.DataFrame(y_data, columns=[f"label_{i}" for i in range(n_labels)])
    label_names = y.columns.tolist()

    return X, y, label_names


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load dataset by name.

    Args:
        dataset_name: 'steel_plates' or 'synthetic'

    Returns:
        X: Feature matrix
        y: Binary label matrix
        label_names: List of label names
    """
    if dataset_name == "steel_plates":
        return load_steel_plates_dataset()
    elif dataset_name == "synthetic":
        return generate_multilabel_synthetic()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ========== Multi-Label Classifiers ========== #


class BinaryRelevanceClassifier:
    """Binary Relevance: Train one binary classifier per label.

    Advantages:
        - Simple, scalable
        - Can use any binary classifier
        - Parallelizable

    Disadvantages:
        - Ignores label correlations
    """

    def __init__(self, base_estimator=None):
        if base_estimator is None:
            self.base_estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        else:
            self.base_estimator = base_estimator
        self.classifiers: List[Any] = []
        self.label_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "BinaryRelevanceClassifier":
        """Fit one classifier per label."""
        self.label_names = y.columns.tolist()
        self.classifiers = []

        for label in self.label_names:
            clf = clone(self.base_estimator)
            clf.fit(X, y[label])
            self.classifiers.append(clf)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict all labels independently."""
        predictions = []
        for clf in self.classifiers:
            predictions.append(clf.predict(X))
        return np.column_stack(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for all labels."""
        probas = []
        for clf in self.classifiers:
            # Get probability of positive class
            proba = clf.predict_proba(X)[:, 1]
            probas.append(proba)
        return np.column_stack(probas)


class ClassifierChainsClassifier:
    """Classifier Chains: Chain classifiers to model label dependencies.

    Each classifier in the chain includes predictions of previous labels as features.

    Advantages:
        - Captures label correlations
        - Often higher accuracy than Binary Relevance

    Disadvantages:
        - Sensitive to label order
        - Error propagation down the chain
    """

    def __init__(self, base_estimator=None, random_order: bool = False):
        if base_estimator is None:
            self.base_estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        else:
            self.base_estimator = base_estimator
        self.random_order = random_order
        self.classifiers: List[Any] = []
        self.label_names: List[str] = []
        self.label_order: List[int] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "ClassifierChainsClassifier":
        """Fit classifiers in a chain."""
        self.label_names = y.columns.tolist()
        n_labels = len(self.label_names)

        # Determine label order
        if self.random_order:
            rng = np.random.default_rng(RANDOM_SEED)
            self.label_order = rng.permutation(n_labels).tolist()
        else:
            self.label_order = list(range(n_labels))

        self.classifiers = []
        X_augmented = X.copy()

        for idx in self.label_order:
            label = self.label_names[idx]
            clf = clone(self.base_estimator)
            clf.fit(X_augmented, y[label])
            self.classifiers.append(clf)

            # Add this label's predictions to features for next classifier
            X_augmented[f"_chain_{label}"] = y[label]

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict labels in chain order."""
        n_samples = X.shape[0]
        n_labels = len(self.label_names)
        predictions = np.zeros((n_samples, n_labels), dtype=int)

        X_augmented = X.copy()

        for i, idx in enumerate(self.label_order):
            clf = self.classifiers[i]
            pred = clf.predict(X_augmented)
            predictions[:, idx] = pred

            # Add prediction to features for next classifier
            label = self.label_names[idx]
            X_augmented[f"_chain_{label}"] = pred

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities in chain order."""
        n_samples = X.shape[0]
        n_labels = len(self.label_names)
        probas = np.zeros((n_samples, n_labels))

        X_augmented = X.copy()

        for i, idx in enumerate(self.label_order):
            clf = self.classifiers[i]
            proba = clf.predict_proba(X_augmented)[:, 1]
            probas[:, idx] = proba

            # Use hard predictions (not probabilities) for chaining
            pred = (proba > 0.5).astype(int)
            label = self.label_names[idx]
            X_augmented[f"_chain_{label}"] = pred

        return probas


class NativeMultiOutputClassifier:
    """Wrapper for scikit-learn classifiers with native multi-output support.

    Random Forest and Extra Trees support multi-output classification natively.
    """

    def __init__(self, base_estimator=None):
        if base_estimator is None:
            self.base_estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        else:
            self.base_estimator = base_estimator
        self.classifier: Any = None
        self.label_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "NativeMultiOutputClassifier":
        """Fit single multi-output classifier."""
        self.label_names = y.columns.tolist()
        self.classifier = clone(self.base_estimator)
        self.classifier.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict all labels simultaneously."""
        return self.classifier.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for all labels."""
        # Random Forest returns list of arrays (one per output)
        probas_list = self.classifier.predict_proba(X)
        probas = np.column_stack([p[:, 1] for p in probas_list])
        return probas


# ========== Metadata & Metrics ========== #


@dataclass
class MultiLabelMetadata:
    trained_at: str
    method: str  # binary_relevance, classifier_chains, native
    base_model: str
    n_features: int
    n_labels: int
    label_names: List[str]
    label_frequencies: Dict[str, float]
    label_cardinality: float  # Average labels per instance
    metrics: Optional[Dict[str, Any]] = None


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Dict[str, Any]:
    """Compute comprehensive multi-label metrics.

    Args:
        y_true: True binary labels (n_samples, n_labels)
        y_pred: Predicted binary labels (n_samples, n_labels)
        label_names: List of label names

    Returns:
        Dictionary with:
            - subset_accuracy: Exact match ratio
            - hamming_loss: Fraction of incorrect labels
            - micro_f1: Micro-averaged F1
            - macro_f1: Macro-averaged F1
            - per_label: Per-label precision, recall, F1
    """
    metrics = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-label metrics
    per_label = {}
    for i, label in enumerate(label_names):
        per_label[label] = {
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "support": int(y_true[:, i].sum()),
        }

    metrics["per_label"] = per_label
    return metrics


def analyze_label_distribution(y: pd.DataFrame) -> Dict[str, Any]:
    """Analyze multi-label distribution characteristics.

    Args:
        y: Binary label matrix

    Returns:
        Dictionary with label frequencies, cardinality, density
    """
    n_samples = len(y)
    label_frequencies = (y.sum(axis=0) / n_samples).to_dict()
    label_cardinality = float(y.sum(axis=1).mean())  # Avg labels per instance
    label_density = label_cardinality / y.shape[1]

    # Label co-occurrence (top 5 pairs)
    co_occurrence = y.T.dot(y).values
    np.fill_diagonal(co_occurrence, 0)
    label_names = y.columns.tolist()

    co_occur_pairs = []
    for i in range(len(label_names)):
        for j in range(i + 1, len(label_names)):
            co_occur_pairs.append(
                {
                    "label1": label_names[i],
                    "label2": label_names[j],
                    "count": int(co_occurrence[i, j]),
                }
            )

    co_occur_pairs = sorted(co_occur_pairs, key=lambda x: x["count"], reverse=True)[:5]

    return {
        "n_samples": n_samples,
        "n_labels": y.shape[1],
        "label_frequencies": label_frequencies,
        "label_cardinality": label_cardinality,
        "label_density": label_density,
        "top_co_occurrences": co_occur_pairs,
    }


# ========== Pipeline ========== #


class MultiLabelPipeline:
    """Multi-label classification pipeline."""

    def __init__(
        self,
        method: str = "binary_relevance",
        model: str = "rf",
        n_estimators: int = 100,
    ):
        """Initialize pipeline.

        Args:
            method: 'binary_relevance', 'classifier_chains', or 'native'
            model: 'rf' (RandomForest) - extensible to other models
            n_estimators: Number of trees
        """
        self.method = method
        self.model_name = model
        self.n_estimators = n_estimators
        self.classifier: Any = None
        self.metadata: Optional[MultiLabelMetadata] = None

    def _build_base_estimator(self):
        """Build base classifier."""
        if self.model_name == "rf":
            return RandomForestClassifier(n_estimators=self.n_estimators, random_state=RANDOM_SEED, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _build_classifier(self):
        """Build multi-label classifier."""
        base = self._build_base_estimator()

        if self.method == "binary_relevance":
            return BinaryRelevanceClassifier(base_estimator=base)
        elif self.method == "classifier_chains":
            return ClassifierChainsClassifier(base_estimator=base, random_order=False)
        elif self.method == "native":
            return NativeMultiOutputClassifier(base_estimator=base)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "MultiLabelPipeline":
        """Fit multi-label classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary label matrix (n_samples, n_labels)
        """
        self.classifier = self._build_classifier()
        self.classifier.fit(X, y)

        # Create metadata
        label_frequencies = (y.sum(axis=0) / len(y)).to_dict()
        label_cardinality = float(y.sum(axis=1).mean())

        self.metadata = MultiLabelMetadata(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            method=self.method,
            base_model=self.model_name,
            n_features=X.shape[1],
            n_labels=y.shape[1],
            label_names=y.columns.tolist(),
            label_frequencies=label_frequencies,
            label_cardinality=label_cardinality,
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary labels.

        Returns:
            Binary predictions (n_samples, n_labels)
        """
        if self.classifier is None:
            raise RuntimeError("Model not fitted")
        return self.classifier.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict label probabilities.

        Returns:
            Probabilities (n_samples, n_labels)
        """
        if self.classifier is None:
            raise RuntimeError("Model not fitted")
        return self.classifier.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate on test data.

        Args:
            X: Feature matrix
            y: True binary labels

        Returns:
            Dictionary of multi-label metrics
        """
        y_pred = self.predict(X)
        label_names = self.metadata.label_names if self.metadata else y.columns.tolist()
        metrics = compute_multilabel_metrics(y.values, y_pred, label_names)

        if self.metadata:
            self.metadata.metrics = metrics

        return metrics

    def save(self, path: Path):
        """Save pipeline to disk."""
        if self.classifier is None or self.metadata is None:
            raise RuntimeError("Nothing to save")

        joblib.dump({"classifier": self.classifier, "metadata": asdict(self.metadata)}, path)

    @staticmethod
    def load(path: Path) -> "MultiLabelPipeline":
        """Load pipeline from disk."""
        obj = joblib.load(path)
        metadata = obj["metadata"]

        pipeline = MultiLabelPipeline(
            method=metadata["method"],
            model=metadata["base_model"],
            n_estimators=100,  # Not stored in metadata
        )
        pipeline.classifier = obj["classifier"]
        pipeline.metadata = MultiLabelMetadata(**metadata)

        return pipeline


# ========== CLI Actions ========== #


def action_train(args):
    """Train multi-label classifier."""
    # Load data
    if args.dataset:
        X, y, label_names = load_dataset(args.dataset)
    elif args.train_features and args.train_labels:
        X = pd.read_csv(args.train_features)
        y = pd.read_csv(args.train_labels)
        label_names = y.columns.tolist()
    else:
        # Default to synthetic
        X, y, label_names = generate_multilabel_synthetic()

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.valid_ratio, random_state=RANDOM_SEED)

    # Train pipeline
    pipeline = MultiLabelPipeline(method=args.method, model=args.model, n_estimators=args.n_estimators)
    pipeline.fit(X_train, y_train)

    # Evaluate on validation
    metrics_val = pipeline.evaluate(X_val, y_val)

    # Save model
    if args.model_out:
        pipeline.save(Path(args.model_out))

    # Output JSON
    output = {
        "status": "trained",
        "method": args.method,
        "model": args.model,
        "dataset": args.dataset or "custom",
        "n_train": len(X_train),
        "n_val": len(X_val),
        "metrics_validation": metrics_val,
        "metadata": asdict(pipeline.metadata) if pipeline.metadata else None,
    }

    print(json.dumps(output, indent=2))


def action_evaluate(args):
    """Evaluate trained model."""
    # Load pipeline
    pipeline = MultiLabelPipeline.load(Path(args.model_path))

    # Load test data
    if args.dataset:
        X, y, _ = load_dataset(args.dataset)
    elif args.test_features and args.test_labels:
        X = pd.read_csv(args.test_features)
        y = pd.read_csv(args.test_labels)
    else:
        raise ValueError("Must provide --dataset or --test-features/--test-labels")

    # Evaluate
    metrics = pipeline.evaluate(X, y)

    # Output JSON
    output = {
        "status": "evaluated",
        "method": pipeline.method,
        "model": pipeline.model_name,
        "n_test": len(X),
        "metrics": metrics,
    }

    print(json.dumps(output, indent=2))


def action_predict(args):
    """Predict single instance."""
    # Load pipeline
    pipeline = MultiLabelPipeline.load(Path(args.model_path))

    # Parse input
    record = json.loads(args.input_json)
    X = pd.DataFrame([record])

    # Predict
    y_pred = pipeline.predict(X)[0]
    y_proba = pipeline.predict_proba(X)[0]

    # Format output
    label_names = pipeline.metadata.label_names if pipeline.metadata else []
    predicted_labels = [label_names[i] for i, val in enumerate(y_pred) if val == 1]

    label_scores = {label_names[i]: float(y_proba[i]) for i in range(len(label_names))}

    output = {
        "status": "predicted",
        "method": pipeline.method,
        "predicted_labels": predicted_labels,
        "label_scores": label_scores,
    }

    print(json.dumps(output, indent=2))


def action_analyze(args):
    """Analyze dataset label distribution."""
    # Load data
    X, y, label_names = load_dataset(args.dataset)

    # Analyze
    analysis = analyze_label_distribution(y)
    analysis["label_names"] = label_names

    # Output JSON
    output = {"status": "analyzed", "dataset": args.dataset, "analysis": analysis}

    print(json.dumps(output, indent=2))


# ========== CLI Parser ========== #


def build_parser():
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description="Module 4.3 Multi-Label Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = p.add_subparsers(dest="command", required=True)

    # ===== Train ===== #
    p_train = sub.add_parser("train", help="Train multi-label classifier")
    p_train.add_argument(
        "--method",
        choices=["binary_relevance", "classifier_chains", "native"],
        default="binary_relevance",
        help="Multi-label method",
    )
    p_train.add_argument("--model", choices=["rf"], default="rf", help="Base classifier")
    p_train.add_argument(
        "--dataset",
        choices=["steel_plates", "synthetic"],
        help="Dataset to use",
    )
    p_train.add_argument("--train-features", type=str, help="Training features CSV")
    p_train.add_argument("--train-labels", type=str, help="Training labels CSV")
    p_train.add_argument("--n-estimators", type=int, default=100)
    p_train.add_argument("--valid-ratio", type=float, default=0.2)
    p_train.add_argument("--model-out", type=str, help="Output model path")
    p_train.set_defaults(func=action_train)

    # ===== Evaluate ===== #
    p_eval = sub.add_parser("evaluate", help="Evaluate trained model")
    p_eval.add_argument("--model-path", required=True, help="Path to saved model")
    p_eval.add_argument(
        "--dataset",
        choices=["steel_plates", "synthetic"],
        help="Dataset to evaluate",
    )
    p_eval.add_argument("--test-features", type=str, help="Test features CSV")
    p_eval.add_argument("--test-labels", type=str, help="Test labels CSV")
    p_eval.set_defaults(func=action_evaluate)

    # ===== Predict ===== #
    p_pred = sub.add_parser("predict", help="Predict single instance")
    p_pred.add_argument("--model-path", required=True, help="Path to saved model")
    p_pred.add_argument(
        "--input-json",
        required=True,
        help="JSON record of features (e.g., '{\"feature_0\": 1.2, ...}')",
    )
    p_pred.set_defaults(func=action_predict)

    # ===== Analyze ===== #
    p_analyze = sub.add_parser("analyze", help="Analyze dataset label distribution")
    p_analyze.add_argument(
        "--dataset",
        required=True,
        choices=["steel_plates", "synthetic"],
        help="Dataset to analyze",
    )
    p_analyze.set_defaults(func=action_analyze)

    return p


def main():  # pragma: no cover
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
