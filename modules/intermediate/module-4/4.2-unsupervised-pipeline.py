"""Production Unsupervised Learning Pipeline Script for Module 4.2

Provides a CLI to train, evaluate, and predict using clustering + anomaly detection
for semiconductor manufacturing feature matrices (synthetic teaching dataset or future real data).

Features:
- Flexible models: kmeans, gmm, dbscan, iso_forest (anomaly only)
    + Hybrid variant: kmeans_iso (kmeans + isolation forest)
- Optional PCA dimensionality reduction retaining specified variance
- Internal validation metrics: silhouette, calinski_harabasz, davies_bouldin
- Manufacturing diagnostics: cluster_size_entropy, anomaly_ratio,
    largest_cluster_fraction
- Guardrails/warnings for degenerate clustering & low structure
- JSON output for train/evaluate/predict
- Model persistence (save/load) with metadata
- Synthetic data generator (latent factors, drift, anomalies)

Example usage:
    python 4.2-unsupervised-pipeline.py train \
        --dataset synthetic_process --model kmeans --k 6 --save kmeans.joblib
    python 4.2-unsupervised-pipeline.py evaluate \
        --model-path kmeans.joblib --dataset synthetic_process
    python 4.2-unsupervised-pipeline.py predict \
        --model-path kmeans.joblib --input-json '{"f1":0.12, "f2":-1.3, ...}'

JSON output keys (train/evaluate):
  status, model, metrics{...}, metadata{...}, cluster_distribution, warnings[]

Limitations:
  * DBSCAN parameters are intentionally simple (eps,min_samples) for teaching.
  * GMM uses full covariance; could add options later.
  * IsolationForest used for anomaly scoring only; combined with clustering (> hybrid).

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------- Synthetic Data Generator ---------------- #


def generate_synthetic_process(
    n_samples: int = 1200,
    n_latent: int = 5,
    n_features: int = 18,
    drift_fraction: float = 0.15,
    anomaly_fraction: float = 0.03,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate semi-realistic semiconductor process feature matrix.

    Latent factors -> linear + mild nonlinear mixture -> observed features.
    Inject drift in subset + isolated anomalies.
    """
    rng = np.random.default_rng(seed)

    # Latent factors (e.g., tool health, ambient, recipe window,
    # contamination noise, shift)
    latent = rng.normal(0, 1, size=(n_samples, n_latent))

    # Mixing matrix
    mix_linear = rng.normal(0, 1, size=(n_latent, n_features))
    base = latent @ mix_linear

    # Mild nonlinear interaction: quadratic of first two latent factors
    if n_latent >= 2:
        interaction = 0.1 * (latent[:, 0:1] ** 2) - 0.05 * (latent[:, 1:2] * latent[:, 0:1])
        # Keep interaction as (n_samples, 1) to avoid broadcasting to (n_samples, n_samples)
        base[:, :1] += interaction

    # Add structured drift to a fraction of samples (simulate tool drift cluster)
    n_drift = int(drift_fraction * n_samples)
    drift_idx = rng.choice(n_samples, size=n_drift, replace=False)
    drift_shift = rng.normal(1.5, 0.2, size=(n_features,))
    base[drift_idx] += drift_shift  # global shift

    # Inject point anomalies
    n_anom = max(1, int(anomaly_fraction * n_samples))
    anom_idx = rng.choice(n_samples, size=n_anom, replace=False)
    base[anom_idx] += rng.normal(6.0, 1.5, size=(n_anom, n_features))

    # Scale features to moderate ranges
    base += rng.normal(0, 0.3, size=base.shape)
    df = pd.DataFrame(base, columns=[f"f{i+1}" for i in range(n_features)])
    # Initialize indicator columns; use list(index_array) for mypy/pylance compatibility
    df["is_drift"] = 0
    df.loc[list(drift_idx), "is_drift"] = 1  # type: ignore[index]
    df["is_injected_anomaly"] = 0
    df.loc[list(anom_idx), "is_injected_anomaly"] = 1  # type: ignore[index]
    return df


# ---------------- Metadata & Pipeline Classes ---------------- #


@dataclass
class UnsupervisedMetadata:
    trained_at: str
    model: str
    params: Dict[str, Any]
    n_features_in: int
    pca_variance: Optional[float]
    pca_components_: Optional[int]


class UnsupervisedPipeline:
    def __init__(
        self,
        model: str = "kmeans",
        n_clusters: int = 6,
        pca_variance: Optional[float] = 0.95,
        dbscan_eps: float = 0.9,
        dbscan_min_samples: int = 12,
        iso_estimators: int = 200,
        seed: int = RANDOM_SEED,
    ):
        self.model_name = model.lower()
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.iso_estimators = iso_estimators
        self.seed = seed

        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.cluster_model: Any = None
        self.iso_model: Optional[IsolationForest] = None
        self.metadata: Optional[UnsupervisedMetadata] = None
        self.feature_names: Optional[List[str]] = None

    # --------------- Internal Helpers --------------- #
    def _prepare_features(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        """Scale and optionally apply PCA to features.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature frame.
        fit : bool
            If True, fit scaler/PCA; otherwise transform using fitted state.

        Returns
        -------
        np.ndarray
            Transformed feature matrix.
        """
        arr = X.values
        if fit:
            self.scaler = StandardScaler()
            arr = self.scaler.fit_transform(arr)
            if self.pca_variance:
                self.pca = PCA(n_components=self.pca_variance, random_state=self.seed)
                arr = self.pca.fit_transform(arr)
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not fitted")
            arr = self.scaler.transform(arr)
            if self.pca is not None:
                arr = self.pca.transform(arr)
        return arr

    def _build_model(self):
        if self.model_name == "kmeans":
            return KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init="auto")
        if self.model_name == "gmm":
            return GaussianMixture(n_components=self.n_clusters, random_state=self.seed)
        if self.model_name == "dbscan":
            return DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        if self.model_name == "iso_forest":
            return IsolationForest(
                n_estimators=self.iso_estimators,
                random_state=self.seed,
                contamination="auto",
            )
        if self.model_name == "kmeans_iso":  # hybrid pattern
            return KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init="auto")
        raise ValueError(f"Unsupported model: {self.model_name}")

    # --------------- Core API --------------- #
    def fit(self, X: pd.DataFrame) -> "UnsupervisedPipeline":
        """Fit the unsupervised pipeline on features `X`.

        Returns the pipeline instance for chaining.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names = list(X.columns)
        Z = self._prepare_features(X, fit=True)
        self.cluster_model = self._build_model()
        model_type = self.model_name

        if self.model_name == "iso_forest":
            # isolation forest: clusters not meaningful; just fit
            self.cluster_model.fit(Z)
        else:
            self.cluster_model.fit(Z)
            # Obtain labels to validate clustering path (discard variable)
            if hasattr(self.cluster_model, "predict"):
                _ = self.cluster_model.predict(Z)
            elif hasattr(self.cluster_model, "labels_"):
                _ = self.cluster_model.labels_  # DBSCAN
            else:
                raise RuntimeError("Unable to obtain cluster labels")

        # Hybrid anomaly scoring
        if self.model_name in {"kmeans_iso", "kmeans"}:
            self.iso_model = IsolationForest(
                n_estimators=self.iso_estimators,
                random_state=self.seed,
                contamination="auto",
            )
            self.iso_model.fit(Z)

        self.metadata = UnsupervisedMetadata(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            model=model_type,
            params={
                "n_clusters": self.n_clusters,
                "pca_variance": self.pca_variance,
                "dbscan_eps": self.dbscan_eps,
                "dbscan_min_samples": self.dbscan_min_samples,
                "iso_estimators": self.iso_estimators,
            },
            n_features_in=len(self.feature_names),
            pca_variance=self.pca_variance,
            pca_components_=None if self.pca is None else int(self.pca.n_components_),
        )
        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predict cluster labels and optional anomaly flags for `X`."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        Z = self._prepare_features(X, fit=False)
        if self.cluster_model is None:
            raise RuntimeError("Model not fitted")

        if self.model_name == "iso_forest":
            # anomaly scores only
            scores = self.cluster_model.score_samples(Z)
            anomaly_flag = scores < np.percentile(scores, 5)
            return {
                "anomaly_scores": scores.tolist(),
                "anomaly_flag": anomaly_flag.astype(int).tolist(),
            }

        if hasattr(self.cluster_model, "predict"):
            labels = self.cluster_model.predict(Z)
        elif hasattr(self.cluster_model, "labels_"):
            labels = self.cluster_model.fit_predict(Z)  # DBSCAN may recompute
        else:
            raise RuntimeError("Unable to get labels in predict")

        out: Dict[str, Any] = {"labels": labels.tolist()}

        if self.iso_model is not None:
            iso_scores = self.iso_model.score_samples(Z)
            iso_flag = iso_scores < np.percentile(iso_scores, 5)
            out["anomaly_scores"] = iso_scores.tolist()
            out["anomaly_flag"] = iso_flag.astype(int).tolist()
        return out

    # --------------- Metrics & Evaluation --------------- #
    def _cluster_metrics(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        Z: np.ndarray,
    ) -> Dict[str, float]:
        """Compute clustering metrics and manufacturing diagnostics."""
        metrics: Dict[str, float] = {}
        unique = np.unique(labels)
        valid_cluster_labels = unique[unique != -1]
        n_effective = len(valid_cluster_labels)

        if n_effective >= 2 and len(Z) > n_effective:
            try:
                metrics["silhouette"] = float(silhouette_score(Z, labels))
            except Exception:
                metrics["silhouette"] = float("nan")
            try:
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(Z, labels))
            except Exception:
                metrics["calinski_harabasz"] = float("nan")
            try:
                metrics["davies_bouldin"] = float(davies_bouldin_score(Z, labels))
            except Exception:
                metrics["davies_bouldin"] = float("nan")
        else:
            metrics["silhouette"] = float("nan")
            metrics["calinski_harabasz"] = float("nan")
            metrics["davies_bouldin"] = float("nan")

        # Cluster size distribution (excluding noise = -1)
        if n_effective > 0:
            counts = np.array([(labels == c).sum() for c in valid_cluster_labels], dtype=float)
            p = counts / counts.sum()
            entropy = -np.sum(p * np.log(p + 1e-12)) / math.log(len(p))  # normalized 0..1
            metrics["cluster_size_entropy"] = float(entropy)
            metrics["largest_cluster_fraction"] = float(counts.max() / counts.sum())
        else:
            metrics["cluster_size_entropy"] = float("nan")
            metrics["largest_cluster_fraction"] = float("nan")

        # Anomaly ratio
        if -1 in unique:
            metrics["anomaly_ratio"] = float((labels == -1).mean())
        else:
            metrics["anomaly_ratio"] = 0.0
        return metrics

    def evaluate(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate the fitted model on `X` and return metrics + warnings."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        Z = self._prepare_features(X, fit=False)
        if self.cluster_model is None:
            raise RuntimeError("Model not fitted")

        if self.model_name == "iso_forest":
            scores = self.cluster_model.score_samples(Z)
            anomaly_flag = scores < np.percentile(scores, 5)
            metrics = {
                "anomaly_ratio": float(anomaly_flag.mean()),
            }
            warnings: List[str] = []
        else:
            if hasattr(self.cluster_model, "predict"):
                labels = self.cluster_model.predict(Z)
            elif hasattr(self.cluster_model, "labels_"):
                labels = self.cluster_model.fit_predict(Z)
            else:
                raise RuntimeError("Unable to compute labels for evaluation")
            metrics = self._cluster_metrics(X, labels, Z)
            warnings = []
            lcf = metrics.get("largest_cluster_fraction")
            if lcf is not None and not math.isnan(lcf) and lcf > 0.85:
                warnings.append("degenerate_cluster:largest_frac>0.85")
            sil = metrics.get("silhouette")
            if sil is not None and not math.isnan(sil) and sil < 0.05:
                warnings.append("low_structure:silhouette_below_0.05")
            anom = metrics.get("anomaly_ratio")
            if anom is not None and not math.isnan(anom) and anom > 0.15:
                warnings.append("high_anomaly_ratio:possible_global_drift")

        return {"metrics": metrics, "warnings": warnings}

    # --------------- Persistence --------------- #
    def save(self, path: Path) -> None:
        """Persist the trained model, scaler, PCA, and metadata to `path`."""
        if self.cluster_model is None or self.metadata is None:
            raise RuntimeError("Nothing to save")
        joblib.dump(
            {
                "model_name": self.model_name,
                "cluster_model": self.cluster_model,
                "iso_model": self.iso_model,
                "scaler": self.scaler,
                "pca": self.pca,
                "metadata": asdict(self.metadata),
            },
            path,
        )

    @staticmethod
    def load(path: Path) -> "UnsupervisedPipeline":
        """Load a persisted pipeline from `path`."""
        obj = joblib.load(path)
        pipe = UnsupervisedPipeline(model=obj["model_name"])
        pipe.cluster_model = obj["cluster_model"]
        pipe.iso_model = obj.get("iso_model")
        pipe.scaler = obj.get("scaler")
        pipe.pca = obj.get("pca")
        md = obj.get("metadata")
        if md:
            pipe.metadata = UnsupervisedMetadata(**md)
        return pipe


# ---------------- CLI Actions ---------------- #


def _load_dataset(name: str) -> pd.DataFrame:
    if name == "synthetic_process":
        return generate_synthetic_process()
    raise ValueError(f"Unknown dataset: {name}")


def action_train(args):
    df = _load_dataset(args.dataset)
    feature_cols = [c for c in df.columns if not c.startswith("is_")]
    X = df[feature_cols]
    pipe = UnsupervisedPipeline(
        model=args.model,
        n_clusters=args.k,
        pca_variance=None if args.no_pca else args.pca_variance,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        iso_estimators=args.iso_estimators,
    )
    pipe.fit(X)
    eval_out = pipe.evaluate(X)
    metrics = eval_out["metrics"]
    warnings = eval_out["warnings"]

    cluster_distribution = None
    if pipe.model_name != "iso_forest":
        pred = pipe.predict(X)
        labels = pred["labels"] if "labels" in pred else []
        if labels:
            s = pd.Series(labels)
            counts = s.value_counts(normalize=True).sort_index()
            # Cast cluster labels to string keys; avoid int() cast.
            # (k already numeric)
            cluster_distribution = {str(k): float(v) for k, v in counts.items()}

    if args.save:
        pipe.save(Path(args.save))

    print(
        json.dumps(
            {
                "status": "trained",
                "model": pipe.model_name,
                "metrics": metrics,
                "metadata": asdict(pipe.metadata) if pipe.metadata else None,
                "cluster_distribution": cluster_distribution,
                "warnings": warnings,
            },
            indent=2,
        )
    )


def action_evaluate(args):
    pipe = UnsupervisedPipeline.load(Path(args.model_path))
    df = _load_dataset(args.dataset)
    feature_cols = [c for c in df.columns if not c.startswith("is_")]
    X = df[feature_cols]
    eval_out = pipe.evaluate(X)
    metrics = eval_out["metrics"]
    warnings = eval_out["warnings"]
    print(
        json.dumps(
            {
                "status": "evaluated",
                "model": pipe.model_name,
                "metrics": metrics,
                "warnings": warnings,
                "metadata": asdict(pipe.metadata) if pipe.metadata else None,
            },
            indent=2,
        )
    )


def action_predict(args):
    pipe = UnsupervisedPipeline.load(Path(args.model_path))
    if args.input_json is None and args.input_file is None:
        raise SystemExit("Provide --input-json or --input-file")
    if args.input_json:
        record = json.loads(args.input_json)
    else:
        with open(args.input_file, "r", encoding="utf-8") as f:
            record = json.load(f)
    X = pd.DataFrame([record])
    out = pipe.predict(X)
    print(
        json.dumps(
            {
                "status": "predicted",
                "model": pipe.model_name,
                "output": out,
            },
            indent=2,
        )
    )


# ---------------- Argument Parsing ---------------- #


def build_parser():
    """Build the CLI argument parser for the pipeline."""
    parser = argparse.ArgumentParser(description="Module 4.2 Unsupervised Learning Pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train unsupervised pipeline")
    p_train.add_argument("--dataset", default="synthetic_process")
    p_train.add_argument(
        "--model",
        default="kmeans",
        choices=["kmeans", "gmm", "dbscan", "iso_forest", "kmeans_iso"],
    )
    p_train.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of clusters for kmeans/gmm",
    )
    p_train.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="Retained variance for PCA (0-1)",
    )
    p_train.add_argument("--no-pca", action="store_true", help="Disable PCA")
    p_train.add_argument("--dbscan-eps", type=float, default=0.9)
    p_train.add_argument("--dbscan-min-samples", type=int, default=12)
    p_train.add_argument("--iso-estimators", type=int, default=200)
    p_train.add_argument("--save", help="Path to save trained model")
    p_train.set_defaults(func=action_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate an unsupervised model")
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument("--dataset", default="synthetic_process")
    p_eval.set_defaults(func=action_evaluate)

    p_pred = sub.add_parser(
        "predict",
        help="Predict cluster / anomaly for a record",
    )
    p_pred.add_argument("--model-path", required=True)
    p_pred.add_argument("--input-json", help="Single JSON record string")
    p_pred.add_argument("--input-file", help="Path to JSON file containing a single record")
    p_pred.set_defaults(func=action_predict)

    return parser


def main(argv: Optional[List[str]] = None):
    """Entry point for CLI invocations."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
