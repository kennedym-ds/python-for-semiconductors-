"""Production Classification Pipeline Script for Module 3.2

Provides a CLI to train, evaluate, and predict using a standardized classification pipeline
for semiconductor manufacturing datasets (synthetic excursion detection example).

Features:
- Unified preprocessing: impute -> scale -> (optional) sampler -> model
- Supports logistic, linear SVM, tree, random forest, gradient boosting
- Imbalance handling via class weights or (optional) SMOTE (if imbalanced-learn available)
- Metrics: ROC AUC, PR AUC (average precision), MCC, balanced accuracy, precision/recall/F1 at chosen threshold
- Threshold optimization to satisfy minimum precision or recall constraint
- Model persistence (save/load) with metadata (including threshold)

Example usage:
    python 3.2-classification-pipeline.py train --dataset synthetic_events --model logistic --min-precision 0.9 --save clf.joblib
    python 3.2-classification-pipeline.py evaluate --model-path clf.joblib --dataset synthetic_events
    python 3.2-classification-pipeline.py predict --model-path clf.joblib --input-json '{"temperature":455, "pressure":2.6, "flow":118, "time":62}'
"""

# ---------------------------------------------------------------------------
# Future Enhancements (Module 3.2 -> 3.3 bridge)
# ---------------------------------------------------------------------------
# 1. Probability Calibration: integrate optional Platt scaling / Isotonic after
#    base model fit (sklearn.calibration.CalibratedClassifierCV) with metadata
#    capturing calibration method + CV strategy.
# 2. Cost-Sensitive Thresholding: add utility function U = TP*benefit - FP*cost
#    and optimize threshold to maximize expected utility vs. fixed precision/recall
#    constraints. Provide CLI flags: --fp-cost, --tp-benefit, --optimize-utility.
# 3. Drift Monitoring Hooks: persist training feature distribution summary
#    (mean/std, quantiles) enabling later PSI / KS statistic computation.
# 4. Explainability: optional SHAP / permutation importance computation stored
#    in metadata['feature_importance'] (guard behind lightweight flag to avoid
#    latency in production path).
# 5. Sampling Strategy Abstraction: allow choosing among SMOTE, BorderlineSMOTE,
#    ADASYN, or None via --sampler argument (current: boolean SMOTE toggle).
# 6. Evaluation Split: introduce automatic train/validation split or CV metrics
#    (currently metrics are on training set for simplicity in the teaching phase).
# 7. Logging: integrate structured logging (JSON) for pipeline steps and timing.
# 8. Model Card Generation: auto-produce markdown summarizing data, metrics,
#    threshold rationale, calibration, and limitations.
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

try:  # optional dependency
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    from imblearn.over_sampling import SMOTE  # type: ignore

    IMB_AVAILABLE = True
except Exception:  # pragma: no cover
    IMB_AVAILABLE = False
    ImbPipeline = Pipeline  # fallback alias
    SMOTE = None  # type: ignore


# ---------------- Synthetic Data Generator ---------------- #


def generate_synthetic_events(
    n: int = 1200, minority_frac: float = 0.08, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temp = rng.normal(450, 15, n)
    pressure = rng.normal(2.5, 0.3, n)
    flow = rng.normal(120, 10, n)
    time = rng.normal(60, 5, n)
    interaction = 0.001 * (temp - 450) * (flow - 120)
    score = (
        0.04 * (temp - 450)
        - 1.2 * (pressure - 2.5) ** 2
        + 0.03 * flow
        + 0.15 * time
        + interaction
    )
    cutoff = np.quantile(score, 1 - minority_frac)
    y = (score >= cutoff).astype(int)
    df = pd.DataFrame(
        {
            "temperature": temp,
            "pressure": pressure,
            "flow": flow,
            "time": time,
            "rare_event": y,
        }
    )
    # feature engineering consistent with notebook
    df["temp_centered"] = df["temperature"] - df["temperature"].mean()
    df["pressure_sq"] = df["pressure"] ** 2
    df["flow_time_inter"] = df["flow"] * df["time"]
    df["temp_flow_inter"] = df["temperature"] * df["flow"]
    return df


def load_dataset(name: str) -> pd.DataFrame:
    if name == "synthetic_events":
        return generate_synthetic_events()
    raise ValueError(f"Unknown dataset '{name}'. Supported: synthetic_events")


TARGET_COLUMN = "rare_event"


# ---------------- Metadata & Pipeline ---------------- #


@dataclass
class ClassificationMetadata:
    trained_at: str
    model_type: str
    n_features_in: int
    sampler: Optional[str]
    params: Dict[str, Any]
    threshold: float
    metrics: Optional[Dict[str, float]] = None


class ClassificationPipeline:
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
        # Accept only allowed literals to satisfy static typing complaints
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
        self.metadata: Optional[ClassificationMetadata] = None
        self.fitted_threshold: float = 0.5

    def _build_estimator(self):
        if self.model_name == "logistic":
            return LogisticRegression(
                max_iter=1000,
                class_weight=self.class_weight_mode,
                C=self.C,
                random_state=RANDOM_SEED,
            )
        if self.model_name == "linear_svm":
            # LinearSVC does not output probability; we will map decision_function through sigmoid-like scaling if needed
            return LinearSVC(
                class_weight=self.class_weight_mode, C=self.C, random_state=RANDOM_SEED
            )
        if self.model_name == "tree":
            return DecisionTreeClassifier(
                max_depth=self.max_depth,
                class_weight=self.class_weight_mode,
                random_state=RANDOM_SEED,
            )
        if self.model_name == "rf":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight=self.class_weight_mode,  # type: ignore[arg-type]
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        if self.model_name == "gb":
            return HistGradientBoostingClassifier(
                max_depth=self.max_depth, learning_rate=0.1, random_state=RANDOM_SEED
            )
        raise ValueError(f"Unsupported model '{self.model_name}'")

    def build(self):
        steps: List[Tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
        sampler_name = None
        if self.use_smote:
            if not IMB_AVAILABLE:
                raise RuntimeError("SMOTE requested but imbalanced-learn not installed")
            steps.append(("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=self.smote_k_neighbors)))  # type: ignore[arg-type]
            sampler_name = "SMOTE"
        steps.append(("model", self._build_estimator()))
        PipelineCls = ImbPipeline if self.use_smote else Pipeline
        self.pipeline = PipelineCls(steps)
        return sampler_name

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        sampler_name = self.build()
        assert self.pipeline is not None
        self.pipeline.fit(X, y)
        # Determine threshold via constraint if specified (only if probability available)
        if hasattr(self.pipeline.named_steps["model"], "predict_proba"):
            probs = self.pipeline.predict_proba(X)[:, 1]
        else:
            # decision_function fallback scaled to [0,1] via logistic mapping
            decision = self.pipeline.named_steps["model"].decision_function(X)  # type: ignore[index]
            probs = 1 / (1 + np.exp(-decision))

        self.fitted_threshold = 0.5
        if self.min_precision or self.min_recall:
            prec, rec, thr = precision_recall_curve(y, probs)
            # precision_recall_curve returns thresholds shorter by 1 element
            thr_full = np.append(thr, 1.0)
            candidate_idx = []
            for i, (p, r) in enumerate(zip(prec, rec)):
                if self.min_precision and p < self.min_precision:
                    continue
                if self.min_recall and r < self.min_recall:
                    continue
                candidate_idx.append(i)
            if candidate_idx:
                # choose threshold giving highest F1 among feasible (scalar calc)
                def f1_at(i_: int) -> float:
                    return float((2 * prec[i_] * rec[i_]) / (prec[i_] + rec[i_] + 1e-9))

                best_i = max(candidate_idx, key=lambda idx: f1_at(idx))
                self.fitted_threshold = float(thr_full[best_i])

        self.metadata = ClassificationMetadata(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            model_type=type(self.pipeline.named_steps["model"]).__name__,  # type: ignore[index]
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

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted")
        model = self.pipeline.named_steps["model"]  # type: ignore[index]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(self.pipeline[:-1].transform(X))  # type: ignore[index]
        # decision function fallback
        decision = model.decision_function(self.pipeline[:-1].transform(X))  # type: ignore[index]
        probs = 1 / (1 + np.exp(-decision))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[override]
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.fitted_threshold).astype(int)

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray
    ) -> Dict[str, float]:
        return {
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "pr_auc": float(average_precision_score(y_true, probs)),
            "mcc": float(matthews_corrcoef(y_true, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
        }

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        probs = self.predict_proba(X)[:, 1]
        preds = (probs >= self.fitted_threshold).astype(int)
        metrics = self.compute_metrics(y, probs, preds)
        if self.metadata:
            self.metadata.metrics = metrics
        return metrics

    def save(self, path: Path):
        if self.pipeline is None or self.metadata is None:
            raise RuntimeError("Nothing to save")
        joblib.dump(
            {"pipeline": self.pipeline, "metadata": asdict(self.metadata)}, path
        )

    @staticmethod
    def load(path: Path) -> "ClassificationPipeline":
        obj = joblib.load(path)
        inst = ClassificationPipeline(model=obj["metadata"]["params"]["model"])
        inst.pipeline = obj["pipeline"]
        inst.metadata = ClassificationMetadata(**obj["metadata"])
        inst.fitted_threshold = inst.metadata.threshold
        return inst


# ---------------- CLI Actions ---------------- #


def action_train(args):
    df = load_dataset(args.dataset)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    X = df.drop(columns=[TARGET_COLUMN])
    pipe = ClassificationPipeline(
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
    metrics = pipe.evaluate(X, y)  # train metrics; future improvement: holdout split
    if args.save:
        pipe.save(Path(args.save))
    meta_dict = asdict(pipe.metadata) if pipe.metadata else None
    print(
        json.dumps(
            {"status": "trained", "metrics": metrics, "metadata": meta_dict}, indent=2
        )
    )


def action_evaluate(args):
    pipe = ClassificationPipeline.load(Path(args.model_path))
    df = load_dataset(args.dataset)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    X = df.drop(columns=[TARGET_COLUMN])
    metrics = pipe.evaluate(X, y)
    meta_dict = asdict(pipe.metadata) if pipe.metadata else None
    print(
        json.dumps(
            {"status": "evaluated", "metrics": metrics, "metadata": meta_dict}, indent=2
        )
    )


def action_predict(args):
    pipe = ClassificationPipeline.load(Path(args.model_path))
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


def build_parser():
    p = argparse.ArgumentParser(
        description="Module 3.2 Classification Production Pipeline CLI"
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a classification pipeline")
    p_train.add_argument("--dataset", default="synthetic_events")
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

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument("--dataset", default="synthetic_events")
    p_eval.set_defaults(func=action_evaluate)

    p_pred = sub.add_parser("predict", help="Predict with a saved model")
    p_pred.add_argument("--model-path", required=True)
    p_pred.add_argument("--input-json")
    p_pred.add_argument("--input-file")
    p_pred.set_defaults(func=action_predict)

    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
