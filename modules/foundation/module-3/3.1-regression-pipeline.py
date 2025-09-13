"""Production Regression Pipeline Script for Module 3.1

Provides a CLI to train, evaluate, and predict using a standardized regression pipeline
for semiconductor manufacturing datasets (synthetic or SECOM).

Features:
- Unified preprocessing: impute -> scale -> (optional) feature selection -> PCA -> model
- Supports Ridge, Lasso, ElasticNet, LinearRegression, RandomForest
- Manufacturing metrics: MAE, RMSE, R2, Prediction Within Spec (PWS), Estimated Loss
- Model persistence (save/load)
- Reproducibility via fixed random seed
- Synthetic data generators consistent with notebook

Example usage:
    python 3.1-regression-pipeline.py train --dataset synthetic_yield --model ridge --alpha 1.0 --save model.joblib
    python 3.1-regression-pipeline.py evaluate --model-path model.joblib --dataset synthetic_yield
    python 3.1-regression-pipeline.py predict --model-path model.joblib --input-json '{"temperature":455, "pressure":2.6, "flow":118, "time":62}'
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------- Synthetic Data Generators (subset) ---------------- #


def generate_yield_process(n=800, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temp = rng.normal(450, 15, n)
    pressure = rng.normal(2.5, 0.3, n)
    flow = rng.normal(120, 10, n)
    time = rng.normal(60, 5, n)
    noise = rng.normal(0, 3, n)
    yield_pct = (
        70
        + 0.05 * (temp - 450)
        - 1.5 * (pressure - 2.5) ** 2
        + 0.04 * flow
        + 0.2 * time
        + 0.0005 * (temp - 450) * (flow - 120)
        + noise
    )
    yield_pct = np.clip(yield_pct, 0, 100)
    df = pd.DataFrame(
        {
            "temperature": temp,
            "pressure": pressure,
            "flow": flow,
            "time": time,
            "yield_pct": yield_pct,
        }
    )
    # Feature engineering consistent with notebook
    df["temp_centered"] = df["temperature"] - df["temperature"].mean()
    df["pressure_sq"] = df["pressure"] ** 2
    df["flow_time_inter"] = df["flow"] * df["time"]
    df["temp_flow_inter"] = df["temperature"] * df["flow"]
    return df


# ---------------- Data Loading ---------------- #


def load_dataset(name: str) -> pd.DataFrame:
    if name == "synthetic_yield":
        return generate_yield_process()
    raise ValueError(f"Unknown dataset '{name}'. Currently supported: synthetic_yield")


TARGET_COLUMN = "yield_pct"

# ---------------- Pipeline Wrapper ---------------- #


@dataclass
class PipelineMetadata:
    trained_at: str
    model_type: str
    n_features_in: int
    n_components: int
    k_best: int
    params: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None


class RegressionPipeline:
    def __init__(
        self,
        model: str = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        k_best: int = 20,
        pca_components: float | int = 0.95,
        use_feature_selection: bool = True,
    ) -> None:
        # Configuration parameters
        self.model_name = model.lower()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.k_best = k_best
        self.pca_components = pca_components
        self.use_feature_selection = use_feature_selection
        # Runtime objects
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Optional[PipelineMetadata] = None

    def _build_model(self):
        if self.model_name == "ridge":
            return Ridge(alpha=self.alpha, random_state=RANDOM_SEED)
        if self.model_name == "lasso":
            return Lasso(alpha=self.alpha, random_state=RANDOM_SEED, max_iter=10000)
        if self.model_name == "elasticnet":
            return ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=RANDOM_SEED,
                max_iter=10000,
            )
        if self.model_name == "linear":
            return LinearRegression()
        if self.model_name == "rf":
            return RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1)
        raise ValueError(f"Unsupported model '{self.model_name}'")

    def build(self, n_features: int):
        steps = [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
        if self.use_feature_selection:
            steps.append(
                (
                    "select",
                    SelectKBest(score_func=f_regression, k=min(self.k_best, n_features)),
                )
            )
        steps.append(("pca", PCA(n_components=self.pca_components, random_state=RANDOM_SEED)))
        steps.append(("model", self._build_model()))
        self.pipeline = Pipeline(steps)
        return self

    def fit(self, X: pd.DataFrame, y: Any):
        if self.pipeline is None:
            self.build(X.shape[1])
        # Ensure y is ndarray
        y_arr = np.asarray(y)
        assert self.pipeline is not None
        self.pipeline.fit(X, y_arr)
        # Determine n_components real value
        pca_step = self.pipeline.named_steps["pca"]  # type: ignore[assignment]
        n_components_real = pca_step.n_components_ if hasattr(pca_step, "n_components_") else self.pca_components
        self.metadata = PipelineMetadata(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            model_type=type(self.pipeline.named_steps["model"]).__name__,  # type: ignore[index]
            n_features_in=X.shape[1],
            n_components=int(n_components_real),
            k_best=self.k_best if self.use_feature_selection else X.shape[1],
            params={
                "model": self.model_name,
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "k_best": self.k_best,
                "pca_components": self.pca_components,
                "use_feature_selection": self.use_feature_selection,
            },
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[override]
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted")
        pipeline = self.pipeline
        assert pipeline is not None
        preds = pipeline.predict(X)
        if isinstance(preds, tuple):  # safety for estimators returning (pred, var)
            preds = preds[0]
        return np.asarray(preds)

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tolerance: float = 2.0,
        spec_low: float = 60,
        spec_high: float = 100,
        cost_per_unit: float = 1.0,
    ) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        pws = ((y_pred >= spec_low) & (y_pred <= spec_high)).mean()
        loss_components = np.maximum(0, np.abs(y_true - y_pred) - tolerance)
        est_loss = float(np.sum(loss_components) * cost_per_unit)
        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "PWS": pws,
            "Estimated_Loss": est_loss,
        }

    def evaluate(self, X: pd.DataFrame, y: Any) -> Dict[str, float]:
        y_arr = np.asarray(y)
        preds = self.predict(X)
        metrics = self.compute_metrics(y_arr, preds)
        if self.metadata:
            self.metadata.metrics = metrics
        return metrics

    def save(self, path: Path):
        if self.pipeline is None or self.metadata is None:
            raise RuntimeError("Nothing to save; fit the pipeline first")
        joblib.dump({"pipeline": self.pipeline, "metadata": asdict(self.metadata)}, path)

    @staticmethod
    def load(path: Path) -> "RegressionPipeline":
        obj = joblib.load(path)
        inst = RegressionPipeline(model=obj["metadata"]["params"]["model"])
        inst.pipeline = obj["pipeline"]
        inst.metadata = PipelineMetadata(**obj["metadata"])
        return inst


# ---------------- CLI Actions ---------------- #


def action_train(args):
    df = load_dataset(args.dataset)
    y = df[TARGET_COLUMN].values
    X = df.drop(columns=[TARGET_COLUMN])

    pipeline = RegressionPipeline(
        model=args.model,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        k_best=args.k_best,
        pca_components=args.pca_components,
        use_feature_selection=not args.no_feature_selection,
    )
    pipeline.fit(X, y)
    metrics = pipeline.evaluate(X, y)  # train metrics (could add split later)
    if args.save:
        pipeline.save(Path(args.save))
    meta_dict = asdict(pipeline.metadata) if pipeline.metadata else None
    print(json.dumps({"status": "trained", "metrics": metrics, "metadata": meta_dict}, indent=2))


def action_evaluate(args):
    pipeline = RegressionPipeline.load(Path(args.model_path))
    df = load_dataset(args.dataset)
    y = df[TARGET_COLUMN].values
    X = df.drop(columns=[TARGET_COLUMN])
    metrics = pipeline.evaluate(X, y)
    meta_dict = asdict(pipeline.metadata) if pipeline.metadata else None
    print(json.dumps({"status": "evaluated", "metrics": metrics, "metadata": meta_dict}, indent=2))


def action_predict(args):
    pipeline = RegressionPipeline.load(Path(args.model_path))
    # Input can be JSON string or path
    if args.input_json:
        record = json.loads(args.input_json)
    elif args.input_file:
        record = json.loads(Path(args.input_file).read_text())
    else:
        raise ValueError("Provide --input-json or --input-file")

    df = pd.DataFrame([record])
    pred = pipeline.predict(df)[0]
    meta_dict = asdict(pipeline.metadata) if pipeline.metadata else None
    print(
        json.dumps(
            {"prediction": float(pred), "input": record, "model_meta": meta_dict},
            indent=2,
        )
    )


# ---------------- Argument Parsing ---------------- #


def build_parser():
    parser = argparse.ArgumentParser(description="Module 3.1 Regression Production Pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a regression pipeline")
    p_train.add_argument("--dataset", default="synthetic_yield", help="Dataset name")
    p_train.add_argument(
        "--model",
        default="ridge",
        choices=["ridge", "lasso", "elasticnet", "linear", "rf"],
    )
    p_train.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    p_train.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet l1_ratio")
    p_train.add_argument("--k-best", type=int, default=20, help="Number of K best features to select")
    p_train.add_argument(
        "--pca-components",
        type=float,
        default=0.95,
        help="PCA components (float for variance or int)",
    )
    p_train.add_argument(
        "--no-feature-selection",
        action="store_true",
        help="Disable SelectKBest feature selection",
    )
    p_train.add_argument("--save", help="Path to save trained model")
    p_train.set_defaults(func=action_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate an existing model")
    p_eval.add_argument("--model-path", required=True, help="Path to saved model")
    p_eval.add_argument("--dataset", default="synthetic_yield", help="Dataset name")
    p_eval.set_defaults(func=action_evaluate)

    p_pred = sub.add_parser("predict", help="Predict with an existing model")
    p_pred.add_argument("--model-path", required=True, help="Path to saved model")
    p_pred.add_argument("--input-json", help="Single JSON record string")
    p_pred.add_argument("--input-file", help="Path to JSON file containing a single record")
    p_pred.set_defaults(func=action_predict)

    return parser


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
