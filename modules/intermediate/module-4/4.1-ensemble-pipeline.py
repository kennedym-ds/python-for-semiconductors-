"""Module 4.1 Ensemble Regression Pipeline

Educational, lightweight implementation following project conventions.

CLI Subcommands (train/evaluate/predict) produce JSON for automation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
TARGET_COLUMN = 'target'

try:  # optional (xgboost)
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False
    xgb = None  # type: ignore

try:  # optional
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:  # pragma: no cover
    HAS_LGB = False
    lgb = None  # type: ignore


def generate_regression_synthetic(
    n: int = 500,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(4, 1.5, n)
    x3 = rng.uniform(-2, 2, n)
    noise = rng.normal(0, 0.5, n)
    y = 2.5 + 1.6 * x1 - 0.6 * x2 + 0.45 * x1 * x3 + np.sin(x2) + noise
    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x1_x3': x1 * x3,
        'sin_x2': np.sin(x2),
        TARGET_COLUMN: y,
    })


@dataclass
class EnsembleMetadata:
    trained_at: str
    model_type: str
    params: Dict[str, Any]
    n_features_in: int
    metrics: Optional[Dict[str, float]] = None


class EnsemblePipeline:
    def __init__(
        self,
        model: str = 'rf',
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.1,
    ):
        self.model_name = model.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Any = None
        self.metadata: Optional[EnsembleMetadata] = None

    def _build(self):
        if self.model_name == 'rf':
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        if self.model_name == 'hist_gb':
            return HistGradientBoostingRegressor(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=RANDOM_SEED,
            )
        if self.model_name == 'xgb':
            if not HAS_XGB:
                raise RuntimeError('xgboost not installed')
            import xgboost as _xgb  # type: ignore
            return _xgb.XGBRegressor(  # type: ignore[attr-defined]
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=RANDOM_SEED,
                tree_method='hist',
            )
        if self.model_name == 'lgbm':
            if not HAS_LGB:
                raise RuntimeError('lightgbm not installed')
            import lightgbm as _lgb  # type: ignore
            return _lgb.LGBMRegressor(  # type: ignore[attr-defined]
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=RANDOM_SEED,
            )
        raise ValueError(f'Unknown model {self.model_name}')

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model = self._build()
        self.model.fit(X, y)
        self.metadata = EnsembleMetadata(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            model_type=type(self.model).__name__,
            params={
                'model': self.model_name,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
            },
            n_features_in=X.shape[1],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('Model not fitted')
        return self.model.predict(X)

    @staticmethod
    def compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        return {
            'R2': float(r2_score(y_true, preds)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, preds))),
            'MAE': float(mean_absolute_error(y_true, preds)),
        }

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        metrics = self.compute_metrics(y, self.predict(X))
        if self.metadata:
            self.metadata.metrics = metrics
        return metrics

    def save(self, path: Path):
        if self.model is None or self.metadata is None:
            raise RuntimeError('Nothing to save')
        joblib.dump({'model': self.model, 'metadata': asdict(self.metadata)}, path)

    @staticmethod
    def load(path: Path) -> 'EnsemblePipeline':
        obj = joblib.load(path)
        inst = EnsemblePipeline(model=obj['metadata']['params']['model'])
        inst.model = obj['model']
        inst.metadata = EnsembleMetadata(**obj['metadata'])
        return inst


# ---------------- CLI ---------------- #
def action_train(args):
    if args.train:
        df = (
            pd.read_parquet(args.train)
            if args.train.endswith('.parquet')
            else pd.read_csv(args.train)
        )
    else:
        df = generate_regression_synthetic()
    if args.target not in df.columns:
        _msg = json.dumps({'error': f'target {args.target} missing'})
        raise SystemExit(_msg)
    y = df[args.target].to_numpy()
    X = df.drop(columns=[args.target])
    cut = int(0.8 * len(df)) if args.valid_ratio else int(0.8 * len(df))
    X_train, X_val = X.iloc[:cut], X.iloc[cut:]
    y_train, y_val = y[:cut], y[cut:]
    pipe = EnsemblePipeline(
        model=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    ).fit(X_train, y_train)
    metrics_val = pipe.evaluate(X_val, y_val)
    if args.model_out:
        pipe.save(Path(args.model_out))
    out = {
        'status': 'trained',
        'model': args.model,
        'metrics_validation': metrics_val,
        'metadata': asdict(pipe.metadata) if pipe.metadata else None,
    }
    print(json.dumps(out))


def action_evaluate(args):
    pipe = EnsemblePipeline.load(Path(args.model_path))
    df = (
        pd.read_parquet(args.data)
        if args.data.endswith('.parquet')
        else pd.read_csv(args.data)
    )
    if args.target not in df.columns:
        _msg = json.dumps({'error': f'target {args.target} missing'})
        raise SystemExit(_msg)
    y = df[args.target].to_numpy()
    X = df.drop(columns=[args.target])
    metrics = pipe.evaluate(X, y)
    model_name = (
        pipe.metadata.params['model'] if pipe.metadata else 'unknown'
    )
    print(
        json.dumps(
            {
                'status': 'evaluated',
                'model': model_name,
                'metrics': metrics,
            }
        )
    )


def action_predict(args):
    pipe = EnsemblePipeline.load(Path(args.model_path))
    record = json.loads(args.input_json)
    X = pd.DataFrame([record])
    preds = pipe.predict(X)
    model_name = (
        pipe.metadata.params['model'] if pipe.metadata else 'unknown'
    )
    print(
        json.dumps(
            {
                'status': 'predicted',
                'model': model_name,
                'prediction': float(preds[0]),
            }
        )
    )


def build_parser():
    p = argparse.ArgumentParser(
        description='Module 4.1 Ensemble Regression'
    )
    sub = p.add_subparsers(dest='command', required=True)

    p_train = sub.add_parser('train', help='Train a model')
    p_train.add_argument(
        '--model',
        default='rf',
        choices=['rf', 'hist_gb', 'xgb', 'lgbm'],
    )
    p_train.add_argument(
        '--train',
        type=str,
        help='Training data (csv or parquet)',
    )
    p_train.add_argument('--target', type=str, default=TARGET_COLUMN)
    p_train.add_argument('--n-estimators', type=int, default=300)
    p_train.add_argument('--max-depth', type=int, default=8)
    p_train.add_argument('--valid-ratio', type=float, default=0.2)
    p_train.add_argument('--model-out', type=str)
    p_train.set_defaults(func=action_train)

    p_eval = sub.add_parser('evaluate', help='Evaluate model')
    p_eval.add_argument('--model-path', required=True)
    p_eval.add_argument('--data', required=True)
    p_eval.add_argument('--target', type=str, default=TARGET_COLUMN)
    p_eval.set_defaults(func=action_evaluate)

    p_pred = sub.add_parser('predict', help='Make predictions')
    p_pred.add_argument('--model-path', required=True)
    p_pred.add_argument(
        '--input-json',
        required=True,
        help='Single JSON record of features',
    )
    p_pred.set_defaults(func=action_predict)
    return p


def main():  # pragma: no cover
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':  # pragma: no cover
    main()
