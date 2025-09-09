"""Module 4.1 Ensemble Pipeline (skeleton) - will implement RandomForest, HistGradientBoosting, optional XGBoost/LightGBM.

Planned Features:
- Unified regression interface (fit, predict, evaluate, save, load)
- Model choices: rf | hist_gb | xgb | lgbm
- Metrics: R2, RMSE, MAE
- Optional randomized hyperparam search
- Feature importance extraction (gain-based if available)
- Reproducibility via fixed random seed
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

try:  # Optional imports
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False
    xgb = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:  # pragma: no cover
    HAS_LGB = False
    lgb = None  # type: ignore

TARGET_COLUMN = 'target'

def generate_regression_synthetic(n: int = 800, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0,1,n)
    x2 = rng.normal(5,2,n)
    x3 = rng.uniform(-2,2,n)
    noise = rng.normal(0,0.5,n)
    y = 3 + 2*x1 - 0.7*x2 + 0.5*x1*x3 + np.sin(x2) + noise
    df = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3,'x1_x3':x1*x3,'sin_x2':np.sin(x2),'target':y})
    return df

@dataclass
class EnsembleMetadata:
    trained_at: str
    model_type: str
    params: Dict[str, Any]
    n_features_in: int
    metrics: Optional[Dict[str, float]] = None

class EnsemblePipeline:
    def __init__(self, model: str = 'rf', n_estimators: int = 400, max_depth: int = 8,
                 learning_rate: float = 0.1, subsample: float = 1.0, colsample: float = 1.0):
        self.model_name = model.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample = colsample
        self.model = None
        self.metadata: Optional[EnsembleMetadata] = None
        self.feature_names: Optional[List[str]] = None
        self._train_X_sample: Optional[pd.DataFrame] = None
        self._train_y_sample: Optional[np.ndarray] = None

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
            if HAS_XGB:
                import xgboost as _xgb  # type: ignore
                return _xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample,
                    random_state=RANDOM_SEED,
                    tree_method='hist',
                )  # type: ignore[attr-defined]
            raise RuntimeError('xgboost not available')
        if self.model_name == 'lgbm':
            if HAS_LGB:
                import lightgbm as _lgb  # type: ignore
                return _lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample,
                    random_state=RANDOM_SEED,
                )  # type: ignore[attr-defined]
            raise RuntimeError('lightgbm not available')
        raise ValueError(f'Unknown model {self.model_name}')

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model = self._build()
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        sample_n = min(256, len(X))
        self._train_X_sample = X.iloc[:sample_n].copy()
        self._train_y_sample = y[:sample_n].copy()
        self.metadata = EnsembleMetadata(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            model_type=type(self.model).__name__,
            params={
                'model': self.model_name,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample': self.colsample,
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
            'MAE': float(mean_absolute_error(y_true, preds))
        }

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        metrics = self.compute_metrics(y, preds)
        if self.metadata:
            self.metadata.metrics = metrics
        return metrics

    def feature_importances(self) -> Optional[pd.DataFrame]:
        if self.model is None:
            return None
        if self.feature_names is None:
            self.feature_names = []
        importances: Optional[np.ndarray] = None
        if hasattr(self.model, 'feature_importances_'):
            try:
                importances = np.asarray(getattr(self.model, 'feature_importances_'), dtype=float)
            except Exception:
                importances = None
        elif hasattr(self.model, 'coef_'):
            try:
                importances = np.abs(np.asarray(getattr(self.model, 'coef_'), dtype=float))
            except Exception:
                importances = None
        if importances is None and self.model_name == 'xgb' and HAS_XGB and hasattr(self.model, 'get_booster'):
            try:  # type: ignore[attr-defined]
                booster = self.model.get_booster()  # type: ignore[attr-defined]
                score_dict = booster.get_score(importance_type='gain')  # type: ignore[attr-defined]
                feat_names = getattr(booster, 'feature_names', [])
                importances = np.array([score_dict.get(f, 0.0) for f in feat_names], dtype=float)
                self.feature_names = list(feat_names)
            except Exception:
                importances = None
        if importances is None and self.model_name == 'lgbm' and HAS_LGB and hasattr(self.model, 'booster_'):
            try:  # type: ignore[attr-defined]
                booster = getattr(self.model, 'booster_', None)
                if booster is not None:
                    importances = np.asarray(booster.feature_importance(importance_type='gain'), dtype=float)  # type: ignore[attr-defined]
                    self.feature_names = list(booster.feature_name())  # type: ignore[attr-defined]
            except Exception:
                importances = None
        if importances is None and self._train_X_sample is not None and self._train_y_sample is not None:
            try:
                from sklearn.inspection import permutation_importance
                perm = permutation_importance(
                    self.model,
                    self._train_X_sample,
                    self._train_y_sample,
                    n_repeats=5,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )
                importances = np.asarray(perm.importances_mean, dtype=float)
            except Exception:
                importances = None
        if importances is None or len(importances) == 0:
            return None
        if not self.feature_names:
            self.feature_names = [f'f{i}' for i in range(len(importances))]
        df_imp = pd.DataFrame({'feature': self.feature_names, 'importance': importances})
        total = float(df_imp['importance'].sum()) + 1e-12
        df_imp['importance_norm'] = df_imp['importance'] / total
        return df_imp.sort_values('importance', ascending=False)

    def randomized_search(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                          n_trials: int = 20, metric: str = 'RMSE', verbose: bool = True,
                          seed: int = RANDOM_SEED) -> Tuple[Dict[str, Any], pd.DataFrame]:
        rng = random.Random(seed)
        best_params: Dict[str, Any] = {}
        best_score = float('inf') if metric.upper() == 'RMSE' else -float('inf')
        rows = []
        def score_from_metrics(m: Dict[str, float]):
            if metric.upper() == 'RMSE':
                return m['RMSE']
            if metric.upper() == 'R2':
                return m['R2']
            if metric.upper() == 'MAE':
                return m['MAE']
            return m['RMSE']
        for t in range(n_trials):
            trial_params = {
                'n_estimators': rng.choice([200, 400, 800, 1000]),
                'max_depth': rng.choice([4, 6, 8, 10]),
                'learning_rate': rng.choice([0.03, 0.05, 0.07, 0.1]),
                'subsample': rng.choice([0.6, 0.7, 0.8, 0.9]),
                'colsample': rng.choice([0.6, 0.7, 0.8, 0.9])
            }
            model_backup = (self.n_estimators, self.max_depth, self.learning_rate, self.subsample, self.colsample)
            self.n_estimators = trial_params['n_estimators']
            self.max_depth = trial_params['max_depth']
            self.learning_rate = trial_params['learning_rate']
            self.subsample = trial_params['subsample']
            self.colsample = trial_params['colsample']
            try:
                self.fit(X, y)
                m = self.evaluate(X_val, y_val)
                trial_score = score_from_metrics(m)
                improve = False
                if metric.upper() in ['R2']:
                    if trial_score > best_score:
                        improve = True
                else:
                    if trial_score < best_score:
                        improve = True
                if improve:
                    best_score = trial_score
                    best_params = {
                        'n_estimators': self.n_estimators,
                        'max_depth': self.max_depth,
                        'learning_rate': self.learning_rate,
                        'subsample': self.subsample,
                        'colsample': self.colsample,
                        'metrics': m
                    }
                rows.append({**trial_params, **m, metric: trial_score, 'improved': improve})
                if verbose:
                    print(f"[Trial {t+1}/{n_trials}] {trial_params} -> {m} {'*' if improve else ''}")
            except Exception as e:
                rows.append({**trial_params, 'error': str(e)})
            finally:
                self.n_estimators, self.max_depth, self.learning_rate, self.subsample, self.colsample = model_backup
        results_df = pd.DataFrame(rows)
        if best_params:
            self.n_estimators = best_params['n_estimators']
            self.max_depth = best_params['max_depth']
            self.learning_rate = best_params['learning_rate']
            self.subsample = best_params['subsample']
            self.colsample = best_params['colsample']
            self.fit(X, y)
        return best_params, results_df

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

def _cli():  # basic CLI
    p = argparse.ArgumentParser(description='Ensemble Pipeline (Module 4.1)')
    p.add_argument('--model', default='rf', choices=['rf', 'hist_gb', 'xgb', 'lgbm'])
    p.add_argument('--train', type=str, help='Path to training parquet/csv')
    p.add_argument('--valid', type=str, help='Path to validation parquet/csv (optional)')
    p.add_argument('--target', type=str, default=TARGET_COLUMN)
    p.add_argument('--search', action='store_true', help='Enable randomized search')
    p.add_argument('--trials', type=int, default=15)
    p.add_argument('--out', type=str, help='Output model path (.joblib)')
    p.add_argument('--report', type=str, help='JSON report path')
    p.add_argument('--importances', type=str, help='CSV feature importance output path')
    args = p.parse_args()

    def _load_frame(path: str) -> pd.DataFrame:
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    if args.train is None:
        df_train = generate_regression_synthetic()
    else:
        df_train = _load_frame(args.train)
    if args.target not in df_train.columns:
        raise ValueError(f'Target {args.target} not in training data')
    y_train = df_train[args.target].to_numpy()
    X_train = df_train.drop(columns=[args.target])

    if args.valid:
        df_val = _load_frame(args.valid)
        if args.target not in df_val.columns:
            raise ValueError(f'Target {args.target} not in validation data')
        y_val = df_val[args.target].to_numpy()
        X_val = df_val.drop(columns=[args.target])
    else:
        # simple split
        cutoff = int(0.8 * len(df_train))
        X_val = X_train.iloc[cutoff:]
        y_val = y_train[cutoff:]
        X_train = X_train.iloc[:cutoff]
        y_train = y_train[:cutoff]

    pipe = EnsemblePipeline(model=args.model).fit(X_train, y_train)
    metrics_val = pipe.evaluate(X_val, y_val)

    search_results_json = None
    if args.search:
        best_params, search_df = pipe.randomized_search(X_train, y_train, X_val, y_val, n_trials=args.trials)
        search_results_json = {
            'best_params': best_params,
            'search_summary': search_df.sort_values('RMSE' if 'RMSE' in search_df.columns else 'R2').head(5).to_dict(orient='records')
        }
        metrics_val = pipe.evaluate(X_val, y_val)  # re-evaluate after refit

    out_report = {
        'model': args.model,
        'metrics_validation': metrics_val,
        'metadata': asdict(pipe.metadata) if pipe.metadata else None,
        'search': search_results_json
    }
    print(json.dumps(out_report, indent=2))

    if args.importances:
        fi = pipe.feature_importances()
        if fi is not None:
            fi.to_csv(args.importances, index=False)
    if args.out:
        pipe.save(Path(args.out))
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(out_report, f, indent=2)


if __name__ == '__main__':  # CLI entry
    _cli()
