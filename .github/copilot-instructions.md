# Copilot Instructions: Python for Semiconductors Learning Series

## Project Architecture

This is a **20-week learning series** for semiconductor engineers, organized as a tiered educational framework with production-ready ML pipelines. The codebase follows a strict **4-content-type pattern** per module.

### Module Content Pattern (CRITICAL)
Every module contains exactly these 4 files:
```
X.Y-topic-analysis.ipynb          # Interactive learning notebook
X.Y-topic-fundamentals.md         # Theory and deep-dive documentation  
X.Y-topic-pipeline.py             # Production-ready CLI script
X.Y-topic-quick-ref.md            # Summary and cheat sheet
```

### Repository Structure
```
modules/
├── foundation/     # Modules 1-3 (Python, stats, basic ML)
├── intermediate/   # Modules 4-5 (ensembles, time series)
├── advanced/       # Modules 6-7 (deep learning, computer vision)
├── cutting-edge/   # Modules 8-9 (generative AI, MLOps)
└── project-dev/    # Module 10 (production projects)
```

## Development Patterns

### Pipeline Script Architecture
All `*-pipeline.py` files follow this standardized pattern:
```python
# 1. Comprehensive docstring with usage examples
# 2. Class-based pipeline with dataclass metadata
# 3. CLI with subcommands: train, evaluate, predict
# 4. JSON output for all operations
# 5. Model persistence with save/load static methods

class SomethingPipeline:
    def fit(X: pd.DataFrame, y: np.ndarray) -> Self
    def predict(X: pd.DataFrame) -> np.ndarray  
    def evaluate(X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]
    def save(path: Path) -> None
    @staticmethod
    def load(path: Path) -> 'SomethingPipeline'
```

### CLI Pattern (MANDATORY)
Every pipeline script uses this exact argparse structure:
```python
def build_parser():
    parser = argparse.ArgumentParser(description='Module X.Y Description')
    sub = parser.add_subparsers(dest='command', required=True)
    
    # train subcommand
    p_train = sub.add_parser('train', help='Train a model')
    p_train.set_defaults(func=action_train)
    
    # evaluate subcommand  
    p_eval = sub.add_parser('evaluate', help='Evaluate model')
    p_eval.set_defaults(func=action_evaluate)
    
    # predict subcommand
    p_pred = sub.add_parser('predict', help='Make predictions')  
    p_pred.set_defaults(func=action_predict)
```

### Dataset Path Resolution (CRITICAL)
Notebooks use relative paths from their module location:
- Module 2/3 notebooks: `DATA_DIR = Path('../../../datasets').resolve()`
- All datasets organized in subfolders: `datasets/secom/secom.data`
- Never use flat paths like `datasets/secom.data`

## Key Conventions

### Constants and Reproducibility
```python
RANDOM_SEED = 42  # Always use this seed
TARGET_COLUMN = 'target'  # Default target name for synthetic data
```

### Semiconductor-Specific Metrics
Include manufacturing metrics in evaluate() methods:
- **PWS (Prediction Within Spec)**: Percentage of predictions within tolerance
- **Estimated Loss**: Cost impact of prediction errors
- Standard metrics: MAE, RMSE, R² for regression; ROC-AUC, PR-AUC for classification

### Environment Management (Tiered Dependencies)
- `requirements-basic.txt` → `intermediate` → `advanced` → `full`
- Use `python env_setup.py --tier <level>` for setup
- Optional dependencies pattern: `HAS_XGB = True; try: import xgboost`

## Testing & CI

### CI Workflow (GitHub Actions)
- Uses basic tier dependencies for speed
- Runs `flake8`, `black --check`, targeted tests
- Smoke tests import key modules
- Matrix testing commented out but available

### Test Patterns
- Test files follow: `test_<module>_pipeline.py` naming
- Located in same directory as pipeline scripts
- Focus on CLI functionality and core pipeline methods

## Critical Files for Understanding

1. **`env_setup.py`**: Tiered dependency management system
2. **`modules/foundation/module-3/3.1-regression-pipeline.py`**: Reference implementation
3. **`datasets/download_semiconductor_datasets.py`**: Dataset management with UCI integration
4. **`.github/workflows/ci.yml`**: Complete CI/CD setup

## When Creating New Modules

1. **Follow the 4-content pattern exactly** - notebook, fundamentals, pipeline, quick-ref
2. **Use semiconductor manufacturing context** - process parameters, yield prediction, defect detection
3. **Include synthetic data generators** for learning when real data unavailable
4. **Implement full CLI interface** with train/evaluate/predict subcommands
5. **Add manufacturing-specific metrics** beyond standard ML metrics
6. **Update tiered requirements** if introducing new dependencies
7. **Test dataset path resolution** from notebook location

## Dependencies and Imports

- Core: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `jupyter`
- Optional: `xgboost`, `lightgbm` (with availability checks)
- Visualization: `seaborn`, `plotly` for advanced plotting
- Data: `ucimlrepo` for dataset downloads
- Structure imports alphabetically within groups (stdlib, third-party, local)

## Error Handling Philosophy

- Fail fast with clear error messages for missing files/data
- Graceful degradation for optional dependencies  
- Comprehensive CLI help text with examples
- JSON error responses for programmatic usage
