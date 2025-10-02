# GitHub Copilot Instructions for Python for Semiconductors

## Project Overview

This is a production-ready **educational platform** teaching ML to semiconductor engineers through an 11-module, 22-week curriculum. The codebase prioritizes **learner accessibility** over production optimization—code must be educational first, efficient second.

## Architecture: The 4-Content-Type System

Every module follows a strict 4-file pattern:

1. **Interactive Notebook** (`.ipynb`) - Hands-on exercises with semiconductor data
2. **Technical Deep-Dive** (`.md`) - Theory, math, flowcharts, case studies  
3. **Production Script** (`.py`) - CLI-based pipeline with argparse, JSON output
4. **Quick Reference** (`.md`) - Commands, troubleshooting, cheat sheets

**Critical**: When creating module content, ALL FOUR files are mandatory. Reference existing modules (e.g., `modules/foundation/module-1/`, `modules/intermediate/module-4/`) for structure.

## Tiered Dependency System

This project uses **hierarchical dependency tiers** managed through `pyproject.toml`:

```python
# Tier hierarchy: basic → intermediate → advanced → full
python env_setup.py --tier basic        # Modules 1-3 (numpy, pandas, sklearn)
python env_setup.py --tier intermediate # + Modules 4-5 (xgboost, lightgbm, statsmodels)
python env_setup.py --tier advanced     # + Modules 6-9 (torch, tensorflow, mlflow)
python env_setup.py --tier full         # + Everything (prophet, PySpice)
```

**Lockfiles**: `requirements-{tier}.txt` are pinned lockfiles generated via `tools/compile_tier_lockfiles.py`. When suggesting dependencies:
- Check which tier the code belongs to (see README module mapping)
- Verify the dependency is available in that tier's group in `pyproject.toml`
- If adding new dependencies, update `pyproject.toml` first, then regenerate lockfiles

## Production Python Scripts: CLI Pattern

All `.py` pipeline scripts follow this structure:

```python
"""Module X.Y Pipeline - Brief Description

CLI Interface for automated workflows with JSON output.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import joblib
import pandas as pd

@dataclass
class PipelineMetadata:
    """Metadata for serialization/reproducibility"""
    trained_at: str
    params: dict
    metrics: dict | None = None

class MyPipeline:
    """Educational pipeline with explicit steps"""
    def fit(self, X: pd.DataFrame, y):
        # Educational code with comments explaining semiconductor context
        pass

    def save(self, path: Path):
        joblib.dump(self.model, path / "model.pkl")
        (path / "metadata.json").write_text(json.dumps(asdict(self.metadata)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], required=True)
    args = parser.parse_args()
    # CLI subcommands with JSON output for automation
    print(json.dumps({"status": "success", ...}))

if __name__ == "__main__":
    main()
```

**Key conventions**:
- Use `@dataclass` for configuration/metadata (not `TypedDict`)
- CLI subcommands: `train`, `evaluate`, `predict` (consistent across all pipelines)
- Output JSON to stdout for automation (enables scripted workflows)
- Use `joblib` for model persistence (sklearn ecosystem standard)
- Include `RANDOM_SEED = 42` for reproducibility

## Assessment System Architecture

The assessment infrastructure uses **SQLite + JSON** for portability:

**Database**: `assessments/assessment_results.db` (SQLite, 3 tables)
- `users` - Student profiles
- `assessment_attempts` - Completion records with scores
- `question_responses` - Granular answer tracking

**Question Banks**: `assessments/module-{N}/{N}.{M}-questions.json`
```json
{
  "module_id": "module-1.1",
  "version": "1.0",
  "questions": [
    {
      "id": "m1.1_q001",
      "type": "multiple_choice",  // or "coding_exercise", "conceptual"
      "difficulty": "medium",      // easy | medium | hard
      "points": 2,
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "B"
    }
  ]
}
```

**Validation**: Run `python assessments/validation/validate_all.py --module 1` before committing. Schema is in `assessments/schema.json`.

**Streamlit App**: `streamlit run assessments/assessment_app.py` launches interactive quiz interface. See `assessments/STREAMLIT_APP_README.md` for features.

## Dataset Conventions

Datasets live in `datasets/` with subdirectories:
- `secom/` - UCI SECOM (semiconductor process control, 1567×590)
- `wm811k/` - Wafer map defect patterns (811K samples)
- `steel-plates/` - UCI steel defects (multi-class classification)
- `vision_defects/` - Synthetic wafer defect images
- `time_series/` - Equipment drift simulation data

**Path references**: Use `Path(__file__).parent.parent / "datasets" / "secom"` for portable paths. Avoid hardcoded absolute paths.

**Download script**: `python datasets/download_semiconductor_datasets.py --dataset secom` handles automatic downloads. See `datasets/DATASET_USAGE_GUIDE.md`.

## Testing & Quality Standards

**Three test phases** (see `.github/workflows/ci.yml`):
1. **Unit tests** (81 tests) - Individual module/pipeline tests
2. **Assessment validation** (32 tests) - JSON schema compliance, grading logic
3. **Notebook execution** (88 tests) - End-to-end notebook smoke tests

**Code quality** (synchronized between CI and pre-commit):
- **Black** (120-char lines) - `black . --check` before committing
- **Flake8** (complexity ≤12) - Errors block CI, warnings pass
- **Pre-commit hooks** - Mirror CI exactly: `pre-commit run --all-files`

**Write tests for**:
- Pipeline `fit()`, `predict()`, `save()`, `load()` methods
- CLI argument parsing (`--help` should not error)
- Dataset loading/validation functions

## VS Code Tasks (20+ automation tasks)

Access via **Terminal > Run Task** or `Ctrl+Shift+B`:

**Environment Setup** (10 tasks)
- `Env: Setup Basic/Intermediate/Advanced/Full` - Tiered installs via `env_setup.py`
- `Env: Create Virtual Environment` - Manual venv creation

**Streamlit App** (3 tasks)
- `Streamlit: Run Assessment App` - Launch on port 8501
- `Streamlit: Auto-Reload Mode` - Development with watch mode

**Database** (2 tasks)
- `Database: Backup Assessment Results` - Timestamped SQLite backup
- `Database: Clear Assessment Results` - Fresh database state

**Assessment Validation** (2 tasks)
- `Assessment: Validate All Modules` - Run validation on all 11 modules
- `Assessment: Validate Specific Module` - Interactive module picker

**Jupyter** (2 tasks)
- `Jupyter: Start Lab` - Launch JupyterLab
- `Jupyter: Start Notebook` - Classic Jupyter Notebook

**Testing** (2 tasks)
- `Tests: Run All` - `pytest tests/ -v`
- `Tests: Run with Coverage` - HTML coverage report in `htmlcov/`

See `.vscode/TASKS_README.md` for detailed documentation.

## Semiconductor-Specific Patterns

**Terminology**:
- **Wafer map** - Spatial defect pattern on silicon wafer (WM-811K dataset)
- **SECOM** - Semiconductor Manufacturing Process Control (UCI dataset)
- **Yield prediction** - Binary pass/fail classification for manufacturing
- **Defect detection** - Computer vision on wafer/die images
- **SPC** - Statistical Process Control (control charts, CUSUM)
- **DOE** - Design of Experiments (factorial designs)

**Common pipelines**:
- Yield regression: Predict pass/fail rates from process parameters
- Defect classification: Categorize defect patterns (scratch, edge-ring, donut)
- Equipment drift: Time series forecasting for tool health monitoring
- Process optimization: Multi-objective optimization of fab parameters

**Industry metrics** (use in evaluation):
- **PWS (Prediction Within Spec)**: % predictions within manufacturing tolerance
- **False alarm rate**: Critical for SPC systems (minimize operator fatigue)
- **Detection recall**: Must be high for defects (safety-critical)

## Common Pitfalls

❌ **Don't**: Use `requirements.txt` directly - Use `env_setup.py --tier X`  
✅ **Do**: Respect the tiered installation system for reproducibility

❌ **Don't**: Hardcode paths like `C:/Users/.../datasets/secom`  
✅ **Do**: Use `Path(__file__).parent.parent / "datasets" / "secom"`

❌ **Don't**: Import production libraries in basic modules (e.g., torch in Module 1)  
✅ **Do**: Check tier requirements - stick to numpy/pandas/sklearn in foundation modules

❌ **Don't**: Create standalone `.py` scripts without CLI interface  
✅ **Do**: Follow the argparse + JSON output pattern for automation

❌ **Don't**: Optimize for performance at expense of readability  
✅ **Do**: Write explicit, commented code that teaches concepts (learners first!)

❌ **Don't**: Skip the 4-content-type pattern when adding modules  
✅ **Do**: Create .ipynb + .md (theory) + .py (pipeline) + .md (quick ref) for every module

## Quick Commands Reference

```powershell
# Setup
python env_setup.py --tier full
python verification.py

# Development
black . --check                  # Format check
flake8 .                         # Lint
pre-commit run --all-files       # Full quality check

# Testing
pytest tests/ -v                             # Run all tests
pytest --cov=modules --cov-report=html       # Coverage report
python assessments/validation/validate_all.py # Validate questions

# Assessment App
streamlit run assessments/assessment_app.py
python datasets/download_semiconductor_datasets.py --dataset secom

# Lockfile Management
python tools/compile_tier_lockfiles.py --tiers full --extras docs dev
```

## Key Files Reference

- **`env_setup.py`** - Tiered environment bootstrapper (use this, not pip directly)
- **`pyproject.toml`** - Dependency groups, pip-tools config, pytest markers
- **`assessments/schema.json`** - Question bank validation schema
- **`.vscode/tasks.json`** - 22 automation tasks for learners
- **`datasets/DATASET_USAGE_GUIDE.md`** - Dataset documentation + examples
- **`docs/TROUBLESHOOTING.md`** - Common issues + solutions

## For AI Agents: Educational Code Style

When generating code for this project:

1. **Prioritize clarity over cleverness** - Learners must understand the "why" behind every line
2. **Add semiconductor context** - Explain how code relates to fab processes
3. **Include docstrings with examples** - Show typical semiconductor use cases
4. **Use explicit variable names** - `wafer_pass_fail_labels` not `y`
5. **Comment ML decisions** - Why XGBoost over Random Forest? Explain for domain experts learning ML
6. **Provide CLI examples in docstrings** - Show how to run the pipeline
7. **Output JSON for automation** - Every CLI tool should support scripting

This is an educational platform where code quality = teaching quality. Favor explicitness, documentation, and reproducibility over brevity or performance optimization.
