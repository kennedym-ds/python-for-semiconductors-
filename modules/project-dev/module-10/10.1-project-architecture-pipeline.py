"""Production Project Architecture Pipeline Script for Module 10.1

Provides a CLI to scaffold, validate, and manage semiconductor ML project structures
following best practices for production environments.

Features:
- Project scaffolding with semiconductor-specific templates
- Structure validation against established patterns
- Configuration management and .env setup
- Data versioning patterns and path resolution
- Logging configuration and structured outputs
- JSON responses for all operations

Example usage:
    python 10.1-project-architecture-pipeline.py scaffold --name wafer_yield_predictor --type classification --output ./projects/
    python 10.1-project-architecture-pipeline.py validate --project-path ./projects/wafer_yield_predictor/
    python 10.1-project-architecture-pipeline.py lint-structure --project-path ./projects/wafer_yield_predictor/
"""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import os
import shutil

RANDOM_SEED = 42

# Project structure templates for semiconductor ML projects
SEMICONDUCTOR_PROJECT_TYPES = {
    "classification": {
        "description": "Binary/multi-class defect detection, yield classification",
        "sample_datasets": ["secom", "defect_detection", "wafer_maps"],
        "key_metrics": ["ROC-AUC", "PR-AUC", "PWS", "False_Positive_Rate"],
    },
    "regression": {
        "description": "Process parameter optimization, yield prediction",
        "sample_datasets": ["process_params", "yield_prediction", "sensor_data"],
        "key_metrics": ["MAE", "RMSE", "R2", "PWS", "Estimated_Loss"],
    },
    "time_series": {
        "description": "Equipment monitoring, sensor drift detection",
        "sample_datasets": ["sensor_logs", "equipment_health", "process_monitoring"],
        "key_metrics": ["MAE", "RMSE", "MAPE", "Anomaly_Detection_Rate"],
    },
    "computer_vision": {
        "description": "Wafer defect inspection, die-level analysis",
        "sample_datasets": ["wafer_images", "die_photos", "microscopy_data"],
        "key_metrics": ["mIoU", "Pixel_Accuracy", "Defect_Detection_Rate"],
    },
}


@dataclass
class ValidationResult:
    """Results from project structure validation."""

    is_valid: bool
    score: float  # 0-100 compliance score
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    metrics: Dict[str, Any]


@dataclass
class ProjectMetadata:
    """Metadata for scaffolded projects."""

    name: str
    project_type: str
    created_at: str
    structure_version: str
    compliance_features: List[str]
    paths: Dict[str, str]


class ProjectArchitecturePipeline:
    """Pipeline for managing semiconductor ML project architecture."""

    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = (
            template_dir or Path(__file__).parent.parent.parent.parent / "projects" / "starter" / "template"
        )
        self.structure_version = "1.0.0"

    def scaffold(
        self,
        name: str,
        project_type: str,
        output_dir: Path,
        include_notebooks: bool = True,
        include_docker: bool = True,
    ) -> Dict[str, Any]:
        """Create a new semiconductor ML project structure."""
        if project_type not in SEMICONDUCTOR_PROJECT_TYPES:
            raise ValueError(
                f"Unsupported project type '{project_type}'. Choose from: {list(SEMICONDUCTOR_PROJECT_TYPES.keys())}"
            )

        project_path = output_dir / name
        if project_path.exists():
            raise FileExistsError(f"Project directory '{project_path}' already exists")

        # Create base structure
        project_path.mkdir(parents=True)

        # Core directories
        directories = [
            "src",
            "src/data",
            "src/features",
            "src/models",
            "src/visualization",
            "data/raw",
            "data/processed",
            "data/external",
            "models",
            "notebooks/exploratory",
            "notebooks/production",
            "tests",
            "configs",
            "logs",
            "scripts",
            "docs",
        ]

        for directory in directories:
            (project_path / directory).mkdir(parents=True, exist_ok=True)

        # Create essential files
        self._create_project_files(project_path, name, project_type, include_notebooks, include_docker)

        # Create metadata
        metadata = ProjectMetadata(
            name=name,
            project_type=project_type,
            created_at=self._get_timestamp(),
            structure_version=self.structure_version,
            compliance_features=self._get_compliance_features(include_notebooks, include_docker),
            paths={dir: str(project_path / dir) for dir in directories},
        )

        # Save metadata
        with open(project_path / ".project_metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        return {
            "status": "success",
            "project_path": str(project_path),
            "project_type": project_type,
            "metadata": asdict(metadata),
            "next_steps": self._get_next_steps(project_type),
        }

    def validate(self, project_path: Path) -> ValidationResult:
        """Validate project structure against best practices."""
        errors = []
        warnings = []
        suggestions = []

        # Check if project exists
        if not project_path.exists():
            return ValidationResult(
                is_valid=False,
                score=0.0,
                errors=[f"Project path '{project_path}' does not exist"],
                warnings=[],
                suggestions=[],
                metrics={},
            )

        # Load metadata if available
        metadata_file = project_path / ".project_metadata.json"
        metadata = None
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                warnings.append("Invalid .project_metadata.json file")
        else:
            warnings.append("No .project_metadata.json found - project may not be scaffolded")

        # Structure validation
        required_dirs = ["src", "data", "tests", "configs"]
        optional_dirs = ["notebooks", "docs", "scripts", "logs", "models"]

        score_components = {}

        # Check required directories
        missing_required = []
        for req_dir in required_dirs:
            if not (project_path / req_dir).exists():
                missing_required.append(req_dir)

        if missing_required:
            errors.extend([f"Missing required directory: {d}" for d in missing_required])
            score_components["required_dirs"] = 0.0
        else:
            score_components["required_dirs"] = 40.0  # 40% for required structure

        # Check optional directories (bonus points)
        present_optional = []
        for opt_dir in optional_dirs:
            if (project_path / opt_dir).exists():
                present_optional.append(opt_dir)
        score_components["optional_dirs"] = (len(present_optional) / len(optional_dirs)) * 20.0

        # Check essential files
        essential_files = ["README.md", "requirements.txt", ".gitignore"]
        missing_files = []
        for file in essential_files:
            if not (project_path / file).exists():
                missing_files.append(file)

        if missing_files:
            warnings.extend([f"Missing recommended file: {f}" for f in missing_files])
            score_components["essential_files"] = max(
                0, (len(essential_files) - len(missing_files)) / len(essential_files) * 20.0
            )
        else:
            score_components["essential_files"] = 20.0

        # Check naming conventions
        score_components["naming"] = self._validate_naming_conventions(project_path)

        # Check configuration management
        score_components["config"] = self._validate_config_management(project_path)

        # Calculate overall score
        total_score = sum(score_components.values())
        is_valid = len(errors) == 0 and total_score >= 60.0

        # Generate suggestions
        if missing_required:
            suggestions.append("Create missing required directories: " + ", ".join(missing_required))
        if missing_files:
            suggestions.append("Add essential files: " + ", ".join(missing_files))
        if total_score < 80:
            suggestions.append("Consider adding optional directories for better organization")

        return ValidationResult(
            is_valid=is_valid,
            score=total_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metrics=score_components,
        )

    def lint_structure(self, project_path: Path) -> Dict[str, Any]:
        """Perform detailed linting of project structure and naming conventions."""
        validation = self.validate(project_path)

        # Additional linting checks
        lint_results = {"validation": asdict(validation), "detailed_checks": {}}

        # Check for common anti-patterns
        anti_patterns = self._check_anti_patterns(project_path)
        lint_results["detailed_checks"]["anti_patterns"] = anti_patterns

        # Check file naming conventions
        naming_issues = self._check_detailed_naming(project_path)
        lint_results["detailed_checks"]["naming_issues"] = naming_issues

        # Check import structure
        import_issues = self._check_import_structure(project_path)
        lint_results["detailed_checks"]["import_issues"] = import_issues

        return lint_results

    def _create_project_files(
        self, project_path: Path, name: str, project_type: str, include_notebooks: bool, include_docker: bool
    ):
        """Create essential project files."""
        project_info = SEMICONDUCTOR_PROJECT_TYPES[project_type]

        # README.md
        readme_content = self._generate_readme(name, project_type, project_info)
        with open(project_path / "README.md", "w") as f:
            f.write(readme_content)

        # requirements.txt
        requirements_content = self._generate_requirements(project_type)
        with open(project_path / "requirements.txt", "w") as f:
            f.write(requirements_content)

        # .gitignore
        gitignore_content = self._generate_gitignore()
        with open(project_path / ".gitignore", "w") as f:
            f.write(gitignore_content)

        # .env template
        env_content = self._generate_env_template(project_type)
        with open(project_path / ".env.template", "w") as f:
            f.write(env_content)

        # Basic configuration
        config_content = self._generate_config(name, project_type)
        with open(project_path / "configs" / "config.yaml", "w") as f:
            f.write(config_content)

        # Basic src structure with __init__.py files
        src_dirs = ["data", "features", "models", "visualization"]
        for src_dir in src_dirs:
            with open(project_path / "src" / src_dir / "__init__.py", "w") as f:
                f.write(f'"""Package for {src_dir} functionality."""\n')

        # Basic pipeline template
        pipeline_content = self._generate_pipeline_template(name, project_type)
        with open(project_path / "src" / "models" / "pipeline.py", "w") as f:
            f.write(pipeline_content)

        # Basic test
        test_content = self._generate_test_template(name)
        with open(project_path / "tests" / "test_pipeline.py", "w") as f:
            f.write(test_content)

        if include_docker:
            self._create_docker_files(project_path)

        if include_notebooks:
            self._create_notebook_templates(project_path, project_type)

    def _generate_readme(self, name: str, project_type: str, project_info: Dict[str, Any]) -> str:
        """Generate README.md content."""
        return f"""# {name.replace('_', ' ').title()}

## Overview
{project_info['description']} project for semiconductor manufacturing.

## Project Type
**{project_type.title()}** - {project_info['description']}

## Structure
```
{name}/
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations
│   └── visualization/     # Plotting and visualization
├── data/                  # Data storage
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned data for analysis
│   └── external/         # Third-party data
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/      # EDA and experimentation
│   └── production/       # Production-ready notebooks
├── tests/                 # Unit and integration tests
├── configs/               # Configuration files
├── models/                # Trained model artifacts
├── logs/                  # Application logs
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

## Key Metrics
This {project_type} project focuses on:
{chr(10).join(f'- **{metric}**: Key performance indicator' for metric in project_info['key_metrics'])}

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Copy environment template: `cp .env.template .env`
3. Configure your settings in `.env`
4. Run tests: `python -m pytest tests/`
5. Train model: `python src/models/pipeline.py train --help`

## Sample Datasets
Recommended datasets for this project type:
{chr(10).join(f'- {dataset}' for dataset in project_info['sample_datasets'])}

## Development Guidelines
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write tests for new functionality
- Update documentation for API changes
- Use structured logging with JSON output

## Manufacturing Context
This project template is optimized for semiconductor manufacturing workflows:
- **PWS (Prediction Within Spec)**: Percentage of predictions within tolerance
- **Process Parameters**: Temperature, pressure, flow rates, time
- **Equipment State**: Real-time monitoring and anomaly detection
- **Quality Metrics**: Yield, defect rates, specification compliance
"""

    def _generate_requirements(self, project_type: str) -> str:
        """Generate requirements.txt content."""
        base_requirements = """# Core ML and Data Science
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
joblib>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuration and Utilities
PyYAML>=6.0.0
python-dotenv>=1.0.0
tqdm>=4.65.0
click>=8.1.0

# Logging and Monitoring
structlog>=23.0.0

# Testing and Quality
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0

# Notebooks (optional)
jupyter>=1.0.0
ipykernel>=6.25.0
"""

        if project_type == "computer_vision":
            base_requirements += "\n# Computer Vision\nopencv-python>=4.8.0\nPillow>=10.0.0\n"
        elif project_type == "time_series":
            base_requirements += "\n# Time Series\nstatsmodels>=0.14.0\n"

        return base_requirements

    def _generate_gitignore(self) -> str:
        """Generate .gitignore content."""
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env

# Model artifacts and data
models/*.joblib
models/*.pkl
models/*.h5
data/raw/*
data/processed/*
data/external/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep

# Logs
logs/*.log
logs/*.out

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Documentation
docs/_build/
"""

    def _generate_env_template(self, project_type: str) -> str:
        """Generate .env.template content."""
        return f"""# Environment Configuration Template
# Copy this file to .env and fill in your values

# Project Configuration
PROJECT_NAME={project_type}_project
PROJECT_TYPE={project_type}
ENVIRONMENT=development

# Data Configuration
DATA_DIR=./data
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
MODEL_DIR=./models

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=./logs
STRUCTURED_LOGS=true

# Model Configuration
RANDOM_SEED=42
MODEL_SAVE_FORMAT=joblib

# Semiconductor Manufacturing Settings
TOLERANCE=2.0
SPEC_LOW=60.0
SPEC_HIGH=100.0
COST_PER_UNIT=1.0

# Database (if needed)
# DATABASE_URL=sqlite:///data/database.db

# API Configuration (if needed)
# API_KEY=your_api_key_here
# API_BASE_URL=https://api.example.com

# Monitoring (if needed)
# MONITORING_ENABLED=false
# METRICS_ENDPOINT=http://localhost:8080/metrics
"""

    def _generate_config(self, name: str, project_type: str) -> str:
        """Generate config.yaml content."""
        return f"""# Configuration for {name}

project:
  name: {name}
  type: {project_type}
  version: "1.0.0"

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  external_dir: "data/external"

model:
  save_dir: "models"
  format: "joblib"

training:
  random_seed: 42
  test_size: 0.2
  validation_size: 0.2
  cross_validation_folds: 5

manufacturing:
  tolerance: 2.0
  spec_limits:
    low: 60.0
    high: 100.0
  cost_per_unit: 1.0

logging:
  level: "INFO"
  format: "json"
  file: "logs/app.log"

features:
  engineering:
    enabled: true
    methods: ["scaling", "pca"]
  selection:
    enabled: true
    method: "k_best"
    k: 20
"""

    def _generate_pipeline_template(self, name: str, project_type: str) -> str:
        """Generate basic pipeline template."""
        return f'''"""Pipeline implementation for {name}

This module provides the core ML pipeline for {project_type} in semiconductor manufacturing.
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

RANDOM_SEED = 42

@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline."""
    model_type: str = "default"
    random_seed: int = RANDOM_SEED
    test_size: float = 0.2

class {name.replace("_", "").title()}Pipeline:
    """Main ML pipeline for {project_type}."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "{"".join(name.split("_")).title()}Pipeline":
        """Train the model."""
        # TODO: Implement your training logic here
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        # TODO: Implement your prediction logic here
        return np.zeros(len(X))

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)

        # Standard metrics
        metrics = {{
            "accuracy": accuracy_score(y, predictions),
            # TODO: Add semiconductor-specific metrics like PWS
        }}

        return metrics

    def save(self, path: Path) -> None:
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline")
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "{name.replace("_", "").title()}Pipeline":
        """Load a trained model."""
        return joblib.load(path)

if __name__ == "__main__":
    # Basic CLI would go here
    print("Run this pipeline using the CLI: python -m src.models.pipeline")
'''

    def _generate_test_template(self, name: str) -> str:
        """Generate basic test template."""
        return f'''"""Tests for {name} pipeline."""
import pytest
import numpy as np
import pandas as pd
from src.models.pipeline import {name.replace("_", "").title()}Pipeline, PipelineConfig

class Test{name.replace("_", "").title()}Pipeline:
    """Test cases for the main pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X = pd.DataFrame({{
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
        }})
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        pipeline = {name.replace("_", "").title()}Pipeline()
        assert pipeline.config.random_seed == 42
        assert not pipeline.is_fitted

    def test_pipeline_fit(self, sample_data):
        """Test pipeline fitting."""
        X, y = sample_data
        pipeline = {name.replace("_", "").title()}Pipeline()
        fitted_pipeline = pipeline.fit(X, y)
        assert fitted_pipeline.is_fitted

    def test_pipeline_predict(self, sample_data):
        """Test pipeline prediction."""
        X, y = sample_data
        pipeline = {name.replace("_", "").title()}Pipeline()
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)

    def test_pipeline_evaluate(self, sample_data):
        """Test pipeline evaluation."""
        X, y = sample_data
        pipeline = {name.replace("_", "").title()}Pipeline()
        pipeline.fit(X, y)
        metrics = pipeline.evaluate(X, y)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
'''

    def _create_docker_files(self, project_path: Path):
        """Create Docker configuration files."""
        dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed models

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Expose port (if needed)
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.models.pipeline", "--help"]
"""

        docker_compose_content = """version: '3.8'

services:
  ml-pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"

  # Optional: Add database, monitoring, etc.
  # database:
  #   image: postgres:15
  #   environment:
  #     POSTGRES_DB: semiconductor_ml
  #     POSTGRES_USER: user
  #     POSTGRES_PASSWORD: password
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   ports:
  #     - "5432:5432"

# volumes:
#   postgres_data:
"""

        with open(project_path / "Dockerfile", "w") as f:
            f.write(dockerfile_content)

        with open(project_path / "docker-compose.yml", "w") as f:
            f.write(docker_compose_content)

    def _create_notebook_templates(self, project_path: Path, project_type: str):
        """Create notebook templates."""
        exploratory_notebook = f"""{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# Exploratory Data Analysis\\n",
    "\\n",
    "## Project: {project_type.title()} for Semiconductor Manufacturing\\n",
    "\\n",
    "This notebook provides initial exploration and analysis."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# Standard imports\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from pathlib import Path\\n",
    "\\n",
    "# Set up paths\\n",
    "DATA_DIR = Path('../../../data').resolve()\\n",
    "\\n",
    "# Set random seed\\n",
    "RANDOM_SEED = 42\\n",
    "np.random.seed(RANDOM_SEED)\\n",
    "\\n",
    "print(f\\"Data directory: {{DATA_DIR}}\\")\\n",
    "print(f\\"Directory exists: {{DATA_DIR.exists()}}\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Data Loading\\n",
    "\\n",
    "Load and inspect the semiconductor manufacturing data."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# TODO: Load your data here\\n",
    "# df = pd.read_csv(DATA_DIR / 'raw' / 'your_data.csv')\\n",
    "# print(f\\"Data shape: {{df.shape}}\\")\\n",
    "# df.head()"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.11.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}"""

        with open(project_path / "notebooks" / "exploratory" / "01_initial_exploration.ipynb", "w") as f:
            f.write(exploratory_notebook)

    def _validate_naming_conventions(self, project_path: Path) -> float:
        """Validate naming conventions (0-10 points)."""
        score = 10.0

        # Check for Python naming conventions
        python_files = list(project_path.rglob("*.py"))
        for py_file in python_files:
            if py_file.name != py_file.name.lower():
                score -= 1.0
                break

        return max(0, score)

    def _validate_config_management(self, project_path: Path) -> float:
        """Validate configuration management (0-10 points)."""
        score = 0.0

        if (project_path / "configs").exists():
            score += 3.0
        if (project_path / ".env.template").exists():
            score += 3.0
        if (project_path / "configs" / "config.yaml").exists():
            score += 4.0

        return score

    def _check_anti_patterns(self, project_path: Path) -> List[str]:
        """Check for common anti-patterns."""
        issues = []

        # Check for hardcoded paths
        python_files = list(project_path.rglob("*.py"))
        for py_file in python_files:
            try:
                content = py_file.read_text()
                if "/Users/" in content or "C:\\" in content:
                    issues.append(f"Hardcoded path found in {py_file.relative_to(project_path)}")
            except UnicodeDecodeError:
                pass

        # Check for missing __init__.py in src
        src_dirs = [d for d in (project_path / "src").rglob("*") if d.is_dir()]
        for src_dir in src_dirs:
            if not (src_dir / "__init__.py").exists():
                issues.append(f"Missing __init__.py in {src_dir.relative_to(project_path)}")

        return issues

    def _check_detailed_naming(self, project_path: Path) -> List[str]:
        """Check detailed naming conventions."""
        issues = []

        # Check directory naming
        for item in project_path.rglob("*"):
            if item.is_dir() and " " in item.name:
                issues.append(f"Directory with spaces: {item.relative_to(project_path)}")

        return issues

    def _check_import_structure(self, project_path: Path) -> List[str]:
        """Check import structure patterns."""
        issues = []

        # This is a simplified check - in practice you'd use AST parsing
        python_files = list(project_path.rglob("*.py"))
        for py_file in python_files:
            try:
                content = py_file.read_text()
                lines = content.split("\\n")
                for i, line in enumerate(lines[:20]):  # Check first 20 lines
                    if line.strip().startswith("from") and "import *" in line:
                        issues.append(f"Wildcard import in {py_file.relative_to(project_path)}:{i+1}")
            except UnicodeDecodeError:
                pass

        return issues

    def _get_compliance_features(self, include_notebooks: bool, include_docker: bool) -> List[str]:
        """Get list of compliance features included."""
        features = [
            "standard_structure",
            "configuration_management",
            "environment_templates",
            "testing_framework",
            "documentation",
        ]

        if include_notebooks:
            features.append("jupyter_notebooks")
        if include_docker:
            features.append("docker_support")

        return features

    def _get_next_steps(self, project_type: str) -> List[str]:
        """Get recommended next steps after scaffolding."""
        return [
            "Copy .env.template to .env and configure your settings",
            "Install dependencies: pip install -r requirements.txt",
            "Add your data to data/raw/",
            "Implement your pipeline logic in src/models/pipeline.py",
            f"Focus on {project_type}-specific metrics and validation",
            "Write tests in tests/",
            "Document your API in docs/",
            "Set up version control: git init && git add . && git commit -m 'Initial scaffold'",
        ]

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------- CLI Functions ---------------- #


def action_scaffold(args):
    """Handle scaffold command."""
    try:
        pipeline = ProjectArchitecturePipeline()
        result = pipeline.scaffold(
            name=args.name,
            project_type=args.type,
            output_dir=Path(args.output),
            include_notebooks=not args.no_notebooks,
            include_docker=not args.no_docker,
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        error_result = {"status": "error", "message": str(e)}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def action_validate(args):
    """Handle validate command."""
    try:
        pipeline = ProjectArchitecturePipeline()
        result = pipeline.validate(Path(args.project_path))
        output = {"status": "success", "validation": asdict(result)}
        print(json.dumps(output, indent=2))
        if not result.is_valid:
            sys.exit(1)
    except Exception as e:
        error_result = {"status": "error", "message": str(e)}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def action_lint_structure(args):
    """Handle lint-structure command."""
    try:
        pipeline = ProjectArchitecturePipeline()
        result = pipeline.lint_structure(Path(args.project_path))
        result["status"] = "success"
        print(json.dumps(result, indent=2))
    except Exception as e:
        error_result = {"status": "error", "message": str(e)}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(description="Module 10.1 Project Architecture Production Pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # scaffold subcommand
    p_scaffold = sub.add_parser("scaffold", help="Create a new semiconductor ML project structure")
    p_scaffold.add_argument("--name", required=True, help="Project name (use snake_case)")
    p_scaffold.add_argument(
        "--type", required=True, choices=list(SEMICONDUCTOR_PROJECT_TYPES.keys()), help="Project type"
    )
    p_scaffold.add_argument("--output", default="./projects", help="Output directory for project")
    p_scaffold.add_argument("--no-notebooks", action="store_true", help="Skip notebook templates")
    p_scaffold.add_argument("--no-docker", action="store_true", help="Skip Docker configuration")
    p_scaffold.set_defaults(func=action_scaffold)

    # validate subcommand
    p_validate = sub.add_parser("validate", help="Validate project structure against best practices")
    p_validate.add_argument("--project-path", required=True, help="Path to project directory")
    p_validate.set_defaults(func=action_validate)

    # lint-structure subcommand
    p_lint = sub.add_parser("lint-structure", help="Perform detailed linting of project structure")
    p_lint.add_argument("--project-path", required=True, help="Path to project directory")
    p_lint.set_defaults(func=action_lint_structure)

    return parser


def main(argv: Optional[List[str]] = None):
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
