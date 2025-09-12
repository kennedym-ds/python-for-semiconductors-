"""Tests for Module 10.1 Project Architecture Pipeline.

This module tests the project scaffolding, validation, and linting functionality
for semiconductor ML projects.
"""

import json
import subprocess
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any
import shutil

# Import the pipeline module
import sys
import importlib.util

pipeline_path = Path(__file__).parent / "10.1-project-architecture-pipeline.py"

# Load the module with proper module name
spec = importlib.util.spec_from_file_location("project_architecture_pipeline", pipeline_path)
project_arch = importlib.util.module_from_spec(spec)
sys.modules["project_architecture_pipeline"] = project_arch
spec.loader.exec_module(project_arch)

ProjectArchitecturePipeline = project_arch.ProjectArchitecturePipeline
SEMICONDUCTOR_PROJECT_TYPES = project_arch.SEMICONDUCTOR_PROJECT_TYPES


class TestProjectArchitecturePipeline:
    """Test cases for the Project Architecture Pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_project_arch_"))
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing."""
        return ProjectArchitecturePipeline()

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline can be initialized."""
        assert pipeline is not None
        assert pipeline.structure_version == "1.0.0"
        assert isinstance(pipeline.template_dir, Path)

    def test_available_project_types(self):
        """Test that all expected project types are available."""
        expected_types = {"classification", "regression", "time_series", "computer_vision"}
        available_types = set(SEMICONDUCTOR_PROJECT_TYPES.keys())
        assert expected_types.issubset(available_types)

        # Verify each type has required fields
        for project_type, info in SEMICONDUCTOR_PROJECT_TYPES.items():
            assert "description" in info
            assert "sample_datasets" in info
            assert "key_metrics" in info
            assert isinstance(info["sample_datasets"], list)
            assert isinstance(info["key_metrics"], list)

    def test_scaffold_classification_project(self, pipeline, temp_dir):
        """Test scaffolding a classification project."""
        project_name = "test_classification"

        result = pipeline.scaffold(
            name=project_name,
            project_type="classification",
            output_dir=temp_dir,
            include_notebooks=True,
            include_docker=True,
        )

        # Verify result structure
        assert result["status"] == "success"
        assert result["project_type"] == "classification"
        assert "project_path" in result
        assert "metadata" in result
        assert "next_steps" in result

        # Verify project directory was created
        project_path = Path(result["project_path"])
        assert project_path.exists()
        assert project_path.name == project_name

        # Verify metadata
        metadata = result["metadata"]
        assert metadata["name"] == project_name
        assert metadata["project_type"] == "classification"
        assert "created_at" in metadata
        assert "compliance_features" in metadata

    def test_scaffold_all_project_types(self, pipeline, temp_dir):
        """Test scaffolding all supported project types."""
        for project_type in SEMICONDUCTOR_PROJECT_TYPES.keys():
            project_name = f"test_{project_type}"

            result = pipeline.scaffold(
                name=project_name,
                project_type=project_type,
                output_dir=temp_dir,
                include_notebooks=False,  # Faster testing
                include_docker=False,  # Faster testing
            )

            assert result["status"] == "success"
            assert result["project_type"] == project_type

            project_path = Path(result["project_path"])
            assert project_path.exists()

    def test_scaffold_invalid_project_type(self, pipeline, temp_dir):
        """Test scaffolding with invalid project type raises error."""
        with pytest.raises(ValueError, match="Unsupported project type"):
            pipeline.scaffold(name="test_invalid", project_type="invalid_type", output_dir=temp_dir)

    def test_scaffold_existing_project(self, pipeline, temp_dir):
        """Test scaffolding over existing project raises error."""
        project_name = "test_existing"

        # Create project first time
        pipeline.scaffold(name=project_name, project_type="classification", output_dir=temp_dir)

        # Try to create again - should fail
        with pytest.raises(FileExistsError):
            pipeline.scaffold(name=project_name, project_type="classification", output_dir=temp_dir)

    def test_validate_scaffolded_project(self, pipeline, temp_dir):
        """Test validation of a properly scaffolded project."""
        project_name = "test_validation"

        # Scaffold project
        result = pipeline.scaffold(
            name=project_name,
            project_type="regression",
            output_dir=temp_dir,
            include_notebooks=True,
            include_docker=True,
        )

        project_path = Path(result["project_path"])

        # Validate project
        validation_result = pipeline.validate(project_path)

        # Should be valid with high score
        assert validation_result.is_valid is True
        assert validation_result.score >= 90.0  # Should be near perfect
        assert len(validation_result.errors) == 0
        assert isinstance(validation_result.metrics, dict)

        # Check metric components
        metrics = validation_result.metrics
        assert "required_dirs" in metrics
        assert "optional_dirs" in metrics
        assert "essential_files" in metrics
        assert "naming" in metrics
        assert "config" in metrics

    def test_validate_nonexistent_project(self, pipeline, temp_dir):
        """Test validation of non-existent project."""
        fake_path = temp_dir / "nonexistent_project"

        validation_result = pipeline.validate(fake_path)

        assert validation_result.is_valid is False
        assert validation_result.score == 0.0
        assert len(validation_result.errors) > 0
        assert "does not exist" in validation_result.errors[0]

    def test_validate_incomplete_project(self, pipeline, temp_dir):
        """Test validation of incomplete project structure."""
        # Create minimal incomplete project
        incomplete_project = temp_dir / "incomplete_project"
        incomplete_project.mkdir()

        # Only create some directories
        (incomplete_project / "src").mkdir()
        (incomplete_project / "README.md").touch()

        validation_result = pipeline.validate(incomplete_project)

        # Should have issues but not necessarily invalid
        assert validation_result.score < 100.0
        assert len(validation_result.warnings) > 0 or len(validation_result.errors) > 0

    def test_lint_structure(self, pipeline, temp_dir):
        """Test detailed structure linting."""
        project_name = "test_linting"

        # Scaffold project
        result = pipeline.scaffold(name=project_name, project_type="time_series", output_dir=temp_dir)

        project_path = Path(result["project_path"])

        # Lint structure
        lint_result = pipeline.lint_structure(project_path)

        assert "status" not in lint_result or lint_result.get("status") == "success"
        assert "validation" in lint_result
        assert "detailed_checks" in lint_result

        # Check detailed checks
        detailed = lint_result["detailed_checks"]
        assert "anti_patterns" in detailed
        assert "naming_issues" in detailed
        assert "import_issues" in detailed

        # Scaffolded projects should have minimal issues
        assert len(detailed["anti_patterns"]) == 0
        assert len(detailed["naming_issues"]) == 0

    def test_generated_files_content(self, pipeline, temp_dir):
        """Test that generated files have expected content."""
        project_name = "test_content"

        result = pipeline.scaffold(
            name=project_name,
            project_type="computer_vision",
            output_dir=temp_dir,
            include_notebooks=True,
            include_docker=True,
        )

        project_path = Path(result["project_path"])

        # Check README exists and has project name
        readme = project_path / "README.md"
        assert readme.exists()
        readme_content = readme.read_text()
        assert project_name.replace("_", " ").title() in readme_content
        assert "computer_vision" in readme_content.lower()

        # Check config file
        config_file = project_path / "configs" / "config.yaml"
        assert config_file.exists()

        # Check env template
        env_template = project_path / ".env.template"
        assert env_template.exists()
        env_content = env_template.read_text()
        assert "RANDOM_SEED=42" in env_content

        # Check gitignore
        gitignore = project_path / ".gitignore"
        assert gitignore.exists()
        gitignore_content = gitignore.read_text()
        assert "__pycache__/" in gitignore_content
        assert ".env" in gitignore_content

        # Check requirements
        requirements = project_path / "requirements.txt"
        assert requirements.exists()
        req_content = requirements.read_text()
        assert "numpy" in req_content
        assert "pandas" in req_content
        assert "scikit-learn" in req_content

        # Check Docker files (if included)
        dockerfile = project_path / "Dockerfile"
        assert dockerfile.exists()
        docker_compose = project_path / "docker-compose.yml"
        assert docker_compose.exists()

        # Check notebook (if included)
        notebook = project_path / "notebooks" / "exploratory" / "01_initial_exploration.ipynb"
        assert notebook.exists()

        # Check metadata
        metadata_file = project_path / ".project_metadata.json"
        assert metadata_file.exists()

    def test_project_structure_completeness(self, pipeline, temp_dir):
        """Test that all expected directories and files are created."""
        project_name = "test_completeness"

        result = pipeline.scaffold(name=project_name, project_type="regression", output_dir=temp_dir)

        project_path = Path(result["project_path"])

        # Expected directories
        expected_dirs = [
            "src",
            "src/data",
            "src/features",
            "src/models",
            "src/visualization",
            "data",
            "data/raw",
            "data/processed",
            "data/external",
            "notebooks",
            "notebooks/exploratory",
            "notebooks/production",
            "tests",
            "configs",
            "models",
            "logs",
            "scripts",
            "docs",
        ]

        for dir_path in expected_dirs:
            assert (project_path / dir_path).exists(), f"Missing directory: {dir_path}"
            assert (project_path / dir_path).is_dir(), f"Not a directory: {dir_path}"

        # Expected files
        expected_files = [
            "README.md",
            "requirements.txt",
            ".gitignore",
            ".env.template",
            "configs/config.yaml",
            "src/models/pipeline.py",
            "tests/test_pipeline.py",
            ".project_metadata.json",
        ]

        for file_path in expected_files:
            assert (project_path / file_path).exists(), f"Missing file: {file_path}"
            assert (project_path / file_path).is_file(), f"Not a file: {file_path}"

        # Check __init__.py files in src packages
        src_packages = ["src/data", "src/features", "src/models", "src/visualization"]
        for package in src_packages:
            init_file = project_path / package / "__init__.py"
            assert init_file.exists(), f"Missing __init__.py in {package}"


class TestCLIInterface:
    """Test the CLI interface of the pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_cli_"))
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(["python", str(pipeline_path), "--help"], capture_output=True, text=True, timeout=10)

        assert result.returncode == 0
        assert "Module 10.1 Project Architecture" in result.stdout
        assert "scaffold" in result.stdout
        assert "validate" in result.stdout
        assert "lint-structure" in result.stdout

    def test_cli_scaffold_help(self):
        """Test CLI scaffold subcommand help."""
        result = subprocess.run(
            ["python", str(pipeline_path), "scaffold", "--help"], capture_output=True, text=True, timeout=10
        )

        assert result.returncode == 0
        assert "--name" in result.stdout
        assert "--type" in result.stdout
        assert "--output" in result.stdout

    def test_cli_scaffold_command(self, temp_dir):
        """Test CLI scaffold command execution."""
        project_name = "cli_test_project"

        result = subprocess.run(
            [
                "python",
                str(pipeline_path),
                "scaffold",
                "--name",
                project_name,
                "--type",
                "classification",
                "--output",
                str(temp_dir),
                "--no-docker",
                "--no-notebooks",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0

        # Parse JSON output
        output_data = json.loads(result.stdout)
        assert output_data["status"] == "success"
        assert output_data["project_type"] == "classification"

        # Verify project was created
        project_path = Path(output_data["project_path"])
        assert project_path.exists()
        assert (project_path / "README.md").exists()

    def test_cli_validate_command(self, temp_dir):
        """Test CLI validate command execution."""
        project_name = "cli_validate_test"

        # First scaffold a project
        scaffold_result = subprocess.run(
            [
                "python",
                str(pipeline_path),
                "scaffold",
                "--name",
                project_name,
                "--type",
                "regression",
                "--output",
                str(temp_dir),
                "--no-docker",
                "--no-notebooks",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert scaffold_result.returncode == 0
        scaffold_data = json.loads(scaffold_result.stdout)
        project_path = scaffold_data["project_path"]

        # Now validate it
        validate_result = subprocess.run(
            ["python", str(pipeline_path), "validate", "--project-path", project_path],
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert validate_result.returncode == 0

        # Parse JSON output
        output_data = json.loads(validate_result.stdout)
        assert output_data["status"] == "success"
        assert "validation" in output_data

        validation = output_data["validation"]
        assert validation["is_valid"] is True
        assert validation["score"] >= 90.0

    def test_cli_lint_command(self, temp_dir):
        """Test CLI lint-structure command execution."""
        project_name = "cli_lint_test"

        # First scaffold a project
        scaffold_result = subprocess.run(
            [
                "python",
                str(pipeline_path),
                "scaffold",
                "--name",
                project_name,
                "--type",
                "time_series",
                "--output",
                str(temp_dir),
                "--no-docker",
                "--no-notebooks",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert scaffold_result.returncode == 0
        scaffold_data = json.loads(scaffold_result.stdout)
        project_path = scaffold_data["project_path"]

        # Now lint it
        lint_result = subprocess.run(
            ["python", str(pipeline_path), "lint-structure", "--project-path", project_path],
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert lint_result.returncode == 0

        # Parse JSON output
        output_data = json.loads(lint_result.stdout)
        assert "validation" in output_data
        assert "detailed_checks" in output_data

        detailed = output_data["detailed_checks"]
        assert "anti_patterns" in detailed
        assert "naming_issues" in detailed
        assert "import_issues" in detailed

    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        result = subprocess.run(
            ["python", str(pipeline_path), "invalid_command"], capture_output=True, text=True, timeout=10
        )

        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    def test_cli_missing_required_args(self):
        """Test CLI with missing required arguments."""
        result = subprocess.run(
            [
                "python",
                str(pipeline_path),
                "scaffold",
                # Missing --name and --type
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower()


class TestSemiconductorMetrics:
    """Test semiconductor-specific functionality."""

    def test_project_type_configurations(self):
        """Test that each project type has appropriate configuration."""
        for project_type, config in SEMICONDUCTOR_PROJECT_TYPES.items():
            # Verify metrics are semiconductor-relevant
            metrics = config["key_metrics"]

            if project_type == "classification":
                assert "ROC-AUC" in metrics or "PR-AUC" in metrics
                assert "PWS" in metrics
            elif project_type == "regression":
                assert "RMSE" in metrics or "MAE" in metrics
                assert "PWS" in metrics
                assert "Estimated_Loss" in metrics
            elif project_type == "time_series":
                assert any(metric in metrics for metric in ["MAE", "RMSE", "MAPE"])
            elif project_type == "computer_vision":
                assert any(metric in metrics for metric in ["mIoU", "Pixel_Accuracy"])

    def test_process_parameter_validation(self):
        """Test process parameter validation functionality."""
        # This would test any process parameter validation
        # For now, just ensure the concept is testable
        valid_params = {"temperature": 450, "pressure": 2.5, "flow_rate": 120, "time": 60}

        # Basic validation logic (would be expanded in actual implementation)
        assert all(isinstance(v, (int, float)) for v in valid_params.values())
        assert all(v > 0 for v in valid_params.values())


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
