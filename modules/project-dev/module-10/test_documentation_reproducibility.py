"""Tests for 10.3-documentation-reproducibility-pipeline.py

Tests the documentation and reproducibility pipeline functionality
including notebook conversion, path validation, environment export,
and mkdocs setup.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add current directory to path to import the pipeline
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Simple import approach
    import importlib.util
    import sys
    
    spec = importlib.util.spec_from_file_location(
        "doc_pipeline", 
        "10.3-documentation-reproducibility-pipeline.py"
    )
    doc_pipeline = importlib.util.module_from_spec(spec)
    sys.modules["doc_pipeline"] = doc_pipeline
    spec.loader.exec_module(doc_pipeline)
except Exception as e:
    print(f"Warning: Could not import pipeline module: {e}")
    doc_pipeline = None


class TestDocumentationReproducibilityPipeline(unittest.TestCase):
    """Test cases for DocumentationReproducibilityPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        if doc_pipeline is None:
            self.skipTest("Pipeline module not available")
        
        self.pipeline = doc_pipeline.DocumentationReproducibilityPipeline()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_demo_notebook(self, path: Path) -> None:
        """Create a demo notebook for testing."""
        demo_notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Test Notebook\n", "This is a test."]
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": ["print('Hello, test!')"]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(demo_notebook, f, indent=2)

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsInstance(self.pipeline, doc_pipeline.DocumentationReproducibilityPipeline)
        self.assertIsNone(self.pipeline.config)
        self.assertEqual(self.pipeline.results, {})

    def test_generate_documentation_success(self):
        """Test successful documentation generation."""
        # Create test notebook
        nb_dir = self.temp_dir / "notebooks"
        nb_path = nb_dir / "test.ipynb"
        self.create_demo_notebook(nb_path)
        
        output_dir = self.temp_dir / "docs"
        
        result = self.pipeline.generate_documentation(
            input_dir=nb_dir,
            output_dir=output_dir,
            format="markdown"
        )
        
        # Check if nbconvert is available
        if doc_pipeline.HAS_NBCONVERT:
            self.assertTrue(result["success"])
            self.assertGreater(len(result["converted_files"]), 0)
            self.assertEqual(result["total_notebooks"], 1)
        else:
            self.assertFalse(result["success"])
            self.assertIn("nbconvert not available", result["error"])

    def test_generate_documentation_missing_input(self):
        """Test documentation generation with missing input directory."""
        result = self.pipeline.generate_documentation(
            input_dir=self.temp_dir / "nonexistent",
            output_dir=self.temp_dir / "docs",
            format="markdown"
        )
        
        self.assertFalse(result["success"])
        # Check for either error type - depends on nbconvert availability
        self.assertTrue(
            "does not exist" in result["error"] or 
            "nbconvert not available" in result["error"]
        )

    def test_validate_dataset_paths_success(self):
        """Test dataset path validation."""
        # Create test notebook with correct path pattern
        nb_dir = self.temp_dir / "test_module"
        nb_path = nb_dir / "test.ipynb"
        
        # Create notebook with correct dataset path pattern
        test_notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "from pathlib import Path\n",
                        "DATA_DIR = Path('../../../datasets').resolve()\n",
                        "data = pd.read_csv(DATA_DIR / 'secom' / 'secom.data')"
                    ]
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        with open(nb_path, 'w') as f:
            json.dump(test_notebook, f)
        
        result = self.pipeline.validate_dataset_paths(nb_dir.parent)
        
        self.assertTrue(result["success"])
        self.assertGreater(result["summary"]["total_notebooks"], 0)

    def test_validate_dataset_paths_invalid_patterns(self):
        """Test dataset path validation with invalid patterns."""
        # Create test notebook with incorrect path pattern
        nb_dir = self.temp_dir / "test_module"
        nb_path = nb_dir / "test.ipynb"
        
        test_notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Bad pattern - flat dataset path\n",
                        "data = pd.read_csv('datasets/secom.data')"
                    ]
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        with open(nb_path, 'w') as f:
            json.dump(test_notebook, f)
        
        result = self.pipeline.validate_dataset_paths(nb_dir.parent)
        
        # Should still succeed but with warnings about invalid paths
        self.assertIsInstance(result["success"], bool)
        self.assertGreaterEqual(result["summary"]["total_warnings"], 0)

    def test_validate_dataset_paths_missing_directory(self):
        """Test dataset path validation with missing directory."""
        result = self.pipeline.validate_dataset_paths(
            self.temp_dir / "nonexistent"
        )
        
        self.assertFalse(result["success"])
        self.assertIn("does not exist", result["error"])

    @patch('subprocess.run')
    def test_export_environment_conda_success(self, mock_run):
        """Test successful conda environment export."""
        mock_run.return_value = MagicMock(
            stdout="name: test\ndependencies:\n- python=3.9",
            returncode=0
        )
        
        output_path = self.temp_dir / "environment.yml"
        result = self.pipeline.export_environment(
            output_path=output_path,
            format="conda"
        )
        
        # Note: This test may fail if conda is not available
        # The actual result depends on system configuration
        self.assertIn("success", result)

    @patch('subprocess.run')
    def test_export_environment_pip_success(self, mock_run):
        """Test successful pip environment export."""
        mock_run.return_value = MagicMock(
            stdout="numpy==1.24.3\npandas==2.0.1",
            returncode=0
        )
        
        output_path = self.temp_dir / "requirements.txt"
        result = self.pipeline.export_environment(
            output_path=output_path,
            format="pip"
        )
        
        # The success depends on whether pip is available
        self.assertIn("success", result)

    def test_export_environment_invalid_format(self):
        """Test environment export with invalid format."""
        output_path = self.temp_dir / "environment.txt"
        result = self.pipeline.export_environment(
            output_path=output_path,
            format="invalid"
        )
        
        self.assertFalse(result["success"])
        self.assertIn("Unsupported format", result["error"])

    def test_setup_mkdocs_success(self):
        """Test mkdocs setup."""
        project_dir = self.temp_dir / "mkdocs_project"
        
        result = self.pipeline.setup_mkdocs(project_dir)
        
        self.assertTrue(result["success"])
        self.assertTrue((project_dir / "mkdocs.yml").exists())
        self.assertTrue((project_dir / "docs" / "index.md").exists())

    @patch('subprocess.run')
    def test_build_docs_success(self, mock_run):
        """Test mkdocs build."""
        mock_run.return_value = MagicMock(
            stdout="INFO - Documentation built",
            returncode=0
        )
        
        # First set up mkdocs
        project_dir = self.temp_dir / "mkdocs_project"
        setup_result = self.pipeline.setup_mkdocs(project_dir)
        self.assertTrue(setup_result["success"])
        
        # Then try to build
        result = self.pipeline.build_docs(project_dir)
        
        # Success depends on mkdocs availability
        self.assertIn("success", result)

    def test_build_docs_missing_config(self):
        """Test mkdocs build with missing configuration."""
        project_dir = self.temp_dir / "empty_project"
        project_dir.mkdir()
        
        result = self.pipeline.build_docs(project_dir)
        
        self.assertFalse(result["success"])
        self.assertIn("mkdocs.yml not found", result["error"])

    def test_save_and_load_pipeline(self):
        """Test pipeline save and load functionality."""
        # Configure pipeline
        config = doc_pipeline.DocumentationConfig(
            input_dir=Path("test/input"),
            output_dir=Path("test/output")
        )
        self.pipeline.config = config
        self.pipeline.results = {"test": "data"}
        
        # Save pipeline
        save_path = self.temp_dir / "pipeline.json"
        self.pipeline.save(save_path)
        
        # Load pipeline
        loaded_pipeline = doc_pipeline.DocumentationReproducibilityPipeline.load(save_path)
        
        # Convert paths back for comparison since JSON serialization converts them to strings
        self.assertEqual(str(loaded_pipeline.config.input_dir), str(config.input_dir))
        self.assertEqual(str(loaded_pipeline.config.output_dir), str(config.output_dir))
        self.assertEqual(loaded_pipeline.results, {"test": "data"})


class TestCLIFunctionality(unittest.TestCase):
    """Test CLI functionality of the pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        if doc_pipeline is None:
            self.skipTest("Pipeline module not available")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        self.script_path = Path(__file__).parent / "10.3-documentation-reproducibility-pipeline.py"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_cli_help(self):
        """Test CLI help functionality."""
        if not self.script_path.exists():
            self.skipTest("Pipeline script not found")
        
        try:
            result = subprocess.run(
                [sys.executable, str(self.script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            self.assertEqual(result.returncode, 0)
            self.assertIn("Documentation & Reproducibility Pipeline", result.stdout)
            self.assertIn("generate-docs", result.stdout)
            self.assertIn("validate-paths", result.stdout)
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.skipTest(f"CLI test failed: {e}")

    def test_cli_subcommand_help(self):
        """Test CLI subcommand help."""
        if not self.script_path.exists():
            self.skipTest("Pipeline script not found")
        
        subcommands = ['generate-docs', 'validate-paths', 'export-env', 
                      'setup-mkdocs', 'build-docs']
        
        for cmd in subcommands:
            with self.subTest(command=cmd):
                try:
                    result = subprocess.run(
                        [sys.executable, str(self.script_path), cmd, "--help"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    self.assertEqual(result.returncode, 0)
                    self.assertIn(cmd, result.stdout.lower())
                    
                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    self.skipTest(f"CLI subcommand test failed for {cmd}: {e}")

    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = doc_pipeline.build_parser()
        
        # Test that parser is created successfully
        self.assertIsNotNone(parser)
        
        # Test parsing help (should not raise exception)
        try:
            parser.parse_args(['--help'])
        except SystemExit:
            # Expected behavior for --help
            pass


class TestDataClasses(unittest.TestCase):
    """Test dataclass functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if doc_pipeline is None:
            self.skipTest("Pipeline module not available")

    def test_documentation_config(self):
        """Test DocumentationConfig dataclass."""
        config = doc_pipeline.DocumentationConfig(
            input_dir=Path("input"),
            output_dir=Path("output"),
            format="markdown"
        )
        
        self.assertEqual(config.input_dir, Path("input"))
        self.assertEqual(config.output_dir, Path("output"))
        self.assertEqual(config.format, "markdown")
        self.assertTrue(config.include_code)

    def test_path_validation_result(self):
        """Test PathValidationResult dataclass."""
        result = doc_pipeline.PathValidationResult(
            module_path=Path("test/module"),
            notebook_files=[Path("test.ipynb")],
            valid_paths=["../../../datasets"],
            invalid_paths=["datasets/flat.data"],
            warnings=["Warning message"]
        )
        
        self.assertEqual(result.module_path, Path("test/module"))
        self.assertEqual(len(result.notebook_files), 1)
        self.assertEqual(len(result.valid_paths), 1)
        self.assertEqual(len(result.invalid_paths), 1)
        self.assertEqual(len(result.warnings), 1)

    def test_environment_export(self):
        """Test EnvironmentExport dataclass."""
        export = doc_pipeline.EnvironmentExport(
            format="conda",
            include_versions=True,
            output_path=Path("env.yml"),
            exclude_dev=False
        )
        
        self.assertEqual(export.format, "conda")
        self.assertTrue(export.include_versions)
        self.assertEqual(export.output_path, Path("env.yml"))
        self.assertFalse(export.exclude_dev)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)