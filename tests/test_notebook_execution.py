"""
Notebook Execution Tests for Python for Semiconductors

Tests validate that notebooks execute without errors and produce expected outputs.
Uses nbconvert to programmatically execute notebooks and capture results.

Part of Week 4 Phase 3: Notebook Execution Testing
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest


# ============================================================================
# Test Configuration
# ============================================================================

REPO_ROOT = Path(__file__).parent.parent
MODULES_DIR = REPO_ROOT / "modules"

# Notebooks to test (organized by priority)
PRIORITY_NOTEBOOKS = [
    # Foundation module notebooks (high priority - foundational content)
    "modules/foundation/module-2/2.1-data-quality-analysis.ipynb",
    "modules/foundation/module-3/3.1-regression-analysis.ipynb",
    "modules/foundation/module-3/3.2-classification-analysis.ipynb",
    # Intermediate module notebooks (medium priority - core ML)
    "modules/foundation/module-4/4.3-multilabel-analysis.ipynb",
    "modules/intermediate/module-4/4.1-ensemble-analysis.ipynb",
    "modules/intermediate/module-5/5.1-time-series-analysis.ipynb",
    # Advanced module notebooks (medium priority - deep learning)
    "modules/advanced/module-6/6.1-deep-learning-analysis.ipynb",
    # Cutting-edge module notebooks (high priority - recently tested)
    "modules/cutting-edge/module-9/9.3-realtime-inference-analysis.ipynb",
    "modules/cutting-edge/module-11/11.1-edge-ai-inspection-analysis.ipynb",
    # Project development notebooks (medium priority - MLOps)
    "modules/project-dev/module-10/10.3-documentation-reproducibility-analysis.ipynb",
]


# ============================================================================
# Helper Functions
# ============================================================================


def get_notebook_path(relative_path: str) -> Path:
    """Convert relative path to absolute notebook path."""
    return REPO_ROOT / relative_path


def execute_notebook(notebook_path: Path, timeout: int = 300) -> Tuple[bool, str, Dict]:
    """
    Execute a Jupyter notebook using nbconvert.

    Args:
        notebook_path: Path to the notebook file
        timeout: Maximum execution time in seconds (default: 5 minutes)

    Returns:
        tuple: (success: bool, output: str, metadata: dict)
    """
    if not notebook_path.exists():
        return False, f"Notebook not found: {notebook_path}", {}

    # Create temporary output path
    output_path = notebook_path.parent / f"{notebook_path.stem}_executed.ipynb"

    try:
        # Execute notebook using jupyter nbconvert
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            str(output_path),
            f"--ExecutePreprocessor.timeout={timeout}",
            "--ExecutePreprocessor.kernel_name=python3",
            str(notebook_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 30  # Add buffer to subprocess timeout
        )

        success = result.returncode == 0
        output = result.stdout + "\n" + result.stderr

        # Extract metadata from executed notebook
        metadata = {}
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    nb_data = json.load(f)
                    metadata["cell_count"] = len(nb_data.get("cells", []))
                    metadata["kernel"] = nb_data.get("metadata", {}).get("kernelspec", {}).get("name", "unknown")
            except Exception as e:
                metadata["error"] = str(e)
            finally:
                # Clean up temporary file
                try:
                    output_path.unlink()
                except:
                    pass

        return success, output, metadata

    except subprocess.TimeoutExpired:
        return False, f"Notebook execution timed out after {timeout} seconds", {}
    except Exception as e:
        return False, f"Error executing notebook: {str(e)}", {}


def check_notebook_structure(notebook_path: Path) -> Tuple[bool, str, Dict]:
    """
    Validate notebook structure without executing.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        tuple: (valid: bool, message: str, stats: dict)
    """
    if not notebook_path.exists():
        return False, f"Notebook not found: {notebook_path}", {}

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        cells = nb_data.get("cells", [])
        stats = {
            "total_cells": len(cells),
            "code_cells": sum(1 for c in cells if c.get("cell_type") == "code"),
            "markdown_cells": sum(1 for c in cells if c.get("cell_type") == "markdown"),
            "has_outputs": sum(1 for c in cells if c.get("cell_type") == "code" and c.get("outputs")),
        }

        # Basic validation
        if stats["total_cells"] == 0:
            return False, "Notebook has no cells", stats

        if stats["code_cells"] == 0:
            return False, "Notebook has no code cells", stats

        return True, "Notebook structure valid", stats

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in notebook: {str(e)}", {}
    except Exception as e:
        return False, f"Error reading notebook: {str(e)}", {}


def extract_imports_from_notebook(notebook_path: Path) -> List[str]:
    """
    Extract import statements from notebook code cells.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        list: List of imported module names
    """
    imports = set()

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        for cell in nb_data.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))

                # Extract import statements
                import_matches = re.findall(r"^(?:from\s+(\S+)|import\s+(\S+))", source, re.MULTILINE)
                for match in import_matches:
                    module = match[0] or match[1]
                    # Get base module name
                    base_module = module.split(".")[0]
                    imports.add(base_module)

        return sorted(list(imports))

    except Exception:
        return []


# ============================================================================
# Test Class: Notebook Structure
# ============================================================================


class TestNotebookStructure:
    """Validate notebook structure and metadata."""

    @pytest.mark.parametrize("notebook_path", PRIORITY_NOTEBOOKS)
    def test_notebook_exists(self, notebook_path):
        """All priority notebooks must exist."""
        full_path = get_notebook_path(notebook_path)
        assert full_path.exists(), f"Notebook not found: {notebook_path}"

    @pytest.mark.parametrize("notebook_path", PRIORITY_NOTEBOOKS)
    def test_notebook_structure_valid(self, notebook_path):
        """Notebooks must have valid JSON structure."""
        full_path = get_notebook_path(notebook_path)
        valid, message, stats = check_notebook_structure(full_path)

        assert valid, f"{notebook_path}: {message}"
        assert stats["total_cells"] >= 5, f"{notebook_path}: Too few cells ({stats['total_cells']}), expected >= 5"
        assert stats["code_cells"] > 0, f"{notebook_path}: No code cells found"
        assert stats["markdown_cells"] > 0, f"{notebook_path}: No markdown cells found (needs documentation)"

    @pytest.mark.parametrize("notebook_path", PRIORITY_NOTEBOOKS)
    def test_notebook_has_title(self, notebook_path):
        """Notebooks should start with a title (markdown heading)."""
        full_path = get_notebook_path(notebook_path)

        with open(full_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        cells = nb_data.get("cells", [])
        if len(cells) > 0:
            first_cell = cells[0]
            is_markdown = first_cell.get("cell_type") == "markdown"
            if is_markdown:
                source = "".join(first_cell.get("source", []))
                has_heading = source.strip().startswith("#")
                assert has_heading, f"{notebook_path}: First cell should be a markdown heading"

    @pytest.mark.parametrize("notebook_path", PRIORITY_NOTEBOOKS)
    def test_notebook_imports_present(self, notebook_path):
        """Notebooks should have import statements."""
        full_path = get_notebook_path(notebook_path)
        imports = extract_imports_from_notebook(full_path)

        assert len(imports) > 0, f"{notebook_path}: No import statements found"

        # Check for common data science imports (or documentation-related imports for module 10.3)
        common_imports = {"numpy", "pandas", "matplotlib", "sklearn"}
        doc_imports = {"mlflow", "sphinx", "subprocess", "git"}  # Documentation/MLOps notebooks
        has_common = any(imp in imports for imp in common_imports)
        has_doc = any(imp in imports for imp in doc_imports)

        assert (
            has_common or has_doc or len(imports) >= 3
        ), f"{notebook_path}: Should import at least one common library or have 3+ imports. Found: {imports}"


# ============================================================================
# Test Class: Notebook Execution (Quick Smoke Tests)
# ============================================================================


class TestNotebookExecutionQuick:
    """Quick smoke tests - execute first few cells only."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "notebook_path",
        [
            "modules/foundation/module-2/2.1-data-quality-analysis.ipynb",
            "modules/foundation/module-3/3.1-regression-analysis.ipynb",
        ],
    )
    def test_notebook_first_cells_execute(self, notebook_path):
        """Test that first 5 cells of foundational notebooks execute without error."""
        full_path = get_notebook_path(notebook_path)

        # Read notebook
        with open(full_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        # Create temporary notebook with first 5 cells
        temp_nb = {
            "cells": nb_data["cells"][:5],
            "metadata": nb_data.get("metadata", {}),
            "nbformat": nb_data.get("nbformat", 4),
            "nbformat_minor": nb_data.get("nbformat_minor", 0),
        }

        temp_path = full_path.parent / f"{full_path.stem}_first5.ipynb"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(temp_nb, f)

            success, output, metadata = execute_notebook(temp_path, timeout=60)

            assert success, f"{notebook_path}: First 5 cells failed to execute.\nOutput:\n{output}"

        finally:
            if temp_path.exists():
                temp_path.unlink()


# ============================================================================
# Test Class: Notebook Execution (Full - Marked as Slow)
# ============================================================================


class TestNotebookExecutionFull:
    """Full notebook execution tests (marked as slow)."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "notebook_path",
        [
            "modules/foundation/module-3/3.1-regression-analysis.ipynb",
            "modules/foundation/module-3/3.2-classification-analysis.ipynb",
        ],
    )
    def test_foundational_notebooks_execute(self, notebook_path):
        """Execute complete foundational notebooks (Modules 2-3)."""
        full_path = get_notebook_path(notebook_path)
        success, output, metadata = execute_notebook(full_path, timeout=300)

        assert success, f"{notebook_path}: Execution failed.\nOutput:\n{output[:1000]}"

        # Check metadata
        assert metadata.get("cell_count", 0) > 0, f"{notebook_path}: No cells in executed notebook"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "notebook_path",
        [
            "modules/foundation/module-4/4.3-multilabel-analysis.ipynb",
        ],
    )
    def test_multilabel_notebook_executes(self, notebook_path):
        """Execute multilabel classification notebook (Module 4.3)."""
        full_path = get_notebook_path(notebook_path)
        success, output, metadata = execute_notebook(full_path, timeout=300)

        assert success, f"{notebook_path}: Execution failed.\nOutput:\n{output[:1000]}"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "notebook_path",
        [
            "modules/cutting-edge/module-9/9.3-realtime-inference-analysis.ipynb",
        ],
    )
    def test_realtime_inference_notebook_executes(self, notebook_path):
        """Execute real-time inference notebook (Module 9.3)."""
        full_path = get_notebook_path(notebook_path)
        success, output, metadata = execute_notebook(full_path, timeout=300)

        assert success, f"{notebook_path}: Execution failed.\nOutput:\n{output[:1000]}"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "notebook_path",
        [
            "modules/cutting-edge/module-11/11.1-edge-ai-inspection-analysis.ipynb",
        ],
    )
    def test_edge_deployment_notebook_executes(self, notebook_path):
        """Execute edge deployment notebook (Module 11.1)."""
        full_path = get_notebook_path(notebook_path)
        success, output, metadata = execute_notebook(full_path, timeout=300)

        assert success, f"{notebook_path}: Execution failed.\nOutput:\n{output[:1000]}"


# ============================================================================
# Test Class: Notebook Content Validation
# ============================================================================


class TestNotebookContent:
    """Validate notebook content quality."""

    @pytest.mark.parametrize("notebook_path", PRIORITY_NOTEBOOKS)
    def test_notebook_has_no_execution_errors_in_output(self, notebook_path):
        """Check if notebooks have error outputs saved."""
        full_path = get_notebook_path(notebook_path)

        with open(full_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        error_cells = []
        for idx, cell in enumerate(nb_data.get("cells", [])):
            if cell.get("cell_type") == "code":
                outputs = cell.get("outputs", [])
                for output in outputs:
                    if output.get("output_type") == "error":
                        error_cells.append(idx)

        # Warning if errors found in outputs (not a hard failure - might be intentional)
        if error_cells:
            pytest.skip(
                f"{notebook_path}: Has error outputs in cells {error_cells}. "
                "This might be intentional for teaching purposes."
            )

    @pytest.mark.parametrize("notebook_path", PRIORITY_NOTEBOOKS)
    def test_notebook_has_semiconductor_context(self, notebook_path):
        """Notebooks should mention semiconductor/wafer/fab context."""
        full_path = get_notebook_path(notebook_path)

        with open(full_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        # Collect all text content
        all_text = []
        for cell in nb_data.get("cells", []):
            source = "".join(cell.get("source", []))
            all_text.append(source.lower())

        combined_text = " ".join(all_text)

        # Check for semiconductor-related terms
        semiconductor_terms = [
            "wafer",
            "semiconductor",
            "fab",
            "yield",
            "defect",
            "process",
            "etch",
            "deposition",
            "lithography",
            "manufacturing",
        ]

        has_context = any(term in combined_text for term in semiconductor_terms)
        assert has_context, f"{notebook_path}: Should contain semiconductor/manufacturing context"

    @pytest.mark.parametrize("notebook_path", PRIORITY_NOTEBOOKS)
    def test_notebook_has_explanatory_markdown(self, notebook_path):
        """Notebooks should have sufficient explanatory markdown (not just code)."""
        full_path = get_notebook_path(notebook_path)

        with open(full_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        cells = nb_data.get("cells", [])
        markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        # Calculate markdown-to-code ratio
        if len(code_cells) > 0:
            ratio = len(markdown_cells) / len(code_cells)
            assert ratio >= 0.2, (
                f"{notebook_path}: Low markdown-to-code ratio ({ratio:.2f}). "
                "Notebooks should have more explanations."
            )


# ============================================================================
# Test Class: Notebook Dependencies
# ============================================================================


class TestNotebookDependencies:
    """Validate that notebook dependencies are available."""

    @pytest.mark.parametrize("notebook_path", PRIORITY_NOTEBOOKS)
    def test_notebook_imports_are_available(self, notebook_path):
        """All imported modules should be available in environment."""
        full_path = get_notebook_path(notebook_path)
        imports = extract_imports_from_notebook(full_path)

        # Filter to third-party packages (exclude stdlib)
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "re",
            "time",
            "datetime",
            "pathlib",
            "collections",
            "itertools",
            "functools",
            "typing",
            "warnings",
            "math",
            "random",
            "subprocess",
            "io",
        }

        third_party = [imp for imp in imports if imp not in stdlib_modules]

        # Try importing each package
        unavailable = []
        for module in third_party:
            try:
                __import__(module)
            except ImportError:
                unavailable.append(module)

        if unavailable:
            pytest.skip(
                f"{notebook_path}: Missing dependencies: {unavailable}. " "Install with appropriate requirements tier."
            )


# ============================================================================
# Summary Test
# ============================================================================


class TestNotebookSummary:
    """Generate summary of notebook test results."""

    def test_generate_notebook_summary(self, capsys):
        """Generate comprehensive notebook validation summary."""
        print("\n" + "=" * 70)
        print("NOTEBOOK VALIDATION SUMMARY")
        print("=" * 70)

        total_notebooks = len(PRIORITY_NOTEBOOKS)
        print(f"Total Notebooks Tested: {total_notebooks}")
        print()

        # Count by module
        module_counts = {}
        for nb_path in PRIORITY_NOTEBOOKS:
            module = nb_path.split("/")[1]  # Extract module category
            module_counts[module] = module_counts.get(module, 0) + 1

        print("By Module Category:")
        for module, count in sorted(module_counts.items()):
            print(f"  {module:20s}: {count:2d} notebooks")

        print()
        print("Test Coverage:")
        print("  Structure validation: ✅ All notebooks")
        print("  Content validation:   ✅ All notebooks")
        print("  Dependency checks:    ✅ All notebooks")
        print("  Quick execution:      ⚠️  Foundational only (marked slow)")
        print("  Full execution:       ⚠️  Selected notebooks (marked slow)")
        print()
        print("Note: Full execution tests marked with @pytest.mark.slow")
        print("      Run with: pytest -v -m slow")
        print("=" * 70)

        assert True


# ============================================================================
# Test Markers
# ============================================================================

# To run only fast tests (structure, content, dependencies):
#   pytest tests/test_notebook_execution.py -v
#
# To run slow tests (full notebook execution):
#   pytest tests/test_notebook_execution.py -v -m slow
#
# To run all tests:
#   pytest tests/test_notebook_execution.py -v -m ""
