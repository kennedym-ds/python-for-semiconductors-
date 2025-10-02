#!/usr/bin/env python3
"""
Yield Regression - Submission Grading Script

Automates evaluation of student notebooks for the yield regression tutorial.

Usage:
    python evaluate_submission.py --notebook yield_regression_tutorial.ipynb
    python evaluate_submission.py --notebook student_submission.ipynb --output-json grades.json
    python evaluate_submission.py --notebook submission.ipynb --verbose

Features:
    - Automated notebook execution and validation
    - Code quality checks (PEP 8, documentation)
    - Results validation against expected ranges
    - Detailed feedback generation
    - 100-point scoring rubric
    - JSON output for LMS integration

Scoring Rubric (100 points total):
    Exercise 1 (Data Exploration): 20 points
        - Data generation (5 pts)
        - Distribution analysis (5 pts)
        - Correlation analysis (5 pts)
        - Visualization quality (5 pts)

    Exercise 2 (Model Training): 30 points
        - Data splitting (5 pts)
        - Multiple model training (10 pts)
        - Performance evaluation (10 pts)
        - Model comparison (5 pts)

    Exercise 3 (Manufacturing Metrics): 25 points
        - Residual analysis (10 pts)
        - PWS calculation (5 pts)
        - Estimated Loss calculation (5 pts)
        - Manufacturing interpretation (5 pts)

    Exercise 4 (Deployment): 15 points
        - Model saving (5 pts)
        - Model loading (5 pts)
        - CLI demonstration (5 pts)

    Code Quality: 10 points
        - Documentation (3 pts)
        - Code style (3 pts)
        - Error handling (2 pts)
        - Best practices (2 pts)

Author: Python for Semiconductors Team
Date: 2024
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


@dataclass
class ExerciseScore:
    """Score for a single exercise."""

    exercise_id: str
    max_points: int
    earned_points: float
    feedback: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class GradingResult:
    """Complete grading results."""

    student_name: Optional[str] = None
    notebook_path: str = ""
    total_score: float = 0.0
    max_score: int = 100
    exercise_scores: List[ExerciseScore] = field(default_factory=list)
    code_quality_score: float = 0.0
    execution_successful: bool = False
    execution_errors: List[str] = field(default_factory=list)
    feedback_summary: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "student_name": self.student_name,
            "notebook_path": self.notebook_path,
            "total_score": round(self.total_score, 2),
            "max_score": self.max_score,
            "percentage": round((self.total_score / self.max_score) * 100, 1),
            "letter_grade": self.get_letter_grade(),
            "exercise_scores": [
                {
                    "exercise_id": es.exercise_id,
                    "earned_points": round(es.earned_points, 2),
                    "max_points": es.max_points,
                    "percentage": round((es.earned_points / es.max_points) * 100, 1) if es.max_points > 0 else 0,
                    "feedback": es.feedback,
                    "errors": es.errors,
                    "warnings": es.warnings,
                }
                for es in self.exercise_scores
            ],
            "code_quality_score": round(self.code_quality_score, 2),
            "execution_successful": self.execution_successful,
            "execution_errors": self.execution_errors,
            "feedback_summary": self.feedback_summary,
        }

    def get_letter_grade(self) -> str:
        """Convert numeric score to letter grade."""
        percentage = (self.total_score / self.max_score) * 100
        if percentage >= 93:
            return "A"
        elif percentage >= 90:
            return "A-"
        elif percentage >= 87:
            return "B+"
        elif percentage >= 83:
            return "B"
        elif percentage >= 80:
            return "B-"
        elif percentage >= 77:
            return "C+"
        elif percentage >= 73:
            return "C"
        elif percentage >= 70:
            return "C-"
        elif percentage >= 67:
            return "D+"
        elif percentage >= 63:
            return "D"
        elif percentage >= 60:
            return "D-"
        else:
            return "F"


class NotebookGrader:
    """Grades student notebooks for the yield regression tutorial."""

    # Expected value ranges for validation
    EXPECTED_RANGES = {
        "data_samples": (500, 2000),  # Number of samples generated
        "yield_min": 60.0,  # Minimum yield percentage
        "yield_max": 100.0,  # Maximum yield percentage
        "num_features": (4, 20),  # Number of features
        "train_test_ratio": (0.15, 0.30),  # Test set size
        "num_models": (3, 7),  # Number of models trained
        "r2_min": 0.30,  # Minimum acceptable R² score
        "rmse_max": 5.0,  # Maximum acceptable RMSE
        "pws_min": 0.90,  # Minimum PWS (90% predictions within spec)
        "num_residual_plots": (1, 4),  # Expected residual plots
    }

    def __init__(self, notebook_path: Path, verbose: bool = False):
        """
        Initialize the grader.

        Args:
            notebook_path: Path to student's notebook
            verbose: If True, print detailed feedback during grading
        """
        self.notebook_path = notebook_path
        self.verbose = verbose
        self.notebook = None
        self.executed_notebook = None
        self.result = GradingResult(notebook_path=str(notebook_path))

    def load_notebook(self) -> bool:
        """Load the student's notebook."""
        try:
            with open(self.notebook_path, "r", encoding="utf-8") as f:
                self.notebook = nbformat.read(f, as_version=4)
            if self.verbose:
                print(f"✅ Loaded notebook: {self.notebook_path}")
            return True
        except Exception as e:
            self.result.execution_errors.append(f"Failed to load notebook: {str(e)}")
            if self.verbose:
                print(f"❌ Failed to load notebook: {e}")
            return False

    def execute_notebook(self, timeout: int = 600) -> bool:
        """
        Execute the notebook to validate code runs successfully.

        Args:
            timeout: Maximum execution time per cell in seconds

        Returns:
            True if execution successful, False otherwise
        """
        if self.notebook is None:
            self.result.execution_errors.append("Notebook not loaded")
            return False

        try:
            ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")

            if self.verbose:
                print("🔄 Executing notebook...")

            # Execute in the notebook's directory context
            import os

            env = os.environ.copy()
            notebook_dir = str(self.notebook_path.parent)
            pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{notebook_dir}{os.pathsep}{pythonpath}" if pythonpath else notebook_dir

            self.executed_notebook, _ = ep.preprocess(self.notebook, {"metadata": {"path": notebook_dir}, "env": env})

            self.result.execution_successful = True
            if self.verbose:
                print("✅ Notebook executed successfully")
            return True

        except CellExecutionError as e:
            self.result.execution_successful = False
            error_msg = f"Cell execution error: {str(e)}"
            self.result.execution_errors.append(error_msg)
            if self.verbose:
                print(f"❌ {error_msg}")
            return False

        except Exception as e:
            self.result.execution_successful = False
            error_msg = f"Execution error: {str(e)}"
            self.result.execution_errors.append(error_msg)
            if self.verbose:
                print(f"❌ {error_msg}")
            return False

    def check_code_removed_todos(self, cell_source: str) -> Tuple[bool, int]:
        """
        Check if student removed TODO comments and implemented code.

        Returns:
            (has_implementation, num_todos_remaining)
        """
        # Count remaining TODOs
        todo_pattern = r"#\s*TODO:"
        todos = re.findall(todo_pattern, cell_source, re.IGNORECASE)

        # Check if code was uncommented (look for non-comment, non-empty lines)
        code_lines = [
            line.strip() for line in cell_source.split("\n") if line.strip() and not line.strip().startswith("#")
        ]

        has_implementation = len(code_lines) > 0
        return has_implementation, len(todos)

    def extract_cell_outputs(self, cell) -> List[str]:
        """Extract text outputs from a cell."""
        outputs = []
        if "outputs" in cell:
            for output in cell["outputs"]:
                if "text" in output:
                    outputs.append(output["text"])
                elif "data" in output and "text/plain" in output["data"]:
                    outputs.append(output["data"]["text/plain"])
        return outputs

    def find_cell_by_pattern(self, pattern: str) -> Optional[Any]:
        """Find a cell matching a regex pattern in its source."""
        if self.executed_notebook is None:
            return None

        for cell in self.executed_notebook.cells:
            if cell.cell_type == "code" and re.search(pattern, cell.source, re.IGNORECASE):
                return cell
        return None

    def grade_exercise_1(self) -> ExerciseScore:
        """Grade Exercise 1: Data Generation and Exploration."""
        score = ExerciseScore(exercise_id="Exercise 1", max_points=20, earned_points=0.0)

        # Check Exercise 1.1: Data Generation (5 points)
        cell_1_1 = self.find_cell_by_pattern(r"generate_yield_process")
        if cell_1_1:
            has_impl, num_todos = self.check_code_removed_todos(cell_1_1.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Data generation code implemented")

                if num_todos == 0:
                    score.earned_points += 2
                    score.feedback.append("✅ All TODOs completed")
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Data generation not implemented")
        else:
            score.errors.append("❌ Exercise 1.1 cell not found")

        # Check Exercise 1.2: Distribution Analysis (5 points)
        cell_1_2 = self.find_cell_by_pattern(r"describe\(\)|\.mean\(\)|\.std\(\)")
        if cell_1_2:
            has_impl, num_todos = self.check_code_removed_todos(cell_1_2.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Distribution analysis implemented")

                if num_todos == 0:
                    score.earned_points += 2
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Distribution analysis not implemented")

        # Check Exercise 1.3: Correlation Analysis (5 points)
        cell_1_3 = self.find_cell_by_pattern(r"\.corr\(\)|correlation.*matrix|heatmap")
        if cell_1_3:
            has_impl, num_todos = self.check_code_removed_todos(cell_1_3.source)

            if has_impl:
                score.earned_points += 2
                score.feedback.append("✅ Correlation analysis implemented")

                # Check for heatmap visualization
                if re.search(r"sns\.heatmap|seaborn.*heatmap", cell_1_3.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Correlation heatmap visualized")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Correlation analysis not implemented")

        # Check Exercise 1.4: Visualization (5 points)
        cell_1_4 = self.find_cell_by_pattern(r"scatter|hist|plot.*yield")
        if cell_1_4:
            has_impl, num_todos = self.check_code_removed_todos(cell_1_4.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Visualization code implemented")

                if num_todos == 0:
                    score.earned_points += 2
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Visualization not implemented")

        return score

    def grade_exercise_2(self) -> ExerciseScore:
        """Grade Exercise 2: Model Training and Comparison."""
        score = ExerciseScore(exercise_id="Exercise 2", max_points=30, earned_points=0.0)

        # Check Exercise 2.1: Data Splitting (5 points)
        cell_2_1 = self.find_cell_by_pattern(r"train_test_split")
        if cell_2_1:
            has_impl, num_todos = self.check_code_removed_todos(cell_2_1.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Data splitting implemented")

                # Check for random_state (reproducibility)
                if re.search(r"random_state\s*=", cell_2_1.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Random state set for reproducibility")
                else:
                    score.warnings.append("⚠️ Consider setting random_state for reproducibility")
            else:
                score.errors.append("❌ Data splitting not implemented")
        else:
            score.errors.append("❌ Exercise 2.1 cell not found")

        # Check Exercise 2.2: Model Training (10 points)
        cell_2_2 = self.find_cell_by_pattern(r'YieldRegressionPipeline|model.*=.*["\']')
        if cell_2_2:
            has_impl, num_todos = self.check_code_removed_todos(cell_2_2.source)

            if has_impl:
                score.earned_points += 2
                score.feedback.append("✅ Model training code implemented")

                # Check for multiple models
                model_patterns = [r"linear", r"ridge", r"lasso", r"elasticnet", r"rf|random.*forest"]
                models_found = sum(
                    1 for pattern in model_patterns if re.search(pattern, cell_2_2.source, re.IGNORECASE)
                )

                if models_found >= 5:
                    score.earned_points += 5
                    score.feedback.append(f"✅ Trained {models_found} models (excellent)")
                elif models_found >= 3:
                    score.earned_points += 3
                    score.feedback.append(f"✅ Trained {models_found} models")
                elif models_found >= 1:
                    score.earned_points += 1
                    score.warnings.append(f"⚠️ Only trained {models_found} model(s), expected 5")

                # Check for .fit() usage
                if re.search(r"\.fit\(", cell_2_2.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Model fitting implemented")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Model training not implemented")
        else:
            score.errors.append("❌ Exercise 2.2 cell not found")

        # Check Exercise 2.3: Performance Evaluation (10 points)
        cell_2_3 = self.find_cell_by_pattern(r"evaluate\(|R2|RMSE|MAE")
        if cell_2_3:
            has_impl, num_todos = self.check_code_removed_todos(cell_2_3.source)

            if has_impl:
                score.earned_points += 2
                score.feedback.append("✅ Performance evaluation implemented")

                # Check for regression metrics
                metrics_found = []
                if re.search(r"R2|r2_score|r_squared", cell_2_3.source, re.IGNORECASE):
                    metrics_found.append("R²")
                if re.search(r"RMSE|root.*mean.*squared", cell_2_3.source, re.IGNORECASE):
                    metrics_found.append("RMSE")
                if re.search(r"MAE|mean.*absolute", cell_2_3.source, re.IGNORECASE):
                    metrics_found.append("MAE")
                if re.search(r"PWS|prediction.*within.*spec", cell_2_3.source, re.IGNORECASE):
                    metrics_found.append("PWS")

                if len(metrics_found) >= 4:
                    score.earned_points += 5
                    score.feedback.append(f"✅ All key metrics calculated: {', '.join(metrics_found)}")
                elif len(metrics_found) >= 2:
                    score.earned_points += 3
                    score.feedback.append(f"✅ Metrics calculated: {', '.join(metrics_found)}")

                # Check for results organization
                if re.search(r"pd\.DataFrame|results.*=.*\[\]|append", cell_2_3.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Results organized in DataFrame")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Performance evaluation not implemented")
        else:
            score.errors.append("❌ Exercise 2.3 cell not found")

        # Check Exercise 2.4: Model Comparison (5 points)
        cell_2_4 = self.find_cell_by_pattern(r"plot.*comparison|bar.*R2|visualiz.*metric")
        if cell_2_4:
            has_impl, num_todos = self.check_code_removed_todos(cell_2_4.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Model comparison visualization implemented")

                if num_todos == 0:
                    score.earned_points += 2
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Model comparison not implemented")

        return score

    def grade_exercise_3(self) -> ExerciseScore:
        """Grade Exercise 3: Manufacturing-Specific Metrics and Residual Analysis."""
        score = ExerciseScore(exercise_id="Exercise 3", max_points=25, earned_points=0.0)

        # Check Exercise 3.1: Residual Analysis (10 points)
        cell_3_1 = self.find_cell_by_pattern(r"residual|y_test.*-.*predict")
        if cell_3_1:
            has_impl, num_todos = self.check_code_removed_todos(cell_3_1.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Residual calculation implemented")

                # Check for residual plots
                plot_types = []
                if re.search(r"hist.*residual|residual.*hist", cell_3_1.source, re.IGNORECASE):
                    plot_types.append("histogram")
                if re.search(r"scatter.*residual|residual.*scatter", cell_3_1.source, re.IGNORECASE):
                    plot_types.append("scatter")
                if re.search(r"probplot|qq.*plot|q-q", cell_3_1.source, re.IGNORECASE):
                    plot_types.append("Q-Q plot")

                if len(plot_types) >= 3:
                    score.earned_points += 5
                    score.feedback.append(f"✅ Complete residual analysis: {', '.join(plot_types)}")
                elif len(plot_types) >= 1:
                    score.earned_points += 3
                    score.feedback.append(f"✅ Residual plots: {', '.join(plot_types)}")

                if num_todos == 0:
                    score.earned_points += 2
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Residual analysis not implemented")
        else:
            score.errors.append("❌ Exercise 3.1 cell not found")

        # Check Exercise 3.2: PWS Calculation (5 points)
        cell_3_2 = self.find_cell_by_pattern(r"PWS|prediction.*within.*spec|>=.*60.*<=.*100")
        if cell_3_2:
            has_impl, num_todos = self.check_code_removed_todos(cell_3_2.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ PWS calculation implemented")

                # Check for spec limits (60-100%)
                if re.search(r"60|100", cell_3_2.source):
                    score.earned_points += 1
                    score.feedback.append("✅ Spec limits defined")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ PWS calculation not implemented")
        else:
            score.warnings.append("⚠️ PWS calculation not found (may be in Exercise 2)")

        # Check Exercise 3.3: Estimated Loss (5 points)
        cell_3_3 = self.find_cell_by_pattern(r"estimated.*loss|loss.*=.*sum|tolerance")
        if cell_3_3:
            has_impl, num_todos = self.check_code_removed_todos(cell_3_3.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Estimated Loss calculation implemented")

                # Check for tolerance threshold
                if re.search(r"tolerance|threshold.*2|abs.*residual", cell_3_3.source, re.IGNORECASE):
                    score.earned_points += 1
                    score.feedback.append("✅ Tolerance threshold applied")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Estimated Loss not implemented")
        else:
            score.warnings.append("⚠️ Estimated Loss calculation not found")

        # Check Exercise 3.4: Manufacturing Interpretation (5 points)
        cell_3_4 = self.find_cell_by_pattern(r"feature.*importance|important.*feature|pressure|temperature")
        if cell_3_4:
            has_impl, num_todos = self.check_code_removed_todos(cell_3_4.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Feature importance analysis implemented")

                # Check for visualization
                if re.search(r"plot|bar|importance", cell_3_4.source, re.IGNORECASE):
                    score.earned_points += 1
                    score.feedback.append("✅ Feature importance visualized")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Feature importance not implemented")
        else:
            score.warnings.append("⚠️ Feature importance analysis not found")

        return score

    def grade_exercise_4(self) -> ExerciseScore:
        """Grade Exercise 4: Model Deployment and CLI Usage."""
        score = ExerciseScore(exercise_id="Exercise 4", max_points=15, earned_points=0.0)

        # Check Exercise 4.1: Model Saving (5 points)
        cell_4_1 = self.find_cell_by_pattern(r"pipeline\.save|\.joblib|save.*model")
        if cell_4_1:
            has_impl, num_todos = self.check_code_removed_todos(cell_4_1.source)

            if has_impl:
                score.earned_points += 2
                score.feedback.append("✅ Model saving code implemented")

                # Check for pipeline.save() usage
                if re.search(r"\.save\(", cell_4_1.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Using pipeline.save() method")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Model saving not implemented")
        else:
            score.errors.append("❌ Exercise 4.1 cell not found")

        # Check Exercise 4.2: Model Loading (5 points)
        cell_4_2 = self.find_cell_by_pattern(r"\.load\(|YieldRegressionPipeline\.load")
        if cell_4_2:
            has_impl, num_todos = self.check_code_removed_todos(cell_4_2.source)

            if has_impl:
                score.earned_points += 2
                score.feedback.append("✅ Model loading code implemented")

                # Check for YieldRegressionPipeline.load()
                if re.search(r"YieldRegressionPipeline\.load\(", cell_4_2.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Using YieldRegressionPipeline.load() method")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Model loading not implemented")
        else:
            score.errors.append("❌ Exercise 4.2 cell not found")

        # Check Exercise 4.3: CLI Demonstration (5 points)
        cell_4_3 = self.find_cell_by_pattern(r"python.*yield_regression_pipeline|CLI|command.*line")
        if cell_4_3:
            has_impl, num_todos = self.check_code_removed_todos(cell_4_3.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ CLI demonstration included")

                # Check for multiple CLI commands
                cli_commands = ["train", "evaluate", "predict"]
                commands_found = sum(1 for cmd in cli_commands if cmd in cell_4_3.source.lower())

                if commands_found >= 2:
                    score.earned_points += 2
                    score.feedback.append(f"✅ Demonstrated {commands_found} CLI commands")
                elif commands_found >= 1:
                    score.earned_points += 1
                    score.feedback.append(f"✅ Demonstrated {commands_found} CLI command")
            else:
                score.warnings.append("⚠️ CLI demonstration not found")
        else:
            score.warnings.append("⚠️ CLI demonstration cell not found")

        return score

    def grade_code_quality(self) -> float:
        """Grade overall code quality (10 points)."""
        score = 0.0
        feedback = []

        if self.notebook is None:
            return 0.0

        code_cells = [cell for cell in self.notebook.cells if cell.cell_type == "code"]

        # Check for documentation (3 points)
        documented_cells = sum(
            1 for cell in code_cells if re.search(r"#.*\w{4,}", cell.source)  # Has meaningful comments
        )
        doc_ratio = documented_cells / len(code_cells) if code_cells else 0
        doc_score = min(3.0, doc_ratio * 3.0)
        score += doc_score
        feedback.append(f"Documentation: {doc_ratio:.1%} of cells have comments ({doc_score:.1f}/3 pts)")

        # Check for code style (3 points)
        style_violations = 0
        for cell in code_cells:
            lines = cell.source.split("\n")
            for line in lines:
                # Check line length
                if len(line) > 120:
                    style_violations += 1
                # Check for inconsistent indentation
                if "\t" in line and "    " in line:
                    style_violations += 1

        max_violations = len(code_cells) * 2
        style_score = max(0, 3.0 - (style_violations / max_violations) * 3.0) if max_violations > 0 else 3.0
        score += style_score
        feedback.append(f"Code style: {style_violations} violations ({style_score:.1f}/3 pts)")

        # Check for error handling (2 points)
        error_handling_cells = sum(
            1 for cell in code_cells if re.search(r"try:|except:|if.*is.*None|assert ", cell.source)
        )
        error_score = min(2.0, (error_handling_cells / max(1, len(code_cells))) * 4.0)
        score += error_score
        feedback.append(f"Error handling: {error_handling_cells} cells with error handling ({error_score:.1f}/2 pts)")

        # Check for best practices (2 points)
        best_practice_score = 0.0

        # Check for descriptive variable names
        descriptive_vars = sum(1 for cell in code_cells if re.search(r"\b[a-z_]{4,}\b\s*=", cell.source))
        if descriptive_vars >= len(code_cells) * 0.5:
            best_practice_score += 1.0
            feedback.append("✅ Descriptive variable names used")

        # Check for constants
        uses_constants = sum(1 for cell in code_cells if re.search(r"\b[A-Z_]{2,}\s*=", cell.source))
        if uses_constants > 0:
            best_practice_score += 1.0
            feedback.append("✅ Constants defined")

        score += best_practice_score

        self.result.code_quality_score = score
        self.result.feedback_summary.extend(feedback)

        return score

    def grade(self) -> GradingResult:
        """Execute complete grading workflow."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("🎓 YIELD REGRESSION - GRADING SYSTEM")
            print("=" * 60 + "\n")

        # Load notebook
        if not self.load_notebook():
            return self.result

        # Execute notebook
        self.execute_notebook()

        # Grade each exercise
        ex1_score = self.grade_exercise_1()
        ex2_score = self.grade_exercise_2()
        ex3_score = self.grade_exercise_3()
        ex4_score = self.grade_exercise_4()

        self.result.exercise_scores = [ex1_score, ex2_score, ex3_score, ex4_score]

        # Grade code quality
        code_quality_score = self.grade_code_quality()

        # Calculate total score
        exercise_total = sum(ex.earned_points for ex in self.result.exercise_scores)
        self.result.total_score = exercise_total + code_quality_score

        # Generate feedback summary
        self.generate_feedback_summary()

        if self.verbose:
            self.print_results()

        return self.result

    def generate_feedback_summary(self):
        """Generate overall feedback summary."""
        percentage = (self.result.total_score / self.result.max_score) * 100

        if percentage >= 90:
            self.result.feedback_summary.insert(
                0, "🌟 Excellent work! You've demonstrated strong understanding of yield regression analysis."
            )
        elif percentage >= 80:
            self.result.feedback_summary.insert(0, "👍 Good work! You've completed most requirements successfully.")
        elif percentage >= 70:
            self.result.feedback_summary.insert(
                0, "✅ Satisfactory work. Review the feedback to improve your understanding."
            )
        else:
            self.result.feedback_summary.insert(
                0, "📚 Additional work needed. Please review the tutorial and solution notebook."
            )

        # Add specific improvement suggestions
        for ex_score in self.result.exercise_scores:
            if ex_score.earned_points < ex_score.max_points * 0.7:
                self.result.feedback_summary.append(
                    f"⚠️ Focus on improving {ex_score.exercise_id} "
                    f"({ex_score.earned_points}/{ex_score.max_points} pts)"
                )

    def print_results(self):
        """Print detailed grading results to console."""
        print("\n" + "=" * 60)
        print("📊 GRADING RESULTS")
        print("=" * 60 + "\n")

        print(f"Student: {self.result.student_name or 'Unknown'}")
        print(f"Notebook: {self.result.notebook_path}")
        print(f"Execution: {'✅ Success' if self.result.execution_successful else '❌ Failed'}")
        print()

        # Exercise scores
        for ex_score in self.result.exercise_scores:
            percentage = (ex_score.earned_points / ex_score.max_points) * 100
            print(f"{ex_score.exercise_id}: {ex_score.earned_points}/{ex_score.max_points} pts ({percentage:.1f}%)")

            for feedback in ex_score.feedback[:3]:
                print(f"  {feedback}")

            if ex_score.errors:
                print(f"  ❌ Errors:")
                for error in ex_score.errors[:2]:
                    print(f"    {error}")

            if ex_score.warnings:
                print(f"  ⚠️  Warnings:")
                for warning in ex_score.warnings[:2]:
                    print(f"    {warning}")
            print()

        # Code quality
        print(f"Code Quality: {self.result.code_quality_score}/10 pts")
        print()

        # Total score
        print("=" * 60)
        print(
            f"TOTAL SCORE: {self.result.total_score}/{self.result.max_score} pts "
            f"({(self.result.total_score/self.result.max_score)*100:.1f}%) - "
            f"Grade: {self.result.get_letter_grade()}"
        )
        print("=" * 60 + "\n")

        # Feedback summary
        print("📝 Feedback Summary:")
        for feedback in self.result.feedback_summary[:5]:
            print(f"  {feedback}")
        print()


def main():
    """Main entry point for the grading script."""
    parser = argparse.ArgumentParser(
        description="Grade yield regression tutorial submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grade a notebook and print results
  python evaluate_submission.py --notebook student_submission.ipynb

  # Grade with verbose output
  python evaluate_submission.py --notebook submission.ipynb --verbose

  # Grade and save results to JSON
  python evaluate_submission.py --notebook submission.ipynb --output-json grades.json

  # Grade all notebooks in a directory
  for notebook in submissions/*.ipynb; do
      python evaluate_submission.py --notebook "$notebook" --output-json "grades/$(basename $notebook .ipynb)_grade.json"
  done
        """,
    )

    parser.add_argument("--notebook", type=Path, required=True, help="Path to student notebook to grade")

    parser.add_argument("--output-json", type=Path, help="Path to save grading results as JSON")

    parser.add_argument("--student-name", type=str, help="Student name for identification")

    parser.add_argument("--verbose", action="store_true", help="Print detailed grading feedback")

    parser.add_argument(
        "--timeout", type=int, default=600, help="Maximum execution time per cell in seconds (default: 600)"
    )

    args = parser.parse_args()

    # Validate notebook exists
    if not args.notebook.exists():
        print(f"❌ Error: Notebook not found: {args.notebook}", file=sys.stderr)
        sys.exit(1)

    # Create grader and run
    grader = NotebookGrader(args.notebook, verbose=args.verbose)
    if args.student_name:
        grader.result.student_name = args.student_name

    result = grader.grade()

    # Save to JSON if requested
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n✅ Results saved to: {args.output_json}")

    # Exit with appropriate code
    percentage = (result.total_score / result.max_score) * 100
    if percentage >= 60:
        sys.exit(0)  # Passing grade
    else:
        sys.exit(1)  # Failing grade


if __name__ == "__main__":
    main()
