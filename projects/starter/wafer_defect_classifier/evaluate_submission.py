#!/usr/bin/env python3
"""
Wafer Defect Classifier - Submission Grading Script

Automates evaluation of student notebooks for the wafer defect classification tutorial.

Usage:
    python evaluate_submission.py --notebook wafer_defect_tutorial.ipynb
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
        - Feature visualization (10 pts)
        - Statistical analysis (5 pts)

    Exercise 2 (Model Training): 30 points
        - Data splitting (5 pts)
        - Model training (10 pts)
        - Performance evaluation (10 pts)
        - Visualization (5 pts)

    Exercise 3 (Manufacturing Metrics): 25 points
        - Cost calculation (10 pts)
        - Threshold optimization (10 pts)
        - Analysis quality (5 pts)

    Exercise 4 (CLI Usage): 15 points
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
    """Grades student notebooks for the wafer defect classification tutorial."""

    # Expected value ranges for validation
    EXPECTED_RANGES = {
        "data_samples": (500, 2000),  # Number of samples generated
        "defect_rate": (0.05, 0.30),  # Defect rate (5-30%)
        "num_features": (3, 20),  # Number of features
        "train_test_ratio": (0.15, 0.30),  # Test set size
        "num_models": (3, 7),  # Number of models trained
        "roc_auc_min": 0.60,  # Minimum acceptable ROC-AUC
        "cost_fp": (10, 100),  # False positive cost range
        "cost_fn": (50, 500),  # False negative cost range (should be > FP)
        "num_thresholds": (5, 50),  # Thresholds tested in sweep
        "optimal_threshold_range": (0.15, 0.70),  # Reasonable threshold range
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
            # Add notebook directory to PYTHONPATH for imports
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
        cell_1_1 = self.find_cell_by_pattern(r"generate_wafer_defect_dataset")
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

        # Check Exercise 1.2: Visualization (15 points total)
        cell_1_2 = self.find_cell_by_pattern(r"violin.*plot|feature.*distribution")
        if cell_1_2:
            has_impl, num_todos = self.check_code_removed_todos(cell_1_2.source)

            if has_impl:
                score.earned_points += 7
                score.feedback.append("✅ Visualization code implemented")

                # Check for seaborn/matplotlib usage
                if re.search(r"sns\.violinplot|plt\.violinplot|sns\.boxplot", cell_1_2.source):
                    score.earned_points += 3
                    score.feedback.append("✅ Appropriate plot type used")

                # Check for feature analysis
                if re.search(r"p.*value|statistical|discriminative", cell_1_2.source, re.IGNORECASE):
                    score.earned_points += 3
                    score.feedback.append("✅ Statistical analysis included")

                if num_todos == 0:
                    score.earned_points += 2
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Visualization not implemented")
        else:
            score.errors.append("❌ Exercise 1.2 cell not found")

        return score

    def grade_exercise_2(self) -> ExerciseScore:
        """Grade Exercise 2: Model Training and Comparison."""
        score = ExerciseScore(exercise_id="Exercise 2", max_points=30, earned_points=0.0)

        # Check Exercise 2.1: Model Training (20 points)
        cell_2_1 = self.find_cell_by_pattern(r"train_test_split|WaferDefectPipeline")
        if cell_2_1:
            has_impl, num_todos = self.check_code_removed_todos(cell_2_1.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Model training code implemented")

                # Check for train/test split
                if re.search(r"train_test_split.*stratify", cell_2_1.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Stratified split used")

                # Check for multiple models
                model_types = len(re.findall(r"model_type\s*=", cell_2_1.source))
                if model_types >= 3:
                    score.earned_points += 5
                    score.feedback.append(f"✅ Trained {model_types} models")
                elif model_types >= 1:
                    score.earned_points += 3
                    score.warnings.append(f"⚠️ Only trained {model_types} model(s), expected 5")

                # Check for evaluation metrics
                if re.search(r"roc_auc|pr_auc|f1|pws", cell_2_1.source, re.IGNORECASE):
                    score.earned_points += 5
                    score.feedback.append("✅ Evaluated models with multiple metrics")

                # Check for results DataFrame
                if re.search(r"pd\.DataFrame|results_df", cell_2_1.source):
                    score.earned_points += 3
                    score.feedback.append("✅ Results organized in DataFrame")

                if num_todos == 0:
                    score.earned_points += 2
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Model training not implemented")
        else:
            score.errors.append("❌ Exercise 2.1 cell not found")

        # Check Exercise 2.2: Visualization (10 points)
        cell_2_2 = self.find_cell_by_pattern(r"roc_curve|plot.*roc|RocCurveDisplay")
        if cell_2_2:
            has_impl, num_todos = self.check_code_removed_todos(cell_2_2.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ ROC curve visualization implemented")

                # Check for multi-model ROC plot
                if re.search(r"for.*model|for.*name", cell_2_2.source):
                    score.earned_points += 4
                    score.feedback.append("✅ Multiple models visualized")

                if num_todos == 0:
                    score.earned_points += 3
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ ROC visualization not implemented")
        else:
            score.errors.append("❌ Exercise 2.2 cell not found")

        return score

    def grade_exercise_3(self) -> ExerciseScore:
        """Grade Exercise 3: Manufacturing-Specific Metrics."""
        score = ExerciseScore(exercise_id="Exercise 3", max_points=25, earned_points=0.0)

        # Check Exercise 3.1: Cost Calculation (10 points)
        cell_3_1 = self.find_cell_by_pattern(r"confusion_matrix|FP_COST|FN_COST")
        if cell_3_1:
            has_impl, num_todos = self.check_code_removed_todos(cell_3_1.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Cost calculation implemented")

                # Check for confusion matrix
                if re.search(r"confusion_matrix", cell_3_1.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Confusion matrix calculated")

                # Check for cost assignment
                fp_cost_match = re.search(r"FP_COST\s*=\s*(\d+)", cell_3_1.source)
                fn_cost_match = re.search(r"FN_COST\s*=\s*(\d+)", cell_3_1.source)

                if fp_cost_match and fn_cost_match:
                    fp_cost = int(fp_cost_match.group(1))
                    fn_cost = int(fn_cost_match.group(1))

                    score.earned_points += 2
                    score.feedback.append("✅ Manufacturing costs assigned")

                    # Check if FN > FP (manufacturing reality)
                    if fn_cost > fp_cost:
                        score.earned_points += 2
                        score.feedback.append("✅ FN cost > FP cost (correct asymmetry)")
                    else:
                        score.warnings.append("⚠️ FN cost should be > FP cost (missed defects cost more)")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Cost calculation not implemented")
        else:
            score.errors.append("❌ Exercise 3.1 cell not found")

        # Check Exercise 3.2: Threshold Optimization (15 points)
        cell_3_2 = self.find_cell_by_pattern(r"threshold.*sweep|arange.*0\.\d+.*0\.\d+|optimal.*threshold")
        if cell_3_2:
            has_impl, num_todos = self.check_code_removed_todos(cell_3_2.source)

            if has_impl:
                score.earned_points += 3
                score.feedback.append("✅ Threshold optimization implemented")

                # Check for probability predictions
                if re.search(r"predict_proba", cell_3_2.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Probability predictions used")

                # Check for threshold sweep
                if re.search(r"for.*threshold|arange|linspace", cell_3_2.source):
                    score.earned_points += 3
                    score.feedback.append("✅ Threshold sweep implemented")

                # Check for cost calculation in loop
                if re.search(r"confusion_matrix.*threshold|total_cost", cell_3_2.source):
                    score.earned_points += 3
                    score.feedback.append("✅ Cost calculated per threshold")

                # Check for optimal threshold identification
                if re.search(r"idxmin|argmin|min.*cost", cell_3_2.source):
                    score.earned_points += 3
                    score.feedback.append("✅ Optimal threshold identified")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Threshold optimization not implemented")
        else:
            score.errors.append("❌ Exercise 3.2 cell not found")

        return score

    def grade_exercise_4(self) -> ExerciseScore:
        """Grade Exercise 4: Model Deployment and CLI Usage."""
        score = ExerciseScore(exercise_id="Exercise 4", max_points=15, earned_points=0.0)

        # Check Exercise 4.1: Model Saving (8 points)
        cell_4_1 = self.find_cell_by_pattern(r"pipeline\.save|\.joblib")
        if cell_4_1:
            has_impl, num_todos = self.check_code_removed_todos(cell_4_1.source)

            if has_impl:
                score.earned_points += 2
                score.feedback.append("✅ Model saving code implemented")

                # Check for pipeline.save() usage
                if re.search(r"pipeline\.save\(", cell_4_1.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Using pipeline.save() method")

                # Check for descriptive filename (date/metrics)
                if re.search(r"timestamp|datetime|strftime|roc.*auc", cell_4_1.source, re.IGNORECASE):
                    score.earned_points += 2
                    score.feedback.append("✅ Descriptive filename with metadata")

                if num_todos == 0:
                    score.earned_points += 2
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Model saving not implemented")
        else:
            score.errors.append("❌ Exercise 4.1 cell not found")

        # Check Exercise 4.2: Model Loading (7 points)
        cell_4_2 = self.find_cell_by_pattern(r"\.load\(|WaferDefectPipeline\.load")
        if cell_4_2:
            has_impl, num_todos = self.check_code_removed_todos(cell_4_2.source)

            if has_impl:
                score.earned_points += 2
                score.feedback.append("✅ Model loading code implemented")

                # Check for WaferDefectPipeline.load()
                if re.search(r"WaferDefectPipeline\.load\(", cell_4_2.source):
                    score.earned_points += 2
                    score.feedback.append("✅ Using WaferDefectPipeline.load() method")

                # Check for round-trip verification
                if re.search(r"array_equal|predictions.*match|verify", cell_4_2.source, re.IGNORECASE):
                    score.earned_points += 2
                    score.feedback.append("✅ Round-trip verification implemented")

                if num_todos == 0:
                    score.earned_points += 1
                else:
                    score.warnings.append(f"⚠️ {num_todos} TODO(s) remaining")
            else:
                score.errors.append("❌ Model loading not implemented")
        else:
            score.errors.append("❌ Exercise 4.2 cell not found")

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
        # Basic checks: proper indentation, no excessive line length
        style_violations = 0
        for cell in code_cells:
            lines = cell.source.split("\n")
            for line in lines:
                # Check line length
                if len(line) > 120:
                    style_violations += 1
                # Check for inconsistent indentation (mixing tabs/spaces)
                if "\t" in line and "    " in line:
                    style_violations += 1

        max_violations = len(code_cells) * 2  # Generous threshold
        style_score = max(0, 3.0 - (style_violations / max_violations) * 3.0) if max_violations > 0 else 3.0
        score += style_score
        feedback.append(f"Code style: {style_violations} violations ({style_score:.1f}/3 pts)")

        # Check for error handling (2 points)
        # Look for try/except blocks or defensive programming
        error_handling_cells = sum(
            1 for cell in code_cells if re.search(r"try:|except:|if.*is.*None|assert ", cell.source)
        )
        error_score = min(2.0, (error_handling_cells / max(1, len(code_cells))) * 4.0)
        score += error_score
        feedback.append(f"Error handling: {error_handling_cells} cells with error handling ({error_score:.1f}/2 pts)")

        # Check for best practices (2 points)
        # Look for: descriptive variable names, avoiding magic numbers, using constants
        best_practice_score = 0.0

        # Check for descriptive variable names (not single letters)
        descriptive_vars = sum(1 for cell in code_cells if re.search(r"\b[a-z_]{4,}\b\s*=", cell.source))
        if descriptive_vars >= len(code_cells) * 0.5:
            best_practice_score += 1.0
            feedback.append("✅ Descriptive variable names used")

        # Check for constants (UPPER_CASE variables)
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
            print("🎓 WAFER DEFECT CLASSIFIER - GRADING SYSTEM")
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
                0, "🌟 Excellent work! You've demonstrated strong understanding of wafer defect classification."
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

            for feedback in ex_score.feedback[:3]:  # Show top 3 feedback items
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
        description="Grade wafer defect classifier tutorial submissions",
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
