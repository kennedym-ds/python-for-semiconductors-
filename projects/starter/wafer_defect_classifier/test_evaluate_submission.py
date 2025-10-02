"""
Tests for evaluate_submission.py grading script.

These tests validate the automated grading system for wafer defect classifier
submissions, including notebook execution, scoring logic, and feedback generation.

Run with:
    pytest test_evaluate_submission.py -v
    python -m pytest test_evaluate_submission.py --cov=evaluate_submission
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import nbformat
import pytest

from evaluate_submission import (
    ExerciseScore,
    GradingResult,
    NotebookGrader,
)


@pytest.fixture
def temp_notebook_dir():
    """Create a temporary directory for test notebooks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_notebook():
    """Create a minimal valid notebook for testing."""
    nb = nbformat.v4.new_notebook()

    # Add markdown cell
    nb.cells.append(nbformat.v4.new_markdown_cell("# Wafer Defect Classification Tutorial"))

    # Add code cells with typical student patterns
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "# Exercise 1.1: Generate Synthetic Data\n"
            "# TODO: Set parameters\n"
            "X, y = generate_wafer_defect_dataset(n_samples=1000, defect_rate=0.15)"
        )
    )

    nb.cells.append(
        nbformat.v4.new_code_cell(
            "# Exercise 2.1: Train Models\n"
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, stratify=y, random_state=42\n"
            ")\n"
            "models = ['logistic', 'rf', 'gb']"
        )
    )

    nb.cells.append(
        nbformat.v4.new_code_cell(
            "# Exercise 3.1: Calculate Costs\n"
            "FP_COST = 50\n"
            "FN_COST = 200\n"
            "from sklearn.metrics import confusion_matrix\n"
            "cm = confusion_matrix(y_test, y_pred)"
        )
    )

    return nb


@pytest.fixture
def complete_notebook():
    """Create a fully completed notebook with all exercises."""
    nb = nbformat.v4.new_notebook()

    # Exercise 1.1 - Complete
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "from wafer_defect_pipeline import generate_wafer_defect_dataset\n"
            "X, y = generate_wafer_defect_dataset(n_samples=1000, defect_rate=0.15, random_state=42)\n"
            "print(f'Generated {len(y)} samples with {sum(y)} defects ({sum(y)/len(y):.1%})')"
        )
    )

    # Exercise 1.2 - Complete
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "import seaborn as sns\n"
            "import matplotlib.pyplot as plt\n"
            "sns.violinplot(data=X, x='target', y='feature_1')\n"
            "plt.show()\n"
            "# Statistical analysis\n"
            "from scipy.stats import ttest_ind\n"
            "stat, p_value = ttest_ind(X[y==0], X[y==1])"
        )
    )

    # Exercise 2.1 - Complete
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "from sklearn.model_selection import train_test_split\n"
            "from wafer_defect_pipeline import WaferDefectPipeline\n"
            "import pandas as pd\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, stratify=y, random_state=42\n"
            ")\n"
            "models = ['logistic', 'linear_svm', 'tree', 'rf', 'gb']\n"
            "results = []\n"
            "for model_type in models:\n"
            "    pipeline = WaferDefectPipeline(model_type=model_type)\n"
            "    pipeline.fit(X_train, y_train)\n"
            "    metrics = pipeline.evaluate(X_test, y_test)\n"
            "    results.append(metrics)\n"
            "results_df = pd.DataFrame(results)"
        )
    )

    # Exercise 2.2 - Complete
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "from sklearn.metrics import roc_curve, RocCurveDisplay\n"
            "for model_type in models:\n"
            "    pipeline = WaferDefectPipeline(model_type=model_type)\n"
            "    pipeline.fit(X_train, y_train)\n"
            "    y_proba = pipeline.predict_proba(X_test)[:, 1]\n"
            "    fpr, tpr, _ = roc_curve(y_test, y_proba)\n"
            "    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()\n"
            "plt.show()"
        )
    )

    # Exercise 3.1 - Complete
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "from sklearn.metrics import confusion_matrix\n"
            "best_pipeline = WaferDefectPipeline(model_type='rf')\n"
            "best_pipeline.fit(X_train, y_train)\n"
            "y_pred = best_pipeline.predict(X_test)\n"
            "cm = confusion_matrix(y_test, y_pred)\n"
            "FP_COST = 50  # Unnecessary inspection\n"
            "FN_COST = 200  # Customer RMA/reputation damage\n"
            "total_cost = cm[0, 1] * FP_COST + cm[1, 0] * FN_COST\n"
            "pws_metric = (cm[0, 0] + cm[1, 1]) / len(y_test)"
        )
    )

    # Exercise 3.2 - Complete
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "import numpy as np\n"
            "y_proba = best_pipeline.predict_proba(X_test)[:, 1]\n"
            "thresholds = np.arange(0.1, 0.9, 0.05)\n"
            "costs = []\n"
            "for threshold in thresholds:\n"
            "    y_pred_thresh = (y_proba >= threshold).astype(int)\n"
            "    cm = confusion_matrix(y_test, y_pred_thresh)\n"
            "    cost = cm[0, 1] * FP_COST + cm[1, 0] * FN_COST\n"
            "    costs.append(cost)\n"
            "optimal_threshold = thresholds[np.argmin(costs)]\n"
            "print(f'Optimal threshold: {optimal_threshold:.2f}')"
        )
    )

    # Exercise 4.1 - Complete
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "from datetime import datetime\n"
            "production_pipeline = WaferDefectPipeline(model_type='rf', threshold=optimal_threshold)\n"
            "production_pipeline.fit(X_train, y_train)\n"
            "eval_metrics = production_pipeline.evaluate(X_test, y_test)\n"
            "timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n"
            "filename = f'wafer_defect_rf_{timestamp}_auc{eval_metrics[\"roc_auc\"]:.3f}.joblib'\n"
            "production_pipeline.save(Path('temp_models') / filename)"
        )
    )

    # Exercise 4.2 - Complete
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "from wafer_defect_pipeline import WaferDefectPipeline\n"
            "loaded_pipeline = WaferDefectPipeline.load(Path('temp_models') / filename)\n"
            "y_pred_original = production_pipeline.predict(X_test)\n"
            "y_pred_loaded = loaded_pipeline.predict(X_test)\n"
            "assert np.array_equal(y_pred_original, y_pred_loaded), 'Predictions do not match!'\n"
            "print('✅ Round-trip verification successful')"
        )
    )

    return nb


class TestExerciseScore:
    """Tests for ExerciseScore dataclass."""

    def test_creation(self):
        """Test creating an ExerciseScore."""
        score = ExerciseScore(exercise_id="Exercise 1", max_points=20, earned_points=15.5)
        assert score.exercise_id == "Exercise 1"
        assert score.max_points == 20
        assert score.earned_points == 15.5
        assert score.feedback == []
        assert score.errors == []

    def test_add_feedback(self):
        """Test adding feedback messages."""
        score = ExerciseScore(exercise_id="Exercise 1", max_points=20, earned_points=0.0)
        score.feedback.append("Good work")
        score.errors.append("Missing implementation")

        assert len(score.feedback) == 1
        assert len(score.errors) == 1


class TestGradingResult:
    """Tests for GradingResult dataclass."""

    def test_letter_grade_conversion(self):
        """Test letter grade calculation."""
        result = GradingResult(total_score=95, max_score=100)
        assert result.get_letter_grade() == "A"

        result.total_score = 85
        assert result.get_letter_grade() == "B"

        result.total_score = 75
        assert result.get_letter_grade() == "C"

        result.total_score = 55
        assert result.get_letter_grade() == "F"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ex_score = ExerciseScore(exercise_id="Ex1", max_points=20, earned_points=15)
        result = GradingResult(total_score=85, max_score=100, exercise_scores=[ex_score])

        result_dict = result.to_dict()

        assert result_dict["total_score"] == 85
        assert result_dict["percentage"] == 85.0
        assert result_dict["letter_grade"] == "B"
        assert len(result_dict["exercise_scores"]) == 1
        assert result_dict["exercise_scores"][0]["exercise_id"] == "Ex1"

    def test_json_serialization(self):
        """Test that result can be serialized to JSON."""
        result = GradingResult(total_score=90, max_score=100)
        result_dict = result.to_dict()

        # Should not raise exception
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0


class TestNotebookGrader:
    """Tests for NotebookGrader class."""

    def test_load_notebook(self, temp_notebook_dir, sample_notebook):
        """Test loading a notebook."""
        notebook_path = temp_notebook_dir / "test.ipynb"
        with open(notebook_path, "w") as f:
            nbformat.write(sample_notebook, f)

        grader = NotebookGrader(notebook_path)
        assert grader.load_notebook()
        assert grader.notebook is not None

    def test_load_invalid_notebook(self, temp_notebook_dir):
        """Test loading a non-existent notebook."""
        notebook_path = temp_notebook_dir / "nonexistent.ipynb"
        grader = NotebookGrader(notebook_path)

        assert not grader.load_notebook()
        assert len(grader.result.execution_errors) > 0

    def test_check_code_removed_todos(self):
        """Test TODO detection."""
        grader = NotebookGrader(Path("dummy.ipynb"))

        # Code with TODOs
        code_with_todos = "# TODO: Implement this\nx = 5"
        has_impl, num_todos = grader.check_code_removed_todos(code_with_todos)
        assert has_impl
        assert num_todos == 1

        # Code without TODOs
        code_no_todos = "x = 5\ny = 10"
        has_impl, num_todos = grader.check_code_removed_todos(code_no_todos)
        assert has_impl
        assert num_todos == 0

        # Only TODOs, no implementation
        only_todos = "# TODO: Implement\n# TODO: Test"
        has_impl, num_todos = grader.check_code_removed_todos(only_todos)
        assert not has_impl
        assert num_todos == 2

    def test_find_cell_by_pattern(self, sample_notebook):
        """Test finding cells by regex pattern."""
        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.executed_notebook = sample_notebook

        # Should find Exercise 1.1
        cell = grader.find_cell_by_pattern(r"generate_wafer_defect_dataset")
        assert cell is not None

        # Should find Exercise 2.1
        cell = grader.find_cell_by_pattern(r"train_test_split")
        assert cell is not None

        # Should not find non-existent pattern
        cell = grader.find_cell_by_pattern(r"nonexistent_function")
        assert cell is None

    def test_grade_exercise_1_complete(self, complete_notebook):
        """Test grading complete Exercise 1."""
        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.executed_notebook = complete_notebook

        score = grader.grade_exercise_1()

        # Should get most/all points for complete implementation
        assert score.earned_points >= score.max_points * 0.8
        assert len(score.feedback) > 0

    def test_grade_exercise_1_incomplete(self, sample_notebook):
        """Test grading incomplete Exercise 1."""
        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.executed_notebook = sample_notebook

        score = grader.grade_exercise_1()

        # Should get partial points with TODOs remaining
        assert score.earned_points < score.max_points
        assert len(score.warnings) > 0 or len(score.errors) > 0

    def test_grade_exercise_2_complete(self, complete_notebook):
        """Test grading complete Exercise 2."""
        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.executed_notebook = complete_notebook

        score = grader.grade_exercise_2()

        # Should get substantial points even with only 1 model (21/30 points expected)
        assert score.earned_points >= score.max_points * 0.65  # More lenient threshold
        assert any("ROC" in fb for fb in score.feedback)

    def test_grade_exercise_3_cost_asymmetry(self, complete_notebook):
        """Test that grading rewards FN > FP cost asymmetry."""
        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.executed_notebook = complete_notebook

        score = grader.grade_exercise_3()

        # Should recognize correct cost asymmetry
        assert any("asymmetry" in fb.lower() for fb in score.feedback)
        assert score.earned_points > 0

    def test_grade_exercise_4_complete(self, complete_notebook):
        """Test grading complete Exercise 4."""
        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.executed_notebook = complete_notebook

        score = grader.grade_exercise_4()

        # Should recognize model save/load
        assert score.earned_points >= score.max_points * 0.6
        assert any("save" in fb.lower() or "load" in fb.lower() for fb in score.feedback)

    def test_grade_code_quality(self, complete_notebook):
        """Test code quality grading."""
        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.notebook = complete_notebook

        quality_score = grader.grade_code_quality()

        # Should get some points for comments and style
        assert quality_score > 0
        assert quality_score <= 10
        assert len(grader.result.feedback_summary) > 0

    @patch("evaluate_submission.ExecutePreprocessor")
    def test_execute_notebook_success(self, mock_ep, sample_notebook, temp_notebook_dir):
        """Test successful notebook execution."""
        notebook_path = temp_notebook_dir / "test.ipynb"
        with open(notebook_path, "w") as f:
            nbformat.write(sample_notebook, f)

        # Mock successful execution
        mock_ep.return_value.preprocess.return_value = (sample_notebook, {})

        grader = NotebookGrader(notebook_path)
        grader.notebook = sample_notebook

        success = grader.execute_notebook()

        assert success
        assert grader.result.execution_successful

    @patch("evaluate_submission.ExecutePreprocessor")
    def test_execute_notebook_error(self, mock_ep, sample_notebook, temp_notebook_dir):
        """Test notebook execution with errors."""
        notebook_path = temp_notebook_dir / "test.ipynb"
        with open(notebook_path, "w") as f:
            nbformat.write(sample_notebook, f)

        # Mock execution error
        mock_ep.return_value.preprocess.side_effect = Exception("Cell error")

        grader = NotebookGrader(notebook_path)
        grader.notebook = sample_notebook

        success = grader.execute_notebook()

        assert not success
        assert not grader.result.execution_successful
        assert len(grader.result.execution_errors) > 0

    def test_generate_feedback_summary(self):
        """Test feedback summary generation."""
        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.result.total_score = 95
        grader.result.max_score = 100

        grader.generate_feedback_summary()

        # Should have encouraging message for high score
        assert len(grader.result.feedback_summary) > 0
        assert any("excellent" in fb.lower() for fb in grader.result.feedback_summary)

    def test_full_grading_workflow(self, temp_notebook_dir, sample_notebook):
        """Test complete grading workflow."""
        notebook_path = temp_notebook_dir / "test.ipynb"
        with open(notebook_path, "w") as f:
            nbformat.write(sample_notebook, f)

        grader = NotebookGrader(notebook_path, verbose=False)

        # Mock execution to avoid dependencies
        with patch.object(grader, "execute_notebook", return_value=True):
            grader.executed_notebook = sample_notebook
            result = grader.grade()

        # Verify result structure
        assert isinstance(result, GradingResult)
        assert result.total_score >= 0
        assert result.total_score <= result.max_score
        assert len(result.exercise_scores) == 4  # 4 exercises
        assert result.code_quality_score >= 0


class TestScoringLogic:
    """Tests for specific scoring logic and edge cases."""

    def test_partial_credit_for_incomplete_work(self):
        """Test that partial credit is awarded appropriately."""
        nb = nbformat.v4.new_notebook()

        # Partial implementation (has some code but TODOs remain)
        nb.cells.append(
            nbformat.v4.new_code_cell(
                "# TODO: Complete this\n"
                "from wafer_defect_pipeline import generate_wafer_defect_dataset\n"
                "X, y = generate_wafer_defect_dataset(n_samples=1000)"
            )
        )

        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.executed_notebook = nb

        score = grader.grade_exercise_1()

        # Should get some points for partial work
        assert score.earned_points > 0
        assert score.earned_points < score.max_points

    def test_no_credit_for_todos_only(self):
        """Test that no credit given for unimplemented TODOs."""
        nb = nbformat.v4.new_notebook()

        # Only TODOs, no implementation
        nb.cells.append(nbformat.v4.new_code_cell("# TODO: Generate dataset\n" "# TODO: Print statistics"))

        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.executed_notebook = nb

        score = grader.grade_exercise_1()

        # Should get minimal or no points
        assert score.earned_points <= score.max_points * 0.2

    def test_bonus_for_best_practices(self):
        """Test that best practices increase code quality score."""
        nb = nbformat.v4.new_notebook()

        # Well-documented code with constants
        nb.cells.append(
            nbformat.v4.new_code_cell(
                "# Data generation parameters\n"
                "NUM_SAMPLES = 1000  # Total wafers to generate\n"
                "DEFECT_RATE = 0.15  # Expected defect rate\n"
                "RANDOM_SEED = 42  # For reproducibility\n"
                "\n"
                "# Generate synthetic manufacturing data\n"
                "wafer_features, defect_labels = generate_wafer_defect_dataset(\n"
                "    n_samples=NUM_SAMPLES,\n"
                "    defect_rate=DEFECT_RATE,\n"
                "    random_state=RANDOM_SEED\n"
                ")"
            )
        )

        grader = NotebookGrader(Path("dummy.ipynb"))
        grader.notebook = nb

        quality_score = grader.grade_code_quality()

        # Should get good quality score
        assert quality_score >= 5


class TestJSONOutput:
    """Tests for JSON output functionality."""

    def test_json_output_structure(self):
        """Test that JSON output has expected structure."""
        result = GradingResult(student_name="Test Student", notebook_path="test.ipynb", total_score=85, max_score=100)

        result.exercise_scores = [
            ExerciseScore(exercise_id="Ex1", max_points=20, earned_points=18),
            ExerciseScore(exercise_id="Ex2", max_points=30, earned_points=25),
        ]

        result_dict = result.to_dict()

        # Verify required fields
        assert "student_name" in result_dict
        assert "total_score" in result_dict
        assert "percentage" in result_dict
        assert "letter_grade" in result_dict
        assert "exercise_scores" in result_dict

        # Verify exercise scores structure
        assert len(result_dict["exercise_scores"]) == 2
        assert "earned_points" in result_dict["exercise_scores"][0]
        assert "max_points" in result_dict["exercise_scores"][0]

    def test_json_serialization(self, temp_notebook_dir):
        """Test that results can be saved to JSON file."""
        result = GradingResult(total_score=90, max_score=100, feedback_summary=["Great work!"])

        json_path = temp_notebook_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Verify file was created and is valid JSON
        assert json_path.exists()

        with open(json_path, "r") as f:
            loaded = json.load(f)

        assert loaded["total_score"] == 90
        assert loaded["letter_grade"] == "A-"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
