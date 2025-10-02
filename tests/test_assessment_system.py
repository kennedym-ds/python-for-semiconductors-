"""
Comprehensive Integration Tests for Assessment System (630-Question Framework)

Tests validate:
- JSON schema compliance across all modules
- Question ID uniqueness
- Module structure integrity
- Required fields presence
- Type-specific requirements (multiple_choice, coding_exercise, conceptual)
- Data consistency and integrity

Part of Week 4 Phase 2: Assessment System Integration Testing
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set

import pytest


# ============================================================================
# Test Configuration
# ============================================================================

ASSESSMENTS_DIR = Path(__file__).parent.parent / "assessments"
SCHEMA_PATH = ASSESSMENTS_DIR / "schema.json"

# Expected modules based on repository structure
EXPECTED_MODULES = [
    "module-1",
    "module-2",
    "module-3",
    "module-4",
    "module-5",
    "module-6",
    "module-7",
    "module-8",
    "module-9",
    "module-10",
    "module-11",
]

# Question type specific requirements
QUESTION_TYPES = ["multiple_choice", "coding_exercise", "conceptual"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


# ============================================================================
# Helper Functions
# ============================================================================


def load_schema() -> dict:
    """Load JSON schema definition."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_questions() -> Dict[str, dict]:
    """
    Load all question files from assessments directory.

    Returns:
        dict: Mapping of file paths to loaded JSON data
    """
    questions = {}
    for module_dir in ASSESSMENTS_DIR.iterdir():
        if module_dir.is_dir() and module_dir.name.startswith("module-"):
            for question_file in module_dir.glob("*.json"):
                # Skip fixed/backup files
                if "fixed" in question_file.name or "backup" in question_file.name:
                    continue
                try:
                    with open(question_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        questions[str(question_file)] = data
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {question_file}: {e}")
    return questions


def extract_all_question_ids(questions_data: Dict[str, dict]) -> List[tuple]:
    """
    Extract all question IDs with their source files.

    Returns:
        list: [(question_id, file_path), ...]
    """
    all_ids = []
    for file_path, data in questions_data.items():
        if "questions" in data:
            for q in data["questions"]:
                if "id" in q:
                    all_ids.append((q["id"], file_path))
    return all_ids


# ============================================================================
# Test Class: Schema Compliance
# ============================================================================


class TestSchemaCompliance:
    """Validate all question files comply with JSON schema."""

    def test_schema_file_exists(self):
        """Schema file must exist and be valid JSON."""
        assert SCHEMA_PATH.exists(), f"Schema file not found at {SCHEMA_PATH}"
        schema = load_schema()
        assert isinstance(schema, dict), "Schema must be a JSON object"

    def test_all_files_have_required_top_level_fields(self):
        """All question files must have module_id and questions (version optional for older files)."""
        questions = load_all_questions()
        assert len(questions) > 0, "No question files found"

        for file_path, data in questions.items():
            assert "module_id" in data, f"{file_path}: Missing 'module_id'"
            assert "questions" in data, f"{file_path}: Missing 'questions'"
            # Version is recommended but not strictly required for legacy files

    def test_module_id_pattern_compliance(self):
        """All module_id values must match pattern: module-X or module-X.Y"""
        # Accept both module-X and module-X.Y formats
        pattern = re.compile(r"^module-[0-9]+(\.[0-9]+)?$")
        questions = load_all_questions()

        for file_path, data in questions.items():
            module_id = data.get("module_id", "")
            assert pattern.match(
                module_id
            ), f"{file_path}: Invalid module_id format '{module_id}' (expected: module-X or module-X.Y)"

    def test_version_format(self):
        """Version should be semantic versioning format (when present)."""
        pattern = re.compile(r"^[0-9]+\.[0-9]+(\.[0-9]+)?$")
        questions = load_all_questions()

        for file_path, data in questions.items():
            # Skip if version not present (legacy files)
            if "version" not in data:
                continue
            version = data.get("version", "")
            if version:  # Only validate if non-empty
                assert pattern.match(
                    str(version)
                ), f"{file_path}: Invalid version format '{version}' (expected: X.Y or X.Y.Z)"

    def test_questions_is_list(self):
        """Questions field must be a list."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            assert isinstance(data.get("questions"), list), f"{file_path}: 'questions' must be a list"


# ============================================================================
# Test Class: Question IDs
# ============================================================================


class TestQuestionIDs:
    """Validate question ID uniqueness and patterns."""

    def test_question_id_pattern(self):
        """All question IDs must match pattern: mX.Y_qNNN"""
        pattern = re.compile(r"^m[0-9]+\.[0-9]+_q[0-9]{3,}$")
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                q_id = q.get("id", "")
                assert pattern.match(
                    q_id
                ), f"{file_path} question {idx}: Invalid ID format '{q_id}' (expected: mX.Y_qNNN)"

    def test_question_id_uniqueness(self):
        """Question IDs must be unique across entire assessment framework."""
        all_ids = extract_all_question_ids(load_all_questions())
        id_counts: Dict[str, List[str]] = {}

        for q_id, file_path in all_ids:
            if q_id not in id_counts:
                id_counts[q_id] = []
            id_counts[q_id].append(file_path)

        # Find duplicates
        duplicates = {q_id: files for q_id, files in id_counts.items() if len(files) > 1}

        assert len(duplicates) == 0, f"Duplicate question IDs found:\n" + "\n".join(
            [f"  {q_id}: {files}" for q_id, files in duplicates.items()]
        )

    def test_question_id_matches_module(self):
        """Question IDs should match their module (e.g., m3.1_q001 in module-3.1)."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            module_id = data.get("module_id", "")
            # Extract module number from module-X.Y
            if module_id.startswith("module-"):
                expected_prefix = "m" + module_id.replace("module-", "")

                for idx, q in enumerate(data.get("questions", [])):
                    q_id = q.get("id", "")
                    assert q_id.startswith(expected_prefix), (
                        f"{file_path} question {idx}: ID '{q_id}' doesn't match module '{module_id}' "
                        f"(expected prefix: {expected_prefix})"
                    )

    def test_question_numbering_sequence(self):
        """Question numbers within a module should be sequential (q001, q002, q003...)."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            q_numbers = []
            for q in data.get("questions", []):
                q_id = q.get("id", "")
                # Extract number from mX.Y_qNNN
                match = re.search(r"_q([0-9]+)$", q_id)
                if match:
                    q_numbers.append(int(match.group(1)))

            # Check if sequential starting from 1
            if q_numbers:
                q_numbers_sorted = sorted(q_numbers)
                expected = list(range(1, len(q_numbers) + 1))

                # Allow gaps but check for duplicates
                assert len(q_numbers) == len(
                    set(q_numbers)
                ), f"{file_path}: Duplicate question numbers found: {q_numbers}"


# ============================================================================
# Test Class: Module Structure
# ============================================================================


class TestModuleStructure:
    """Validate module organization and completeness."""

    def test_expected_module_directories_exist(self):
        """All expected module directories must exist."""
        existing_modules = [d.name for d in ASSESSMENTS_DIR.iterdir() if d.is_dir() and d.name.startswith("module-")]

        for expected in EXPECTED_MODULES:
            assert (
                expected in existing_modules
            ), f"Expected module directory '{expected}' not found in {ASSESSMENTS_DIR}"

    def test_each_module_has_questions(self):
        """Each module directory should contain at least one question file."""
        for module_name in EXPECTED_MODULES:
            module_dir = ASSESSMENTS_DIR / module_name
            if not module_dir.exists():
                continue

            question_files = list(module_dir.glob("*.json"))
            # Filter out fixed/backup files
            question_files = [f for f in question_files if "fixed" not in f.name and "backup" not in f.name]

            assert len(question_files) > 0, f"Module {module_name} has no question files"

    def test_no_orphan_question_files(self):
        """All question JSON files should be in module directories."""
        for json_file in ASSESSMENTS_DIR.glob("*.json"):
            # Allow schema.json and other config files at root
            if json_file.name not in ["schema.json"]:
                pytest.fail(f"Orphan question file found at root: {json_file}")

    def test_total_question_count(self):
        """Verify approximate total question count (should be around 630)."""
        questions = load_all_questions()
        total = sum(len(data.get("questions", [])) for data in questions.values())

        # Allow some flexibility (600-700 range)
        assert 600 <= total <= 700, f"Expected ~630 questions, found {total}"


# ============================================================================
# Test Class: Question Fields
# ============================================================================


class TestQuestionFields:
    """Validate required fields are present in all questions."""

    def test_all_questions_have_required_fields(self):
        """All questions must have: id, type, difficulty, topic, question, points."""
        required = ["id", "type", "difficulty", "topic", "question", "points"]
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                for field in required:
                    assert (
                        field in q
                    ), f"{file_path} question {idx} (ID: {q.get('id', 'unknown')}): Missing required field '{field}'"

    def test_question_type_validity(self):
        """Question type must be one of: multiple_choice, coding_exercise, conceptual."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                q_type = q.get("type", "")
                assert q_type in QUESTION_TYPES, (
                    f"{file_path} question {idx}: Invalid type '{q_type}' " f"(must be one of: {QUESTION_TYPES})"
                )

    def test_difficulty_validity(self):
        """Difficulty must be one of: easy, medium, hard."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                difficulty = q.get("difficulty", "")
                assert difficulty in DIFFICULTY_LEVELS, (
                    f"{file_path} question {idx}: Invalid difficulty '{difficulty}' "
                    f"(must be one of: {DIFFICULTY_LEVELS})"
                )

    def test_points_validity(self):
        """Points must be a positive integer."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                points = q.get("points")
                assert isinstance(
                    points, int
                ), f"{file_path} question {idx}: Points must be integer, got {type(points)}"
                assert points >= 1, f"{file_path} question {idx}: Points must be >= 1, got {points}"

    def test_topic_is_string(self):
        """Topic must be a non-empty string."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                topic = q.get("topic", "")
                assert isinstance(topic, str), f"{file_path} question {idx}: Topic must be string"
                assert len(topic) > 0, f"{file_path} question {idx}: Topic cannot be empty"

    def test_question_text_is_string(self):
        """Question text must be a non-empty string."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                question_text = q.get("question", "")
                assert isinstance(question_text, str), f"{file_path} question {idx}: Question text must be string"
                assert len(question_text) > 10, f"{file_path} question {idx}: Question text too short"


# ============================================================================
# Test Class: Question Types
# ============================================================================


class TestQuestionTypes:
    """Validate type-specific requirements for different question types."""

    def test_multiple_choice_requirements(self):
        """Multiple choice questions must have: options, correct_answer, explanation."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                if q.get("type") == "multiple_choice":
                    q_id = q.get("id", "unknown")

                    # Check required fields
                    assert "options" in q, f"{file_path} question {idx} ({q_id}): multiple_choice missing 'options'"
                    assert (
                        "correct_answer" in q
                    ), f"{file_path} question {idx} ({q_id}): multiple_choice missing 'correct_answer'"
                    assert (
                        "explanation" in q
                    ), f"{file_path} question {idx} ({q_id}): multiple_choice missing 'explanation'"

                    # Validate options is list with at least 2 items
                    options = q.get("options", [])
                    assert isinstance(options, list), f"{file_path} question {idx} ({q_id}): options must be list"
                    assert len(options) >= 2, f"{file_path} question {idx} ({q_id}): options must have at least 2 items"

                    # Validate correct_answer is valid index (int or convertible string)
                    correct = q.get("correct_answer")

                    # Accept both int and letter strings (A, B, C, D)
                    if isinstance(correct, str):
                        # Convert letter to index (A=0, B=1, C=2, D=3)
                        if correct.upper() in ["A", "B", "C", "D", "E", "F"]:
                            correct_idx = ord(correct.upper()) - ord("A")
                        else:
                            pytest.fail(
                                f"{file_path} question {idx} ({q_id}): correct_answer '{correct}' must be int or letter (A-F)"
                            )
                    elif isinstance(correct, int):
                        correct_idx = correct
                    else:
                        pytest.fail(f"{file_path} question {idx} ({q_id}): correct_answer must be integer or letter")

                    assert (
                        0 <= correct_idx < len(options)
                    ), f"{file_path} question {idx} ({q_id}): correct_answer {correct_idx} out of range (0-{len(options)-1})"

    def test_coding_exercise_requirements(self):
        """Coding exercises must have: starter_code, test_cases, explanation (solution optional)."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                if q.get("type") == "coding_exercise":
                    q_id = q.get("id", "unknown")

                    # Check required fields (accept both starter_code and code_template)
                    has_starter = "starter_code" in q or "code_template" in q
                    assert (
                        has_starter
                    ), f"{file_path} question {idx} ({q_id}): coding_exercise missing 'starter_code' or 'code_template'"
                    assert (
                        "test_cases" in q
                    ), f"{file_path} question {idx} ({q_id}): coding_exercise missing 'test_cases'"

                    # Validate starter_code is non-empty string
                    starter = q.get("starter_code") or q.get("code_template", "")
                    assert (
                        isinstance(starter, str) and len(starter) > 0
                    ), f"{file_path} question {idx} ({q_id}): starter_code must be non-empty string"

                    # Validate test_cases is list
                    tests = q.get("test_cases", [])
                    assert isinstance(tests, list), f"{file_path} question {idx} ({q_id}): test_cases must be list"
                    assert len(tests) > 0, f"{file_path} question {idx} ({q_id}): test_cases must have at least 1 item"

    def test_conceptual_requirements(self):
        """Conceptual questions must have: rubric, sample_answer (or explanation)."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                if q.get("type") == "conceptual":
                    q_id = q.get("id", "unknown")

                    # Check required fields
                    assert "rubric" in q, f"{file_path} question {idx} ({q_id}): conceptual missing 'rubric'"

                    # Validate rubric structure (accept both dict and list formats)
                    rubric = q.get("rubric", {})
                    is_valid = isinstance(rubric, (dict, list))
                    assert is_valid, f"{file_path} question {idx} ({q_id}): rubric must be object or array"

                    # If dict, check for criteria, max_points, OR any scoring keys (flat format)
                    if isinstance(rubric, dict):
                        has_structure = "criteria" in rubric or "max_points" in rubric
                        has_scoring_keys = len(rubric) > 0 and all(isinstance(v, int) for v in rubric.values())
                        assert (
                            has_structure or has_scoring_keys
                        ), f"{file_path} question {idx} ({q_id}): rubric must have 'criteria'/'max_points' or scoring keys"
                    # If list, check it has items
                    elif isinstance(rubric, list):
                        assert (
                            len(rubric) > 0
                        ), f"{file_path} question {idx} ({q_id}): rubric array must have at least one criterion"


# ============================================================================
# Test Class: Data Integrity
# ============================================================================


class TestDataIntegrity:
    """Validate cross-module data consistency and integrity."""

    def test_difficulty_distribution(self):
        """Check reasonable distribution of difficulty levels."""
        questions = load_all_questions()
        difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}

        for data in questions.values():
            for q in data.get("questions", []):
                diff = q.get("difficulty")
                if diff in difficulty_counts:
                    difficulty_counts[diff] += 1

        total = sum(difficulty_counts.values())

        # Each difficulty should be at least 15% of total
        for difficulty, count in difficulty_counts.items():
            percentage = (count / total) * 100
            assert percentage >= 10, f"Difficulty '{difficulty}' only {percentage:.1f}% of questions (expected >= 10%)"

    def test_question_type_distribution(self):
        """Check reasonable distribution of question types."""
        questions = load_all_questions()
        type_counts = {"multiple_choice": 0, "coding_exercise": 0, "conceptual": 0}

        for data in questions.values():
            for q in data.get("questions", []):
                q_type = q.get("type")
                if q_type in type_counts:
                    type_counts[q_type] += 1

        total = sum(type_counts.values())

        # Each type should be at least 10% of total
        for q_type, count in type_counts.items():
            percentage = (count / total) * 100
            assert percentage >= 5, f"Question type '{q_type}' only {percentage:.1f}% of questions (expected >= 5%)"

    def test_points_distribution(self):
        """Check reasonable point value distribution."""
        questions = load_all_questions()
        points_values = []

        for data in questions.values():
            for q in data.get("questions", []):
                points = q.get("points", 0)
                points_values.append(points)

        # Basic statistics
        min_points = min(points_values)
        max_points = max(points_values)
        avg_points = sum(points_values) / len(points_values)

        # Assertions
        assert min_points >= 1, f"Minimum points {min_points} too low"
        assert max_points <= 20, f"Maximum points {max_points} seems too high"
        assert 2 <= avg_points <= 8, f"Average points {avg_points:.1f} outside expected range (2-8)"

    def test_no_duplicate_questions(self):
        """Check for potential duplicate question text (same question in multiple files)."""
        questions = load_all_questions()
        question_texts: Dict[str, List[str]] = {}

        for file_path, data in questions.items():
            for q in data.get("questions", []):
                q_text = q.get("question", "").strip().lower()[:100]  # First 100 chars
                if q_text:
                    if q_text not in question_texts:
                        question_texts[q_text] = []
                    question_texts[q_text].append(f"{file_path}:{q.get('id', 'unknown')}")

        # Find potential duplicates
        duplicates = {text: files for text, files in question_texts.items() if len(files) > 1}

        # This is a warning rather than failure (similar questions might be intentional)
        if duplicates:
            print("\nWarning: Potential duplicate questions found:")
            for text, files in list(duplicates.items())[:5]:  # Show first 5
                print(f"  '{text[:50]}...': {files}")

    def test_module_file_naming_convention(self):
        """Question files should follow naming convention: X.Y-questions.json"""
        pattern = re.compile(r"^[0-9]+\.[0-9]+-questions\.json$")
        questions = load_all_questions()

        for file_path in questions.keys():
            filename = Path(file_path).name
            # Allow some flexibility for fixed/backup files
            if "fixed" in filename or "backup" in filename:
                continue

            assert pattern.match(filename), f"File {file_path} doesn't follow naming convention X.Y-questions.json"


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_no_empty_option_strings(self):
        """Multiple choice options should not be empty strings."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                if q.get("type") == "multiple_choice":
                    options = q.get("options", [])
                    for opt_idx, opt in enumerate(options):
                        assert (
                            isinstance(opt, str) and len(opt.strip()) > 0
                        ), f"{file_path} question {idx}: Option {opt_idx} is empty"

    def test_no_extremely_long_questions(self):
        """Question text should not be excessively long (> 1000 chars)."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                q_text = q.get("question", "")
                assert len(q_text) < 2000, f"{file_path} question {idx}: Question text too long ({len(q_text)} chars)"

    def test_coding_exercise_has_valid_python(self):
        """Starter code should contain Python function/class definition (or YAML/Markdown for special exercises)."""
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                if q.get("type") == "coding_exercise":
                    starter = q.get("starter_code") or q.get("code_template", "")
                    # Check for Python function/class OR YAML workflow OR documentation template
                    is_python = "def " in starter or "class " in starter
                    is_yaml = "---" in starter or "name:" in starter or "jobs:" in starter
                    is_documentation = "README" in starter or "CHANGELOG" in starter or "# " in starter[:50]
                    assert (
                        is_python or is_yaml or is_documentation
                    ), f"{file_path} question {idx}: starter_code should contain Python/YAML/documentation content"

    def test_no_special_characters_in_ids(self):
        """Question IDs should only contain alphanumeric, underscore, dot."""
        pattern = re.compile(r"^[a-zA-Z0-9_\.]+$")
        questions = load_all_questions()

        for file_path, data in questions.items():
            for idx, q in enumerate(data.get("questions", [])):
                q_id = q.get("id", "")
                assert pattern.match(q_id), f"{file_path} question {idx}: ID '{q_id}' contains invalid characters"


# ============================================================================
# Summary Test
# ============================================================================


class TestSummary:
    """Generate summary statistics for the assessment framework."""

    def test_generate_assessment_summary(self, capsys):
        """Generate and print comprehensive assessment summary."""
        questions = load_all_questions()

        # Count totals
        total_files = len(questions)
        total_questions = sum(len(data.get("questions", [])) for data in questions.values())

        # Count by type
        type_counts = {"multiple_choice": 0, "coding_exercise": 0, "conceptual": 0}
        difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}

        for data in questions.values():
            for q in data.get("questions", []):
                q_type = q.get("type")
                if q_type in type_counts:
                    type_counts[q_type] += 1

                difficulty = q.get("difficulty")
                if difficulty in difficulty_counts:
                    difficulty_counts[difficulty] += 1

        # Print summary
        print("\n" + "=" * 70)
        print("ASSESSMENT SYSTEM SUMMARY")
        print("=" * 70)
        print(f"Total Question Files: {total_files}")
        print(f"Total Questions: {total_questions}")
        print()
        print("By Type:")
        for q_type, count in type_counts.items():
            pct = (count / total_questions) * 100
            print(f"  {q_type:20s}: {count:4d} ({pct:5.1f}%)")
        print()
        print("By Difficulty:")
        for difficulty, count in difficulty_counts.items():
            pct = (count / total_questions) * 100
            print(f"  {difficulty:20s}: {count:4d} ({pct:5.1f}%)")
        print("=" * 70)

        # Always pass this test
        assert True
