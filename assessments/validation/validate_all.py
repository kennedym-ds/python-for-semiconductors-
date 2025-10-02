"""
Unified Assessment Validation Script

Validates assessment question JSON files against the schema.
Can validate all modules or specific modules via CLI arguments.

Usage:
    python validate_all.py                    # Validate all modules
    python validate_all.py --module 1         # Validate Module 1 only
    python validate_all.py --module 4 9       # Validate Modules 4 and 9
    python validate_all.py --verbose          # Show detailed output

Part of the Python for Semiconductors assessment system.
For comprehensive testing, use: pytest tests/test_assessment_system.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# Configuration
# ============================================================================

ASSESSMENTS_DIR = Path(__file__).parent.parent
SCHEMA_PATH = ASSESSMENTS_DIR / "schema.json"

QUESTION_TYPES = ["multiple_choice", "coding_exercise", "conceptual"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Expected modules
ALL_MODULES = list(range(1, 12))  # Modules 1-11


# ============================================================================
# Validation Functions
# ============================================================================


def load_schema() -> dict:
    """Load JSON schema definition."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_question_file(file_path: Path, verbose: bool = False) -> Tuple[bool, Dict]:
    """
    Validate a single assessment JSON file.

    Returns:
        (success, result_dict) where result_dict contains:
        - questions: number of questions
        - errors: list of error messages
        - warnings: list of warning messages
        - breakdown: dict of question types
    """
    result = {
        "questions": 0,
        "errors": [],
        "warnings": [],
        "breakdown": {"multiple_choice": 0, "coding_exercise": 0, "conceptual": 0},
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result["errors"].append(f"JSON decode error: {e}")
        return False, result
    except Exception as e:
        result["errors"].append(f"File read error: {e}")
        return False, result

    # Check required top-level fields
    required_fields = ["module_id", "sub_module", "title", "week", "questions"]
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        result["errors"].append(f"Missing required fields: {missing_fields}")
        return False, result

    # Check questions array
    questions = data["questions"]
    if not isinstance(questions, list):
        result["errors"].append("'questions' must be a list")
        return False, result

    if len(questions) == 0:
        result["warnings"].append("No questions found (empty list)")

    result["questions"] = len(questions)

    # Validate each question
    question_ids = set()

    for i, q in enumerate(questions, 1):
        # Required fields for all questions
        required_q_fields = ["id", "type", "question", "difficulty", "points"]
        missing_q_fields = [field for field in required_q_fields if field not in q]

        if missing_q_fields:
            result["errors"].append(f"Question {i}: Missing fields {missing_q_fields}")
            continue

        # Check question ID uniqueness (within file)
        q_id = q.get("id")
        if q_id in question_ids:
            result["errors"].append(f"Question {i}: Duplicate ID '{q_id}'")
        question_ids.add(q_id)

        # Validate question type
        q_type = q.get("type")
        if q_type not in QUESTION_TYPES:
            result["errors"].append(
                f"Question {i} ({q_id}): Invalid type '{q_type}'. " f"Must be one of {QUESTION_TYPES}"
            )
        else:
            result["breakdown"][q_type] += 1

        # Validate difficulty
        difficulty = q.get("difficulty")
        if difficulty not in DIFFICULTY_LEVELS:
            result["errors"].append(
                f"Question {i} ({q_id}): Invalid difficulty '{difficulty}'. " f"Must be one of {DIFFICULTY_LEVELS}"
            )

        # Type-specific validation
        if q_type == "multiple_choice":
            if "options" not in q:
                result["errors"].append(f"Question {i} ({q_id}): Multiple choice missing 'options'")
            if "correct_answer" not in q:
                result["errors"].append(f"Question {i} ({q_id}): Multiple choice missing 'correct_answer'")
            elif isinstance(q.get("options"), list):
                correct_idx = q.get("correct_answer")
                if not isinstance(correct_idx, int) or correct_idx >= len(q["options"]):
                    result["errors"].append(f"Question {i} ({q_id}): Invalid correct_answer index")

        elif q_type == "coding_exercise":
            if "starter_code" not in q:
                result["warnings"].append(f"Question {i} ({q_id}): Coding exercise missing 'starter_code'")
            if "test_cases" not in q:
                result["errors"].append(f"Question {i} ({q_id}): Coding exercise missing 'test_cases'")

        elif q_type == "conceptual":
            if "rubric" not in q:
                result["warnings"].append(f"Question {i} ({q_id}): Conceptual question missing 'rubric'")

    # Overall validation result
    success = len(result["errors"]) == 0
    return success, result


def validate_module(module_num: int, verbose: bool = False) -> Tuple[bool, Dict]:
    """
    Validate all question files for a specific module.

    Returns:
        (success, results) where results is a dict of file results
    """
    module_dir = ASSESSMENTS_DIR / f"module-{module_num}"

    if not module_dir.exists():
        return False, {"error": f"Module directory not found: {module_dir}"}

    question_files = list(module_dir.glob("*.json"))

    # Filter out backup/fixed files
    question_files = [f for f in question_files if "backup" not in f.name.lower() and "fixed" not in f.name.lower()]

    if not question_files:
        return False, {"error": f"No question files found in {module_dir}"}

    results = {}
    all_success = True

    for qf in sorted(question_files):
        success, result = validate_question_file(qf, verbose)
        results[qf.name] = {"success": success, "data": result}
        if not success:
            all_success = False

    return all_success, results


# ============================================================================
# Display Functions
# ============================================================================


def print_file_result(filename: str, success: bool, data: Dict, verbose: bool):
    """Print validation results for a single file."""
    status = "[PASS]" if success else "[FAIL]"
    print(f"\n{status} {filename}")

    if data["questions"] > 0:
        breakdown = data["breakdown"]
        print(f"   {data['questions']} questions total:")
        print(f"   - {breakdown['multiple_choice']} multiple choice")
        print(f"   - {breakdown['coding_exercise']} coding exercises")
        print(f"   - {breakdown['conceptual']} conceptual")

    if data["errors"]:
        print(f"   [ERROR] {len(data['errors'])} error(s):")
        for error in data["errors"]:
            print(f"      - {error}")

    if verbose and data["warnings"]:
        print(f"   [WARN] {len(data['warnings'])} warning(s):")
        for warning in data["warnings"]:
            print(f"      - {warning}")


def print_module_summary(module_num: int, success: bool, results: Dict, verbose: bool):
    """Print summary for a module."""
    print("\n" + "=" * 80)
    print(f"MODULE {module_num} VALIDATION")
    print("=" * 80)

    if "error" in results:
        print(f"❌ {results['error']}")
        return

    total_questions = 0
    total_errors = 0
    total_warnings = 0

    for filename, result in results.items():
        print_file_result(filename, result["success"], result["data"], verbose)
        total_questions += result["data"]["questions"]
        total_errors += len(result["data"]["errors"])
        total_warnings += len(result["data"]["warnings"])

    print("\n" + "-" * 80)
    status = "[PASSED]" if success else "[FAILED]"
    print(f"{status} - Module {module_num}")
    print(f"Total: {total_questions} questions across {len(results)} file(s)")
    if total_errors > 0:
        print(f"Errors: {total_errors}")
    if verbose and total_warnings > 0:
        print(f"Warnings: {total_warnings}")
    print("=" * 80)


def print_overall_summary(results: Dict[int, Tuple[bool, Dict]], verbose: bool):
    """Print overall validation summary."""
    print("\n" + "=" * 80)
    print("OVERALL VALIDATION SUMMARY")
    print("=" * 80)

    total_modules = len(results)
    passed_modules = sum(1 for success, _ in results.values() if success)
    failed_modules = total_modules - passed_modules

    total_questions = 0
    for success, module_results in results.values():
        if "error" not in module_results:
            for file_result in module_results.values():
                total_questions += file_result["data"]["questions"]

    print(f"\nModules validated: {total_modules}")
    print(f"[PASS] Passed: {passed_modules}")
    if failed_modules > 0:
        print(f"[FAIL] Failed: {failed_modules}")
    print(f"\nTotal questions validated: {total_questions}")

    if failed_modules == 0:
        print("\n*** ALL VALIDATIONS PASSED! ***")
    else:
        print("\n*** SOME VALIDATIONS FAILED - See errors above ***")

    print("=" * 80)


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Main validation function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Validate assessment question JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_all.py                    # Validate all modules
  python validate_all.py --module 1         # Validate Module 1 only
  python validate_all.py --module 4 9 11    # Validate specific modules
  python validate_all.py --verbose          # Show warnings too
        """,
    )

    parser.add_argument(
        "--module",
        "-m",
        type=int,
        nargs="+",
        metavar="N",
        help="Specific module(s) to validate (1-11). If not specified, validates all.",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Show warnings in addition to errors")

    args = parser.parse_args()

    # Determine which modules to validate
    if args.module:
        modules_to_validate = args.module
        # Validate module numbers
        invalid_modules = [m for m in modules_to_validate if m not in ALL_MODULES]
        if invalid_modules:
            print(f"❌ Invalid module number(s): {invalid_modules}")
            print(f"   Valid range: {min(ALL_MODULES)}-{max(ALL_MODULES)}")
            sys.exit(1)
    else:
        modules_to_validate = ALL_MODULES

    # Load schema (validates it exists)
    try:
        load_schema()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Validate each module
    results = {}
    for module_num in modules_to_validate:
        success, module_results = validate_module(module_num, args.verbose)
        results[module_num] = (success, module_results)
        print_module_summary(module_num, success, module_results, args.verbose)

    # Print overall summary
    if len(modules_to_validate) > 1:
        print_overall_summary(results, args.verbose)

    # Exit with appropriate code
    all_passed = all(success for success, _ in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
