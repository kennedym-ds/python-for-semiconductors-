"""
Validation script for Module 10.1 assessment (Project Architecture).
"""

import json
from pathlib import Path
import sys


def validate_assessment_file(file_path):
    """Validate a single assessment JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check required fields
        required_fields = ["module_id", "sub_module", "title", "week", "questions"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return False, f"Missing fields: {missing_fields}"

        # Check module ID
        if data["module_id"] != "module-10.1":
            return False, f"Expected module_id 'module-10.1', got '{data['module_id']}'"

        # Check questions
        questions = data["questions"]
        if not isinstance(questions, list):
            return False, "Questions must be a list"

        if len(questions) == 0:
            return False, "No questions found"

        # Count question types
        mc_count = sum(1 for q in questions if q.get("type") == "multiple_choice")
        coding_count = sum(1 for q in questions if q.get("type") == "coding_exercise")
        conceptual_count = sum(1 for q in questions if q.get("type") == "conceptual")

        # Validate each question has required fields
        for i, q in enumerate(questions, 1):
            if "id" not in q:
                return False, f"Question {i} missing 'id'"
            if "type" not in q:
                return False, f"Question {i} missing 'type'"
            if "question" not in q:
                return False, f"Question {i} missing 'question'"
            if "difficulty" not in q:
                return False, f"Question {i} missing 'difficulty'"
            if "points" not in q:
                return False, f"Question {i} missing 'points'"

            # Type-specific validation
            if q["type"] == "multiple_choice":
                if "options" not in q:
                    return False, f"Question {i} (MC) missing 'options'"
                if "correct_answer" not in q:
                    return False, f"Question {i} (MC) missing 'correct_answer'"
            elif q["type"] == "coding_exercise":
                if "starter_code" not in q:
                    return False, f"Question {i} (Coding) missing 'starter_code'"
                if "test_cases" not in q:
                    return False, f"Question {i} (Coding) missing 'test_cases'"
            elif q["type"] == "conceptual":
                if "rubric" not in q:
                    return False, f"Question {i} (Conceptual) missing 'rubric'"

        return True, {
            "total": len(questions),
            "multiple_choice": mc_count,
            "coding_exercise": coding_count,
            "conceptual": conceptual_count,
        }

    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Validate Module 10.1 assessment file."""
    print("=" * 70)
    print("MODULE 10.1 ASSESSMENT VALIDATION")
    print("=" * 70)

    file_path = Path("assessments/module-10/10.1-questions.json")

    if not file_path.exists():
        print(f"\n❌ {file_path}")
        print(f"   File not found")
        sys.exit(1)

    valid, result = validate_assessment_file(file_path)

    if valid:
        print(f"\n✅ {file_path}")
        print(f"   Total questions: {result['total']}")
        print(f"   Multiple choice: {result['multiple_choice']}")
        print(f"   Coding exercises: {result['coding_exercise']}")
        print(f"   Conceptual: {result['conceptual']}")
        print("\n" + "=" * 70)
        print("✅ VALIDATION PASSED!")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"\n❌ {file_path}")
        print(f"   {result}")
        print("\n" + "=" * 70)
        print("❌ VALIDATION FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
