"""
Validation script for Module 11.1 - Edge AI & Model Deployment assessment.
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
            "title": data["title"],
        }

    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Validate Module 11.1 assessment file."""
    print("=" * 80)
    print("MODULE 11.1 ASSESSMENT VALIDATION - EDGE AI & MODEL DEPLOYMENT")
    print("=" * 80)

    file_path = "assessments/module-11/11.1-questions.json"
    path = Path(file_path)

    if not path.exists():
        print(f"\n❌ {file_path}")
        print(f"   File not found")
        sys.exit(1)

    valid, result = validate_assessment_file(path)

    if valid:
        print(f"\n✅ {file_path}")
        print(f"   Title: {result['title']}")  # type: ignore
        print(f"   Total: {result['total']} questions")  # type: ignore
        print(f"   - Multiple choice: {result['multiple_choice']}")  # type: ignore
        print(f"   - Coding exercises: {result['coding_exercise']}")  # type: ignore
        print(f"   - Conceptual: {result['conceptual']}")  # type: ignore

        # Verify expected counts
        expected_mc = 12
        expected_coding = 8
        expected_conceptual = 5
        expected_total = 25

        if (
            result["total"] == expected_total
            and result["multiple_choice"] == expected_mc  # type: ignore
            and result["coding_exercise"] == expected_coding  # type: ignore
            and result["conceptual"] == expected_conceptual  # type: ignore
        ):  # type: ignore
            print("\n" + "=" * 80)
            print("✅ ALL VALIDATIONS PASSED!")
            print(f"\n   Module 11.1 (Edge AI & Model Deployment) is COMPLETE!")
            print(f"   ✓ 25 questions (12 MC, 8 coding, 5 conceptual)")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("⚠️  WARNING: Question counts don't match expected")
            print(
                f"   Expected: {expected_total} total ({expected_mc} MC, {expected_coding} coding, {expected_conceptual} conceptual)"
            )
            print(f"   Found: {result['total']} total ({result['multiple_choice']} MC, {result['coding_exercise']} coding, {result['conceptual']} conceptual)")  # type: ignore
            print("=" * 80)
            sys.exit(1)
    else:
        print(f"\n❌ {file_path}")
        print(f"   {result}")
        print("\n" + "=" * 80)
        print("❌ VALIDATION FAILED")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
