"""
Comprehensive validation script for Phase 2.5 - Module 10 assessments.
Validates all four Module 10 assessment files.
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
    """Validate all Phase 2.5 (Module 10) assessment files."""
    print("=" * 80)
    print("PHASE 2.5 ASSESSMENT VALIDATION - MODULE 10: PROJECT DEVELOPMENT")
    print("=" * 80)

    # Define all Module 10 assessment files
    assessment_files = [
        "assessments/module-10/10.1-questions.json",
        "assessments/module-10/10.2-questions.json",
        "assessments/module-10/10.3-questions.json",
        "assessments/module-10/10.4-questions.json",
    ]

    total_questions = 0
    total_mc = 0
    total_coding = 0
    total_conceptual = 0
    all_valid = True

    for file_path in assessment_files:
        path = Path(file_path)

        if not path.exists():
            print(f"\n❌ {file_path}")
            print(f"   File not found")
            all_valid = False
            continue

        valid, result = validate_assessment_file(path)

        if valid:
            total_questions += result["total"]
            total_mc += result["multiple_choice"]
            total_coding += result["coding_exercise"]
            total_conceptual += result["conceptual"]

            print(f"\n✅ {file_path}")
            print(f"   Title: {result['title']}")
            print(f"   Total: {result['total']} questions")
            print(f"   - Multiple choice: {result['multiple_choice']}")
            print(f"   - Coding exercises: {result['coding_exercise']}")
            print(f"   - Conceptual: {result['conceptual']}")
        else:
            print(f"\n❌ {file_path}")
            print(f"   {result}")
            all_valid = False

    print("\n" + "=" * 80)
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED!")
        print(f"\n📊 PHASE 2.5 SUMMARY:")
        print(f"   Total questions: {total_questions}")
        print(f"   - Multiple choice: {total_mc}")
        print(f"   - Coding exercises: {total_coding}")
        print(f"   - Conceptual: {total_conceptual}")
        print(f"\n   Module 10 (Project Development) is COMPLETE!")
        print(f"   All 4 sub-modules validated successfully.")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED")
        print("   Please fix errors above")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
