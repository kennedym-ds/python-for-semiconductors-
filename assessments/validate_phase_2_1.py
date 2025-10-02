"""
Quick validation script to verify all Phase 2.1 assessment JSON files load correctly.
"""

import json
from pathlib import Path


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

        # Validate each question has required fields
        for i, q in enumerate(questions):
            if "id" not in q:
                return False, f"Question {i} missing 'id'"
            if "type" not in q:
                return False, f"Question {i} missing 'type'"
            if "question" not in q:
                return False, f"Question {i} missing 'question'"

        return True, f"{len(questions)} questions"

    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Validate all Phase 2.1 assessment files."""
    print("=" * 70)
    print("PHASE 2.1 ASSESSMENT VALIDATION")
    print("=" * 70)

    # Define all Phase 2.1 assessment files
    assessment_files = [
        "assessments/module-1/1.1-questions.json",
        "assessments/module-1/1.2-questions.json",
        "assessments/module-2/2.1-questions.json",
        "assessments/module-2/2.2-questions.json",
        "assessments/module-2/2.3-questions.json",
        "assessments/module-3/3.1-questions.json",
        "assessments/module-3/3.2-questions.json",
    ]

    total_questions = 0
    all_valid = True

    for file_path in assessment_files:
        path = Path(file_path)

        if not path.exists():
            print(f"\n❌ {file_path}")
            print(f"   File not found")
            all_valid = False
            continue

        valid, message = validate_assessment_file(path)

        if valid:
            # Extract question count from message
            q_count = int(message.split()[0])
            total_questions += q_count
            print(f"\n✅ {file_path}")
            print(f"   {message} loaded successfully")
        else:
            print(f"\n❌ {file_path}")
            print(f"   {message}")
            all_valid = False

    print("\n" + "=" * 70)
    if all_valid:
        print(f"✅ ALL VALIDATIONS PASSED!")
        print(f"   Total: {total_questions} questions across 7 modules")
        print(f"   Phase 2.1 is COMPLETE and ready for deployment!")
    else:
        print("❌ VALIDATION FAILED - Please fix errors above")
    print("=" * 70)


if __name__ == "__main__":
    main()
