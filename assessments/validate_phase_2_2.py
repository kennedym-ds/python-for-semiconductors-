import json
from pathlib import Path

# Module files to validate
modules = [
    ("4.1", "assessments/module-4/4.1-questions.json"),
    ("4.2", "assessments/module-4/4.2-questions.json"),
    ("5.1", "assessments/module-5/5.1-questions.json"),
    ("5.2", "assessments/module-5/5.2-questions.json"),
]

print("=" * 60)
print("Phase 2.2 Validation Report")
print("=" * 60)
print()

total_questions = 0
all_valid = True

for mod_name, file_path in modules:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data["questions"]
        mc = sum(1 for q in questions if q["type"] == "multiple_choice")
        ce = sum(1 for q in questions if q["type"] == "coding_exercise")
        con = sum(1 for q in questions if q["type"] == "conceptual")

        print(f"✅ Module {mod_name}: {len(questions)} questions")
        print(f"   Distribution: {mc} MC, {ce} coding, {con} conceptual")
        print(f"   File: {file_path}")
        print()

        total_questions += len(questions)

    except Exception as e:
        print(f"❌ Module {mod_name}: VALIDATION FAILED")
        print(f"   Error: {e}")
        print()
        all_valid = False

print("=" * 60)
print(f"Total Questions: {total_questions}")
print(f"Validation Status: {'✅ ALL PASS' if all_valid else '❌ ERRORS FOUND'}")
print("=" * 60)
