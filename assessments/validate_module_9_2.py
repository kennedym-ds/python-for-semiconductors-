"""
Validation script for Module 9.2 - Production ML
"""
import json
from pathlib import Path
from collections import Counter


def validate_module_9_2():
    """Validate Module 9.2 questions file"""

    file_path = Path(__file__).parent / "module-9" / "9.2-questions.json"

    print("=" * 80)
    print("VALIDATING MODULE 9.2 - Production ML")
    print("=" * 80)

    # Load JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("✓ JSON parsing successful")
    except Exception as e:
        print(f"✗ JSON parsing failed: {e}")
        return False

    # Validate structure
    required_fields = ["module_id", "title", "questions"]
    for field in required_fields:
        if field not in data:
            print(f"✗ Missing required field: {field}")
            return False
    print("✓ Required fields present")

    questions = data["questions"]
    print(f"✓ Total questions: {len(questions)}")

    # Check question count
    if len(questions) != 20:
        print(f"✗ Expected 20 questions, found {len(questions)}")
        return False
    print("✓ Question count matches target (20)")

    # Validate question IDs
    expected_ids = [f"m9.2_q{i:03d}" for i in range(1, 21)]
    actual_ids = [q["id"] for q in questions]

    if actual_ids != expected_ids:
        print("✗ Question IDs not sequential or incorrect format")
        print(f"Expected: {expected_ids[:3]}...{expected_ids[-3:]}")
        print(f"Actual: {actual_ids[:3]}...{actual_ids[-3:]}")
        return False
    print("✓ Question IDs are sequential and properly formatted")

    # Check for unique IDs
    if len(set(actual_ids)) != len(actual_ids):
        print("✗ Duplicate question IDs found")
        return False
    print("✓ All question IDs are unique")

    # Type distribution
    type_counts = Counter(q["type"] for q in questions)
    print("\nQuestion Type Distribution:")
    print(
        f"  Multiple Choice: {type_counts['multiple_choice']} ({type_counts['multiple_choice']/len(questions)*100:.1f}%)"
    )
    print(
        f"  Coding Exercises: {type_counts['coding_exercise']} ({type_counts['coding_exercise']/len(questions)*100:.1f}%)"
    )
    print(f"  Conceptual: {type_counts['conceptual']} ({type_counts['conceptual']/len(questions)*100:.1f}%)")

    # Check distribution targets (44-52% MC, 32-40% coding, 16-24% conceptual)
    mc_pct = type_counts["multiple_choice"] / len(questions) * 100
    code_pct = type_counts["coding_exercise"] / len(questions) * 100
    concept_pct = type_counts["conceptual"] / len(questions) * 100

    distribution_ok = True
    if not (44 <= mc_pct <= 52):
        print(f"  ⚠ MC percentage ({mc_pct:.1f}%) outside target range (44-52%)")
        distribution_ok = False
    if not (32 <= code_pct <= 40):
        print(f"  ⚠ Coding percentage ({code_pct:.1f}%) outside target range (32-40%)")
        distribution_ok = False
    if not (16 <= concept_pct <= 24):
        print(f"  ⚠ Conceptual percentage ({concept_pct:.1f}%) outside target range (16-24%)")
        distribution_ok = False

    if distribution_ok:
        print("✓ Distribution within target ranges")

    # Difficulty distribution
    diff_counts = Counter(q["difficulty"] for q in questions)
    print("\nDifficulty Distribution:")
    print(f"  Easy: {diff_counts['easy']}")
    print(f"  Medium: {diff_counts['medium']}")
    print(f"  Hard: {diff_counts['hard']}")

    # Topics
    topics = [q["topic"] for q in questions]
    unique_topics = set(topics)
    print(f"\n✓ Total unique topics: {len(unique_topics)}")
    print(f"Topics: {', '.join(sorted(unique_topics))}")

    # Point distribution
    total_points = sum(q["points"] for q in questions)
    print(f"\n✓ Total points: {total_points}")
    print(f"  Average points per question: {total_points/len(questions):.1f}")

    # Validate question structure
    for i, q in enumerate(questions, 1):
        required_q_fields = ["id", "type", "difficulty", "topic", "question", "explanation", "points"]
        for field in required_q_fields:
            if field not in q:
                print(f"✗ Question {i} ({q.get('id', 'unknown')}) missing field: {field}")
                return False

        # Type-specific validation
        if q["type"] == "multiple_choice":
            if "options" not in q or "correct_answer" not in q:
                print(f"✗ Question {i} (MC) missing options or correct_answer")
                return False
        elif q["type"] == "coding_exercise":
            if "code_template" not in q:
                print(f"✗ Question {i} (coding) missing code_template")
                return False
        elif q["type"] == "conceptual":
            if "rubric" not in q:
                print(f"✗ Question {i} (conceptual) missing rubric")
                return False

    print("\n✓ All questions have required fields and proper structure")

    print("\n" + "=" * 80)
    print("✅ VALIDATION PASSED - Module 9.2")
    print("=" * 80)
    print(f"20 questions covering Production ML best practices")
    print(
        f"{type_counts['multiple_choice']} MC, {type_counts['coding_exercise']} coding, {type_counts['conceptual']} conceptual"
    )
    print(f"{len(unique_topics)} unique topics, {total_points} total points")

    return True


if __name__ == "__main__":
    success = validate_module_9_2()
    exit(0 if success else 1)
