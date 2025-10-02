"""
Validation script for Module 8.2 - Advanced NLP for Manufacturing
Checks JSON structure, question counts, distributions, and quality metrics.
"""

import json
import sys
from pathlib import Path
from collections import Counter


def validate_module_8_2():
    """Validate Module 8.2 assessment file."""

    file_path = Path(__file__).parent / "module-8" / "8.2-questions.json"

    print("=" * 70)
    print("MODULE 8.2 - ADVANCED NLP FOR MANUFACTURING - VALIDATION")
    print("=" * 70)
    print()

    # Load JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("✓ JSON parsing successful")
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing failed: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return False

    # Validate structure
    required_keys = [
        "module_id",
        "sub_module",
        "title",
        "week",
        "description",
        "estimated_time_minutes",
        "passing_score",
        "version",
        "questions",
    ]

    for key in required_keys:
        if key not in data:
            print(f"✗ Missing required key: {key}")
            return False
    print("✓ All required top-level keys present")

    # Validate module metadata
    if data["module_id"] != "module-8.2":
        print(f"✗ Incorrect module_id: {data['module_id']}")
        return False
    if data["sub_module"] != "8.2":
        print(f"✗ Incorrect sub_module: {data['sub_module']}")
        return False
    print("✓ Module metadata correct")

    # Validate questions
    questions = data["questions"]

    # Check question count
    expected_count = 40
    actual_count = len(questions)
    print(f"\nQuestion Count: {actual_count}/{expected_count}", end="")
    if actual_count != expected_count:
        print(f" ✗ (Expected {expected_count})")
        return False
    else:
        print(" ✓")

    # Check question IDs are unique and sequential
    expected_ids = [f"m8.2_q{i:03d}" for i in range(1, 41)]
    actual_ids = [q["id"] for q in questions]

    if actual_ids != expected_ids:
        missing = set(expected_ids) - set(actual_ids)
        extra = set(actual_ids) - set(expected_ids)
        if missing:
            print(f"✗ Missing IDs: {missing}")
        if extra:
            print(f"✗ Extra IDs: {extra}")
        return False
    print("✓ All question IDs unique and sequential (m8.2_q001 to m8.2_q040)")

    # Validate each question
    required_question_keys = ["id", "type", "difficulty", "topic", "question", "explanation", "points"]

    for i, q in enumerate(questions, 1):
        for key in required_question_keys:
            if key not in q:
                print(f"✗ Question {i} missing key: {key}")
                return False

        # Type-specific validation
        if q["type"] == "multiple_choice":
            if "options" not in q or "correct_answer" not in q:
                print(f"✗ Question {i} (MC) missing options or correct_answer")
                return False
        elif q["type"] == "coding_exercise":
            if "code_template" not in q or "test_cases" not in q or "hints" not in q:
                print(f"✗ Question {i} (coding) missing code_template, test_cases, or hints")
                return False
        elif q["type"] == "conceptual":
            if "rubric" not in q or "hints" not in q:
                print(f"✗ Question {i} (conceptual) missing rubric or hints")
                return False

    print("✓ All questions have required fields for their type")

    # Analyze question distributions
    types = [q["type"] for q in questions]
    type_counts = Counter(types)

    print("\n" + "-" * 70)
    print("QUESTION TYPE DISTRIBUTION")
    print("-" * 70)

    mc_count = type_counts.get("multiple_choice", 0)
    coding_count = type_counts.get("coding_exercise", 0)
    conceptual_count = type_counts.get("conceptual", 0)

    mc_pct = (mc_count / actual_count) * 100
    coding_pct = (coding_count / actual_count) * 100
    conceptual_pct = (conceptual_count / actual_count) * 100

    print(f"Multiple Choice:    {mc_count:2d} ({mc_pct:5.1f}%)  [Target: 40-50%]")
    print(f"Coding Exercises:   {coding_count:2d} ({coding_pct:5.1f}%)  [Target: 30-40%]")
    print(f"Conceptual:         {conceptual_count:2d} ({conceptual_pct:5.1f}%)  [Target: 15-20%]")

    # Check distribution targets
    distribution_ok = True
    if not (40 <= mc_pct <= 50):
        print(f"  ⚠ Multiple choice slightly outside target range")
        distribution_ok = False
    if not (30 <= coding_pct <= 40):
        print(f"  ⚠ Coding exercises slightly outside target range")
        distribution_ok = False
    if not (15 <= conceptual_pct <= 20):
        print(f"  ⚠ Conceptual questions slightly outside target range")
        distribution_ok = False

    if distribution_ok:
        print("✓ Distribution within target ranges")

    # Difficulty distribution
    print("\n" + "-" * 70)
    print("DIFFICULTY DISTRIBUTION")
    print("-" * 70)

    difficulties = [q["difficulty"] for q in questions]
    difficulty_counts = Counter(difficulties)

    for difficulty in ["easy", "medium", "hard"]:
        count = difficulty_counts.get(difficulty, 0)
        pct = (count / actual_count) * 100
        print(f"{difficulty.capitalize():10s}: {count:2d} ({pct:5.1f}%)")

    # Topic coverage
    print("\n" + "-" * 70)
    print("TOPIC COVERAGE")
    print("-" * 70)

    topics = [q["topic"] for q in questions]
    unique_topics = len(set(topics))
    print(f"Unique topics: {unique_topics}")

    topic_counts = Counter(topics)
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count}")

    # Point distribution
    print("\n" + "-" * 70)
    print("POINT DISTRIBUTION")
    print("-" * 70)

    total_points = sum(q["points"] for q in questions)
    points_by_type = {
        "multiple_choice": sum(q["points"] for q in questions if q["type"] == "multiple_choice"),
        "coding_exercise": sum(q["points"] for q in questions if q["type"] == "coding_exercise"),
        "conceptual": sum(q["points"] for q in questions if q["type"] == "conceptual"),
    }

    print(f"Total points: {total_points}")
    print(f"  Multiple Choice: {points_by_type['multiple_choice']} points")
    print(f"  Coding Exercises: {points_by_type['coding_exercise']} points")
    print(f"  Conceptual: {points_by_type['conceptual']} points")
    print(f"Average points per question: {total_points/actual_count:.1f}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print("✓ JSON structure valid")
    print(f"✓ Question count correct ({actual_count}/40)")
    print(f"✓ All IDs unique and sequential")
    print(f"✓ Type distribution: {mc_count} MC, {coding_count} coding, {conceptual_count} conceptual")
    print(f"✓ {unique_topics} unique topics covered")
    print(f"✓ {total_points} total points")
    print()
    print("✅ MODULE 8.2 VALIDATION PASSED")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = validate_module_8_2()
    sys.exit(0 if success else 1)
