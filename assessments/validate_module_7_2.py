#!/usr/bin/env python3
"""
Validation script for Module 7.2 - Pattern Recognition and Wafer Map Analysis
Checks JSON structure, question counts, type distribution, and required fields.
"""

import json
import sys
from pathlib import Path
from collections import Counter


def validate_module_7_2():
    """Validate Module 7.2 assessment structure."""

    # Load the assessment file
    json_path = Path(__file__).parent / "module-7" / "7.2-questions.json"

    print("=" * 70)
    print("MODULE 7.2 VALIDATION - Pattern Recognition and Wafer Map Analysis")
    print("=" * 70)
    print()

    # Check 1: File exists and is valid JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("✅ CHECK 1: Valid JSON format")
    except FileNotFoundError:
        print("⚠️  CHECK 1 FAILED: File not found")
        return False
    except json.JSONDecodeError as e:
        print(f"⚠️  CHECK 1 FAILED: Invalid JSON - {e}")
        return False

    # Check 2: Expected question count (25 questions)
    questions = data.get("questions", [])
    expected_count = 25
    actual_count = len(questions)

    if actual_count == expected_count:
        print(f"✅ CHECK 2: Question count = {actual_count} (target: {expected_count})")
    else:
        print(f"⚠️  CHECK 2 FAILED: Question count = {actual_count}, expected {expected_count}")
        return False

    # Check 3: Question type distribution
    type_counts = Counter(q.get("type") for q in questions)
    mc_count = type_counts.get("multiple_choice", 0)
    coding_count = type_counts.get("coding_exercise", 0)
    conceptual_count = type_counts.get("conceptual", 0)

    mc_pct = (mc_count / actual_count * 100) if actual_count > 0 else 0
    coding_pct = (coding_count / actual_count * 100) if actual_count > 0 else 0
    conceptual_pct = (conceptual_count / actual_count * 100) if actual_count > 0 else 0

    print(f"\n📊 Question Type Distribution:")
    print(f"   Multiple Choice: {mc_count} ({mc_pct:.1f}%)")
    print(f"   Coding Exercise: {coding_count} ({coding_pct:.1f}%)")
    print(f"   Conceptual:      {conceptual_count} ({conceptual_pct:.1f}%)")

    # Distribution targets: 40-50% MC, 30-40% coding, 15-20% conceptual
    distribution_ok = True
    if not (40 <= mc_pct <= 50):
        print(f"   ⚠️  Multiple choice should be 40-50%, got {mc_pct:.1f}%")
        distribution_ok = False
    if not (30 <= coding_pct <= 40):
        print(f"   ⚠️  Coding should be 30-40%, got {coding_pct:.1f}%")
        distribution_ok = False
    if not (15 <= conceptual_pct <= 25):
        print(f"   ⚠️  Conceptual should be 15-25%, got {conceptual_pct:.1f}%")
        distribution_ok = False

    if distribution_ok:
        print("✅ CHECK 3: Type distribution within target ranges")
    else:
        print("⚠️  CHECK 3 FAILED: Type distribution outside target ranges")
        return False

    # Check 4: Required fields for each question type
    required_common = ["id", "type", "difficulty", "topic", "question", "explanation", "points"]
    required_mc = required_common + ["options", "correct_answer"]
    required_coding = required_common + ["code_template", "test_cases", "hints"]
    required_conceptual = required_common + ["rubric", "hints"]

    missing_fields = []
    for q in questions:
        q_type = q.get("type")
        q_id = q.get("id", "unknown")

        if q_type == "multiple_choice":
            required = required_mc
        elif q_type == "coding_exercise":
            required = required_coding
        elif q_type == "conceptual":
            required = required_conceptual
        else:
            missing_fields.append(f"{q_id}: Unknown type '{q_type}'")
            continue

        for field in required:
            if field not in q:
                missing_fields.append(f"{q_id}: Missing '{field}'")

    if not missing_fields:
        print("✅ CHECK 4: All required fields present")
    else:
        print("⚠️  CHECK 4 FAILED: Missing required fields:")
        for msg in missing_fields[:10]:  # Show first 10
            print(f"   - {msg}")
        if len(missing_fields) > 10:
            print(f"   ... and {len(missing_fields) - 10} more")
        return False

    # Check 5: Metadata and statistics
    print(f"\n📋 Module Metadata:")
    print(f"   Module ID:        {data.get('module_id')}")
    print(f"   Title:            {data.get('title')}")
    print(f"   Week:             {data.get('week')}")
    print(f"   Time (minutes):   {data.get('estimated_time_minutes')}")
    print(f"   Passing Score:    {data.get('passing_score')}%")

    # Total points
    total_points = sum(q.get("points", 0) for q in questions)
    print(f"\n🎯 Assessment Statistics:")
    print(f"   Total Points:     {total_points}")
    print(f"   Questions:        {actual_count}")
    print(f"   Avg Points/Q:     {total_points/actual_count:.1f}")

    # Difficulty distribution
    difficulty_counts = Counter(q.get("difficulty") for q in questions)
    print(f"\n📈 Difficulty Distribution:")
    for diff in ["easy", "medium", "hard"]:
        count = difficulty_counts.get(diff, 0)
        pct = (count / actual_count * 100) if actual_count > 0 else 0
        print(f"   {diff.capitalize():8s}: {count:2d} ({pct:.1f}%)")

    # Topic coverage
    topics = [q.get("topic") for q in questions]
    unique_topics = len(set(topics))
    print(f"\n🔍 Topic Coverage:")
    print(f"   Unique Topics:    {unique_topics}")

    if unique_topics == actual_count:
        print("   ✅ Each question covers a distinct topic")
    else:
        print(f"   ⚠️  Some topics may be repeated ({unique_topics} unique vs {actual_count} questions)")

    print(f"\n{'=' * 70}")
    print("✅ ALL CHECKS PASSED - Module 7.2 validation successful!")
    print(f"{'=' * 70}")
    return True


if __name__ == "__main__":
    success = validate_module_7_2()
    sys.exit(0 if success else 1)
