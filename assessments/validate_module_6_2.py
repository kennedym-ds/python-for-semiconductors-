#!/usr/bin/env python3
"""Validate Module 6.2 assessment structure."""

import json
from pathlib import Path
from collections import Counter


def validate_module_6_2():
    """Validate Module 6.2 question bank."""

    file_path = Path("assessments/module-6/6.2-questions.json")

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ Valid JSON structure")
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return False

    # Count questions
    questions = data.get("questions", [])
    total = len(questions)
    print(f"\n📊 Total Questions: {total}")

    # Count by type
    types = Counter(q["type"] for q in questions)
    print(f"\n📝 Question Types:")
    print(f"   Multiple Choice: {types['multiple_choice']}")
    print(f"   Coding Exercises: {types['coding_exercise']}")
    print(f"   Conceptual: {types['conceptual']}")

    # Calculate distribution
    mc_pct = (types["multiple_choice"] / total) * 100
    coding_pct = (types["coding_exercise"] / total) * 100
    conceptual_pct = (types["conceptual"] / total) * 100

    print(f"\n📈 Distribution:")
    print(f"   MC: {mc_pct:.1f}% (Target: 40-50%)")
    print(f"   Coding: {coding_pct:.1f}% (Target: 30-40%)")
    print(f"   Conceptual: {conceptual_pct:.1f}% (Target: 15-20%)")

    # Count by difficulty
    difficulties = Counter(q["difficulty"] for q in questions)
    print(f"\n🎯 Difficulty Levels:")
    for diff, count in sorted(difficulties.items()):
        print(f"   {diff.capitalize()}: {count}")

    # Count by topic
    topics = Counter(q["topic"] for q in questions)
    print(f"\n🏷️  Topics Covered ({len(topics)} unique):")
    for topic, count in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"   {topic.replace('_', ' ').title()}: {count}")

    # Calculate total points
    total_points = sum(q["points"] for q in questions)
    print(f"\n💯 Total Points: {total_points}")

    # Check metadata
    print(f"\n📋 Metadata:")
    print(f"   Module: {data.get('module_id')}")
    print(f"   Title: {data.get('title')}")
    print(f"   Week: {data.get('week')}")
    print(f"   Estimated Time: {data.get('estimated_time_minutes')} minutes")
    print(f"   Passing Score: {data.get('passing_score')}%")

    # Validation checks
    print(f"\n✔️  Validation Results:")
    checks_passed = 0
    checks_total = 5

    if total == 30:
        print(f"   ✅ Question count: {total} (Target: 30)")
        checks_passed += 1
    else:
        print(f"   ⚠️  Question count: {total} (Target: 30)")

    if 40 <= mc_pct <= 50:
        print(f"   ✅ MC distribution: {mc_pct:.1f}% (Target: 40-50%)")
        checks_passed += 1
    else:
        print(f"   ⚠️  MC distribution: {mc_pct:.1f}% (Target: 40-50%)")

    if 30 <= coding_pct <= 40:
        print(f"   ✅ Coding distribution: {coding_pct:.1f}% (Target: 30-40%)")
        checks_passed += 1
    else:
        print(f"   ⚠️  Coding distribution: {coding_pct:.1f}% (Target: 30-40%)")

    if 15 <= conceptual_pct <= 20:
        print(f"   ✅ Conceptual distribution: {conceptual_pct:.1f}% (Target: 15-20%)")
        checks_passed += 1
    else:
        print(f"   ⚠️  Conceptual distribution: {conceptual_pct:.1f}% (Target: 15-20%)")

    # Check all questions have required fields
    required_fields = ["id", "type", "difficulty", "topic", "question", "explanation", "points"]
    all_valid = True
    for i, q in enumerate(questions, 1):
        for field in required_fields:
            if field not in q:
                print(f"   ❌ Question {i} missing field: {field}")
                all_valid = False

    if all_valid:
        print(f"   ✅ All questions have required fields")
        checks_passed += 1

    print(f"\n🏆 Validation Score: {checks_passed}/{checks_total} checks passed")

    return checks_passed == checks_total


if __name__ == "__main__":
    success = validate_module_6_2()
    exit(0 if success else 1)
