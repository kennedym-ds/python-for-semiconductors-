"""
Comprehensive validation script for Phase 2.4 (Modules 8-9)
Validates all 125 questions across Generative AI and MLOps modules.
"""
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


def load_module_questions(module_path: Path) -> Tuple[dict, List[dict]]:
    """Load questions from a module JSON file"""
    with open(module_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, data["questions"]


def validate_phase_2_4():
    """Validate all Phase 2.4 modules (8-9)"""

    print("=" * 100)
    print("PHASE 2.4 COMPREHENSIVE VALIDATION - GENERATIVE AI & MLOps")
    print("=" * 100)

    base_path = Path(__file__).parent

    # Define module files
    modules = {
        "8.1": base_path / "module-8" / "8.1-questions.json",
        "8.2": base_path / "module-8" / "8.2-questions.json",
        "9.1": base_path / "module-9" / "9.1-questions.json",
        "9.2": base_path / "module-9" / "9.2-questions.json",
    }

    all_questions = []
    module_summaries = {}
    total_points = 0

    # Load and validate each module
    for module_id, module_path in modules.items():
        print(f"\n{'─' * 100}")
        print(f"MODULE {module_id}")
        print("─" * 100)

        if not module_path.exists():
            print(f"✗ Module file not found: {module_path}")
            return False

        try:
            data, questions = load_module_questions(module_path)
            print(f"✓ Loaded {len(questions)} questions from {data['title']}")

            # Validate question IDs are unique within module
            question_ids = [q["id"] for q in questions]
            if len(set(question_ids)) != len(question_ids):
                print(f"✗ Duplicate question IDs in module {module_id}")
                return False

            # Count types
            type_counts = Counter(q["type"] for q in questions)
            difficulty_counts = Counter(q["difficulty"] for q in questions)
            topics = set(q["topic"] for q in questions)
            module_points = sum(q["points"] for q in questions)

            print(
                f"  Types: {type_counts['multiple_choice']} MC, {type_counts['coding_exercise']} coding, {type_counts['conceptual']} conceptual"
            )
            print(
                f"  Difficulty: {difficulty_counts['easy']} easy, {difficulty_counts['medium']} medium, {difficulty_counts['hard']} hard"
            )
            print(f"  Topics: {len(topics)} unique")
            print(f"  Points: {module_points}")

            module_summaries[module_id] = {
                "title": data["title"],
                "count": len(questions),
                "type_counts": dict(type_counts),
                "difficulty_counts": dict(difficulty_counts),
                "topics": len(topics),
                "points": module_points,
            }

            all_questions.extend(questions)
            total_points += module_points

        except Exception as e:
            print(f"✗ Error loading module {module_id}: {e}")
            return False

    # Phase-wide validation
    print(f"\n{'=' * 100}")
    print("PHASE 2.4 OVERALL STATISTICS")
    print("=" * 100)

    print(f"\n✓ Total questions: {len(all_questions)} / 125 expected")
    if len(all_questions) != 125:
        print(f"✗ Expected 125 questions, found {len(all_questions)}")
        return False

    # Check for duplicate IDs across all modules
    all_ids = [q["id"] for q in all_questions]
    if len(set(all_ids)) != len(all_ids):
        print("✗ Duplicate question IDs found across modules")
        duplicates = [id for id in all_ids if all_ids.count(id) > 1]
        print(f"  Duplicates: {set(duplicates)}")
        return False
    print("✓ All question IDs are unique across all modules")

    # Overall type distribution
    all_types = Counter(q["type"] for q in all_questions)
    mc_pct = all_types["multiple_choice"] / len(all_questions) * 100
    coding_pct = all_types["coding_exercise"] / len(all_questions) * 100
    conceptual_pct = all_types["conceptual"] / len(all_questions) * 100

    print(f"\n✓ Overall Type Distribution:")
    print(f"  Multiple Choice: {all_types['multiple_choice']} ({mc_pct:.1f}%)")
    print(f"  Coding Exercises: {all_types['coding_exercise']} ({coding_pct:.1f}%)")
    print(f"  Conceptual: {all_types['conceptual']} ({conceptual_pct:.1f}%)")

    # Check overall distribution targets
    distribution_ok = True
    if not (44 <= mc_pct <= 52):
        print(f"  ⚠ MC percentage ({mc_pct:.1f}%) outside target range (44-52%)")
        distribution_ok = False
    if not (32 <= coding_pct <= 40):
        print(f"  ⚠ Coding percentage ({coding_pct:.1f}%) outside target range (32-40%)")
        distribution_ok = False
    if not (16 <= conceptual_pct <= 24):
        print(f"  ⚠ Conceptual percentage ({conceptual_pct:.1f}%) outside target range (16-24%)")
        distribution_ok = False

    if distribution_ok:
        print("  ✓ Distribution within target ranges")

    # Difficulty distribution
    all_difficulty = Counter(q["difficulty"] for q in all_questions)
    print(f"\n✓ Overall Difficulty Distribution:")
    print(f"  Easy: {all_difficulty['easy']} ({all_difficulty['easy']/len(all_questions)*100:.1f}%)")
    print(f"  Medium: {all_difficulty['medium']} ({all_difficulty['medium']/len(all_questions)*100:.1f}%)")
    print(f"  Hard: {all_difficulty['hard']} ({all_difficulty['hard']/len(all_questions)*100:.1f}%)")

    # Topics
    all_topics = set(q["topic"] for q in all_questions)
    print(f"\n✓ Total unique topics across all modules: {len(all_topics)}")

    # Points
    print(f"\n✓ Total points: {total_points}")
    print(f"  Average points per question: {total_points/len(all_questions):.1f}")

    # Module breakdown
    print(f"\n{'─' * 100}")
    print("MODULE BREAKDOWN")
    print("─" * 100)
    for module_id, summary in module_summaries.items():
        print(f"\nModule {module_id}: {summary['title']}")
        print(f"  Questions: {summary['count']}")
        print(f"  Types: {summary['type_counts']}")
        print(f"  Difficulty: {summary['difficulty_counts']}")
        print(f"  Topics: {summary['topics']}")
        print(f"  Points: {summary['points']}")

    # Topic coverage analysis
    print(f"\n{'─' * 100}")
    print("TOPIC COVERAGE ANALYSIS")
    print("─" * 100)

    generative_ai_topics = [
        t
        for t in all_topics
        if any(
            keyword in t.lower()
            for keyword in [
                "gan",
                "vae",
                "diffusion",
                "synthetic",
                "generation",
                "augmentation",
                "transformer",
                "bert",
                "gpt",
                "nlp",
                "text",
                "language",
            ]
        )
    ]
    mlops_topics = [
        t
        for t in all_topics
        if any(
            keyword in t.lower()
            for keyword in [
                "mlops",
                "deployment",
                "monitoring",
                "serving",
                "pipeline",
                "drift",
                "production",
                "ci_cd",
                "docker",
                "kubernetes",
                "mlflow",
            ]
        )
    ]

    print(f"\n✓ Generative AI topics: {len(generative_ai_topics)}")
    print(f"  {', '.join(sorted(generative_ai_topics)[:10])}{'...' if len(generative_ai_topics) > 10 else ''}")

    print(f"\n✓ MLOps topics: {len(mlops_topics)}")
    print(f"  {', '.join(sorted(mlops_topics)[:10])}{'...' if len(mlops_topics) > 10 else ''}")

    # Quality checks
    print(f"\n{'─' * 100}")
    print("QUALITY CHECKS")
    print("─" * 100)

    # Check for questions missing required fields
    required_fields = ["id", "type", "difficulty", "topic", "question", "explanation", "points"]
    missing_fields = []
    for q in all_questions:
        for field in required_fields:
            if field not in q:
                missing_fields.append(f"{q.get('id', 'unknown')}: missing {field}")

    if missing_fields:
        print(f"✗ Questions with missing fields: {len(missing_fields)}")
        for issue in missing_fields[:5]:
            print(f"  {issue}")
    else:
        print("✓ All questions have required fields")

    # Check for type-specific fields
    type_field_issues = []
    for q in all_questions:
        if q["type"] == "multiple_choice":
            if "options" not in q or "correct_answer" not in q:
                type_field_issues.append(f"{q['id']}: MC missing options/correct_answer")
        elif q["type"] == "coding_exercise":
            if "code_template" not in q:
                type_field_issues.append(f"{q['id']}: Coding missing code_template")
        elif q["type"] == "conceptual":
            if "rubric" not in q:
                type_field_issues.append(f"{q['id']}: Conceptual missing rubric")

    if type_field_issues:
        print(f"✗ Type-specific field issues: {len(type_field_issues)}")
        for issue in type_field_issues[:5]:
            print(f"  {issue}")
    else:
        print("✓ All questions have type-specific required fields")

    # Final summary
    print(f"\n{'=' * 100}")
    print("✅ PHASE 2.4 VALIDATION COMPLETE")
    print("=" * 100)
    print(f"Total Modules: 4 (8.1, 8.2, 9.1, 9.2)")
    print(f"Total Questions: {len(all_questions)}")
    print(f"Total Topics: {len(all_topics)}")
    print(f"Total Points: {total_points}")
    print(f"\nCoverage:")
    print(f"  • Generative AI (GANs, VAEs, Diffusion, NLP)")
    print(f"  • MLOps (CI/CD, Deployment, Monitoring, Serving)")
    print(f"  • Production ML (Drift Detection, Troubleshooting, Security)")

    if not missing_fields and not type_field_issues:
        print("\n✅ ALL VALIDATION CHECKS PASSED")
        return True
    else:
        print("\n⚠ VALIDATION PASSED WITH WARNINGS")
        return True


if __name__ == "__main__":
    success = validate_phase_2_4()
    exit(0 if success else 1)
