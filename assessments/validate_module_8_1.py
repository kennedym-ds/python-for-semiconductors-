"""
Validation script for Module 8.1 Assessment: Generative Models
"""

import json
import sys
from pathlib import Path
from collections import Counter


def validate_module_8_1():
    """Validate Module 8.1 questions structure and content."""

    assessment_file = Path(__file__).parent / "module-8" / "8.1-questions.json"

    print("=" * 70)
    print("MODULE 8.1 VALIDATION: Generative Models for Semiconductor Manufacturing")
    print("=" * 70)

    # Check file exists
    if not assessment_file.exists():
        print(f"❌ FAIL: Assessment file not found: {assessment_file}")
        return False

    # Load and parse JSON
    try:
        with open(assessment_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("✓ JSON structure valid")
    except json.JSONDecodeError as e:
        print(f"❌ FAIL: Invalid JSON: {e}")
        return False

    # Validate required top-level fields
    required_fields = ["module_id", "title", "questions"]
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        print(f"❌ FAIL: Missing required fields: {missing_fields}")
        return False
    print("✓ Required top-level fields present")

    # Validate module metadata
    if data["module_id"] != "module-8.1":
        print(f"❌ FAIL: Incorrect module_id: {data['module_id']}")
        return False
    print(f"✓ Module ID correct: {data['module_id']}")

    if "Generative Models" not in data["title"]:
        print(f"⚠ Warning: Title may not match content: {data['title']}")
    print(f"✓ Title: {data['title']}")

    # Validate questions
    questions = data["questions"]

    # Check question count
    expected_count = 40
    actual_count = len(questions)
    if actual_count != expected_count:
        print(f"❌ FAIL: Expected {expected_count} questions, found {actual_count}")
        return False
    print(f"✓ Question count: {actual_count}/{expected_count}")

    # Check question IDs are unique and sequential
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        duplicates = [id for id in ids if ids.count(id) > 1]
        print(f"❌ FAIL: Duplicate question IDs: {duplicates}")
        return False

    expected_ids = [f"m8.1_q{i:03d}" for i in range(1, expected_count + 1)]
    if ids != expected_ids:
        print(f"❌ FAIL: Question IDs not sequential or incorrectly formatted")
        print(f"  Expected: {expected_ids[:3]} ... {expected_ids[-3:]}")
        print(f"  Got: {ids[:3]} ... {ids[-3:]}")
        return False
    print("✓ Question IDs unique and sequential")

    # Validate each question
    required_q_fields = ["id", "type", "difficulty", "topic", "question", "points"]
    type_specific_fields = {
        "multiple_choice": ["options", "correct_answer", "explanation"],
        "coding_exercise": ["code_template", "test_cases", "hints", "explanation"],
        "conceptual": ["rubric", "hints", "explanation"],
    }

    question_types = Counter()
    difficulties = Counter()
    topics = set()
    total_points = 0

    for i, q in enumerate(questions, 1):
        # Check required fields
        missing = [f for f in required_q_fields if f not in q]
        if missing:
            print(f"❌ FAIL: Question {i} ({q.get('id', 'NO_ID')}) missing fields: {missing}")
            return False

        # Check type-specific fields
        q_type = q["type"]
        if q_type not in type_specific_fields:
            print(f"❌ FAIL: Question {i} has invalid type: {q_type}")
            return False

        type_fields = type_specific_fields[q_type]
        missing_type_fields = [f for f in type_fields if f not in q]
        if missing_type_fields:
            print(f"❌ FAIL: Question {i} ({q['id']}) missing {q_type} fields: {missing_type_fields}")
            return False

        # Validate multiple choice specifics
        if q_type == "multiple_choice":
            if not isinstance(q["options"], list) or len(q["options"]) != 4:
                print(f"❌ FAIL: Question {i} must have exactly 4 options")
                return False
            if not isinstance(q["correct_answer"], int) or q["correct_answer"] not in [0, 1, 2, 3]:
                print(f"❌ FAIL: Question {i} correct_answer must be 0-3")
                return False

        # Validate coding exercise specifics
        if q_type == "coding_exercise":
            if not isinstance(q["test_cases"], list) or len(q["test_cases"]) == 0:
                print(f"❌ FAIL: Question {i} must have test_cases")
                return False
            if not isinstance(q["hints"], list) or len(q["hints"]) == 0:
                print(f"❌ FAIL: Question {i} must have hints")
                return False

        # Validate conceptual specifics
        if q_type == "conceptual":
            if not isinstance(q["rubric"], list) or len(q["rubric"]) == 0:
                print(f"❌ FAIL: Question {i} must have rubric")
                return False

        # Check points
        if not isinstance(q["points"], (int, float)) or q["points"] <= 0:
            print(f"❌ FAIL: Question {i} has invalid points: {q['points']}")
            return False

        # Check difficulty
        if q["difficulty"] not in ["easy", "medium", "hard"]:
            print(f"❌ FAIL: Question {i} has invalid difficulty: {q['difficulty']}")
            return False

        question_types[q_type] += 1
        difficulties[q["difficulty"]] += 1
        topics.add(q["topic"])
        total_points += q["points"]

    print(f"✓ All questions have required fields and valid structure")

    # Check type distribution
    print(f"\n📊 Question Type Distribution:")
    mc_count = question_types["multiple_choice"]
    coding_count = question_types["coding_exercise"]
    conceptual_count = question_types["conceptual"]

    mc_pct = (mc_count / actual_count) * 100
    coding_pct = (coding_count / actual_count) * 100
    conceptual_pct = (conceptual_count / actual_count) * 100

    print(f"  Multiple Choice: {mc_count} ({mc_pct:.1f}%) - Target: 40-50%")
    print(f"  Coding Exercise: {coding_count} ({coding_pct:.1f}%) - Target: 30-40%")
    print(f"  Conceptual: {conceptual_count} ({conceptual_pct:.1f}%) - Target: 15-20%")

    # Validate distribution
    distribution_ok = True
    if not (40 <= mc_pct <= 50):
        print(f"  ⚠ Warning: Multiple choice percentage {mc_pct:.1f}% outside 40-50% target")
        distribution_ok = False
    if not (30 <= coding_pct <= 40):
        print(f"  ⚠ Warning: Coding exercise percentage {coding_pct:.1f}% outside 30-40% target")
        distribution_ok = False
    if not (15 <= conceptual_pct <= 20):
        print(f"  ⚠ Warning: Conceptual percentage {conceptual_pct:.1f}% outside 15-20% target")
        distribution_ok = False

    if distribution_ok:
        print("  ✓ Type distribution within targets")

    # Check difficulty distribution
    print(f"\n📊 Difficulty Distribution:")
    for diff in ["easy", "medium", "hard"]:
        count = difficulties[diff]
        pct = (count / actual_count) * 100
        print(f"  {diff.capitalize()}: {count} ({pct:.1f}%)")

    # Check topics
    print(f"\n📊 Topics Covered ({len(topics)} unique topics):")
    expected_topics = [
        "gan_basics",
        "gan_loss_function",
        "conditional_gan",
        "mode_collapse",
        "vae_basics",
        "vae_loss_function",
        "vae_vs_gan",
        "diffusion_models",
        "synthetic_data_quality",
        "latent_space_manipulation",
        "class_imbalance_augmentation",
        "privacy_preservation",
        "simple_gan_implementation",
        "conditional_gan_implementation",
        "vae_implementation",
        "fid_score_calculation",
        "data_augmentation_pipeline",
        "vae_anomaly_detection",
        "latent_space_visualization",
        "synthetic_data_validation",
        "gan_training_challenges",
        "generative_vs_discriminative",
        "conditional_generation_applications",
        "privacy_differential_privacy",
        "evaluation_metrics_comparison",
        "mode_collapse_detection",
        "disentangled_vae",
        "synthetic_real_mixing_strategy",
        "transfer_learning_gan",
        "diffusion_vs_gan",
        "gan_architecture_choices",
        "generative_pipeline_production",
        "rare_defect_synthesis",
        "synthetic_temporal_consistency",
        "federated_learning_gan",
        "vae_reparameterization_trick",
        "inception_score",
        "gan_equilibrium",
        "spectral_normalization",
        "synthetic_data_ablation",
    ]

    topics_list = sorted(topics)
    print(f"  Topics: {', '.join(topics_list[:5])}, ...")

    missing_topics = set(expected_topics) - topics
    extra_topics = topics - set(expected_topics)

    if missing_topics:
        print(f"  ⚠ Some expected topics missing: {missing_topics}")
    if extra_topics:
        print(f"  ℹ Additional topics included: {extra_topics}")

    # Check total points
    print(f"\n📊 Total Points: {total_points}")
    if total_points < 120 or total_points > 160:
        print(f"  ⚠ Warning: Total points {total_points} outside typical range (120-160)")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"✓ Module 8.1 assessment structure valid")
    print(f"✓ {actual_count} questions with unique IDs")
    print(f"✓ {len(topics)} unique topics covering generative models comprehensively")
    print(f"✓ Total points: {total_points}")
    print(f"✓ Distribution: {mc_count} MC, {coding_count} Coding, {conceptual_count} Conceptual")

    if not distribution_ok:
        print(f"\n⚠ Note: Type distribution slightly outside targets but acceptable")

    print(f"\n🎉 Module 8.1 validation PASSED!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = validate_module_8_1()
    sys.exit(0 if success else 1)
