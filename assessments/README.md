## Assessment System

This directory contains the assessment infrastructure for the Python for Semiconductors learning series.

### Directory Structure

```text
assessments/
├── schema.json                    # JSON schema for question banks
├── templates/                     # Question templates
│   ├── multiple_choice.json      # Multiple choice template
│   ├── coding_exercise.json      # Coding exercise template
│   └── conceptual_question.json  # Conceptual question template
├── module-1/                      # Module 1 question banks
├── module-2/                      # Module 2 question banks
├── module-3/                      # Module 3 question banks
├── module-4/                      # Module 4 question banks
├── module-5/                      # Module 5 question banks
├── module-6/                      # Module 6 question banks
├── module-7/                      # Module 7 question banks
├── module-8/                      # Module 8 question banks
├── module-9/                      # Module 9 question banks
├── module-10/                     # Module 10 question banks
└── module-11/                     # Module 11 question banks
```

### Question Types

#### 1. Multiple Choice

Test conceptual understanding and knowledge recall.

**Format**:

- Question text
- 2-6 answer options
- Single correct answer (by index)
- Detailed explanation
- Optional hints

**Example**:

```json
{
  "id": "m1.1_q001",
  "type": "multiple_choice",
  "difficulty": "easy",
  "topic": "python_basics",
  "question": "What is the primary advantage of using NumPy arrays?",
  "options": [
    "Better performance for numerical operations",
    "Built-in plotting capabilities",
    "Automatic data cleaning",
    "Native JSON serialization"
  ],
  "correct_answer": 0,
  "explanation": "NumPy provides vectorized operations...",
  "points": 2
}
```

#### 2. Coding Exercises

Test practical implementation skills.

**Format**:

- Problem statement
- Starter code template
- Multiple test cases with inputs/outputs
- Detailed explanation of solution approach
- Optional hints

**Example**:

```json
{
  "id": "m1.1_q002",
  "type": "coding_exercise",
  "difficulty": "medium",
  "topic": "data_manipulation",
  "question": "Write a function to calculate yield percentage...",
  "starter_code": "import pandas as pd\n\ndef calculate_yield(df):\n    pass",
  "test_cases": [
    {
      "input": {"data": [...]},
      "expected_output": 96.0,
      "description": "Basic yield calculation"
    }
  ],
  "points": 5
}
```

#### 3. Conceptual Questions

Test deep understanding and ability to apply concepts.

**Format**:

- Open-ended question
- Grading rubric with criteria and points
- Keywords to look for (optional)
- Model answer in explanation
- Optional hints

**Example**:

```json
{
  "id": "m1.1_q003",
  "type": "conceptual",
  "difficulty": "hard",
  "topic": "ml_application",
  "question": "Explain how you would approach detecting anomalous wafer patterns...",
  "rubric": [
    {
      "criteria": "Identifies appropriate ML algorithms",
      "points": 3,
      "keywords": ["isolation forest", "autoencoder", "svm"]
    }
  ],
  "points": 10
}
```

### Creating Questions

#### Step 1: Choose a Template

Copy the appropriate template from `templates/`:

```powershell
# For Module 1.1 multiple choice questions
Copy-Item assessments/templates/multiple_choice.json assessments/module-1/1.1-questions.json
```

#### Step 2: Follow the Schema

All questions must validate against `schema.json`. Key requirements:

- **Unique IDs**: Use format `mX.Y_q###` (e.g., `m1.1_q001`)
- **Difficulty**: Must be `easy`, `medium`, or `hard`
- **Points**: Positive integer
- **Type-specific fields**:
  - Multiple choice: `options`, `correct_answer`, `explanation`
  - Coding exercise: `starter_code`, `test_cases`, `explanation`
  - Conceptual: `rubric`, `explanation`

#### Step 3: Write High-Quality Questions

**Best Practices**:

- **Clear and specific**: Avoid ambiguous wording
- **Semiconductor-relevant**: Use real-world manufacturing contexts
- **Progressive difficulty**: Start with easier questions
- **Comprehensive explanations**: Help students learn from mistakes
- **Actionable hints**: Guide without giving away the answer

**Example of Good vs. Bad**:

❌ **Bad**: "What is NumPy?"  
✅ **Good**: "When processing 10,000 wafer measurement readings, which NumPy feature provides the greatest performance advantage over standard Python lists?"

#### Step 4: Validate Questions

Use the assessment system's validation:

```python
from modules.foundation.assessment_system import ModuleAssessment

# Load and validate question bank
assessment = ModuleAssessment.load_from_json("assessments/module-1/1.1-questions.json")
print(f"Loaded {len(assessment.questions)} valid questions")
```

### Using Assessments

#### For Students

Take an assessment:

```python
from modules.foundation.assessment_system import ModuleAssessment

# Load assessment
assessment = ModuleAssessment.load_from_json("assessments/module-1/1.1-questions.json")

# Take the assessment interactively
result = assessment.take_assessment()

# View results
print(f"Score: {result.score}/{result.total_points}")
print(f"Percentage: {result.percentage:.1f}%")

# Review incorrect answers
for q_id, feedback in result.feedback.items():
    if not feedback['correct']:
        print(f"\nQuestion {q_id}: {feedback['explanation']}")
```

#### For Instructors

Grade submissions:

```python
# Load assessment
assessment = ModuleAssessment.load_from_json("assessments/module-1/1.1-questions.json")

# Grade student responses
student_answers = {
    "m1.1_q001": 0,  # Answer to multiple choice
    "m1.1_q002": "def calculate_yield(df): ...",  # Code submission
    "m1.1_q003": "My approach would be..."  # Conceptual answer
}

result = assessment.grade(student_answers)
print(f"Grade: {result.percentage:.1f}%")
```

### Question Bank Goals

Target question counts per module:

| Module | Multiple Choice | Coding Exercise | Conceptual | Total |
|--------|----------------|-----------------|------------|-------|
| 1.1 | 10 | 5 | 5 | 20 |
| 1.2 | 12 | 8 | 5 | 25 |
| 2.1 | 12 | 8 | 5 | 25 |
| 2.2 | 10 | 7 | 3 | 20 |
| 2.3 | 12 | 8 | 5 | 25 |
| 3.1 | 15 | 10 | 5 | 30 |
| 3.2 | 15 | 10 | 5 | 30 |
| 4.1 | 12 | 8 | 5 | 25 |
| 4.2 | 12 | 8 | 5 | 25 |
| 5.1 | 15 | 10 | 5 | 30 |
| 5.2 | 12 | 8 | 5 | 25 |
| 6.1 | 15 | 10 | 5 | 30 |
| 6.2 | 15 | 10 | 5 | 30 |
| 7.1 | 12 | 8 | 5 | 25 |
| 7.2 | 12 | 8 | 5 | 25 |
| 8.1 | 12 | 8 | 5 | 25 |
| 8.2 | 12 | 8 | 5 | 25 |
| 9.1 | 15 | 10 | 5 | 30 |
| 9.2 | 12 | 8 | 5 | 25 |
| 9.3 | 10 | 7 | 3 | 20 |
| 10.x | 10 per sub | 10 per sub | 5 per sub | ~40 |
| 11.1 | 12 | 8 | 5 | 25 |

**Total**: ~600 questions across all modules

### Development Status

- ✅ Schema defined
- ✅ Templates created
- ✅ Module directories created
- 🔄 Question banks in development (ETA: 6 weeks)
- 🔄 Automated grading engine in development
- 🔄 Progress tracking in development

### Contributing

When adding questions:

1. Follow the schema strictly
2. Use templates as starting points
3. Test questions with real students if possible
4. Include semiconductor-specific context
5. Provide comprehensive explanations
6. Add progressive hints for difficult questions

### Support

For questions about the assessment system:

- Review `modules/foundation/assessment_system.py` for the framework
- Check `docs/architecture/002-assessment-system-design.md` for design details
- See templates for examples
- Validate against schema before committing

---

**Status**: Infrastructure complete, question bank development in progress
