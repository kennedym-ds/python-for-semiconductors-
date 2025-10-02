# Assessment Validation Tools

This directory contains validation tools for the assessment system.

## validate_all.py

Unified validation script that checks all assessment question JSON files against the schema.

### Usage

**Validate all modules:**

```powershell
python validation/validate_all.py
```

**Validate specific module(s):**

```powershell
# Single module
python validation/validate_all.py --module 3

# Multiple modules
python validation/validate_all.py --module 4 9 11
```

**Show warnings in addition to errors:**

```powershell
python validation/validate_all.py --verbose
```

### What It Checks

- **JSON Syntax**: Valid JSON format
- **Required Fields**: All required top-level and question fields present
- **Question Types**: Must be `multiple_choice`, `coding_exercise`, or `conceptual`
- **Difficulty Levels**: Must be `easy`, `medium`, or `hard`
- **Type-Specific Requirements**:
  - Multiple choice: `options` and `correct_answer` present and valid
  - Coding exercise: `starter_code` and `test_cases` present
  - Conceptual: `rubric` present
- **ID Uniqueness**: No duplicate IDs within a file

### Example Output

```
================================================================================
MODULE 3 VALIDATION
================================================================================

[PASS] 3.1-questions.json
   30 questions total:
   - 14 multiple choice
   - 11 coding exercises
   - 5 conceptual

[PASS] 3.2-questions.json
   30 questions total:
   - 15 multiple choice
   - 9 coding exercises
   - 6 conceptual

--------------------------------------------------------------------------------
[PASSED] - Module 3
Total: 60 questions across 2 file(s)
================================================================================
```

### Comprehensive Testing

For more comprehensive testing including cross-file ID uniqueness checks:

```powershell
pytest tests/test_assessment_system.py -v
```

The pytest test suite (`tests/test_assessment_system.py`) provides:

- Cross-file question ID uniqueness validation
- Schema compliance verification
- Module structure integrity checks
- Complete coverage of all 685 questions across 11 modules
- Detailed reporting and statistics

### When to Use Each Tool

**Use `validate_all.py` when:**

- Quickly checking a single module during development
- Validating before committing changes
- Getting immediate feedback without pytest overhead
- Running from the command line

**Use `pytest tests/test_assessment_system.py` when:**

- Running CI/CD pipeline
- Comprehensive validation of entire framework
- Checking cross-file constraints
- Generating detailed test reports

## Development Notes

This validation script is a lightweight alternative to the comprehensive pytest suite. It provides quick feedback for contributors working on individual modules while maintaining the same validation logic for core requirements.

For production releases, always run the full pytest suite to ensure complete validation.
