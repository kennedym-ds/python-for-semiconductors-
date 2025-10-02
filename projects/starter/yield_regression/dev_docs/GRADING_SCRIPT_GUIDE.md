# Yield Regression Grading Script

**Created**: September 30, 2025  
**Status**: Complete - Ready for Testing  
**File**: `evaluate_submission.py`

---

## Overview

Automated grading script for yield regression tutorial submissions. Executes student notebooks and validates completion of all 4 exercises with a 100-point rubric.

---

## Features

### Automated Assessment
- **Notebook Execution**: Runs student code in isolated environment
- **Pattern Matching**: Detects implementation vs TODO comments
- **Metric Validation**: Checks for correct regression metrics (R², RMSE, MAE, PWS)
- **Output Validation**: Verifies plots and results

### Scoring System
- **Exercise 1** (20 pts): Data Generation & Exploration
  - Data generation (5 pts)
  - Distribution analysis (5 pts)
  - Correlation analysis (5 pts)
  - Visualization quality (5 pts)

- **Exercise 2** (30 pts): Model Training & Comparison
  - Data splitting (5 pts)
  - Multiple model training (10 pts)
  - Performance evaluation (10 pts)
  - Model comparison visualization (5 pts)

- **Exercise 3** (25 pts): Manufacturing Metrics & Residual Analysis
  - Residual analysis (10 pts)
  - PWS calculation (5 pts)
  - Estimated Loss calculation (5 pts)
  - Manufacturing interpretation (5 pts)

- **Exercise 4** (15 pts): Model Deployment
  - Model saving (5 pts)
  - Model loading (5 pts)
  - CLI demonstration (5 pts)

- **Code Quality** (10 pts): Overall assessment
  - Documentation (3 pts)
  - Code style (3 pts)
  - Error handling (2 pts)
  - Best practices (2 pts)

### Output Formats
- **Console**: Human-readable results with emoji feedback
- **JSON**: LMS-compatible format with detailed breakdowns

---

## Usage

### Basic Grading
```bash
# Grade with console output
python evaluate_submission.py --notebook student_submission.ipynb

# Verbose mode (detailed feedback)
python evaluate_submission.py --notebook submission.ipynb --verbose
```

### Save Results to JSON
```bash
# Single submission
python evaluate_submission.py \
    --notebook student_submission.ipynb \
    --output-json grades/student123.json \
    --student-name "Alice Johnson"

# Batch grading
for notebook in submissions/*.ipynb; do
    python evaluate_submission.py \
        --notebook "$notebook" \
        --output-json "grades/$(basename $notebook .ipynb)_grade.json"
done
```

### Advanced Options
```bash
# Custom timeout (default: 600s)
python evaluate_submission.py \
    --notebook submission.ipynb \
    --timeout 1200 \
    --verbose

# Check grading script help
python evaluate_submission.py --help
```

---

## Grading Logic

### Exercise 1: Data Generation & Exploration

**What's Checked**:
- ✅ `generate_yield_process()` function call
- ✅ Statistical analysis (`.describe()`, `.mean()`, `.std()`)
- ✅ Correlation matrix with heatmap
- ✅ Scatter plots or histograms

**Patterns Detected**:
```python
# Data generation
r'generate_yield_process'

# Distribution analysis
r'describe\(\)|\.mean\(\)|\.std\(\)'

# Correlation
r'\.corr\(\)|correlation.*matrix|heatmap'

# Visualization
r'scatter|hist|plot.*yield'
```

**Scoring Example**:
- Data generated: 3 pts
- TODOs removed: 2 pts
- Stats computed: 3 pts
- Heatmap shown: 2 pts
- **Total**: 10/20 pts (if partial implementation)

---

### Exercise 2: Model Training & Comparison

**What's Checked**:
- ✅ `train_test_split()` with `random_state`
- ✅ Multiple models trained (Linear, Ridge, Lasso, ElasticNet, RF)
- ✅ Regression metrics calculated (R², RMSE, MAE, PWS)
- ✅ Results organized in DataFrame
- ✅ Comparison visualization

**Model Detection**:
```python
# Counts occurrences of model types
patterns = [
    r'linear',
    r'ridge',
    r'lasso',
    r'elasticnet',
    r'rf|random.*forest'
]

# Scoring:
# 5 models = 5 pts
# 3 models = 3 pts
# 1 model = 1 pt
```

**Metrics Detection**:
```python
# Checks for metric usage
metrics = {
    'R²': r'R2|r2_score|r_squared',
    'RMSE': r'RMSE|root.*mean.*squared',
    'MAE': r'MAE|mean.*absolute',
    'PWS': r'PWS|prediction.*within.*spec'
}

# Scoring:
# All 4 metrics = 5 pts
# 2+ metrics = 3 pts
```

---

### Exercise 3: Manufacturing Metrics & Residual Analysis

**What's Checked**:
- ✅ Residual calculation (`y_test - predictions`)
- ✅ Residual plots (histogram, scatter, Q-Q plot)
- ✅ PWS calculation (predictions within 60-100% spec)
- ✅ Estimated Loss (errors beyond ±2% tolerance)
- ✅ Feature importance analysis

**Residual Plot Detection**:
```python
plot_types = {
    'histogram': r'hist.*residual|residual.*hist',
    'scatter': r'scatter.*residual|residual.*scatter',
    'Q-Q plot': r'probplot|qq.*plot|q-q'
}

# Scoring:
# All 3 plots = 5 pts
# 1+ plots = 3 pts
```

**Manufacturing Metrics**:
```python
# PWS: Prediction Within Spec
r'PWS|prediction.*within.*spec|>=.*60.*<=.*100'

# Estimated Loss: Cost beyond tolerance
r'estimated.*loss|loss.*=.*sum|tolerance'

# Scoring: 3 pts each if implemented
```

---

### Exercise 4: Model Deployment

**What's Checked**:
- ✅ Model saved with `.save()` method
- ✅ Model loaded with `.load()` method
- ✅ CLI commands demonstrated (train, evaluate, predict)

**Patterns**:
```python
# Saving
r'pipeline\.save|\.joblib|save.*model'

# Loading
r'\.load\(|YieldRegressionPipeline\.load'

# CLI demo
r'python.*yield_regression_pipeline|CLI|command.*line'

# CLI commands
cli_commands = ['train', 'evaluate', 'predict']
# 2+ commands = 2 pts
# 1+ command = 1 pt
```

---

### Code Quality Assessment

**Documentation** (3 pts):
- Ratio of cells with meaningful comments
- Formula: `documented_cells / total_cells * 3.0`

**Code Style** (3 pts):
- Line length violations (> 120 chars)
- Inconsistent indentation (mixing tabs/spaces)
- Formula: `max(0, 3.0 - (violations / max_violations) * 3.0)`

**Error Handling** (2 pts):
- Cells with try/except, None checks, or assertions
- Formula: `min(2.0, (error_handling_cells / total_cells) * 4.0)`

**Best Practices** (2 pts):
- Descriptive variable names (≥4 chars): 1 pt
- Constants defined (UPPER_CASE): 1 pt

---

## JSON Output Format

```json
{
  "student_name": "Alice Johnson",
  "notebook_path": "student_submission.ipynb",
  "total_score": 87.5,
  "max_score": 100,
  "percentage": 87.5,
  "letter_grade": "B+",
  "exercise_scores": [
    {
      "exercise_id": "Exercise 1",
      "earned_points": 18.0,
      "max_points": 20,
      "percentage": 90.0,
      "feedback": [
        "✅ Data generation code implemented",
        "✅ All TODOs completed",
        "✅ Correlation heatmap visualized"
      ],
      "errors": [],
      "warnings": []
    },
    {
      "exercise_id": "Exercise 2",
      "earned_points": 27.0,
      "max_points": 30,
      "percentage": 90.0,
      "feedback": [
        "✅ Data splitting implemented",
        "✅ Random state set for reproducibility",
        "✅ Trained 5 models (excellent)",
        "✅ All key metrics calculated: R², RMSE, MAE, PWS"
      ],
      "errors": [],
      "warnings": []
    },
    {
      "exercise_id": "Exercise 3",
      "earned_points": 22.0,
      "max_points": 25,
      "percentage": 88.0,
      "feedback": [
        "✅ Residual calculation implemented",
        "✅ Complete residual analysis: histogram, scatter, Q-Q plot",
        "✅ PWS calculation implemented"
      ],
      "errors": [],
      "warnings": [
        "⚠️ Feature importance analysis not found"
      ]
    },
    {
      "exercise_id": "Exercise 4",
      "earned_points": 13.0,
      "max_points": 15,
      "percentage": 86.7,
      "feedback": [
        "✅ Model saving code implemented",
        "✅ Using pipeline.save() method",
        "✅ Model loading code implemented",
        "✅ CLI demonstration included"
      ],
      "errors": [],
      "warnings": []
    }
  ],
  "code_quality_score": 7.5,
  "execution_successful": true,
  "execution_errors": [],
  "feedback_summary": [
    "🌟 Excellent work! You've demonstrated strong understanding of yield regression analysis.",
    "Documentation: 80.0% of cells have comments (2.4/3 pts)",
    "Code style: 3 violations (2.7/3 pts)",
    "Error handling: 2 cells with error handling (1.6/2 pts)",
    "✅ Descriptive variable names used"
  ]
}
```

---

## Differences from Classification Grading

| Aspect | Wafer Defect (Classification) | Yield Regression |
|--------|------------------------------|------------------|
| **Metrics** | ROC-AUC, Precision, Recall, F1 | R², RMSE, MAE, PWS |
| **Visualization** | ROC curve, Precision-Recall curve | Residual plots, Q-Q plot, Actual vs Predicted |
| **Manufacturing** | Confusion matrix, FP/FN costs | PWS, Estimated Loss, Spec limits |
| **Exercise 3** | Threshold optimization | Residual analysis + manufacturing metrics |
| **Models** | Logistic, SVM, RF, GradientBoosting | Linear, Ridge, Lasso, ElasticNet, RF |

---

## Testing Checklist

### Unit Testing
- [ ] Test on tutorial notebook (should fail - has TODOs)
- [ ] Test on solution notebook (should score 95-100/100)
- [ ] Test with incomplete notebook (should give partial credit)
- [ ] Test with syntax errors (should report execution failure)
- [ ] Test JSON output format

### Validation
- [ ] Verify all exercise patterns match actual notebook structure
- [ ] Check metric thresholds are reasonable (R² > 0.30)
- [ ] Confirm feedback messages are helpful
- [ ] Test batch grading workflow

### Integration
- [ ] Test with different Python environments
- [ ] Verify PYTHONPATH handling for imports
- [ ] Check timeout handling (600s default)
- [ ] Test with various notebook formats

---

## Expected Scores

### Tutorial Notebook
- **Expected**: 20-40/100
- **Reason**: Contains TODO comments, incomplete implementations
- **Use Case**: Baseline for students to improve

### Solution Notebook
- **Expected**: 95-100/100
- **Reason**: Complete implementations, all TODOs removed, comprehensive analysis
- **Use Case**: Reference for full credit

### Common Student Scores
- **90-100**: Exceptional (A/A-)
- **80-89**: Proficient (B+/B/B-)
- **70-79**: Developing (C+/C/C-)
- **60-69**: Needs improvement (D)
- **Below 60**: Incomplete (F)

---

## Troubleshooting

### "Notebook not found" Error
```bash
# Check file path
ls -la yield_regression_tutorial.ipynb

# Use absolute path
python evaluate_submission.py --notebook "$(pwd)/student_submission.ipynb"
```

### Execution Timeout
```bash
# Increase timeout for slow models
python evaluate_submission.py --notebook submission.ipynb --timeout 1200
```

### Import Errors
```bash
# Ensure notebook directory is in PYTHONPATH
# Script handles this automatically, but verify:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python evaluate_submission.py --notebook submission.ipynb
```

### Dependencies
```bash
# Install required packages
pip install nbformat nbconvert jupyter

# Or use project requirements
pip install -r requirements-basic.txt
```

---

## Future Enhancements

### Planned Improvements
- [ ] Metric threshold validation (warn if R² < 0.30)
- [ ] Cell output analysis (check plot generation)
- [ ] Plagiarism detection (code similarity)
- [ ] Auto-feedback generation (specific improvement tips)
- [ ] Performance benchmarking (execution time tracking)

### Optional Features
- [ ] HTML report generation
- [ ] Side-by-side comparison with solution
- [ ] Interactive grading dashboard
- [ ] Email notifications for batch grading

---

## Maintenance Notes

### When to Update

**New Exercise Added**:
1. Add `grade_exercise_N()` method
2. Update `EXPECTED_RANGES` if needed
3. Add to `self.result.exercise_scores` in `grade()`
4. Update rubric in docstring

**Metrics Changed**:
1. Update regex patterns in `grade_exercise_2()` or `grade_exercise_3()`
2. Adjust scoring thresholds
3. Update documentation

**Notebook Structure Changed**:
1. Review all `find_cell_by_pattern()` calls
2. Update regex patterns to match new structure
3. Test on updated tutorial/solution notebooks

---

## Success Criteria

✅ **Grading script is ready when**:
1. Syntax validated (`python -m py_compile evaluate_submission.py`)
2. Runs on tutorial notebook without errors
3. Produces expected score range (20-40/100 for tutorial)
4. Runs on solution notebook (95-100/100 expected)
5. JSON output validates against schema
6. Batch grading workflow tested
7. Documentation complete

---

## Next Steps

1. **Test on tutorial notebook**:
   ```bash
   python evaluate_submission.py --notebook yield_regression_tutorial.ipynb --verbose
   ```

2. **Test on solution notebook** (when complete):
   ```bash
   python evaluate_submission.py --notebook yield_regression_solution.ipynb --verbose --output-json test_grade.json
   ```

3. **Create test suite**:
   - Unit tests for each `grade_exercise_N()` method
   - Integration tests for full grading workflow
   - Validation tests for JSON schema

4. **Document in README**:
   - Add "For Instructors" section
   - Include grading workflow examples
   - Link to this documentation

---

## Related Files

- `evaluate_submission.py` - Main grading script (THIS FILE)
- `yield_regression_tutorial.ipynb` - Student template (has TODOs)
- `yield_regression_solution.ipynb` - Reference solution (no TODOs)
- `yield_regression_pipeline.py` - Production pipeline (imported by notebooks)
- `test_yield_regression_pipeline.py` - Unit tests for pipeline

---

**Status**: ✅ Complete and ready for testing
**Estimated Testing Time**: 30 minutes
**Next Priority**: Test on tutorial notebook
