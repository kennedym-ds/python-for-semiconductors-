# VS Code Tasks Guide

This document explains all available VS Code tasks for the Python for Semiconductors project.

## How to Run Tasks

1. **Command Palette**: Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type: `Tasks: Run Task`
3. Select the task you want to run

Alternatively, use the keyboard shortcut: `Ctrl+Shift+B` (Windows/Linux) or `Cmd+Shift+B` (Mac)

---

## Available Tasks

### 🔧 Environment Setup Tasks

#### **Env: Create Virtual Environment**

Creates a new Python virtual environment in the `.venv` directory.

```bash
python -m venv .venv
```

**When to use**: First time setup or when you need a fresh environment.

#### **Env: Activate Virtual Environment (Info)**

Displays instructions for activating the virtual environment in your terminal.

- **PowerShell**: `.venv\Scripts\Activate.ps1`
- **CMD**: `.venv\Scripts\activate.bat`
- **Mac/Linux**: `source .venv/bin/activate`

#### **Env: Setup Basic**

Installs basic tier dependencies (core Python libraries).

```bash
python env_setup.py --tier basic
```

**Includes**: numpy, pandas, matplotlib, scikit-learn, jupyter

#### **Env: Setup Intermediate**

Installs intermediate tier dependencies (adds time series and additional ML tools).

```bash
python env_setup.py --tier intermediate
```

**Includes**: Basic tier + statsmodels, pmdarima, prophet

#### **Env: Setup Advanced**

Installs advanced tier dependencies (adds deep learning).

```bash
python env_setup.py --tier advanced
```

**Includes**: Intermediate tier + tensorflow, keras, torch, opencv

#### **Env: Setup Full**

Installs ALL dependencies including MLOps, visualization, and advanced tools.

```bash
python env_setup.py --tier full
```

**Includes**: Everything (MLflow, Streamlit, Plotly, Dash, etc.)

#### **Env: Recreate Full (Force)**

Forcefully recreates the full environment from scratch.

```bash
python env_setup.py --tier full --force
```

**When to use**: When your environment is corrupted or you need a clean slate.

#### **Env: Install Streamlit Requirements**

Installs only the requirements needed for the Streamlit assessment app.

```bash
pip install -r requirements-streamlit.txt
```

**When to use**: Quick setup if you only want to run the assessment application.

#### **Env: Install All Requirements**

Directly installs all requirements from requirements-full.txt.

```bash
pip install -r requirements-full.txt
```

#### **Env: Upgrade pip**

Upgrades pip to the latest version.

```bash
pip install --upgrade pip
```

---

### 📊 Streamlit Assessment App Tasks

#### **Streamlit: Run Assessment App**

Launches the interactive assessment application in your browser.

```bash
streamlit run assessments/assessment_app.py
```

**Features**:

- Take module assessments interactively
- Track your progress over time
- View performance analytics with charts
- SQLite database for persistent results

**Access**: Opens automatically at `http://localhost:8501`

#### **Streamlit: Run Assessment App (with auto-reload)**

Runs the app with automatic reloading when you save changes to the code.

```bash
streamlit run assessments/assessment_app.py --server.runOnSave=true
```

**When to use**: Development mode when making changes to the app.

#### **Streamlit: Clear Cache and Run**

Clears the assessment database and then launches the app with a fresh start.

**When to use**: Testing or when you want to reset all progress data.

---

### 🗄️ Database Management Tasks

#### **Database: Clear Assessment Results**

Deletes the SQLite database containing all assessment attempts and user data.

**PowerShell**:

```powershell
Remove-Item 'assessments/assessment_results.db'
```

**⚠️ Warning**: This permanently deletes all assessment history!

#### **Database: Backup Assessment Results**

Creates a timestamped backup of the assessment database.

**Output**: `assessments/assessment_results_backup_YYYYMMDD_HHMMSS.db`

**When to use**: Before clearing database or performing major updates.

---

### ✅ Assessment Validation Tasks

#### **Assessment: Validate All**

Validates all assessment JSON files across all modules for schema compliance.

```bash
python assessments/validation/validate_all.py --verbose
```

**Checks**:

- JSON syntax validity
- Required fields present
- Correct data types
- Question structure integrity

#### **Assessment: Validate Specific Module**

Validates assessments for a single module (interactive selection).

```bash
python assessments/validation/validate_all.py --module <module-id> --verbose
```

**When to use**: After creating or editing assessment files for a specific module.

---

### 🚀 Pipeline Script Tasks

#### **Pipeline: Run Module 3.1 Regression**

Example task for running the regression pipeline from Module 3.

```bash
python modules/foundation/module-3/3.1-regression-pipeline.py train \
  --data datasets/secom/secom.data \
  --labels datasets/secom/secom_labels.data \
  --output models/regression_model.pkl
```

**When to use**: Training production-ready regression models on semiconductor data.

---

### 🧪 Testing Tasks

All existing test tasks for the grading system and notebook validation are preserved.

#### **Test Grading Script**

Runs pytest on the grading script test suite.

```bash
pytest projects/starter/wafer_defect_classifier/test_evaluate_submission.py -v
```

---

## Quick Start Workflows

### First Time Setup

1. **Create Virtual Environment**: Run `Env: Create Virtual Environment`
2. **Activate Environment**: Follow instructions from `Env: Activate Virtual Environment (Info)`
3. **Install Dependencies**: Run `Env: Setup Full` or `Env: Install Streamlit Requirements`
4. **Launch Assessment App**: Run `Streamlit: Run Assessment App`

### Daily Development Workflow

1. **Activate Environment**: Activate `.venv` in your terminal
2. **Run App**: Use `Streamlit: Run Assessment App (with auto-reload)`
3. **Validate Changes**: Run `Assessment: Validate All` after editing assessment files

### Clean Slate Workflow

1. **Backup Database**: Run `Database: Backup Assessment Results`
2. **Recreate Environment**: Run `Env: Recreate Full (Force)`
3. **Clear Data**: Run `Database: Clear Assessment Results`
4. **Fresh Start**: Run `Streamlit: Clear Cache and Run`

---

## Tips and Best Practices

### Environment Management

- **Always activate your virtual environment** before running Python commands manually
- Use tiered installation (`basic` → `intermediate` → `advanced` → `full`) to save disk space if you don't need all features
- Run `Env: Upgrade pip` first if you encounter installation issues

### Streamlit App

- The app runs in the background - stop it with `Ctrl+C` in the terminal panel
- Database file is created automatically on first run
- Each user's progress is tracked by their unique user ID
- Export your database file before major updates

### Assessment Validation

- Always validate after creating new assessment files
- Use `--verbose` flag to see detailed validation results
- Schema validation catches common mistakes early

### Task Customization

You can modify tasks in `.vscode/tasks.json` to:

- Change default parameters
- Add new pipeline scripts
- Create custom workflows
- Modify keyboard shortcuts

---

## Troubleshooting

### Task Not Found

**Issue**: Task doesn't appear in the task list  
**Solution**: Reload VS Code window (`Ctrl+Shift+P` → "Developer: Reload Window")

### Permission Errors (Windows)

**Issue**: PowerShell script execution blocked  
**Solution**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Streamlit Won't Start

**Issue**: "streamlit: command not found"  
**Solution**:

1. Ensure virtual environment is activated
2. Run `Env: Install Streamlit Requirements`
3. Verify installation: `pip list | grep streamlit`

### Database Locked Error

**Issue**: SQLite database is locked  
**Solution**:

1. Close all Streamlit app instances
2. Run `Database: Backup Assessment Results`
3. Run `Database: Clear Assessment Results`
4. Restart the app

---

## Additional Resources

- **Streamlit Documentation**: [STREAMLIT_APP_README.md](../assessments/STREAMLIT_APP_README.md)
- **Quick Start Guide**: [STREAMLIT_QUICKSTART.md](../STREAMLIT_QUICKSTART.md)
- **Environment Setup**: [docs/setup-guide.md](../docs/setup-guide.md)
- **Assessment Framework**: [docs/assessment-framework.md](../docs/assessment-framework.md)

---

## Contributing

When adding new tasks:

1. Use descriptive labels with category prefixes (e.g., `Env:`, `Streamlit:`, `Database:`)
2. Add comments to organize task sections
3. Include `problemMatcher` when appropriate
4. Set `isBackground: true` for long-running processes
5. Document the task in this README

---

**Last Updated**: January 2025  
**Version**: 1.0.0
