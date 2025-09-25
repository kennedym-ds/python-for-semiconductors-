# Troubleshooting Guide

This guide helps resolve common issues with the Python for Semiconductors learning platform.

## 🚀 Quick Diagnostics

### Run System Check
```bash
# Check Python environment
python --version  # Should be 3.10+

# Verify key dependencies
python -c "import numpy, pandas, sklearn, matplotlib; print('Core dependencies OK')"

# Test interactive widgets
python -c "import ipywidgets; print('Widgets available')"

# Check assessment system
python modules/foundation/assessment_system.py dashboard --student-id test-user
```

## 📦 Installation Issues

### Problem: Dependencies Not Installing
```bash
# Error: Could not find a version that satisfies the requirement
```

**Solutions:**
1. **Update pip and setuptools**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Use tiered installation**
   ```bash
   # Start with basic tier
   python env_setup.py --tier basic
   
   # Upgrade incrementally
   python env_setup.py --tier intermediate
   ```

3. **Platform-specific issues**
   ```bash
   # Skip problematic packages
   pip install -r requirements-basic.txt --skip-pywin32
   
   # Use conda for complex dependencies
   conda install numpy pandas scikit-learn matplotlib
   pip install -r requirements-basic.txt --no-deps
   ```

### Problem: Import Errors
```python
ImportError: No module named 'sklearn'
```

**Solutions:**
1. **Verify installation**
   ```bash
   pip list | grep scikit-learn
   python -c "import sklearn; print(sklearn.__version__)"
   ```

2. **Reinstall if needed**
   ```bash
   pip uninstall scikit-learn
   pip install scikit-learn>=1.3.0
   ```

3. **Check Python path**
   ```python
   import sys
   print(sys.path)
   # Ensure your package locations are included
   ```

## 🎮 Interactive Widgets Issues

### Problem: Widgets Not Displaying
```
Widget output not showing in Jupyter
```

**Solutions:**
1. **Enable Jupyter widgets**
   ```bash
   pip install ipywidgets
   jupyter nbextension enable --py widgetsnbextension
   jupyter labextension install @jupyter-widgets/jupyterlab-manager
   ```

2. **Restart Jupyter**
   ```bash
   # Kill all Jupyter processes
   jupyter lab stop
   
   # Start fresh
   jupyter lab
   ```

3. **Check browser compatibility**
   - Use Chrome, Firefox, or Safari
   - Disable ad blockers temporarily
   - Clear browser cache

### Problem: Widget Interactions Not Working
```
Sliders/buttons not responding
```

**Solutions:**
1. **Check JavaScript console**
   - F12 → Console tab
   - Look for error messages

2. **Verify widget communication**
   ```python
   import ipywidgets as widgets
   from IPython.display import display
   
   # Test basic widget
   slider = widgets.IntSlider(value=50)
   display(slider)
   ```

3. **Reset widget state**
   ```python
   # In Jupyter cell
   from IPython.display import clear_output
   clear_output(wait=True)
   ```

## 📊 Visualization Problems

### Problem: Plots Not Showing
```
Matplotlib/Plotly plots empty or not displaying
```

**Solutions:**
1. **Check backend configuration**
   ```python
   import matplotlib
   print(matplotlib.get_backend())
   
   # Set appropriate backend
   matplotlib.use('TkAgg')  # For desktop
   matplotlib.use('webagg')  # For web/Jupyter
   ```

2. **Enable inline plotting**
   ```python
   %matplotlib inline
   import matplotlib.pyplot as plt
   
   # Test plot
   plt.plot([1,2,3], [1,4,2])
   plt.show()
   ```

3. **Plotly configuration**
   ```python
   import plotly.io as pio
   pio.renderers.default = "notebook"  # For Jupyter
   pio.renderers.default = "browser"   # For standalone
   ```

### Problem: 3D Visualizations Slow/Unresponsive
```
Wafer visualization widget performance issues
```

**Solutions:**
1. **Reduce data points**
   ```python
   # In wafer visualization widget
   # Sample data for better performance
   sampled_data = data.sample(n=500, random_state=42)
   ```

2. **Lower visual quality**
   ```python
   # Reduce marker size and opacity
   marker=dict(size=2, opacity=0.5)
   ```

3. **Use static fallbacks**
   ```python
   # Check if plotly available, use matplotlib backup
   try:
       import plotly.graph_objects as go
       # Use interactive plot
   except ImportError:
       import matplotlib.pyplot as plt
       # Use static plot
   ```

## 🧪 Assessment System Issues

### Problem: Assessments Not Running
```bash
python assessment_system.py assess --module module-3
# Error: Module assessment failed
```

**Solutions:**
1. **Check module path**
   ```bash
   # Verify you're in correct directory
   pwd
   ls modules/foundation/
   
   # Run from project root
   cd /path/to/python-for-semiconductors-
   python modules/foundation/assessment_system.py assess --module module-3
   ```

2. **Debug mode**
   ```bash
   # Enable detailed error messages
   python -v modules/foundation/assessment_system.py assess --module module-3
   ```

3. **Check dependencies**
   ```python
   # Verify sklearn available
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_val_score
   ```

### Problem: Progress Not Saving
```
Dashboard shows no progress data
```

**Solutions:**
1. **Check data directory permissions**
   ```bash
   ls -la assessment_data/
   # Should be writable
   chmod 755 assessment_data/
   ```

2. **Verify JSON files**
   ```bash
   # Check for valid JSON
   python -m json.tool assessment_data/profiles/test-user.json
   ```

3. **Manual data reset**
   ```bash
   # Backup and reset if corrupted
   mv assessment_data assessment_data_backup
   mkdir assessment_data
   ```

## 🎯 Gamification System Issues

### Problem: Achievements Not Unlocking
```
Completed activities but no badges earned
```

**Solutions:**
1. **Check unlock conditions**
   ```python
   from modules.foundation.gamification_system import GamificationSystem
   
   gamification = GamificationSystem()
   profile = gamification.get_or_create_student('your-id')
   
   # Debug achievement checking
   engine = gamification.achievement_engine
   stats = engine._calculate_student_stats(profile)
   print(stats)
   ```

2. **Trigger manual check**
   ```bash
   python modules/foundation/gamification_system.py track \
     --student-id your-id --activity assessment --score 95
   ```

3. **Reset achievement state**
   ```bash
   # Remove student profile to start fresh
   rm gamification_data/profiles/your-id.json
   ```

## 🗄️ Data and Dataset Issues

### Problem: Dataset Loading Errors
```
FileNotFoundError: datasets/secom/secom.data not found
```

**Solutions:**
1. **Check dataset path resolution**
   ```python
   from pathlib import Path
   
   # From notebook location
   DATA_DIR = Path('../../../datasets').resolve()
   print(f"Looking for data in: {DATA_DIR}")
   print(f"Exists: {DATA_DIR.exists()}")
   ```

2. **Download missing datasets**
   ```bash
   cd datasets
   python download_semiconductor_datasets.py --dataset secom
   ```

3. **Verify dataset structure**
   ```bash
   ls -la datasets/
   # Should see: secom/, wm811k/, etc.
   
   ls -la datasets/secom/
   # Should see: secom.data, secom.names
   ```

### Problem: Data Quality Issues
```
Empty dataframe or missing columns
```

**Solutions:**
1. **Inspect raw data**
   ```python
   import pandas as pd
   
   # Check file contents
   with open('datasets/secom/secom.data', 'r') as f:
       print(f.read()[:500])  # First 500 chars
   
   # Load with explicit parameters
   df = pd.read_csv('datasets/secom/secom.data', 
                    sep='\s+', na_values=['?', 'NaN'])
   print(df.info())
   ```

2. **Handle missing values**
   ```python
   # Check for missing data patterns
   missing_percent = (df.isnull().sum() / len(df)) * 100
   print(missing_percent.sort_values(ascending=False).head())
   ```

## 🌐 Documentation and API Issues

### Problem: API Documentation Not Generating
```bash
python docs/api-documentation.py generate
# Error: Module analysis failed
```

**Solutions:**
1. **Check source paths**
   ```bash
   # Verify modules directory exists
   ls -la modules/
   
   # Run with explicit paths
   python docs/api-documentation.py generate \
     --source modules/ --output docs/api-docs/
   ```

2. **Debug module loading**
   ```python
   import sys
   sys.path.append('modules/')
   
   # Test individual module import
   from foundation.assessment_system import ModuleAssessment
   ```

3. **Skip problem modules**
   ```python
   # Modify api-documentation.py to skip failing modules
   # Add try/except around module analysis
   ```

### Problem: MkDocs Build Fails
```bash
mkdocs build
# Error: Configuration file malformed
```

**Solutions:**
1. **Validate YAML syntax**
   ```bash
   python -c "import yaml; yaml.safe_load(open('mkdocs.yml'))"
   ```

2. **Check file paths**
   ```bash
   # Verify all referenced files exist
   ls -la docs-site/docs/
   ```

3. **Test with minimal config**
   ```yaml
   # mkdocs.yml
   site_name: Test Site
   nav:
     - Home: index.md
   ```

## 🔧 Development Environment Issues

### Problem: Pre-commit Hooks Failing
```bash
git commit -m "Update"
# Error: pre-commit hook failed
```

**Solutions:**
1. **Run hooks manually**
   ```bash
   pre-commit run --all-files
   pre-commit run black --all-files
   pre-commit run flake8 --all-files
   ```

2. **Fix common issues**
   ```bash
   # Auto-fix formatting
   black modules/foundation/
   
   # Check specific linting errors
   flake8 modules/foundation/ --show-source
   ```

3. **Skip hooks temporarily**
   ```bash
   # For emergency commits only
   git commit -m "Fix" --no-verify
   ```

### Problem: Tests Failing
```bash
python -m pytest modules/foundation/test_*.py
# Multiple test failures
```

**Solutions:**
1. **Run specific test files**
   ```bash
   # Test one file at a time
   python -m pytest modules/foundation/test_assessment_system.py -v
   ```

2. **Check test dependencies**
   ```bash
   # Ensure test requirements installed
   pip install pytest pytest-cov
   ```

3. **Debug test environment**
   ```python
   # In test file
   import sys
   print(f"Python path: {sys.path}")
   print(f"Working directory: {os.getcwd()}")
   ```

## 🚨 Performance Issues

### Problem: Slow Notebook Execution
```
Jupyter cells taking too long to run
```

**Solutions:**
1. **Monitor resource usage**
   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
   ```

2. **Optimize data loading**
   ```python
   # Use chunking for large datasets
   chunk_size = 10000
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       # Process chunk
       pass
   ```

3. **Clear memory periodically**
   ```python
   import gc
   
   # Delete large variables
   del large_dataframe
   gc.collect()
   ```

## 📞 Getting Additional Help

### Create Detailed Bug Report
When filing issues, include:

1. **Environment information**
   ```bash
   python --version
   pip list | grep -E "(pandas|numpy|sklearn|matplotlib|ipywidgets)"
   jupyter --version
   ```

2. **Exact error message**
   ```bash
   # Copy complete stack trace
   ```

3. **Reproduction steps**
   ```bash
   # Minimal example that fails
   ```

4. **Expected vs actual behavior**

### Community Resources
- **GitHub Issues**: Report bugs and feature requests
- **GitHub Discussions**: Ask questions and share ideas
- **Documentation**: Check existing guides and tutorials

### Emergency Workarounds
If you need to continue learning while troubleshooting:

1. **Use static alternatives**: Replace interactive widgets with static plots
2. **Skip problematic modules**: Continue with working modules
3. **Use cloud environments**: Try Google Colab or similar platforms
4. **Minimal installation**: Use only core dependencies

Remember: Most issues have been encountered before and have solutions! Don't hesitate to ask for help.