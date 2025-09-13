# Semiconductor ML Project Template

This directory contains a reusable template for creating standardized semiconductor ML projects. The template follows industry best practices and provides a solid foundation for any machine learning project in semiconductor manufacturing.

## Quick Start

Use the project architecture pipeline to create a new project from this template:

```bash
# Navigate to the project root
cd /path/to/python-for-semiconductors

# Create a new semiconductor ML project
python modules/project-dev/module-10/10.1-project-architecture-pipeline.py scaffold \
  --name my_wafer_classifier \
  --type classification \
  --output ./projects/

# Validate the created project
python modules/project-dev/module-10/10.1-project-architecture-pipeline.py validate \
  --project-path ./projects/my_wafer_classifier/
```

## Template Features

### 🏗️ **Standardized Structure**
- **`src/`** - Source code organized by functionality (data, features, models, visualization)
- **`data/`** - Data storage with clear separation (raw, processed, external)  
- **`notebooks/`** - Jupyter notebooks organized by purpose (exploratory, production)
- **`tests/`** - Unit and integration tests for all components
- **`configs/`** - Configuration management with YAML and environment variables
- **`docs/`** - Documentation and API references

### 🔧 **Configuration Management**
- **Environment Variables** - Production settings that vary by deployment (.env)
- **YAML Configuration** - Application settings that are version-controlled
- **Code Constants** - Hard-coded values that should never change
- **Secrets Management** - Secure handling of credentials and API keys

### 🏭 **Semiconductor Manufacturing Focus**
- **Process Parameters** - Standard templates for temperature, pressure, flow rates
- **Manufacturing Metrics** - PWS (Prediction Within Spec), Estimated Loss, Yield Rate
- **Quality Standards** - Specification compliance and tolerance validation
- **Equipment Integration** - Patterns for sensor data and equipment monitoring

### 🛠️ **Production Ready**
- **CLI Interface** - Consistent train/evaluate/predict commands with JSON outputs
- **Docker Support** - Container deployment with docker-compose orchestration
- **Testing Framework** - Comprehensive test coverage with pytest
- **Logging** - Structured JSON logging for production monitoring
- **CI/CD Ready** - GitHub Actions workflow templates

## Project Types Supported

| Type | Use Case | Key Metrics | Sample Applications |
|------|----------|-------------|-------------------|
| **Classification** | Defect detection, yield classification | ROC-AUC, PR-AUC, PWS, False Positive Rate | Wafer defect detection, pass/fail prediction |
| **Regression** | Process optimization, yield prediction | MAE, RMSE, R², PWS, Estimated Loss | Process parameter optimization, yield forecasting |
| **Time Series** | Equipment monitoring, drift detection | MAE, RMSE, MAPE, Anomaly Detection Rate | Sensor monitoring, predictive maintenance |
| **Computer Vision** | Visual inspection, die analysis | mIoU, Pixel Accuracy, Defect Detection Rate | Wafer surface inspection, die-level defect detection |

## Template Customization

### Adding New Project Types

To add a new project type to the template system:

1. **Update Project Types Dictionary** in `10.1-project-architecture-pipeline.py`:
```python
SEMICONDUCTOR_PROJECT_TYPES['new_type'] = {
    'description': 'Your project type description',
    'sample_datasets': ['dataset1', 'dataset2'],
    'key_metrics': ['Metric1', 'Metric2', 'PWS']
}
```

2. **Customize Generated Files** by modifying the generation methods:
   - `_generate_readme()` - Custom README content
   - `_generate_requirements()` - Type-specific dependencies
   - `_generate_config()` - Type-specific configuration
   - `_generate_pipeline_template()` - Custom pipeline implementation

3. **Add Validation Rules** in `_validate_*()` methods for type-specific validation

### Modifying File Templates

All generated files are created by template methods in the pipeline:

- **README.md** - `_generate_readme()`
- **requirements.txt** - `_generate_requirements()`
- **config.yaml** - `_generate_config()`
- **pipeline.py** - `_generate_pipeline_template()`
- **Dockerfile** - `_create_docker_files()`
- **Notebooks** - `_create_notebook_templates()`

## Integration with Existing Projects

### Retrofitting Existing Projects

To apply this template structure to an existing project:

1. **Backup your existing project**
2. **Run structure validation** to identify gaps:
```bash
python 10.1-project-architecture-pipeline.py validate --project-path /path/to/existing/project
```
3. **Create missing directories and files** based on validation suggestions
4. **Migrate existing code** to the standardized structure
5. **Update imports and paths** to match new organization

### Migration Checklist

- [ ] Move data files to `data/raw/` and `data/processed/`
- [ ] Organize source code into `src/` packages
- [ ] Extract configuration to `configs/config.yaml` and `.env`
- [ ] Add `__init__.py` files to all Python packages
- [ ] Create standardized CLI interface
- [ ] Add comprehensive tests
- [ ] Update documentation

## Best Practices Implemented

### 🔒 **Security**
- Secrets managed through environment variables
- No credentials committed to version control
- `.env.template` provides secure examples
- Gitignore configured for sensitive files

### 📊 **Data Management**
- Clear separation of raw and processed data
- Data versioning patterns with semantic versioning
- Consistent path resolution across notebooks and scripts
- External data sources properly organized

### 🧪 **Testing**
- Unit tests for all core functionality
- Integration tests for complete workflows
- CLI interface testing with subprocess
- Performance regression tests for critical paths

### 📝 **Documentation**
- Comprehensive README with setup instructions
- API documentation for all modules
- Configuration reference and examples
- Manufacturing context and domain knowledge

### 🔄 **Reproducibility**
- Fixed random seeds (RANDOM_SEED = 42)
- Version-controlled configuration
- Dependency management with requirements files
- Docker containers for consistent environments

## Validation and Compliance

The template includes comprehensive validation to ensure compliance with best practices:

### Structure Validation (40 points)
- Required directories: `src/`, `data/`, `tests/`, `configs/`
- Optional directories: `notebooks/`, `docs/`, `scripts/`, `logs/`, `models/`

### File Validation (20 points)  
- Essential files: `README.md`, `requirements.txt`, `.gitignore`
- Configuration files: `configs/config.yaml`, `.env.template`

### Naming Conventions (10 points)
- Python files use snake_case
- No spaces in directory names
- Consistent package structure

### Configuration Management (10 points)
- Environment variable templates
- YAML configuration files
- Proper secrets management

### Code Quality (20 points)
- No hardcoded paths
- Proper import structure
- `__init__.py` files in all packages
- No wildcard imports

## Manufacturing Integration Examples

### Process Parameter Monitoring
```python
# Example: Temperature monitoring pipeline
PROCESS_LIMITS = {
    'temperature': {'min': 400, 'max': 500, 'tolerance': 5},
    'pressure': {'min': 1.0, 'max': 5.0, 'tolerance': 0.1},
    'flow_rate': {'min': 50, 'max': 200, 'tolerance': 10}
}

def validate_process_window(measurements):
    """Validate measurements against process windows."""
    for param, limits in PROCESS_LIMITS.items():
        if param in measurements:
            value = measurements[param]
            if not (limits['min'] <= value <= limits['max']):
                return False, f"{param} out of range: {value}"
    return True, "All parameters within specification"
```

### Yield Prediction Integration
```python
# Example: Yield prediction with manufacturing metrics
def compute_manufacturing_metrics(y_true, y_pred, spec_limits):
    """Compute semiconductor-specific metrics."""
    # Prediction Within Spec (PWS)
    pws = np.mean((y_pred >= spec_limits['low']) &
                  (y_pred <= spec_limits['high']))

    # Estimated financial loss from prediction errors
    tolerance = spec_limits.get('tolerance', 2.0)
    cost_per_unit = spec_limits.get('cost_per_unit', 1.0)
    loss_components = np.maximum(0, np.abs(y_true - y_pred) - tolerance)
    estimated_loss = np.sum(loss_components) * cost_per_unit

    return {
        'PWS': pws,
        'Estimated_Loss': estimated_loss,
        'Yield_Rate': np.mean(y_pred >= spec_limits['low'])
    }
```

## Troubleshooting

### Common Issues and Solutions

**Q: Project validation fails with "Missing required directory" errors**
A: Run the scaffold command to create the complete structure, or manually create the missing directories.

**Q: CLI commands return JSON parsing errors**
A: Ensure you're using Python 3.8+ and all dependencies are installed. Check that the pipeline script is executable.

**Q: Notebooks can't find data files**
A: Use the path resolution pattern: `DATA_DIR = Path('../../../data').resolve()` from notebook location.

**Q: Docker build fails**
A: Verify Docker is installed and running. Check that requirements.txt contains all necessary dependencies.

**Q: Tests fail with import errors**
A: Ensure all `__init__.py` files exist in src packages and PYTHONPATH includes the project root.

## Contributing

To contribute improvements to this template:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/template-improvement`
3. **Make your changes** to the template generation code
4. **Add tests** for any new functionality
5. **Update documentation** as needed
6. **Submit a pull request** with a clear description of improvements

### Template Development Guidelines

- Maintain backward compatibility with existing projects
- Add comprehensive tests for new features
- Document all configuration options
- Follow semiconductor industry standards
- Include examples for complex features

## License and Usage

This template is part of the Python for Semiconductors learning series and is designed for educational and commercial use in semiconductor manufacturing environments. Please refer to the main repository license for usage terms.

## Support and Resources

- **Documentation**: See `modules/project-dev/module-10/` for comprehensive guides
- **Examples**: Check the `10.1-project-architecture.ipynb` notebook for interactive examples
- **Quick Reference**: Use `10.1-project-architecture-quick-ref.md` for command reference
- **Issues**: Report problems through the main repository issue tracker

---

**Happy Building! 🚀**

*This template helps you create production-ready semiconductor ML projects that follow industry best practices and maintain consistency across teams and deployments.*
