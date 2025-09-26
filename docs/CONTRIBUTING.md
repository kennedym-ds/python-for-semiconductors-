# Contributing to Python for Semiconductors

Thank you for your interest in contributing to the Python for Semiconductors learning series! This document provides guidelines for contributing to this educational platform.

## 🎯 Project Vision

This project aims to be the industry-leading educational platform for semiconductor engineers learning ML/AI techniques. We welcome contributions that enhance the learning experience, improve code quality, or expand the curriculum.

## 🚀 Getting Started

### Development Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/kennedym-ds/python-for-semiconductors-.git
   cd python-for-semiconductors-
   ```

2. **Set up Python environment**
   ```bash
   # Using the tiered dependency system
   python env_setup.py --tier intermediate
   
   # Or manually with pip
   pip install -r requirements-intermediate.txt
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   python -m pytest modules/foundation/test_*.py -v
   flake8 modules/foundation/ --max-line-length=120
   black --check modules/foundation/
   ```

## 📋 Contribution Types

### 1. Educational Content
- **New modules**: Follow the 4-content pattern (notebook, fundamentals, pipeline, quick-ref)
- **Assessment questions**: Add to existing assessment systems
- **Interactive widgets**: Enhance learning with ipywidgets
- **Case studies**: Real semiconductor industry examples

### 2. Code Improvements
- **Bug fixes**: Error corrections and edge case handling
- **Performance optimization**: Speed and memory improvements
- **Documentation**: API docs, tutorials, examples
- **Testing**: Unit tests, integration tests, validation scripts

### 3. Platform Features
- **Gamification**: New achievements, progress tracking
- **Visualization**: Enhanced charts, 3D models, dashboards
- **Assessment tools**: New question types, rubrics
- **Infrastructure**: CI/CD, deployment, monitoring

## 🏗️ Development Guidelines

### Code Style and Standards

#### Python Code
- **PEP 8 compliance**: Use `flake8` with max line length 120
- **Type hints**: Required for all new functions and classes
- **Docstrings**: Google-style docstrings for all public APIs
- **Error handling**: Comprehensive try/catch with meaningful messages

```python
def process_wafer_data(
    data: pd.DataFrame, 
    quality_threshold: float = 0.8
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Process wafer manufacturing data for ML analysis.
    
    Args:
        data: Raw wafer measurement data
        quality_threshold: Minimum quality score for inclusion
        
    Returns:
        Tuple of processed data and quality metrics
        
    Raises:
        ValueError: If data format is invalid
        DataQualityError: If quality threshold not met
    """
```

#### Module Content Pattern (CRITICAL)
Every new module must contain exactly these 4 files:
```
X.Y-topic-analysis.ipynb          # Interactive learning notebook
X.Y-topic-fundamentals.md         # Theory and deep-dive documentation  
X.Y-topic-pipeline.py             # Production-ready CLI script
X.Y-topic-quick-ref.md            # Summary and cheat sheet
```

#### Pipeline Script Requirements
All `*-pipeline.py` files must follow this pattern:
```python
class SomethingPipeline:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]: ...
    def save(self, path: Path) -> None: ...
    @staticmethod
    def load(path: Path) -> 'SomethingPipeline': ...

# CLI with subcommands: train, evaluate, predict
def build_parser() -> argparse.ArgumentParser: ...
```

### Testing Requirements

#### Test Coverage
- **Minimum 80%** coverage for new code
- **Pipeline tests**: CLI functionality validation
- **Widget tests**: Interactive component verification
- **Assessment tests**: Question and scoring accuracy

#### Test Structure
```python
# modules/foundation/test_module_pipeline.py
import pytest
from module_pipeline import ModulePipeline

class TestModulePipeline:
    def test_pipeline_creation(self):
        """Test pipeline initialization."""
        pipeline = ModulePipeline()
        assert pipeline is not None
    
    def test_fit_predict_workflow(self):
        """Test complete ML workflow."""
        # Test implementation
        pass
        
    def test_cli_interface(self):
        """Test command line interface."""
        # Test CLI commands
        pass
```

### Documentation Standards

#### API Documentation
- **Automated generation**: Use the api-documentation.py tool
- **Examples**: Include usage examples in docstrings
- **Type information**: Complete parameter and return type docs

#### User Guides
- **Step-by-step tutorials**: Clear learning progression
- **Screenshots**: Visual guides for complex procedures
- **Troubleshooting**: Common issues and solutions
- **Industry context**: Real semiconductor applications

## 🔄 Contribution Workflow

### 1. Issue Creation
Before starting work, create or find an issue:
- **Bug reports**: Use the bug report template
- **Feature requests**: Use the feature request template
- **Educational content**: Use the content proposal template

### 2. Branch Strategy
```bash
# Create feature branch
git checkout -b feature/assessment-enhancement
git checkout -b bugfix/widget-loading-error
git checkout -b content/module-8-gan-augmentation
```

### 3. Development Process
1. **Write tests first** (TDD approach recommended)
2. **Implement feature** following coding standards
3. **Run full test suite** and ensure no regressions
4. **Update documentation** including docstrings and user guides
5. **Test interactive components** in Jupyter environment

### 4. Pull Request Process

#### PR Checklist
- [ ] Tests pass (`python -m pytest`)
- [ ] Code style compliant (`flake8`, `black`)
- [ ] Documentation updated
- [ ] Interactive widgets tested in Jupyter
- [ ] Assessment questions validated by domain expert
- [ ] No security vulnerabilities introduced

#### PR Description Template
```markdown
## Changes Made
- Brief description of changes
- Reference to issue number

## Testing Performed
- Unit tests added/updated
- Integration testing completed
- Manual testing in Jupyter environment

## Documentation Updates
- API documentation updated
- User guide sections modified
- Examples added/updated

## Screenshots (if applicable)
- Before/after images for UI changes
- Widget functionality demonstrations
```

### 5. Review Process
- **Automated checks**: CI/CD pipeline validation
- **Code review**: At least one maintainer approval
- **Educational review**: Content reviewed by domain expert
- **Testing verification**: All tests must pass

## 🎓 Educational Content Guidelines

### Learning Objectives
Each module should have:
- **Clear objectives**: What students will learn
- **Prerequisites**: Required background knowledge
- **Assessments**: Knowledge validation methods  
- **Industry applications**: Real-world relevance

### Semiconductor Context
All examples should relate to:
- **Manufacturing processes**: Wafer fabrication, testing
- **Quality control**: Defect detection, yield optimization
- **Process optimization**: Parameter tuning, DOE
- **Equipment monitoring**: Predictive maintenance

### Interactive Elements
- **Parameter tuning widgets**: Let students experiment
- **3D visualizations**: Wafer maps, defect patterns
- **Real-time feedback**: Immediate assessment results
- **Progress tracking**: Gamification elements

## 🏆 Recognition

### Contributor Levels
- **Community Contributor**: Bug fixes, small improvements
- **Content Creator**: New modules, comprehensive examples
- **Core Maintainer**: Regular contributions, code reviews
- **Domain Expert**: Technical accuracy validation

### Attribution
- Contributors listed in module credits
- GitHub contributor recognition
- Professional LinkedIn recommendations
- Conference presentation opportunities

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Code Reviews**: Pull request feedback
- **Documentation**: Inline comments and guides

### Expert Support
- **Semiconductor domain**: Process engineering questions
- **Machine learning**: Algorithm and implementation help
- **Educational design**: Learning theory and assessment
- **Platform development**: Architecture and infrastructure

## 🔒 Security and Privacy

### Code Security
- **No secrets in code**: Use environment variables
- **Input validation**: Sanitize all user inputs
- **Dependency management**: Regular security updates
- **Code scanning**: Automated vulnerability detection

### Student Privacy
- **Data minimization**: Collect only necessary information
- **Anonymization**: Remove personal identifiers where possible
- **Consent**: Clear opt-in for progress tracking
- **GDPR/FERPA compliance**: Educational data protection

## 📜 License and Legal

This project is licensed under the MIT License. By contributing, you agree:
- Your contributions will be licensed under MIT
- You have the right to submit the contribution
- Your contribution is your original work

## 🙏 Acknowledgments

We appreciate all contributors who help make this the best educational platform for semiconductor engineers entering the ML/AI field. Every contribution, from typo fixes to major features, makes a difference!

---

## Quick Start Checklist for New Contributors

- [ ] Read this contributing guide
- [ ] Set up development environment
- [ ] Run tests to verify setup
- [ ] Choose an issue to work on
- [ ] Create feature branch
- [ ] Make changes following guidelines
- [ ] Add/update tests
- [ ] Update documentation
- [ ] Submit pull request
- [ ] Respond to review feedback

Welcome to the team! 🎉