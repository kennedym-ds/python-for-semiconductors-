# Module 8 - Cutting Edge: Generative AI and Large Language Models

This module explores cutting-edge AI technologies applied to semiconductor manufacturing, focusing on generative AI, large language models (LLMs), and their practical applications in fab environments.

## Module Structure

### 8.2 - LLMs for Manufacturing NLP

A comprehensive implementation of natural language processing pipelines for manufacturing text analysis, including maintenance logs, shift reports, and equipment alerts.

#### 📂 Files

- **`8.2-llm-manufacturing-nlp.ipynb`** - Interactive analysis notebook demonstrating text classification and summarization
- **`8.2-llm-manufacturing-nlp-fundamentals.md`** - Deep-dive into NLP theory and manufacturing applications
- **`8.2-llm-manufacturing-nlp-pipeline.py`** - Production-ready CLI pipeline with dual backends
- **`8.2-llm-manufacturing-nlp-quick-ref.md`** - Commands, troubleshooting, and integration guide
- **`test_llm_nlp_pipeline.py`** - Comprehensive test suite

#### 🚀 Quick Start

```bash
# Train a severity classification model
python 8.2-llm-manufacturing-nlp-pipeline.py train \
    --task classification \
    --backend classical \
    --target-type severity \
    --save severity_model.joblib

# Evaluate the model
python 8.2-llm-manufacturing-nlp-pipeline.py evaluate \
    --model-path severity_model.joblib

# Make predictions
python 8.2-llm-manufacturing-nlp-pipeline.py predict \
    --model-path severity_model.joblib \
    --input-json '{"text":"Pump P-101 emergency shutdown due to overheating"}'
```

#### 🎯 Key Features

**Text Classification:**
- **Severity prediction**: Low/Medium/High urgency levels
- **Tool area classification**: Wet Bench, Lithography, Etch, Deposition, Metrology
- **Manufacturing-specific metrics**: PWS (Prediction Within Spec), Estimated Loss

**Text Summarization:**
- **Shift report summarization**: Extract key operational insights
- **Maintenance log condensation**: Highlight critical information
- **Actionable summary generation**: Enable quick decision-making

**Dual Backend Support:**
- **Classical**: TF-IDF + scikit-learn (always available, fast, interpretable)
- **Transformers**: BERT/RoBERTa models (optional, higher accuracy when available)

**Production Features:**
- **JSON API**: Ready for system integration
- **Model persistence**: Save/load trained models
- **Graceful fallbacks**: Classical methods when transformers unavailable
- **Privacy-first**: On-premise deployment, no external APIs required

#### 📊 Synthetic Data

The module includes realistic synthetic data generators:

**Maintenance Logs (800 samples):**
```
"Reactor R-204 emergency shutdown triggered due to overheating exceeded safety limits"  # High severity
"Pump P-101 showing unusual vibration patterns during night shift"                        # Medium severity  
"CVD-301 completed routine maintenance check successfully"                                # Low severity
```

**Shift Reports (300 samples):**
```
Day Shift Report - Lithography Area

All lithography tools operating within normal parameters. Completed 12 wafer lots 
successfully. Tool A experienced minor alarm, resolved by technician. Overall yield: 96.2%
```

#### 🔧 Architecture

```python
class ManufacturingNLPPipeline:
    def fit(X: pd.DataFrame, y: Optional[pd.Series] = None) -> Self
    def predict(X: pd.DataFrame) -> np.ndarray  
    def evaluate(X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, float]
    def save(path: Path) -> None
    @staticmethod
    def load(path: Path) -> 'ManufacturingNLPPipeline'
```

**CLI Commands:**
- `train`: Train models with configurable parameters
- `evaluate`: Assess model performance on test data  
- `predict`: Make predictions on new text inputs

#### 📈 Performance Benchmarks

- **Runtime**: < 45 seconds for full test suite
- **Accuracy**: > 90% for classification tasks
- **Model size**: < 1MB per model
- **Inference speed**: < 1 second per prediction

#### 🏭 Manufacturing Integration

**Use Cases:**
- **Real-time alert classification**: Route maintenance requests by urgency
- **Automated shift report processing**: Extract KPIs and issues
- **Knowledge base search**: Find relevant procedures and solutions
- **Predictive maintenance**: Identify equipment degradation patterns

**Integration Points:**
- **MES Systems**: Manufacturing Execution System integration
- **CMMS**: Computerized Maintenance Management Systems
- **Alert platforms**: Automated notification systems
- **Quality systems**: Non-conformance report processing

#### 💰 Business Impact

Based on typical semiconductor fab operations:

- **Annual labor savings**: ~$400K+ from automated text processing
- **Incident cost reduction**: ~$190K+ from faster response times
- **Payback period**: < 3 months for implementation
- **Daily time savings**: 8+ hours of technician time

#### 🔒 Privacy & Security

- **On-premise deployment**: No external API dependencies
- **Data anonymization**: Equipment ID and personnel masking
- **Configurable sensitivity**: Adjustable confidence thresholds
- **Audit trails**: Full prediction logging and traceability

#### 🧪 Testing

Run the comprehensive test suite:

```bash
python -m pytest test_llm_nlp_pipeline.py -v
```

**Test Coverage:**
- CLI functionality for all commands
- Model save/load round-trip testing
- Both classical and transformers backends
- Error handling and edge cases
- Performance benchmarks

#### 📚 Learning Path

1. **Start with fundamentals**: Read `8.2-llm-manufacturing-nlp-fundamentals.md`
2. **Hands-on exploration**: Work through `8.2-llm-manufacturing-nlp.ipynb`
3. **Production deployment**: Use `8.2-llm-manufacturing-nlp-pipeline.py`
4. **Reference guide**: Keep `8.2-llm-manufacturing-nlp-quick-ref.md` handy

#### 🚀 Advanced Topics

**Model Optimization:**
- Hyperparameter tuning for manufacturing datasets
- Custom preprocessing for equipment-specific terminology
- Ensemble methods combining multiple approaches
- Active learning for continuous model improvement

**Production Deployment:**
- Docker containerization for scalable deployment
- API server implementation with FastAPI
- Database integration for persistent storage
- Monitoring and alerting for model performance

**Domain Adaptation:**
- Fine-tuning transformer models on manufacturing text
- Custom vocabulary expansion for semiconductor terms
- Transfer learning from general NLP to manufacturing domain
- Multi-task learning for related NLP tasks

## Dependencies

**Required (Classical Backend):**
- scikit-learn >= 1.7.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- joblib >= 1.3.0

**Optional (Transformers Backend):**
- transformers >= 4.20.0
- torch >= 2.0.0

**Development:**
- pytest >= 7.0.0
- jupyter >= 1.0.0

## Installation

```bash
# Install advanced tier dependencies
python env_setup.py --tier advanced

# Or install manually
pip install -r requirements-advanced.txt
```

## Future Modules

This module sets the foundation for advanced AI applications in manufacturing:

- **8.3**: Computer Vision for Defect Detection
- **8.4**: Reinforcement Learning for Process Optimization  
- **8.5**: MLOps for Manufacturing AI Systems

The NLP capabilities developed here integrate with vision and optimization modules to create comprehensive AI-driven manufacturing solutions.

---

*This module demonstrates how modern NLP and LLM technologies can be practically applied to semiconductor manufacturing environments, balancing accuracy, efficiency, and operational requirements.*