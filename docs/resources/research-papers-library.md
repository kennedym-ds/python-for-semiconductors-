# Research Papers Library: AI and Machine Learning for Semiconductor Manufacturing

**Last Updated**: January 2025  
**Total Papers**: 15 curated foundational and cutting-edge papers  
**Topics**: Defect detection, yield prediction, process control, lithography optimization, MLOps for semiconductor manufacturing

---

## Overview

This curated library contains essential research papers that bridge AI/ML techniques with semiconductor manufacturing applications. Papers are organized by topic and include practical relevance notes for learners.

---

## 1. Defect Detection & Classification

### 1.1 Wafer Defect Root Cause Analysis (2025)
**Title**: Wafer Defect Root Cause Analysis with Partial Trajectory Regression  
**Authors**: Kohei Miyaguchi, Masao Joko, Rebekah Sheraw, Tsuyoshi Idé  
**Link**: [arXiv:2507.20357](https://arxiv.org/abs/2507.20357)  
**Published**: ASMC 2025 (Advanced Semiconductor Manufacturing Conference)

**Key Contributions**:
- Introduces `proc2vec` and `route2vec` representation learning methods
- Compares counterfactual outcomes from partial process trajectories
- Enables root cause analysis by comparing defective vs non-defective wafer paths
- Validated on real wafer history data from NY CREATES fab

**Relevance to Course**:
- **Module 3**: Applies regression techniques to trajectory data
- **Module 6**: Uses representation learning (embedding techniques)
- **Module 9**: Real-time defect attribution in production

**Implementation Concepts**:
```python
# Conceptual approach from paper
class WaferDefectRCA:
    def __init__(self):
        self.proc2vec = ProcessEmbedding()  # Embeds individual processes
        self.route2vec = RouteEmbedding()   # Embeds process sequences

    def compare_trajectories(self, defective_wafer, reference_wafers):
        # Compare partial trajectories to identify problematic steps
        return self.counterfactual_analysis(defective_wafer, reference_wafers)
```

**Key Takeaways**:
- Process sequence matters more than individual process parameters
- Representation learning enables comparison of complex manufacturing paths
- Counterfactual reasoning helps isolate root causes

---

### 1.2 Few-Shot Defect Detection with CLIP (2025)
**Title**: SEM-CLIP: Precise Few-Shot Learning for Nanoscale Defect Detection  
**Authors**: Qian Jin, Yuqi Jiang, Xudong Lu, et al.  
**Link**: [arXiv:2502.14884](https://arxiv.org/abs/2502.14884)  
**Published**: ICCAD 2024

**Key Contributions**:
- Adapts CLIP (Contrastive Language-Image Pre-training) for SEM images
- Achieves high accuracy with only 5-10 labeled examples per defect type
- Handles complex background patterns in scanning electron microscope images
- Addresses the challenge of rare defect types

**Relevance to Course**:
- **Module 7**: Few-shot learning and transfer learning
- **Module 8**: Vision-language models for inspection
- **Module 11**: Edge deployment of lightweight models

**Implementation Concepts**:
```python
# CLIP-based approach for SEM images
from transformers import CLIPModel, CLIPProcessor

class SEMDefectDetector:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def few_shot_classify(self, sem_image, defect_types, examples_per_type=5):
        # Use CLIP's zero/few-shot capabilities
        # Compare image embeddings with text descriptions
        return self.model.compare_embeddings(sem_image, defect_types)
```

**Key Takeaways**:
- Pre-trained vision-language models reduce labeling requirements
- Few-shot learning is critical for rare defect types
- Domain adaptation from natural images to SEM images is feasible

---

### 1.3 Continual Learning for Defect Inspection (2024)
**Title**: An Evaluation of Continual Learning for Advanced Node Semiconductor Defect Inspection  
**Authors**: Amit Prasad, Bappaditya Dey, Victor Blanco, Sandip Halder  
**Link**: [arXiv:2407.12724](https://arxiv.org/abs/2407.12724)  
**Published**: ECML-PKDD 2024 Industry Track

**Key Contributions**:
- Evaluates continual learning strategies for evolving defect patterns
- Addresses catastrophic forgetting in production environments
- Compares memory replay, elastic weight consolidation, and progressive neural networks
- Benchmarked on real semiconductor manufacturing data

**Relevance to Course**:
- **Module 7**: Advanced deep learning techniques
- **Module 9**: Production ML systems that adapt over time
- **Module 10**: MLOps for model retraining strategies

**Implementation Concepts**:
```python
# Continual learning approach
class ContinualDefectClassifier:
    def __init__(self):
        self.model = DefectCNN()
        self.memory_buffer = []  # Store exemplars from previous tasks

    def learn_new_defect_type(self, new_data):
        # Mix new data with memory buffer to prevent forgetting
        combined_data = new_data + self.sample_from_memory()
        self.model.fit(combined_data)
        self.update_memory(new_data)
```

**Key Takeaways**:
- New defect types emerge constantly in semiconductor fabs
- Continual learning prevents retraining from scratch
- Memory-based approaches outperform regularization methods

---

## 2. Statistical Process Control & Yield Prediction

### 2.1 Proactive SPC Using Time Series Forecasting (2025)
**Title**: Proactive Statistical Process Control Using AI: A Time Series Forecasting Approach  
**Authors**: Mohammad Iqbal Rasul Seeam, Victor S. Sheng  
**Link**: [arXiv:2509.16431](https://arxiv.org/abs/2509.16431)  
**Published**: September 2025

**Key Contributions**:
- Predicts process drift before control limits are breached
- Uses LSTM and Transformer models for multivariate time series
- Reduces scrap by enabling proactive interventions
- Validated on real fab sensor data (temperature, pressure, gas flow)

**Relevance to Course**:
- **Module 5**: Time series analysis and forecasting
- **Module 9**: Real-time inference for process control
- **Module 10**: Production deployment of predictive maintenance

**Implementation Concepts**:
```python
# Proactive SPC with LSTM
class ProactiveSPC:
    def __init__(self, lookback_window=50):
        self.model = LSTMForecaster(lookback=lookback_window)
        self.control_limits = {"UCL": 3.0, "LCL": -3.0}

    def predict_drift(self, sensor_data):
        # Forecast next N steps
        forecast = self.model.predict(sensor_data, steps_ahead=10)

        # Check if forecast breaches control limits
        if any(forecast > self.control_limits["UCL"]):
            return "ALERT: Process drift predicted in 10 steps"
        return "NORMAL"
```

**Key Takeaways**:
- Reactive SPC catches issues too late
- Time series forecasting enables proactive intervention
- Multi-sensor fusion improves prediction accuracy

---

### 2.2 Transfer Learning for Minimum Operating Voltage (2025)
**Title**: Transfer Learning for Minimum Operating Voltage Prediction in Advanced Technology Nodes  
**Authors**: Yuxuan Yin, Rebecca Chen, Boxun Xu, Chen He, Peng Li  
**Link**: [arXiv:2509.00035](https://arxiv.org/abs/2509.00035)  
**Published**: September 2025

**Key Contributions**:
- Predicts chip Vmin (minimum operating voltage) at advanced nodes (5nm, 3nm)
- Leverages transfer learning from legacy technology nodes (28nm, 14nm)
- Uses "silicon odometer" sensing (chip degradation indicators)
- Critical for energy efficiency and reliability prediction

**Relevance to Course**:
- **Module 3**: Regression for voltage prediction
- **Module 4**: Ensemble methods for robust predictions
- **Module 7**: Transfer learning across technology nodes

**Implementation Concepts**:
```python
# Transfer learning for Vmin prediction
class VminPredictor:
    def __init__(self, source_model_28nm):
        self.base_model = source_model_28nm  # Pre-trained on 28nm data
        self.fine_tune_layers = [...]  # Last 2 layers

    def adapt_to_advanced_node(self, limited_5nm_data):
        # Freeze early layers, fine-tune final layers
        self.base_model.freeze_layers(except_last=2)
        self.base_model.fit(limited_5nm_data)
        return self.base_model
```

**Key Takeaways**:
- Limited training data at advanced nodes (expensive to collect)
- Transfer learning from mature nodes reduces data requirements
- Silicon odometer features capture aging effects

---

### 2.3 Tool-to-Tool Matching Analysis (2025)
**Title**: Tool-to-Tool Matching Analysis Based Difference Score Computation  
**Authors**: Sameera Bharadwaja H., Siddhrath Jandial, et al.  
**Link**: [arXiv:2507.10564](https://arxiv.org/abs/2507.10564)  
**Published**: July 2025

**Key Contributions**:
- Addresses chamber-to-chamber variation in process equipment
- Proposes difference score metrics for tool matching
- Reduces systematic variation between identical tools
- Enables balanced tool utilization and yield optimization

**Relevance to Course**:
- **Module 2**: Data quality and variance analysis
- **Module 3**: Regression for tool matching
- **Module 9**: Real-time tool assignment decisions

**Implementation Concepts**:
```python
# Tool matching analysis
class ToolMatcher:
    def __init__(self, reference_tool):
        self.reference = reference_tool

    def compute_difference_score(self, target_tool):
        # Compare output distributions between tools
        ref_outputs = self.reference.get_historical_outputs()
        target_outputs = target_tool.get_historical_outputs()

        # Statistical distance metrics
        return {
            "mean_diff": np.mean(target_outputs - ref_outputs),
            "variance_ratio": np.var(target_outputs) / np.var(ref_outputs),
            "distribution_distance": wasserstein_distance(ref_outputs, target_outputs)
        }
```

**Key Takeaways**:
- Tool-to-tool variation is a major source of yield loss
- Matching tools enables predictable scheduling
- Statistical process control requires tool-specific models

---

## 3. Optical Lithography & Patterning

### 3.1 Physics-Informed Neural Networks for Lithography (2025)
**Title**: Physics-Informed Neural Networks For Semiconductor Film Deposition: A Review  
**Authors**: Tao Han, Zahra Taheri, Hyunwoong Ko  
**Link**: [arXiv:2507.10983](https://arxiv.org/abs/2507.10983)  
**Published**: IDETC-CIE 2025

**Key Contributions**:
- Integrates physical laws (diffusion, reaction kinetics) into neural networks
- Reduces training data requirements by embedding domain knowledge
- Accelerates lithography simulation (1000x faster than finite element methods)
- Enables inverse design (desired pattern → optimal process parameters)

**Relevance to Course**:
- **Module 6**: Physics-informed neural networks
- **Module 7**: Advanced deep learning architectures
- **Module 9**: Fast simulation for real-time optimization

**Implementation Concepts**:
```python
# Physics-informed NN for lithography
class PhysicsInformedLithography(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = DeepNeuralNet()

    def forward(self, x):
        prediction = self.network(x)
        return prediction

    def compute_loss(self, x, y_true):
        y_pred = self.forward(x)

        # Data loss
        data_loss = mse_loss(y_pred, y_true)

        # Physics loss (e.g., diffusion equation)
        physics_loss = self.diffusion_residual(y_pred, x)

        return data_loss + lambda_physics * physics_loss
```

**Key Takeaways**:
- Physics-based constraints improve generalization
- Reduces reliance on expensive simulation data
- Enables real-time what-if analysis in production

---

### 3.2 Differentiable Resist Simulator (2025)
**Title**: TorchResist: Open-Source Differentiable Resist Simulator  
**Authors**: Zixiao Wang, Jieya Zhou, Su Zheng, et al.  
**Link**: [arXiv:2502.06838](https://arxiv.org/abs/2502.06838)  
**Published**: SPIE Advanced Lithography 2025

**Key Contributions**:
- End-to-end differentiable optical lithography simulation
- Enables gradient-based optimization of mask patterns
- Open-source PyTorch implementation
- 100x faster than traditional simulators

**Relevance to Course**:
- **Module 6**: Differentiable programming
- **Module 8**: Generative models for inverse design
- **Module 9**: Fast optimization loops

**Implementation Concepts**:
```python
# Differentiable lithography simulation
import torch
import torch.nn as nn

class DifferentiableResistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.optical_model = OpticalPropagation()
        self.resist_model = ChemicalReaction()

    def forward(self, mask_pattern):
        # Differentiable forward pass
        aerial_image = self.optical_model(mask_pattern)
        resist_profile = self.resist_model(aerial_image)
        return resist_profile

    def optimize_mask(self, target_profile):
        mask = nn.Parameter(torch.randn(size))
        optimizer = torch.optim.Adam([mask], lr=0.01)

        for epoch in range(1000):
            predicted = self.forward(mask)
            loss = mse_loss(predicted, target_profile)
            loss.backward()
            optimizer.step()

        return mask
```

**Key Takeaways**:
- Differentiable simulators enable inverse design
- Gradient-based optimization finds optimal process parameters
- Critical for advanced node manufacturing (<7nm)

---

## 4. MLOps & Production Deployment

### 4.1 Capacity Planning with Reinforcement Learning (2025)
**Title**: Learning to Optimize Capacity Planning in Semiconductor Manufacturing  
**Authors**: Philipp Andelfinger, Jieyi Bi, Qiuyu Zhu, et al.  
**Link**: [arXiv:2509.15767](https://arxiv.org/abs/2509.15767)  
**Published**: September 2025

**Key Contributions**:
- Applies reinforcement learning to fab capacity allocation
- Handles variable demand and multi-product lines
- Reduces cycle time by 15% compared to heuristic scheduling
- Deployed in production semiconductor fab

**Relevance to Course**:
- **Module 9**: Real-time decision making
- **Module 10**: Production ML systems
- **Module 11**: Edge deployment for manufacturing control

**Implementation Concepts**:
```python
# RL for capacity planning
import gym
from stable_baselines3 import PPO

class FabCapacityEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(n_tools)
        self.observation_space = gym.spaces.Box(...)

    def step(self, action):
        # Allocate wafer to tool 'action'
        reward = -cycle_time  # Minimize cycle time
        return next_state, reward, done, info

# Train RL agent
env = FabCapacityEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

**Key Takeaways**:
- Traditional heuristics (FIFO, EDD) are suboptimal
- RL agents learn complex scheduling policies
- Simulation-based training before deployment

---

### 4.2 Few-Shot Recipe Generation (2025)
**Title**: Few-Shot Test-Time Optimization Without Retraining for Semiconductor Recipe Generation  
**Authors**: Shangding Gu, Donghao Ying, Ming Jin, et al.  
**Link**: [arXiv:2505.16060](https://arxiv.org/abs/2505.16060)  
**Published**: May 2025

**Key Contributions**:
- Optimizes process recipes at test time without retraining
- Model Feedback Learning (MFL) framework
- Adapts to new equipment or materials with 5-10 examples
- Reduces recipe development time from weeks to hours

**Relevance to Course**:
- **Module 4**: Hyperparameter optimization
- **Module 7**: Meta-learning and few-shot adaptation
- **Module 9**: Real-time recipe optimization

**Implementation Concepts**:
```python
# Model Feedback Learning
class RecipeOptimizer:
    def __init__(self, forward_model):
        self.model = forward_model  # Pre-trained process model

    def optimize_recipe(self, target_output, n_examples=5):
        # Initialize recipe parameters
        recipe = np.random.randn(n_params)

        for iter in range(100):
            # Forward pass: predict output
            predicted_output = self.model.predict(recipe)

            # Compute gradient via model feedback
            gradient = self.model.compute_gradient(recipe, target_output)

            # Update recipe
            recipe -= learning_rate * gradient

        return recipe
```

**Key Takeaways**:
- Test-time optimization avoids expensive retraining
- Leverages pre-trained models as differentiable simulators
- Critical for rapid process development

---

### 4.3 Explainable AutoML for Yield Enhancement (2024)
**Title**: Explainable AutoML (xAutoML) with adaptive modeling for yield enhancement  
**Authors**: Weihong Zhai, Xiupeng Shi, Yiik Diew Wong, et al.  
**Link**: [arXiv:2403.12381](https://arxiv.org/abs/2403.12381)  
**Published**: March 2024

**Key Contributions**:
- Combines AutoML with SHAP explainability
- Identifies top process parameters impacting yield
- Adaptive model selection based on data characteristics
- Validated on real fab data (12% yield improvement)

**Relevance to Course**:
- **Module 4**: Ensemble methods and model selection
- **Module 9**: Explainable AI for manufacturing
- **Module 10**: Production ML pipelines

**Implementation Concepts**:
```python
# Explainable AutoML
from autosklearn.classification import AutoSklearnClassifier
import shap

class ExplainableYieldPredictor:
    def __init__(self):
        self.automl = AutoSklearnClassifier(time_left_for_this_task=3600)

    def fit_and_explain(self, X, y):
        # AutoML model selection
        self.automl.fit(X, y)
        best_model = self.automl.get_models_with_weights()[0][1]

        # SHAP explainability
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X)

        return {
            "model": best_model,
            "feature_importance": shap_values,
            "top_features": self.get_top_features(shap_values)
        }
```

**Key Takeaways**:
- AutoML reduces ML expertise requirements
- Explainability builds trust with process engineers
- Feature importance guides process improvement

---

## 5. Anomaly Detection & Quality Control

### 5.1 Wavelet Transform for Anomaly Detection (2025)
**Title**: Continuous Wavelet Transform and Siamese Network-Based Anomaly Detection  
**Authors**: Bappaditya Dey, Daniel Sorensen, Minjin Hwang, Sandip Halder  
**Link**: [arXiv:2507.01999](https://arxiv.org/abs/2507.01999)  
**Published**: IEEE TSM (under review)

**Key Contributions**:
- Combines wavelet transforms with Siamese networks
- Detects subtle anomalies in multivariate sensor data
- Handles high-frequency sampling (1000+ Hz)
- Achieves 95%+ anomaly detection rate

**Relevance to Course**:
- **Module 5**: Signal processing and time series
- **Module 7**: Siamese networks for similarity learning
- **Module 9**: Real-time anomaly detection

**Implementation Concepts**:
```python
# Wavelet-based anomaly detection
import pywt
from tensorflow.keras import Model, layers

class WaveletAnomalyDetector(Model):
    def __init__(self):
        super().__init__()
        self.siamese = self.build_siamese_network()

    def preprocess(self, time_series):
        # Continuous wavelet transform
        coeffs, freqs = pywt.cwt(time_series, scales=np.arange(1, 128), wavelet='morl')
        return coeffs

    def detect_anomaly(self, test_sample, normal_samples):
        # Transform to wavelet domain
        test_wavelet = self.preprocess(test_sample)

        # Compute similarity with normal samples
        distances = [self.siamese.compare(test_wavelet, normal)
                    for normal in normal_samples]

        # Anomaly if distance exceeds threshold
        return np.min(distances) > self.threshold
```

**Key Takeaways**:
- Wavelet transforms capture transient anomalies
- Siamese networks learn similarity metrics
- Effective for rare anomaly types

---

### 5.2 Cross-Process Defect Attribution (2025)
**Title**: Cross-Process Defect Attribution using Potential Loss Analysis  
**Authors**: Tsuyoshi Idé, Kohei Miyaguchi  
**Link**: [arXiv:2508.00895](https://arxiv.org/abs/2508.00895)  
**Published**: WSC 2025 (Winter Simulation Conference)

**Key Contributions**:
- Attributes defects to upstream process steps
- Handles combinatorial complexity of process routes
- Potential Loss Analysis framework
- Reduces time-to-diagnosis by 40%

**Relevance to Course**:
- **Module 3**: Root cause analysis with regression
- **Module 6**: Graph neural networks for process paths
- **Module 9**: Real-time attribution systems

**Implementation Concepts**:
```python
# Potential Loss Analysis
class PotentialLossAnalyzer:
    def __init__(self, process_graph):
        self.graph = process_graph  # Directed acyclic graph of processes

    def attribute_defect(self, defective_wafer):
        # Compute potential loss for each process step
        loss_scores = {}

        for process in defective_wafer.history:
            # Counterfactual: what if this step was "good"?
            counterfactual_wafer = self.simulate_alternative(defective_wafer, process)

            # Potential loss = reduction in defect probability
            loss_scores[process] = self.defect_prob(defective_wafer) - \
                                  self.defect_prob(counterfactual_wafer)

        return sorted(loss_scores.items(), key=lambda x: x[1], reverse=True)
```

**Key Takeaways**:
- Attribution is harder than detection
- Counterfactual reasoning isolates causal processes
- Simulation enables what-if analysis

---

## 6. Edge AI & Deployment

### 6.1 Modular Networks for Defect Detection (2025)
**Title**: Detecting Defective Wafers Via Modular Networks  
**Authors**: Yifeng Zhang, Bryan Baker, Shi Chen, et al.  
**Link**: [arXiv:2501.03368](https://arxiv.org/abs/2501.03368)  
**Published**: January 2025

**Key Contributions**:
- Modular architecture for different defect types
- Reduces inference latency by 60% vs monolithic models
- Enables selective module updates without retraining all
- Deployed on edge devices (NVIDIA Jetson)

**Relevance to Course**:
- **Module 7**: Modular neural network architectures
- **Module 11**: Edge deployment and optimization
- **Module 9**: Real-time inference

**Implementation Concepts**:
```python
# Modular network architecture
class ModularDefectDetector:
    def __init__(self):
        self.shared_backbone = ResNet50(pretrained=True)
        self.defect_modules = {
            "scratch": DefectModule(type="scratch"),
            "particle": DefectModule(type="particle"),
            "stain": DefectModule(type="stain")
        }

    def detect(self, wafer_image):
        # Shared feature extraction
        features = self.shared_backbone(wafer_image)

        # Parallel defect-specific detection
        results = {}
        for defect_type, module in self.defect_modules.items():
            results[defect_type] = module(features)

        return results

    def update_module(self, defect_type, new_data):
        # Update single module without retraining others
        self.defect_modules[defect_type].fine_tune(new_data)
```

**Key Takeaways**:
- Monolithic models are slow and inflexible
- Modular design enables independent updates
- Critical for evolving defect landscapes

---

## Additional Resources

### Conferences to Follow
1. **ASMC** (Advanced Semiconductor Manufacturing Conference) - Annual, US-based
2. **ISSM** (International Symposium on Semiconductor Manufacturing) - IEEE sponsored
3. **SPIE Advanced Lithography** - Optical/EUV lithography focus
4. **ICML/NeurIPS ML for Science Workshops** - Emerging ML applications

### Online Resources
1. **SEMATECH** (sematech.org) - Industry consortium research
2. **IEEE Xplore** - Search "semiconductor manufacturing machine learning"
3. **ArXiv** - cs.LG, cs.CV, eess.SY categories
4. **Semiconductor Today** - Industry news and trends

### Key Journals
1. **IEEE Transactions on Semiconductor Manufacturing** - Flagship journal
2. **Journal of Vacuum Science & Technology B** - Nanoscale processing
3. **Microelectronic Engineering** - Device fabrication
4. **IEEE Transactions on Automation Science and Engineering** - Manufacturing automation

---

## How to Use This Library

### For Beginners (Modules 1-3)
Start with papers on:
- Defect detection basics (#1.1, #1.2)
- Statistical process control (#2.1)
- Tool-to-tool matching (#2.3)

### For Intermediate Learners (Modules 4-6)
Explore:
- Transfer learning (#2.2)
- Physics-informed networks (#3.1)
- Continual learning (#1.3)

### For Advanced Learners (Modules 7-11)
Deep dive into:
- Reinforcement learning (#4.1)
- Few-shot optimization (#4.2)
- Edge deployment (#6.1)

---

## Citation Format

When referencing these papers in your projects:

```bibtex
@inproceedings{miyaguchi2025wafer,
  title={Wafer Defect Root Cause Analysis with Partial Trajectory Regression},
  author={Miyaguchi, Kohei and Joko, Masao and Sheraw, Rebekah and Idé, Tsuyoshi},
  booktitle={Proceedings of ASMC 2025},
  year={2025}
}
```

---

## Updates

This library is curated from recent arXiv papers (2024-2025). For the latest research:
1. Monitor [arXiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent)
2. Search: "semiconductor manufacturing machine learning"
3. Check ASMC/ISSM/SPIE conference proceedings

**Curator Note**: Papers selected based on relevance to practical manufacturing applications, code availability, and real-world validation.

---

**Last Updated**: January 2025  
**Next Review**: June 2025  
**Maintainer**: Python for Semiconductors Learning Series
