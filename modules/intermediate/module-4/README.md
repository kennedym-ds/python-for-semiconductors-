# Module 4 (Intermediate): Ensembles & Unsupervised Learning

This intermediate module advances from supervised fundamentals (Modules 1–3) into:

- 4.1 Ensembles (Random Forest, Gradient Boosting, stacking concepts)
- 4.2 Unsupervised Learning (clustering + anomaly detection for process monitoring)
- **4.3 Multi-Label Classification** (Steel Plates defect detection with multiple simultaneous faults)

## 4.1 Ensembles Recap

Artifacts:

- `4.1-ensemble-analysis.ipynb`
- `4.1-ensemble-fundamentals.md`
- `4.1-ensemble-pipeline.py`
- `4.1-ensemble-quick-ref.md`

Focus:

- Bias–variance reduction strategies
- Manufacturing yield regression/classification with tree ensembles
- Feature importance & stability considerations

## 4.2 Unsupervised Learning Overview

Artifacts:

- `4.2-unsupervised-analysis.ipynb`
- `4.2-unsupervised-fundamentals.md`
- `4.2-unsupervised-pipeline.py`
- `4.2-unsupervised-quick-ref.md`

Capabilities:

- Clustering models: KMeans, Gaussian Mixture, DBSCAN
- Anomaly detection: IsolationForest + hybrid (KMeans + IsolationForest)
- Synthetic dataset simulating drift and injected anomalies
- Internal validation metrics + manufacturing guardrails

Key Metrics:

- `silhouette`, `calinski_harabasz`, `davies_bouldin`
- `cluster_size_entropy`, `largest_cluster_fraction`, `anomaly_ratio`

Guardrail Warnings:

- Degenerate cluster dominance (`largest_cluster_fraction > 0.85`)
- Low structural separation (`silhouette < 0.05`)
- Excess anomaly rate (`anomaly_ratio > 0.15`)

## 4.3 Multi-Label Classification (NEW)

**Status**: ✅ Complete (Added September 2025)

Artifacts:

- `4.3-multilabel-fundamentals.md` (35+ pages, comprehensive theory)
- `4.3-multilabel-pipeline.py` (production CLI with Steel Plates integration)
- `4.3-multilabel-quick-ref.md` (practical reference guide)

**Dataset**: Steel Plates Faults (UCI ML Repository)
- 1,941 steel plates
- 27 geometric/luminosity features
- 7 binary fault indicators (Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps, Other_Faults)
- Real manufacturing quality control data

**Methods Implemented**:
1. **Binary Relevance**: One classifier per label (simple, scalable)
2. **Classifier Chains**: Sequential dependency modeling (captures correlations)
3. **Native Multi-Output**: Random Forest with built-in multi-output support

**Key Capabilities**:
- Multiple defect detection on single products
- Label correlation analysis and co-occurrence detection
- Comprehensive multi-label metrics (Subset Accuracy, Hamming Loss, Micro/Macro-F1)
- Per-label performance analysis
- Handle imbalanced labels (frequencies: 2.8% to 34.7%)

**CLI Commands**:
```bash
# Analyze Steel Plates dataset
python 4.3-multilabel-pipeline.py analyze --dataset steel_plates

# Train Binary Relevance model
python 4.3-multilabel-pipeline.py train \
    --dataset steel_plates \
    --method binary_relevance \
    --model-out models/steel_br.joblib

# Train Classifier Chains
python 4.3-multilabel-pipeline.py train \
    --dataset steel_plates \
    --method classifier_chains \
    --model-out models/steel_cc.joblib

# Evaluate model
python 4.3-multilabel-pipeline.py evaluate \
    --model-path models/steel_br.joblib \
    --dataset steel_plates

# Predict single instance
python 4.3-multilabel-pipeline.py predict \
    --model-path models/steel_br.joblib \
    --input-json '{"X_Min": 42, "X_Max": 308, ...}'
```

**Manufacturing Applications**:
- Wafer defect detection (multiple patterns per wafer)
- Equipment health monitoring (multiple subsystem failures)
- Process fault diagnosis (concurrent process issues)

**Evaluation Metrics**:
- **Subset Accuracy**: Fraction of perfect predictions (strict)
- **Hamming Loss**: Average per-label error rate
- **Micro-F1**: Aggregate performance (frequent labels weighted more)
- **Macro-F1**: Equal weight to all labels
- **Per-Label Metrics**: Precision, Recall, F1 for each fault type

## Educational Goals

| Aspect | Ensembles | Unsupervised | Multi-Label |
|--------|-----------|--------------|-------------|
| Core Skill | Model aggregation & variance control | Structure discovery & anomaly surfacing | Multiple simultaneous predictions |
| Manufacturing Use | Yield drivers, defect classification | Tool health, drift detection | Multi-defect quality control |
| Metrics Emphasis | Predictive accuracy + feature importance | Internal cluster quality + stability | Subset accuracy + per-label F1 |
| Risks Highlighted | Overfitting via deep trees | Mode collapse / false anomalies | Label imbalance, correlations |

## CLI Pattern Consistency

All pipelines expose: `train`, `evaluate`, `predict` with JSON outputs for downstream MLOps integration.

Module 4.3 adds: `analyze` command for dataset exploration.

## Next Module Preview

Module 5 will extend time series / temporal degradation modeling leveraging unsupervised signals (drift and anomaly sequences) as engineered features for predictive maintenance.

---
Maintain reproducibility with `RANDOM_SEED = 42` and follow dataset path conventions from project root.
