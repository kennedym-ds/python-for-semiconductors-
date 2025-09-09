# Module 4 (Intermediate): Ensembles & Unsupervised Learning

This intermediate module advances from supervised fundamentals (Modules 1–3) into:

- 4.1 Ensembles (Random Forest, Gradient Boosting, stacking concepts)
- 4.2 Unsupervised Learning (clustering + anomaly detection for process monitoring)

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

## Educational Goals

| Aspect | Ensembles | Unsupervised |
|--------|-----------|--------------|
| Core Skill | Model aggregation & variance control | Structure discovery & anomaly surfacing |
| Manufacturing Use | Yield drivers, defect classification | Tool health, drift detection |
| Metrics Emphasis | Predictive accuracy + feature importance | Internal cluster quality + stability |
| Risks Highlighted | Overfitting via deep trees | Mode collapse / false anomalies |

## CLI Pattern Consistency

All pipelines expose: `train`, `evaluate`, `predict` with JSON outputs for downstream MLOps integration.

## Next Module Preview

Module 5 will extend time series / temporal degradation modeling leveraging unsupervised signals (drift and anomaly sequences) as engineered features for predictive maintenance.

---
Maintain reproducibility with `RANDOM_SEED = 42` and follow dataset path conventions from project root.
