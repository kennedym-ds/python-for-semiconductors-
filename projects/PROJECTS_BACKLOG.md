# Projects Backlog

This backlog organizes work to build out the `projects/` area with production-grade, semiconductor-focused projects. Each task is written as a GitHub issue-ready item with clear outcomes and acceptance criteria.

## Phases Overview
- Foundation templates and scaffolding
- Starter reference projects (classification, regression, time-series, vision)
- Advanced projects (GAN augmentation, anomaly detection, MLOps/MLflow)
- Documentation, testing, and CI integration

## 1) Scaffolding & Templates

1. Initialize project scaffolding pipeline docs
- Summary: Document usage of `10.1-project-architecture-pipeline.py` for scaffolding and validation.
- Acceptance:
  - README updates in `projects/starter/template/`
  - Example commands verified locally

2. Add cookiecutter-compatible template export
- Summary: Enable exporting the template as a Cookiecutter-ready skeleton to speed up new projects.
- Acceptance:
  - `template/` export command documented
  - Example repo created from export

3. Add `.env.template` and secrets guidance
- Summary: Provide environment variables and secrets handling instructions for all generated projects.
- Acceptance:
  - `.env.template` added to scaffold output
  - README section on secrets management

## 2) Starter Projects

4. Starter classification project: wafer defect classifier
- Path: `projects/starter/wafer_defect_classifier/`
- Summary: Binary classification using `datasets/vision_defects` (placeholder synthetic if absent).
- Acceptance:
  - CLI: train/evaluate/predict with JSON outputs
  - Metrics: ROC-AUC, PR-AUC, PWS
  - Tests for CLI and pipeline class

5. Starter regression project: yield prediction
- Path: `projects/starter/yield_regression/`
- Summary: Regression using `datasets/secom` or synthetic data generator.
- Acceptance:
  - CLI with full pipeline
  - Metrics: MAE, RMSE, R², PWS, Estimated Loss
  - Reproducible seed and model persistence

6. Starter time-series project: equipment drift monitoring
- Path: `projects/starter/equipment_drift_monitor/`
- Summary: Forecasting/anomaly detection pipeline on `datasets/time_series` or synthetic signals.
- Acceptance:
  - Train/evaluate/predict commands
  - Metrics: MAE, MAPE, anomaly rate
  - Sliding window and feature extraction utilities

7. Starter computer-vision project: die defect segmentation
- Path: `projects/starter/die_defect_segmentation/`
- Summary: Lightweight U-Net or classical CV baseline with `datasets/vision_defects`.
- Acceptance:
  - Data loader and augmentation
  - Metrics: mIoU, pixel accuracy
  - Inference script and visualization

## 3) Advanced Projects

8. GAN-based data augmentation for defects
- Path: `projects/advanced/gan_defect_augmentation/`
- Summary: Implement training/inference for simple GAN to augment defect images.
- Acceptance:
  - Optional dependency checks (torch)
  - CLI to generate augmented dataset
  - Before/after evaluation on CV starter project

9. Unsupervised anomaly detection for equipment
- Path: `projects/advanced/anomaly_detection_equipment/`
- Summary: Isolation Forest/GMM on multivariate sensor data.
- Acceptance:
  - Train/evaluate anomaly detection
  - Threshold tuning and ROC analysis
  - Export detected intervals and scores

10. MLOps integration with MLflow
- Path: `projects/advanced/mlops_mlflow_integration/`
- Summary: Wrap training/evaluation to log params, metrics, and artifacts into MLflow.
- Acceptance:
  - `mlflow` optional dependency
  - Start/stop tracking scripts (local server ok)
  - Example run with artifacts logged

## 4) Documentation & CI

11. Project READMEs and quick-starts
- Summary: Add high-quality READMEs and quick commands for each starter/advanced project.
- Acceptance:
  - Clear run instructions per OS
  - Troubleshooting and dataset notes

12. Tests for generated projects
- Summary: Template-level tests to validate scaffold outputs and minimal run.
- Acceptance:
  - Pytest covering CLI invocation and paths
  - Dataset path convention checks

13. CI workflows for projects
- Summary: Add GitHub Actions jobs to lint, test, and smoke-run starter projects using basic tier.
- Acceptance:
  - Workflow under `.github/workflows/projects-ci.yml`
  - Uses `requirements-basic.txt`

14. Docs site integration for projects
- Summary: Expose project guides via `docs-site` and ensure navigation.
- Acceptance:
  - MkDocs pages linking to each project
  - Build verification

## 5) Stretch Goals

15. Dataset download helpers per project
- Summary: Project-level script to fetch sample/synthetic data.
- Acceptance:
  - `scripts/download_data.py` with retries and checks

16. Benchmark harness and leaderboards
- Summary: Standard harness to compare baselines across projects.
- Acceptance:
  - `benchmarks/` with JSON results
  - README leaderboard table

---

To create issues from this backlog, copy the relevant section into a new GitHub issue using the Project Task template.
