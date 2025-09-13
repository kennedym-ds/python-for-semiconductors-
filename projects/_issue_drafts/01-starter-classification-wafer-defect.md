# [Project] Starter classification: wafer defect classifier

## Summary

Baseline binary classifier for wafer defects using standardized CLI and metrics.

## Project Context

- Project type: classification
- Location: `projects/starter/wafer_defect_classifier/`
- Related module(s): module-6 (vision), module-10.1

## Problem / Why

Provide a ready-to-run baseline project aligned with dataset path and CLI standards.

## Scope

- In scope: data loading, training, evaluation, prediction, metrics, tests
- Out of scope: deep learning models (use classical baseline first)

## Deliverables

- [ ] Project skeleton with README
- [ ] Pipeline CLI (train/evaluate/predict) with JSON outputs
- [ ] Unit tests for CLI and pipeline
- [ ] Metrics: ROC-AUC, PR-AUC, PWS

## Acceptance Criteria

- [ ] CLI runs end-to-end on sample/synthetic data
- [ ] Tests pass locally and in CI
- [ ] Follows `DATA_DIR = Path('../../../datasets').resolve()` conventions
