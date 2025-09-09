# Semiconductor Time Series Dataset (Placeholder)

This directory is reserved for an open manufacturing time series dataset (e.g., simulated fab benchmark or publicly released process telemetry). Candidate sources include recent benchmark simulation datasets (arXiv:2408.09307) or curated industrial process monitoring sets.

## Use Cases

- Drift detection methods
- Predictive maintenance / early warning
- Multivariate forecasting

## Acquisition (Example Workflow)

1. Identify dataset from curated list (e.g., Industrial ML Datasets GitHub).
2. Review license and usage terms.
3. Download raw CSV / Parquet and place under `raw/`.
4. Document schema in a new `SCHEMA.md`.

## Suggested Layout

```text
time_series/
  README.md
  raw/
    *.csv
  processed/
    (generated features / windows)
```

## Notes

- Keep raw immutable; all transformations reproducible via scripts.
- Add checksum file (SHA256) for integrity if distributing internally.
