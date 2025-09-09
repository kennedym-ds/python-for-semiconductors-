# Semiconductor / PCB Vision Defect Datasets (Placeholder)

This directory is for image-based defect detection datasets relevant to semiconductor wafers or analogous PCB defect corpora when wafer images are limited.

## Candidate Sources

- Wafer Surface Defect (IEEE DataPort)
- Mixed-type Wafer Defect Datasets (IEEE DataPort)
- Wafer Map GitHub repositories (processed pattern matrices)
- PCB Defects (Kaggle) as proxy for classification pipeline demonstrations

## Use Cases

- Convolutional / Vision Transformer experiments
- Data augmentation & class imbalance strategies
- Transfer learning baseline comparisons

## Acquisition

1. Select dataset ensuring redistribution terms allow local educational use.
2. Download archives (do not commit) and extract under `raw/`.
3. Convert proprietary formats (MAT, pickled numpy) to standardized PNG/NumPy arrays if needed.
4. Create `labels.csv` mapping file name to defect class.

## Suggested Layout

```text
vision_defects/
  README.md
  raw/
    classA/
    classB/
  annotations/
    labels.csv
  samples/
    thumbnail_grid.png
```

## Notes

- Keep a 10-image representative sample grid (committable) under `samples/`.
- For large archives provide checksum (SHA256) text file.
- Document any preprocessing (normalization, resizing) for reproducibility.
