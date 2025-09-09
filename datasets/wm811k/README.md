# WM-811K Wafer Map Dataset

Source: Kaggle (WM-811K Wafer Map) and academic references citing Taiwan Semiconductor Manufacturing Co. internal patterns. Public derivative hosted on Kaggle by community contributor.

## Use Cases

- Wafer defect pattern classification
- Imbalanced class handling (rare spatial patterns)
- Spatial feature extraction / CNN models

## Acquisition

1. Visit Kaggle dataset page: [WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
2. Accept terms (requires Kaggle account).
3. Download archive and extract under this directory maintaining original folder hierarchy.
4. Do NOT commit extracted images or large numpy arrays.

Suggested Directory Layout (after extraction):

```text
wm811k/
  README.md
  data/
    wafer_map/  # raw pattern arrays or images
    label/      # labels / metadata csv
```

## Licensing & Terms

Check Kaggle page license field; typically for research/educational use. Include citation if publishing results.

### Citation (example placeholder – verify actual)

```bibtex
@dataset{wm811k,
  title={WM-811K Wafer Map Dataset},
  year={2018},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map}
}
```

## Notes

- Highly imbalanced – consider focal loss or class-weighting.
- Some patterns ambiguous; manual inspection recommended for top model errors.
