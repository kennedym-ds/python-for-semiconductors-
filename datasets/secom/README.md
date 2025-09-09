# SECOM Dataset

**Source**: UCI Machine Learning Repository  
**Domain**: Semiconductor Manufacturing Process Control  
**Task**: Binary Classification (Pass/Fail prediction)

## Description

The SECOM dataset contains semiconductor manufacturing data collected from a semi-conductor fabrication process. Each row represents a sensor measurement from the manufacturing process, with 590 features representing various process parameters and measurements.

## Files

- `secom.data`: Feature matrix (1567 instances × 590 features)
- `secom_labels.data`: Binary target labels (-1 = Pass, +1 = Fail)  
- `secom.names`: Dataset description and attribute information

## Usage Notes

- High dimensionality: 590 features for quality control sensors
- Class imbalance: Most wafers pass (majority class -1)
- Missing values: Some sensors may have null readings
- Real semiconductor fab data with anonymized feature names

## Citation

```
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
School of Information and Computer Science.
```

Original source: https://archive.ics.uci.edu/ml/datasets/SECOM