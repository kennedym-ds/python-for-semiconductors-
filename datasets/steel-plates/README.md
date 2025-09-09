# Steel Plates Faults Dataset

**Source**: UCI Machine Learning Repository (ID: 198)  
**Domain**: Manufacturing Quality Control  
**Task**: Multi-class Classification (7 fault types)

## Description

A dataset of steel plates' faults, classified into 7 different types. The goal was to train machine learning for automatic pattern recognition in steel manufacturing.

## Fault Types

1. **Pastry** - Surface texture defects
2. **Z_Scratch** - Longitudinal scratches  
3. **K_Scratch** - Transverse scratches
4. **Stains** - Discoloration defects
5. **Dirtiness** - Contamination
6. **Bumps** - Surface elevation defects
7. **Other_Faults** - Miscellaneous defects

## Dataset Characteristics

- **Instances**: 1,941 steel plates
- **Features**: 27 geometric and luminosity measurements
- **Missing Values**: None
- **Feature Types**: Integer and Real

## Key Features

- Geometric measurements (X/Y min/max, areas, perimeters)
- Luminosity statistics (sum, min, max, luminosity index)
- Steel properties (type, thickness)
- Shape indices (edges, orientation, sigmoid transforms)

## Citation

```
Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults [Dataset]. 
UCI Machine Learning Repository. https://doi.org/10.24432/C5J88N
```

Original source: https://archive.ics.uci.edu/dataset/198/steel+plates+faults