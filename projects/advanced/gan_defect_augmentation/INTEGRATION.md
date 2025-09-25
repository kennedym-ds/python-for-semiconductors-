# GAN Defect Augmentation Integration Guide

This document describes how to integrate the GAN-based defect augmentation pipeline with existing computer vision projects, particularly the CV starter project for baseline evaluation.

## Overview

The GAN defect augmentation pipeline is designed to:
1. Generate synthetic defect images that improve CV model performance
2. Provide measurable impact assessment through before/after evaluation
3. Integrate seamlessly with existing CV training pipelines

## Integration with CV Starter Project

### Step 1: Set Up the Environment

```bash
# Clone or navigate to the project
cd projects/advanced/gan_defect_augmentation/

# Install dependencies (choose one):
# Option A: Full advanced environment
python ../../env_setup.py --tier advanced

# Option B: Minimal dependencies
pip install numpy pandas matplotlib Pillow joblib scikit-learn
```

### Step 2: Train the GAN Model

```bash
# Train GAN on defect data (uses rule-based generation if PyTorch unavailable)
python gan_augmentation_pipeline.py train \
  --data-path path/to/defect/images \
  --epochs 100 \
  --save models/defect_gan.joblib \
  --sample-grid outputs/training_samples.png
```

### Step 3: Generate Augmented Dataset

```bash
# Generate synthetic defects for augmentation
python gan_augmentation_pipeline.py generate \
  --model-path models/defect_gan.joblib \
  --num-samples 1000 \
  --output-dir data/augmented/ \
  --sample-grid outputs/generated_samples.png
```

### Step 4: Evaluate Augmentation Impact

```bash
# Evaluate improvement on baseline CV model
python gan_augmentation_pipeline.py evaluate \
  --model-path models/defect_gan.joblib \
  --augmentation-ratio 0.5
```

## Programmatic Integration

### Using with Existing CV Models

```python
from gan_augmentation_pipeline import GANAugmentationPipeline
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your existing data
train_images = load_training_images()  # Shape: (N, H, W)
train_labels = load_training_labels()  # Shape: (N,)
test_images = load_test_images()       # Shape: (M, H, W)
test_labels = load_test_labels()       # Shape: (M,)

# Load trained GAN
gan = GANAugmentationPipeline.load('models/defect_gan.joblib')

# Generate augmented training set
augmented_images = gan.generate_augmented_dataset(
    original_data=train_images,
    augmentation_ratio=0.5,  # 50% synthetic data
    output_dir='data/augmented/'
)

# Create corresponding labels for augmented data
num_synthetic = len(augmented_images) - len(train_images)
synthetic_labels = np.random.choice(train_labels, num_synthetic)
augmented_labels = np.concatenate([train_labels, synthetic_labels])

# Train baseline model
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(train_images.reshape(len(train_images), -1), train_labels)

# Train augmented model
augmented_model = RandomForestClassifier(n_estimators=100, random_state=42)
augmented_model.fit(augmented_images.reshape(len(augmented_images), -1), augmented_labels)

# Compare performance
baseline_pred = baseline_model.predict(test_images.reshape(len(test_images), -1))
augmented_pred = augmented_model.predict(test_images.reshape(len(test_images), -1))

print("Baseline Performance:")
print(classification_report(test_labels, baseline_pred))

print("\nAugmented Performance:")
print(classification_report(test_labels, augmented_pred))
```

### Comprehensive Evaluation Framework

```python
# Automated evaluation with detailed metrics
evaluation_results = gan.evaluate_augmentation_impact(
    baseline_model=baseline_model,
    original_data=train_images,
    original_labels=train_labels,
    test_data=test_images,
    test_labels=test_labels,
    augmentation_ratio=0.5
)

# Access detailed metrics
metrics = evaluation_results['metrics']
print(f"Accuracy improvement: {metrics['accuracy_gain']:.2%}")
print(f"Data efficiency gain: {metrics['data_efficiency_gain']:.2%}")
print(f"Training time ratio: {metrics['training_time_ratio']:.2f}")
```

## Custom Defect Pattern Configuration

### Configuring Defect Types

```python
# Create custom defect configuration
custom_defects = {
    'pattern_types': ['edge', 'center', 'ring', 'scratch', 'cluster'],
    'size_range': (5, 15),
    'intensity_range': (0.5, 1.0),
    'noise_level': 0.03
}

# Initialize pipeline with custom configuration
gan = GANAugmentationPipeline(
    image_size=128,  # Higher resolution
    batch_size=32,
    defect_config=custom_defects
)
```

### Advanced Generation Control

```python
# Generate specific defect types
edge_defects = gan.generate_specific_defects('edge', num_samples=100)
center_defects = gan.generate_specific_defects('center', num_samples=100)
mixed_defects = gan.generate_mixed_defects(
    defect_types=['edge', 'center', 'ring'],
    num_samples=300
)
```

## Integration with Different CV Architectures

### With CNN Models (if PyTorch available)

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DefectDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return torch.FloatTensor(image), torch.LongTensor([label])

# Create augmented dataset
augmented_images = gan.generate_augmented_dataset(train_images, 0.5)
augmented_dataset = DefectDataset(augmented_images, augmented_labels)
augmented_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)

# Train your CNN with augmented data
model = YourCNNModel()
train_cnn(model, augmented_loader)
```

### With Traditional ML Models

```python
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Test multiple models with augmentation
models = {
    'SVM': SVC(random_state=42),
    'MLP': MLPClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    # Evaluate with augmentation
    evaluation = gan.evaluate_augmentation_impact(
        baseline_model=model,
        original_data=train_images,
        original_labels=train_labels,
        test_data=test_images,
        test_labels=test_labels
    )
    results[name] = evaluation['metrics']['accuracy_gain']

# Display results
for model_name, improvement in results.items():
    print(f"{model_name}: {improvement:.2%} improvement")
```

## Quality Control and Validation

### Automated Quality Assessment

```python
# Assess generation quality
quality_metrics = gan.assess_generation_quality(
    num_samples=100,
    reference_data=train_images[:100]
)

print(f"Generation quality score: {quality_metrics['overall_score']:.2f}")
print(f"Diversity score: {quality_metrics['diversity']:.2f}")
print(f"Realism score: {quality_metrics['realism']:.2f}")
```

### Visual Inspection Tools

```python
# Generate comparison grids
gan.create_comparison_grid(
    real_samples=train_images[:64],
    synthetic_samples=gan.generate(64),
    output_path='outputs/comparison_grid.png'
)

# Generate quality progression during training
gan.visualize_training_progression(
    epochs=[10, 25, 50, 100],
    output_path='outputs/training_progression.png'
)
```

## Performance Benchmarking

### Recommended Evaluation Protocol

1. **Baseline Training**: Train CV model on original data only
2. **Augmented Training**: Train identical CV model on augmented data
3. **Cross-Validation**: Use k-fold CV to ensure robust results
4. **Multiple Metrics**: Evaluate accuracy, precision, recall, F1-score
5. **Statistical Significance**: Test significance of improvements

### Example Benchmarking Script

```python
from sklearn.model_selection import cross_val_score
from scipy import stats

def benchmark_augmentation(gan, original_data, labels, cv_folds=5):
    """Comprehensive benchmarking of augmentation impact."""
    
    # Generate augmented data
    augmented_data = gan.generate_augmented_dataset(original_data, 0.5)
    
    # Create models
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    augmented_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Cross-validation scores
    baseline_scores = cross_val_score(
        baseline_model, original_data.reshape(len(original_data), -1), 
        labels, cv=cv_folds
    )
    
    augmented_scores = cross_val_score(
        augmented_model, augmented_data.reshape(len(augmented_data), -1),
        augmented_labels, cv=cv_folds
    )
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(augmented_scores, baseline_scores)
    
    return {
        'baseline_mean': baseline_scores.mean(),
        'baseline_std': baseline_scores.std(),
        'augmented_mean': augmented_scores.mean(),
        'augmented_std': augmented_scores.std(),
        'improvement': augmented_scores.mean() - baseline_scores.mean(),
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Run benchmark
results = benchmark_augmentation(gan, train_images, train_labels)
print(f"Mean improvement: {results['improvement']:.3f} ± {results['augmented_std']:.3f}")
print(f"Statistically significant: {results['significant']} (p={results['p_value']:.3f})")
```

## Troubleshooting

### Common Integration Issues

1. **Data Format Mismatch**
   - Ensure images are in correct format (grayscale, normalized)
   - Check image dimensions match pipeline expectations

2. **Memory Issues**
   - Reduce batch size: `--batch-size 32`
   - Generate data in chunks for large datasets

3. **Poor Augmentation Quality**
   - Increase training epochs: `--epochs 200`
   - Adjust augmentation ratio: `--augmentation-ratio 0.3`

4. **No Performance Improvement**
   - Check data quality and labeling
   - Verify baseline model isn't overfitting
   - Try different augmentation ratios

### Performance Optimization

```python
# Optimize for speed
gan_fast = GANAugmentationPipeline(
    batch_size=128,  # Larger batches
    use_torch=False,  # Rule-based for speed
    image_size=32     # Smaller images
)

# Optimize for quality
gan_quality = GANAugmentationPipeline(
    batch_size=16,    # Smaller batches for stability
    use_torch=True,   # Deep learning if available
    image_size=128    # Higher resolution
)
```

## Next Steps

1. **Experiment with Parameters**: Try different augmentation ratios (0.2, 0.5, 1.0)
2. **Custom Defect Patterns**: Implement domain-specific defect types
3. **Advanced Metrics**: Add manufacturing-specific evaluation metrics
4. **Production Deployment**: Integrate with MLOps pipelines for automated retraining

For more detailed examples and advanced usage, see the example notebooks in the project directory.