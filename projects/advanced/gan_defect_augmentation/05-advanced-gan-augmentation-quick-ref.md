# GAN-based Defect Augmentation - Quick Reference

## Command Line Interface

### Training a GAN Model
```bash
# Basic training
python gan_augmentation_pipeline.py train --data-dir data/defects/ --output-path models/defect_gan.joblib

# Advanced training with custom parameters
python gan_augmentation_pipeline.py train \
  --data-dir data/defects/ \
  --output-path models/defect_gan.joblib \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.0002 \
  --image-size 64
```

### Generating Synthetic Samples
```bash
# Generate samples
python gan_augmentation_pipeline.py generate \
  --model-path models/defect_gan.joblib \
  --num-samples 1000 \
  --output-dir data/augmented/ \
  --sample-grid outputs/generated_samples.png

# Generate with specific seed for reproducibility
python gan_augmentation_pipeline.py generate \
  --model-path models/defect_gan.joblib \
  --num-samples 500 \
  --seed 42 \
  --output-dir data/synthetic/
```

### Evaluating Model Quality
```bash
# Basic evaluation
python gan_augmentation_pipeline.py evaluate \
  --model-path models/defect_gan.joblib

# Evaluate with reference data
python gan_augmentation_pipeline.py evaluate \
  --model-path models/defect_gan.joblib \
  --reference-data data/real_defects/ \
  --metrics fid,is,visual_similarity
```

## Python API

### Basic Usage
```python
from gan_augmentation_pipeline import GANAugmentationPipeline
import numpy as np

# Initialize pipeline
gan = GANAugmentationPipeline(
    image_size=64,
    batch_size=32,
    latent_dim=100
)

# Train on your data
training_images = np.load('defect_images.npy')
gan.fit(training_images)

# Generate synthetic samples
synthetic_defects = gan.generate(num_samples=1000)

# Save model
gan.save('models/defect_gan.joblib')
```

### Advanced Configuration
```python
# Custom defect configuration
defect_config = {
    'pattern_types': ['edge', 'center', 'ring', 'scratch'],
    'size_range': (5, 15),
    'intensity_range': (0.5, 1.0),
    'noise_level': 0.03
}

gan = GANAugmentationPipeline(
    image_size=128,
    batch_size=16,
    defect_config=defect_config,
    pytorch_backend=True  # Enable PyTorch if available
)
```

### Data Augmentation Workflow
```python
# Create augmented training dataset
original_data = load_defect_images('data/real/')
augmented_data = gan.generate_augmented_dataset(
    original_data=original_data,
    augmentation_ratio=0.5,  # 50% synthetic data
    output_dir='data/augmented/'
)

# Evaluate impact on model performance
from sklearn.ensemble import RandomForestClassifier

baseline_model = RandomForestClassifier()
baseline_model.fit(original_data.reshape(len(original_data), -1), labels)

augmented_model = RandomForestClassifier()
augmented_model.fit(augmented_data.reshape(len(augmented_data), -1), augmented_labels)

# Compare performance
baseline_score = baseline_model.score(test_data, test_labels)
augmented_score = augmented_model.score(test_data, test_labels)
improvement = (augmented_score - baseline_score) / baseline_score * 100
print(f"Performance improvement: {improvement:.1f}%")
```

## Configuration Parameters

### Model Architecture
| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 64 | Output image dimensions (image_size x image_size) |
| `latent_dim` | 100 | Dimensionality of noise vector input |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.0002 | Learning rate for both generator and discriminator |
| `beta1` | 0.5 | Adam optimizer beta1 parameter |
| `beta2` | 0.999 | Adam optimizer beta2 parameter |

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 100 | Number of training epochs |
| `save_interval` | 10 | Epochs between model checkpoints |
| `sample_interval` | 5 | Epochs between sample generation |
| `d_train_steps` | 1 | Discriminator updates per generator update |
| `gradient_penalty` | 10.0 | WGAN-GP gradient penalty coefficient |

### Generation Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | 64 | Number of samples to generate |
| `seed` | None | Random seed for reproducible generation |
| `temperature` | 1.0 | Sampling temperature (higher = more diverse) |
| `truncation` | None | Truncation threshold for noise sampling |

## Quality Metrics Reference

### Fréchet Inception Distance (FID)
- **Range**: 0-∞ (lower is better)
- **Good**: < 50
- **Excellent**: < 20
- **Interpretation**: Distance between feature distributions of real and generated images

### Inception Score (IS)
- **Range**: 1-∞ (higher is better)
- **Good**: > 3
- **Excellent**: > 5
- **Interpretation**: Measures both quality and diversity of generated samples

### Manufacturing-Specific Metrics
```python
# Pattern fidelity assessment
pattern_metrics = gan.assess_pattern_quality(
    generated_samples=synthetic_data,
    reference_samples=real_data
)

print(f"Edge similarity: {pattern_metrics['edge_similarity']:.3f}")
print(f"Intensity match: {pattern_metrics['intensity_match']:.3f}")
print(f"Spatial frequency: {pattern_metrics['frequency_match']:.3f}")
```

## Troubleshooting Guide

### Common Issues

#### 1. Mode Collapse
**Symptoms**: Generated samples look very similar
**Solutions**:
```python
# Increase diversity penalty
gan = GANAugmentationPipeline(
    diversity_penalty=0.1,
    feature_matching=True
)

# Use different noise sampling
gan.set_noise_sampling('truncated_normal', truncation=0.8)
```

#### 2. Training Instability
**Symptoms**: Loss oscillates wildly, no convergence
**Solutions**:
```python
# Reduce learning rates
gan = GANAugmentationPipeline(
    learning_rate=0.0001,  # Reduced from 0.0002
    lr_decay=0.95,         # Add learning rate decay
    gradient_clipping=1.0   # Add gradient clipping
)

# Adjust training balance
gan.train(
    d_train_steps=2,  # Train discriminator more often
    g_train_steps=1
)
```

#### 3. Poor Quality Generation
**Symptoms**: Blurry or unrealistic samples
**Solutions**:
```python
# Increase model capacity
gan = GANAugmentationPipeline(
    generator_filters=128,    # Increased from 64
    discriminator_filters=128,
    num_residual_blocks=4    # Add residual connections
)

# Improve training data
# - Ensure high-quality training samples
# - Increase dataset size
# - Balance defect types
```

#### 4. Memory Issues
**Symptoms**: Out of memory errors during training
**Solutions**:
```python
# Reduce batch size
gan = GANAugmentationPipeline(batch_size=16)

# Use gradient checkpointing
gan.enable_gradient_checkpointing()

# Reduce image size temporarily
gan = GANAugmentationPipeline(image_size=32)  # Start smaller
```

### Performance Optimization

#### GPU Acceleration
```python
# Enable GPU if available
gan = GANAugmentationPipeline(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    mixed_precision=True,  # Use automatic mixed precision
    dataloader_workers=4   # Parallel data loading
)
```

#### Memory Optimization
```python
# Optimize for limited memory
gan.optimize_memory_usage(
    gradient_accumulation_steps=2,
    cpu_offloading=True,
    checkpoint_segments=4
)
```

## Integration Examples

### With MLflow Tracking
```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        'image_size': 64,
        'batch_size': 32,
        'learning_rate': 0.0002,
        'epochs': 100
    })
    
    # Train model
    gan.fit(training_data)
    
    # Log metrics
    fid_score = gan.calculate_fid(real_data, generated_data)
    mlflow.log_metric('fid_score', fid_score)
    
    # Save model artifact
    mlflow.sklearn.log_model(gan, 'gan_model')
```

### With Existing CV Pipeline
```python
# Load existing CV model
from your_cv_project import DefectClassifier

cv_model = DefectClassifier.load('models/baseline_cv.joblib')

# Generate augmented training data
augmented_data = gan.generate_augmented_dataset(
    original_data=cv_model.training_data,
    augmentation_ratio=0.3
)

# Retrain with augmented data
cv_model.fit(augmented_data, augmented_labels)
cv_model.save('models/augmented_cv.joblib')

# Compare performance
baseline_accuracy = cv_model.evaluate_baseline()
augmented_accuracy = cv_model.evaluate()
print(f"Improvement: {augmented_accuracy - baseline_accuracy:.3f}")
```

### Production Deployment
```python
# Production inference pipeline
class ProductionGANPipeline:
    def __init__(self, model_path):
        self.gan = GANAugmentationPipeline.load(model_path)
        self.quality_threshold = 0.8
        
    def generate_for_retraining(self, num_samples):
        # Generate samples for model retraining
        samples = self.gan.generate(num_samples)
        
        # Quality check
        quality_score = self.assess_quality(samples)
        if quality_score < self.quality_threshold:
            raise ValueError(f"Quality too low: {quality_score}")
            
        return samples
    
    def daily_augmentation_job(self):
        # Scheduled job for daily model updates
        new_samples = self.generate_for_retraining(1000)
        self.update_production_model(new_samples)
        self.log_performance_metrics()
```

## Best Practices Checklist

### Data Preparation
- [ ] Normalize images to consistent scale (0-1 or -1 to 1)
- [ ] Ensure balanced representation of defect types
- [ ] Remove corrupted or mislabeled samples
- [ ] Standardize image dimensions
- [ ] Apply consistent preprocessing pipeline

### Model Training
- [ ] Monitor both generator and discriminator losses
- [ ] Save regular checkpoints during training
- [ ] Use validation set for early stopping
- [ ] Log training metrics and sample images
- [ ] Test on held-out real data regularly

### Quality Assurance
- [ ] Calculate FID and IS scores
- [ ] Visual inspection of generated samples
- [ ] Test augmentation impact on downstream tasks
- [ ] Validate against manufacturing constraints
- [ ] Document model limitations and assumptions

### Production Deployment
- [ ] Version control for models and configurations
- [ ] A/B testing framework for model updates
- [ ] Monitoring and alerting for quality degradation
- [ ] Fallback to previous model versions
- [ ] Regular retraining schedules
- [ ] Compliance with regulatory requirements