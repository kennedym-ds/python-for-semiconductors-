#!/usr/bin/env python3
"""
Example: Before/After Evaluation of GAN Defect Augmentation

This script demonstrates how to measure the impact of GAN-based defect 
augmentation on baseline computer vision performance. It provides a 
comprehensive evaluation framework that can be used to validate the 
effectiveness of the augmentation pipeline.

Usage:
    python evaluate_augmentation_impact.py --data-path datasets/defects/ --output-dir results/
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check dependencies
HAS_SKLEARN = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split, cross_val_score
    HAS_SKLEARN = True
except ImportError:
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")

try:
    from gan_augmentation_pipeline import GANAugmentationPipeline
    HAS_PIPELINE = True
except ImportError:
    logger.error("GAN augmentation pipeline not available. Ensure dependencies are installed.")
    HAS_PIPELINE = False

# Constants
RANDOM_SEED = 42
DEFECT_TYPES = ['edge', 'center', 'ring', 'random', 'scratch']
AUGMENTATION_RATIOS = [0.0, 0.2, 0.5, 1.0]  # 0%, 20%, 50%, 100% synthetic data

class MockDefectDataset:
    """Mock dataset for demonstration when real data isn't available."""
    
    def __init__(self, num_samples: int = 1000, image_size: int = 64):
        self.num_samples = num_samples
        self.image_size = image_size
        np.random.seed(RANDOM_SEED)
        
    def generate_labeled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mock defect data with labels."""
        images = []
        labels = []
        
        for i in range(self.num_samples):
            # Randomly assign defect type
            defect_type = np.random.choice(['normal', 'defective'])
            label = 0 if defect_type == 'normal' else 1
            
            # Generate mock image
            if defect_type == 'normal':
                # Normal wafer pattern (mostly uniform with noise)
                image = np.random.normal(0.1, 0.05, (self.image_size, self.image_size))
            else:
                # Defective pattern (with defect features)
                image = np.random.normal(0.1, 0.05, (self.image_size, self.image_size))
                
                # Add defect features
                center = self.image_size // 2
                defect_size = np.random.randint(5, 15)
                y, x = np.ogrid[:self.image_size, :self.image_size]
                defect_mask = (x - center) ** 2 + (y - center) ** 2 <= defect_size ** 2
                image[defect_mask] += np.random.uniform(0.5, 1.0)
                
            # Clip to valid range
            image = np.clip(image, 0, 1)
            images.append(image)
            labels.append(label)
            
        return np.array(images), np.array(labels)

class AugmentationEvaluator:
    """Comprehensive evaluation framework for augmentation impact."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def load_or_generate_data(self, data_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load real data or generate mock data for evaluation."""
        if data_path and Path(data_path).exists():
            logger.info(f"Loading data from {data_path}")
            # In real implementation, load actual defect images
            # For now, use mock data
            dataset = MockDefectDataset(num_samples=1000)
            images, labels = dataset.generate_labeled_data()
        else:
            logger.info("Generating mock defect dataset for demonstration")
            dataset = MockDefectDataset(num_samples=1000)
            images, labels = dataset.generate_labeled_data()
            
        # Split into train and test
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.3, random_state=RANDOM_SEED, stratify=labels
        )
        
        logger.info(f"Dataset: {len(train_images)} train, {len(test_images)} test samples")
        logger.info(f"Class distribution - Train: {np.bincount(train_labels)}, Test: {np.bincount(test_labels)}")
        
        return train_images, test_images, train_labels, test_labels
    
    def train_baseline_models(self, train_images: np.ndarray, train_labels: np.ndarray) -> Dict[str, Any]:
        """Train baseline models without augmentation."""
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for baseline evaluation")
            
        logger.info("Training baseline models...")
        
        # Flatten images for traditional ML models
        train_flat = train_images.reshape(len(train_images), -1)
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
            'SVM': SVC(random_state=RANDOM_SEED, probability=True)
        }
        
        baseline_results = {}
        for name, model in models.items():
            start_time = time.time()
            model.fit(train_flat, train_labels)
            training_time = time.time() - start_time
            
            baseline_results[name] = {
                'model': model,
                'training_time': training_time
            }
            
            logger.info(f"Trained {name} in {training_time:.2f} seconds")
            
        return baseline_results
    
    def evaluate_augmentation_ratios(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        test_images: np.ndarray,
        test_labels: np.ndarray,
        ratios: List[float] = None
    ) -> Dict[str, Any]:
        """Evaluate different augmentation ratios."""
        if not HAS_PIPELINE:
            raise RuntimeError("GAN pipeline not available")
            
        if ratios is None:
            ratios = AUGMENTATION_RATIOS
            
        logger.info(f"Evaluating augmentation ratios: {ratios}")
        
        # Train GAN for augmentation
        logger.info("Training GAN for augmentation...")
        gan = GANAugmentationPipeline(
            image_size=train_images.shape[1],
            use_torch=False  # Use rule-based for reliability
        )
        gan.fit(epochs=10)  # Quick training for demo
        
        results = {}
        test_flat = test_images.reshape(len(test_images), -1)
        
        for ratio in ratios:
            logger.info(f"Evaluating augmentation ratio: {ratio}")
            
            if ratio == 0.0:
                # No augmentation (baseline)
                augmented_images = train_images
                augmented_labels = train_labels
            else:
                # Generate augmented dataset
                augmented_images = gan.generate_augmented_dataset(
                    original_data=train_images,
                    augmentation_ratio=ratio
                )
                
                # Create labels for synthetic data
                num_synthetic = len(augmented_images) - len(train_images)
                synthetic_labels = np.random.choice(train_labels, num_synthetic)
                augmented_labels = np.concatenate([train_labels, synthetic_labels])
            
            # Train models with augmented data
            augmented_flat = augmented_images.reshape(len(augmented_images), -1)
            
            ratio_results = {}
            for model_name in ['RandomForest', 'SVM']:
                if model_name == 'RandomForest':
                    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
                else:
                    model = SVC(random_state=RANDOM_SEED, probability=True)
                
                # Train and evaluate
                start_time = time.time()
                model.fit(augmented_flat, augmented_labels)
                training_time = time.time() - start_time
                
                predictions = model.predict(test_flat)
                accuracy = accuracy_score(test_labels, predictions)
                
                ratio_results[model_name] = {
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'num_training_samples': len(augmented_images),
                    'predictions': predictions
                }
                
            results[f"ratio_{ratio}"] = ratio_results
            
        return results
    
    def generate_performance_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        logger.info("Generating performance report...")
        
        report = {
            'summary': {},
            'detailed_results': evaluation_results,
            'improvements': {},
            'recommendations': []
        }
        
        # Extract baseline performance (ratio 0.0)
        baseline = evaluation_results['ratio_0.0']
        
        for model_name in baseline.keys():
            baseline_acc = baseline[model_name]['accuracy']
            
            model_improvements = []
            for ratio_key, ratio_results in evaluation_results.items():
                if ratio_key == 'ratio_0.0':
                    continue
                    
                ratio = float(ratio_key.split('_')[1])
                aug_acc = ratio_results[model_name]['accuracy']
                improvement = aug_acc - baseline_acc
                
                model_improvements.append({
                    'ratio': ratio,
                    'accuracy': aug_acc,
                    'improvement': improvement,
                    'relative_improvement': improvement / baseline_acc if baseline_acc > 0 else 0
                })
            
            # Find best improvement
            best_improvement = max(model_improvements, key=lambda x: x['improvement'])
            
            report['improvements'][model_name] = {
                'baseline_accuracy': baseline_acc,
                'best_accuracy': best_improvement['accuracy'],
                'best_improvement': best_improvement['improvement'],
                'best_ratio': best_improvement['ratio'],
                'all_improvements': model_improvements
            }
            
            # Generate recommendations
            if best_improvement['improvement'] > 0.02:  # 2% improvement threshold
                report['recommendations'].append(
                    f"{model_name}: Use augmentation ratio {best_improvement['ratio']} "
                    f"for {best_improvement['improvement']:.1%} improvement"
                )
            else:
                report['recommendations'].append(
                    f"{model_name}: Augmentation shows minimal benefit, "
                    "consider different approach or more training data"
                )
        
        # Overall summary
        all_improvements = [imp['best_improvement'] for imp in report['improvements'].values()]
        report['summary'] = {
            'average_improvement': np.mean(all_improvements),
            'max_improvement': np.max(all_improvements),
            'models_improved': sum(1 for imp in all_improvements if imp > 0.01),
            'total_models': len(all_improvements)
        }
        
        return report
    
    def create_visualizations(self, evaluation_results: Dict[str, Any], test_labels: np.ndarray):
        """Create visualization plots for the evaluation results."""
        logger.info("Creating visualization plots...")
        
        # Performance comparison plot
        ratios = []
        rf_accuracies = []
        svm_accuracies = []
        
        for ratio_key, results in evaluation_results.items():
            ratio = float(ratio_key.split('_')[1])
            ratios.append(ratio)
            rf_accuracies.append(results['RandomForest']['accuracy'])
            svm_accuracies.append(results['SVM']['accuracy'])
        
        plt.figure(figsize=(12, 5))
        
        # Accuracy vs Augmentation Ratio
        plt.subplot(1, 2, 1)
        plt.plot(ratios, rf_accuracies, 'o-', label='Random Forest', linewidth=2, markersize=6)
        plt.plot(ratios, svm_accuracies, 's-', label='SVM', linewidth=2, markersize=6)
        plt.xlabel('Augmentation Ratio')
        plt.ylabel('Test Accuracy')
        plt.title('Model Performance vs Augmentation Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Improvement over baseline
        plt.subplot(1, 2, 2)
        baseline_rf = rf_accuracies[0]
        baseline_svm = svm_accuracies[0]
        
        rf_improvements = [(acc - baseline_rf) * 100 for acc in rf_accuracies[1:]]
        svm_improvements = [(acc - baseline_svm) * 100 for acc in svm_accuracies[1:]]
        
        x_pos = np.arange(len(ratios[1:]))
        width = 0.35
        
        plt.bar(x_pos - width/2, rf_improvements, width, label='Random Forest', alpha=0.8)
        plt.bar(x_pos + width/2, svm_improvements, width, label='SVM', alpha=0.8)
        
        plt.xlabel('Augmentation Ratio')
        plt.ylabel('Accuracy Improvement (%)')
        plt.title('Performance Improvement over Baseline')
        plt.xticks(x_pos, [f'{r:.1f}' for r in ratios[1:]])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def save_results(self, report: Dict[str, Any]):
        """Save evaluation results to files."""
        # Save JSON report
        with open(self.output_dir / 'evaluation_report.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            json.dump(convert_numpy(report), f, indent=2)
        
        # Save summary text report
        with open(self.output_dir / 'summary.txt', 'w') as f:
            f.write("GAN Defect Augmentation Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            summary = report['summary']
            f.write(f"Average Improvement: {summary['average_improvement']:.1%}\n")
            f.write(f"Maximum Improvement: {summary['max_improvement']:.1%}\n")
            f.write(f"Models Improved: {summary['models_improved']}/{summary['total_models']}\n\n")
            
            f.write("Recommendations:\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")
            
            f.write("\nDetailed Results:\n")
            for model_name, improvements in report['improvements'].items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Baseline Accuracy: {improvements['baseline_accuracy']:.1%}\n")
                f.write(f"  Best Accuracy: {improvements['best_accuracy']:.1%}\n")
                f.write(f"  Best Improvement: {improvements['best_improvement']:.1%}\n")
                f.write(f"  Optimal Ratio: {improvements['best_ratio']}\n")
        
        logger.info(f"Results saved to {self.output_dir}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate GAN defect augmentation impact')
    parser.add_argument('--data-path', type=str, help='Path to defect dataset')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--ratios', type=float, nargs='+', default=AUGMENTATION_RATIOS,
                       help='Augmentation ratios to test')
    
    args = parser.parse_args()
    
    if not HAS_SKLEARN:
        logger.error("scikit-learn is required for evaluation. Install with: pip install scikit-learn")
        return 1
    
    if not HAS_PIPELINE:
        logger.error("GAN pipeline not available. Check dependencies.")
        return 1
    
    # Initialize evaluator
    evaluator = AugmentationEvaluator(args.output_dir)
    
    try:
        # Load or generate data
        train_images, test_images, train_labels, test_labels = evaluator.load_or_generate_data(args.data_path)
        
        # Evaluate different augmentation ratios
        evaluation_results = evaluator.evaluate_augmentation_ratios(
            train_images, train_labels, test_images, test_labels, args.ratios
        )
        
        # Generate comprehensive report
        report = evaluator.generate_performance_report(evaluation_results)
        
        # Create visualizations
        evaluator.create_visualizations(evaluation_results, test_labels)
        
        # Save results
        evaluator.save_results(report)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        summary = report['summary']
        print(f"Average Improvement: {summary['average_improvement']:.1%}")
        print(f"Maximum Improvement: {summary['max_improvement']:.1%}")
        print(f"Models Improved: {summary['models_improved']}/{summary['total_models']}")
        
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"- {rec}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())