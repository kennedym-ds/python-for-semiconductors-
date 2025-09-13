"""Test suite for GAN Defect Augmentation Pipeline

Tests the core functionality of the GAN-based defect augmentation pipeline,
including dependency checking, synthetic data generation, and CLI interface.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Import the pipeline
sys.path.append(str(Path(__file__).parent))
from gan_augmentation_pipeline import (
    GANAugmentationPipeline,
    SyntheticDefectDataset,
    SimpleGenerator,
    HAS_TORCH,
    HAS_JOBLIB,
    HAS_SKLEARN,
    main
)

class TestSyntheticDefectDataset(unittest.TestCase):
    """Test synthetic defect data generation."""
    
    def setUp(self):
        self.dataset = SyntheticDefectDataset(num_samples=10, image_size=32)
    
    def test_defect_generation(self):
        """Test that defect patterns are generated correctly."""
        for defect_type in ['edge', 'center', 'ring', 'random', 'scratch']:
            sample = self.dataset.generate_sample(defect_type)
            
            # Check shape
            self.assertEqual(sample.shape, (32, 32))
            
            # Check value range
            self.assertTrue(np.all(sample >= 0))
            self.assertTrue(np.all(sample <= 1))
            
            # Check that some defects are present (non-zero values)
            self.assertTrue(np.any(sample > 0))
    
    def test_circular_wafer_boundary(self):
        """Test that samples respect circular wafer boundary."""
        sample = self.dataset.generate_sample('center')
        
        # Check corners are mostly zero (outside wafer)
        corners = [
            sample[0, 0], sample[0, -1], 
            sample[-1, 0], sample[-1, -1]
        ]
        for corner in corners:
            self.assertLess(corner, 0.1)  # Should be near zero

class TestSimpleGenerator(unittest.TestCase):
    """Test the rule-based generator."""
    
    def setUp(self):
        self.generator = SimpleGenerator(latent_dim=100, image_size=32)
    
    def test_generation(self):
        """Test sample generation."""
        samples = self.generator.generate(num_samples=5)
        
        # Check shape
        self.assertEqual(samples.shape, (5, 32, 32))
        
        # Check value range
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1))

class TestGANAugmentationPipeline(unittest.TestCase):
    """Test the main pipeline functionality."""
    
    def setUp(self):
        # Force rule-based mode for testing
        self.pipeline = GANAugmentationPipeline(
            image_size=32,
            batch_size=16,
            use_torch=False
        )
    
    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.image_size, 32)
        self.assertEqual(self.pipeline.batch_size, 16)
        self.assertFalse(self.pipeline.use_torch)
        self.assertFalse(self.pipeline.is_trained)
    
    def test_training(self):
        """Test model training."""
        # Train the pipeline
        pipeline = self.pipeline.fit(epochs=1)
        
        # Check training completed
        self.assertTrue(pipeline.is_trained)
        self.assertIsNotNone(pipeline.metadata)
        self.assertEqual(pipeline.metadata.model_type, 'rule_based')
    
    def test_generation_before_training(self):
        """Test that generation fails before training."""
        with self.assertRaises(RuntimeError):
            self.pipeline.generate(10)
    
    def test_generation_after_training(self):
        """Test sample generation after training."""
        # Train first
        self.pipeline.fit(epochs=1)
        
        # Generate samples
        samples = self.pipeline.generate(5)
        
        # Check output
        self.assertEqual(samples.shape, (5, 32, 32))
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1))
    
    def test_augmented_dataset_generation(self):
        """Test augmented dataset creation."""
        # Train first
        self.pipeline.fit(epochs=1)
        
        # Create mock original data
        original_data = np.random.rand(10, 32, 32)
        
        # Generate augmented dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            augmented_data = self.pipeline.generate_augmented_dataset(
                original_data=original_data,
                augmentation_ratio=0.5,
                output_dir=temp_dir
            )
            
            # Check augmented data shape
            expected_size = 10 + int(10 * 0.5)  # original + 50% synthetic
            self.assertEqual(len(augmented_data), expected_size)
            
            # Check files were saved
            saved_files = list(Path(temp_dir).glob("synthetic_*.png"))
            self.assertEqual(len(saved_files), 5)  # 50% of 10
    
    @unittest.skipIf(not HAS_SKLEARN, "scikit-learn not available")
    def test_evaluation_with_sklearn(self):
        """Test augmentation impact evaluation."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Train pipeline
        self.pipeline.fit(epochs=1)
        
        # Create mock data
        original_data = np.random.rand(20, 32, 32)
        original_labels = np.random.randint(0, 2, 20)
        test_data = np.random.rand(10, 32, 32)
        test_labels = np.random.randint(0, 2, 10)
        
        # Create baseline model
        baseline_model = RandomForestClassifier(n_estimators=3, random_state=42)
        
        # Evaluate
        results = self.pipeline.evaluate_augmentation_impact(
            baseline_model=baseline_model,
            original_data=original_data,
            original_labels=original_labels,
            test_data=test_data,
            test_labels=test_labels,
            augmentation_ratio=0.5
        )
        
        # Check results structure
        self.assertIn('metrics', results)
        self.assertIn('baseline_report', results)
        self.assertIn('augmented_report', results)
        
        metrics = results['metrics']
        self.assertIn('baseline_accuracy', metrics)
        self.assertIn('augmented_accuracy', metrics)
        self.assertIn('accuracy_gain', metrics)
    
    @unittest.skipIf(not HAS_JOBLIB, "joblib not available")
    def test_save_load(self):
        """Test model saving and loading."""
        # Train pipeline
        self.pipeline.fit(epochs=1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            # Save model
            self.pipeline.save(model_path)
            self.assertTrue(model_path.exists())
            
            # Load model
            loaded_pipeline = GANAugmentationPipeline.load(model_path)
            
            # Check loaded model
            self.assertTrue(loaded_pipeline.is_trained)
            self.assertEqual(loaded_pipeline.image_size, self.pipeline.image_size)
            self.assertEqual(loaded_pipeline.batch_size, self.pipeline.batch_size)

class TestCLIInterface(unittest.TestCase):
    """Test the command-line interface."""
    
    def setUp(self):
        self.pipeline_script = Path(__file__).parent / "gan_augmentation_pipeline.py"
    
    def test_train_command(self):
        """Test the train CLI command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            grid_path = Path(temp_dir) / "test_grid.png"
            
            cmd = [
                sys.executable, str(self.pipeline_script),
                "train",
                "--epochs", "1",
                "--batch-size", "16",
                "--image-size", "32",
                "--no-torch",  # Force rule-based mode
                "--save", str(model_path),
                "--sample-grid", str(grid_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check command succeeded
            self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
            
            # Check JSON output
            output_data = json.loads(result.stdout)
            self.assertEqual(output_data['status'], 'trained')
            self.assertEqual(output_data['epochs'], 1)
            self.assertFalse(output_data['use_torch'])
            
            # Check files were created
            if HAS_JOBLIB:
                self.assertTrue(model_path.exists())
            self.assertTrue(grid_path.exists())
    
    @unittest.skipIf(not HAS_JOBLIB, "joblib not available")
    def test_generate_command(self):
        """Test the generate CLI command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            output_dir = Path(temp_dir) / "generated"
            grid_path = Path(temp_dir) / "grid.png"
            
            # First train a model
            train_cmd = [
                sys.executable, str(self.pipeline_script),
                "train",
                "--epochs", "1",
                "--no-torch",
                "--save", str(model_path)
            ]
            
            train_result = subprocess.run(train_cmd, capture_output=True, text=True)
            self.assertEqual(train_result.returncode, 0)
            
            # Then generate samples
            generate_cmd = [
                sys.executable, str(self.pipeline_script),
                "generate",
                "--model-path", str(model_path),
                "--num-samples", "5",
                "--output-dir", str(output_dir),
                "--sample-grid", str(grid_path)
            ]
            
            generate_result = subprocess.run(generate_cmd, capture_output=True, text=True)
            
            # Check command succeeded
            self.assertEqual(generate_result.returncode, 0, f"stderr: {generate_result.stderr}")
            
            # Check JSON output
            output_data = json.loads(generate_result.stdout)
            self.assertEqual(output_data['status'], 'generated')
            self.assertEqual(output_data['num_samples'], 5)
            
            # Check files were created
            generated_files = list(output_dir.glob("generated_*.png"))
            self.assertEqual(len(generated_files), 5)
            self.assertTrue(grid_path.exists())
    
    @unittest.skipIf(not (HAS_JOBLIB and HAS_SKLEARN), "joblib and sklearn required")
    def test_evaluate_command(self):
        """Test the evaluate CLI command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            # First train a model
            train_cmd = [
                sys.executable, str(self.pipeline_script),
                "train",
                "--epochs", "1",
                "--no-torch",
                "--save", str(model_path)
            ]
            
            train_result = subprocess.run(train_cmd, capture_output=True, text=True)
            self.assertEqual(train_result.returncode, 0)
            
            # Then evaluate
            evaluate_cmd = [
                sys.executable, str(self.pipeline_script),
                "evaluate",
                "--model-path", str(model_path),
                "--augmentation-ratio", "0.3"
            ]
            
            evaluate_result = subprocess.run(evaluate_cmd, capture_output=True, text=True)
            
            # Check command succeeded
            self.assertEqual(evaluate_result.returncode, 0, f"stderr: {evaluate_result.stderr}")
            
            # Check JSON output
            output_data = json.loads(evaluate_result.stdout)
            self.assertEqual(output_data['status'], 'evaluated')
            self.assertEqual(output_data['augmentation_ratio'], 0.3)
            self.assertIn('evaluation', output_data)

class TestDependencyHandling(unittest.TestCase):
    """Test graceful handling of optional dependencies."""
    
    def test_torch_dependency_check(self):
        """Test PyTorch dependency detection."""
        # This test just verifies the import check works
        # The actual value depends on the environment
        self.assertIsInstance(HAS_TORCH, bool)
    
    def test_joblib_dependency_check(self):
        """Test joblib dependency detection."""
        self.assertIsInstance(HAS_JOBLIB, bool)
    
    def test_sklearn_dependency_check(self):
        """Test scikit-learn dependency detection."""
        self.assertIsInstance(HAS_SKLEARN, bool)
    
    def test_pipeline_without_torch(self):
        """Test pipeline works without PyTorch."""
        pipeline = GANAugmentationPipeline(use_torch=False)
        self.assertFalse(pipeline.use_torch)
        
        # Should still be able to train and generate
        pipeline.fit(epochs=1)
        self.assertTrue(pipeline.is_trained)
        
        samples = pipeline.generate(5)
        self.assertEqual(samples.shape, (5, 64, 64))

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)