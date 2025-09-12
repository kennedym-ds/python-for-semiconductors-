"""Tests for Advanced Defect Detection Pipeline (Module 7.1)

Tests the defect detection pipeline with synthetic data generation,
multiple backends, and CLI functionality.
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys
import os
import numpy as np

# Add the module directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the pipeline module directly
import importlib.util
pipeline_path = Path(__file__).parent / "7.1-advanced-defect-detection-pipeline.py"
spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
sys.modules["pipeline"] = importlib.util.module_from_spec(spec)
pipeline = sys.modules["pipeline"]
spec.loader.exec_module(pipeline)

AdvancedDefectDetectionPipeline = pipeline.AdvancedDefectDetectionPipeline
generate_synthetic_wafer_defects = pipeline.generate_synthetic_wafer_defects

class TestSyntheticDataGeneration(unittest.TestCase):
    """Test synthetic wafer defect generation."""
    
    def test_synthetic_data_generation(self):
        """Test that synthetic data is generated correctly."""
        images, annotations = generate_synthetic_wafer_defects(
            n_images=3,
            image_size=(200, 200),
            n_defects_range=(1, 3),
            seed=42
        )
        
        self.assertEqual(len(images), 3)
        self.assertEqual(len(annotations), 3)
        
        # Check image properties
        for img in images:
            self.assertEqual(img.shape[:2], (200, 200))
            self.assertEqual(len(img.shape), 3)  # RGB
        
        # Check annotations
        for ann_list in annotations:
            self.assertIsInstance(ann_list, list)
            for ann in ann_list:
                self.assertIn('bbox', ann)
                self.assertIn('class', ann)
                self.assertIn('class_id', ann)
                
                bbox = ann['bbox']
                self.assertEqual(len(bbox), 4)
                self.assertTrue(all(isinstance(x, (int, float, np.integer, np.floating)) for x in bbox))

class TestClassicalDetector(unittest.TestCase):
    """Test classical OpenCV-based detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.images, self.annotations = generate_synthetic_wafer_defects(
            n_images=5, seed=42
        )
    
    def test_classical_pipeline_creation(self):
        """Test creating classical pipeline."""
        pipeline = AdvancedDefectDetectionPipeline(backend='classical')
        self.assertEqual(pipeline.backend, 'classical')
        self.assertIsNotNone(pipeline.detector)
    
    def test_classical_fit_predict(self):
        """Test fitting and prediction with classical detector."""
        pipeline = AdvancedDefectDetectionPipeline(backend='classical')
        
        # Fit (no-op for classical)
        pipeline.fit(self.images, self.annotations)
        self.assertIsNotNone(pipeline.metadata)
        
        # Predict
        predictions = pipeline.predict(self.images[:2])
        self.assertEqual(len(predictions), 2)
        
        for pred_list in predictions:
            self.assertIsInstance(pred_list, list)
            for pred in pred_list:
                self.assertIn('bbox', pred)
                self.assertIn('class', pred)
                self.assertIn('confidence', pred)
    
    def test_classical_evaluate(self):
        """Test evaluation with classical detector."""
        pipeline = AdvancedDefectDetectionPipeline(backend='classical')
        pipeline.fit(self.images, self.annotations)
        
        metrics = pipeline.evaluate(self.images[:3], self.annotations[:3])
        
        expected_metrics = [
            'precision', 'recall', 'f1_score', 'map_50', 'pws_percent',
            'estimated_loss_usd', 'true_positives', 'false_positives', 'false_negatives'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

class TestPipelinePersistence(unittest.TestCase):
    """Test model saving and loading."""
    
    def test_save_load_classical(self):
        """Test saving and loading classical pipeline."""
        images, annotations = generate_synthetic_wafer_defects(n_images=3, seed=42)
        
        # Create and train pipeline
        pipeline = AdvancedDefectDetectionPipeline(
            backend='classical',
            blur_kernel=7,
            threshold_value=60
        )
        pipeline.fit(images, annotations)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            pipeline.save(temp_path)
            self.assertTrue(temp_path.exists())
            
            # Load pipeline
            loaded_pipeline = AdvancedDefectDetectionPipeline.load(temp_path)
            
            self.assertEqual(loaded_pipeline.backend, 'classical')
            self.assertEqual(loaded_pipeline.kwargs['blur_kernel'], 7)
            self.assertEqual(loaded_pipeline.kwargs['threshold_value'], 60)
            
            # Test prediction still works
            predictions = loaded_pipeline.predict(images[:1])
            self.assertEqual(len(predictions), 1)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()

class TestCLIInterface(unittest.TestCase):
    """Test command-line interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline_script = Path(__file__).parent / "7.1-advanced-defect-detection-pipeline.py"
        self.assertTrue(self.pipeline_script.exists())
    
    def test_cli_train(self):
        """Test CLI training command."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            model_path = f.name
        
        try:
            cmd = [
                'python3', str(self.pipeline_script), 'train',
                '--backend', 'classical',
                '--n-images', '3',
                '--epochs', '1',
                '--save', model_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            self.assertEqual(result.returncode, 0)
            
            # Parse JSON output
            output = json.loads(result.stdout)
            self.assertEqual(output['status'], 'success')
            self.assertEqual(output['backend'], 'classical')
            self.assertEqual(output['n_images_trained'], 3)
            self.assertTrue(output['model_saved'])
            
            # Check model file was created
            self.assertTrue(Path(model_path).exists())
            
        finally:
            if Path(model_path).exists():
                Path(model_path).unlink()
    
    def test_cli_evaluate(self):
        """Test CLI evaluation command."""
        # First train a model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            model_path = f.name
        
        try:
            # Train
            cmd_train = [
                'python3', str(self.pipeline_script), 'train',
                '--backend', 'classical',
                '--n-images', '3',
                '--epochs', '1',
                '--save', model_path
            ]
            result = subprocess.run(cmd_train, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0)
            
            # Evaluate
            cmd_eval = [
                'python3', str(self.pipeline_script), 'evaluate',
                '--model-path', model_path,
                '--dataset', 'synthetic',
                '--n-images', '3'
            ]
            result = subprocess.run(cmd_eval, capture_output=True, text=True, timeout=60)
            
            self.assertEqual(result.returncode, 0)
            
            # Parse JSON output
            output = json.loads(result.stdout)
            self.assertEqual(output['status'], 'success')
            self.assertEqual(output['backend'], 'classical')
            self.assertIn('metrics', output)
            
            metrics = output['metrics']
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('pws_percent', metrics)
            
        finally:
            if Path(model_path).exists():
                Path(model_path).unlink()
    
    def test_cli_predict(self):
        """Test CLI prediction command."""
        # First train a model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            model_path = f.name
        
        try:
            # Train
            cmd_train = [
                'python3', str(self.pipeline_script), 'train',
                '--backend', 'classical',
                '--n-images', '3',
                '--epochs', '1',
                '--save', model_path
            ]
            result = subprocess.run(cmd_train, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0)
            
            # Predict
            cmd_pred = [
                'python3', str(self.pipeline_script), 'predict',
                '--model-path', model_path,
                '--dataset', 'synthetic'
            ]
            result = subprocess.run(cmd_pred, capture_output=True, text=True, timeout=60)
            
            self.assertEqual(result.returncode, 0)
            
            # Parse JSON output
            output = json.loads(result.stdout)
            self.assertEqual(output['status'], 'success')
            self.assertEqual(output['backend'], 'classical')
            self.assertIn('predictions', output)
            
        finally:
            if Path(model_path).exists():
                Path(model_path).unlink()

class TestBackendFallbacks(unittest.TestCase):
    """Test backend fallback behavior."""
    
    def test_yolo_fallback_to_classical(self):
        """Test YOLO fallback to classical when ultralytics unavailable."""
        # This test assumes ultralytics is not installed in CI environment
        # If it were installed, the test would still pass but use YOLO
        pipeline = AdvancedDefectDetectionPipeline(backend='yolo')
        
        # Should fallback to classical if ultralytics not available
        # or use yolo if available - either is acceptable
        self.assertIn(pipeline.backend, ['yolo', 'classical'])
    
    def test_fasterrcnn_fallback_to_classical(self):
        """Test Faster R-CNN fallback to classical when torchvision unavailable."""
        # Similar to above - test fallback behavior
        pipeline = AdvancedDefectDetectionPipeline(backend='fasterrcnn')
        
        # Should fallback to classical if torchvision not available
        # or use fasterrcnn if available - either is acceptable
        self.assertIn(pipeline.backend, ['fasterrcnn', 'classical'])

def run_tests():
    """Run all tests and return success status."""
    # Create test suite
    test_classes = [
        TestSyntheticDataGeneration,
        TestClassicalDetector,
        TestPipelinePersistence,
        TestCLIInterface,
        TestBackendFallbacks
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)