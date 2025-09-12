"""Tests for Module 8.1 GANs for Data Augmentation Pipeline."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Get the directory containing this test file
TEST_DIR = Path(__file__).parent
SCRIPT = TEST_DIR / '8.1-gans-data-augmentation-pipeline.py'

def run_cmd(args, check=True):
    """Helper to run pipeline commands."""
    cmd = [sys.executable, str(SCRIPT)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return result

class TestGANsPipeline(unittest.TestCase):
    """Test GANs pipeline functionality."""

    def test_train_small_model(self):
        """Test training a very small model for quick validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'test_model.joblib'
            
            # Train for minimal epochs with small batch
            result = run_cmd([
                'train',
                '--epochs', '2',
                '--batch-size', '8',
                '--image-size', '32',  # Smaller for speed
                '--save', str(model_path)
            ])
            
            # Parse JSON output
            output = json.loads(result.stdout)
            self.assertEqual(output['status'], 'trained')
            self.assertEqual(output['model_type'], 'dcgan')
            self.assertEqual(output['epochs'], 2)
            
            # Check model file exists
            self.assertTrue(model_path.exists())

    def test_generate_samples(self):
        """Test sample generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'test_model.joblib'
            grid_path = Path(temp_dir) / 'samples.png'
            
            # First train a small model
            run_cmd([
                'train',
                '--epochs', '1',
                '--batch-size', '4',
                '--image-size', '32',
                '--save', str(model_path)
            ])
            
            # Generate samples
            result = run_cmd([
                'generate',
                '--model-path', str(model_path),
                '--num-samples', '16',
                '--output-grid', str(grid_path),
                '--grid-nrow', '4'
            ])
            
            # Parse JSON output
            output = json.loads(result.stdout)
            self.assertEqual(output['status'], 'generated')
            self.assertEqual(output['num_samples'], 16)
            
            # Check grid file exists
            self.assertTrue(grid_path.exists())

    def test_evaluate_model(self):
        """Test model evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'test_model.joblib'
            
            # First train a small model
            run_cmd([
                'train',
                '--epochs', '1',
                '--batch-size', '4',
                '--image-size', '32',
                '--save', str(model_path)
            ])
            
            # Evaluate model
            result = run_cmd([
                'evaluate',
                '--model-path', str(model_path)
            ])
            
            # Parse JSON output
            output = json.loads(result.stdout)
            self.assertEqual(output['status'], 'evaluated')
            self.assertIn('metrics', output)
            self.assertIn('warnings', output)
            
            # Check expected metrics exist
            metrics = output['metrics']
            self.assertIn('num_generated_samples', metrics)
            self.assertIn('sample_mean', metrics)
            self.assertIn('sample_std', metrics)

    def test_save_load_roundtrip(self):
        """Test saving and loading model works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'test_model.joblib'
            
            # Train and save model
            train_result = run_cmd([
                'train',
                '--epochs', '1',
                '--batch-size', '4',
                '--image-size', '32',
                '--seed', '123',
                '--save', str(model_path)
            ])
            
            train_output = json.loads(train_result.stdout)
            
            # Generate samples with loaded model
            generate_result = run_cmd([
                'generate',
                '--model-path', str(model_path),
                '--num-samples', '4'
            ])
            
            generate_output = json.loads(generate_result.stdout)
            self.assertEqual(generate_output['status'], 'generated')
            
            # Evaluate loaded model
            eval_result = run_cmd([
                'evaluate',
                '--model-path', str(model_path)
            ])
            
            eval_output = json.loads(eval_result.stdout)
            self.assertEqual(eval_output['status'], 'evaluated')

    def test_cli_error_handling(self):
        """Test CLI error handling for invalid inputs."""
        # Test generate without model
        result = run_cmd([
            'generate',
            '--model-path', 'nonexistent.joblib'
        ], check=False)
        
        self.assertNotEqual(result.returncode, 0)
        error_output = json.loads(result.stdout)
        self.assertEqual(error_output['status'], 'error')
        
        # Test evaluate without model
        result = run_cmd([
            'evaluate',
            '--model-path', 'nonexistent.joblib'
        ], check=False)
        
        self.assertNotEqual(result.returncode, 0)
        error_output = json.loads(result.stdout)
        self.assertEqual(error_output['status'], 'error')

if __name__ == '__main__':
    unittest.main()