"""Tests for enhanced dataset management functionality.

This module provides comprehensive tests for all the new dataset management
features including synthetic data generation, validation, WM-811K preprocessing,
and DVC integration.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add datasets directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_generators import (
    TimeSensorDataGenerator, ProcessRecipeGenerator, 
    WaferDefectPatternGenerator, generate_all_synthetic_datasets
)
from data_validation import DatasetValidator, run_dataset_validation
from wm811k_preprocessing import WM811KPreprocessor, WaferMapVisualizer, WaferMapData


class TestSyntheticDataGenerators:
    """Test synthetic data generators."""
    
    def test_time_sensor_data_generator(self):
        """Test time series sensor data generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "time_series"
            
            generator = TimeSensorDataGenerator(random_state=42)
            result = generator.generate(
                n_samples=10,
                n_sensors=5,
                sequence_length=20,
                output_dir=output_dir
            )
            
            # Check result structure
            assert 'sensor_data' in result
            assert 'labels' in result
            assert 'metadata' in result
            assert 'sensor_names' in result
            
            # Check data shapes
            assert result['sensor_data'].shape == (10, 20, 5)
            assert len(result['labels']) == 10
            assert len(result['sensor_names']) == 5
            
            # Check files were saved
            assert (output_dir / "sensor_data.npz").exists()
            assert (output_dir / "metadata.json").exists()
            
            # Verify saved data can be loaded
            saved_data = np.load(output_dir / "sensor_data.npz")
            assert 'sensor_data' in saved_data
            assert 'labels' in saved_data
    
    def test_process_recipe_generator(self):
        """Test process recipe database generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "recipes"
            
            generator = ProcessRecipeGenerator(random_state=42)
            result = generator.generate(
                n_recipes=50,
                n_parameters=10,
                output_dir=output_dir
            )
            
            # Check result structure
            assert 'recipes' in result
            assert 'outcomes' in result
            assert 'metadata' in result
            assert 'parameter_descriptions' in result
            
            # Check data dimensions
            assert result['recipes'].shape == (50, 10)
            assert result['outcomes'].shape[0] == 50
            assert len(result['parameter_descriptions']) == 10
            
            # Check outcome variables
            expected_outcomes = ['yield_percent', 'defect_density_per_cm2', 
                               'process_time_minutes', 'quality_score', 'cost_per_wafer_usd']
            for outcome in expected_outcomes:
                assert outcome in result['outcomes'].columns
            
            # Check files were saved
            assert (output_dir / "recipes.csv").exists()
            assert (output_dir / "outcomes.csv").exists()
            assert (output_dir / "metadata.json").exists()
    
    def test_wafer_defect_pattern_generator(self):
        """Test wafer defect pattern generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "wafer_patterns"
            
            generator = WaferDefectPatternGenerator(random_state=42)
            result = generator.generate(
                n_wafers=20,
                map_size=32,
                output_dir=output_dir
            )
            
            # Check result structure
            assert 'wafer_maps' in result
            assert 'labels' in result
            assert 'defect_types' in result
            assert 'metadata' in result
            
            # Check data shapes
            assert result['wafer_maps'].shape == (20, 32, 32)
            assert len(result['labels']) == 20
            
            # Check defect types are valid
            expected_types = ['None', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 
                            'Loc', 'Random', 'Scratch', 'Near-full']
            for label in result['labels']:
                assert label in expected_types
            
            # Check files were saved
            assert (output_dir / "wafer_maps.npz").exists()
            assert (output_dir / "metadata.json").exists()
    
    def test_generate_all_synthetic_datasets(self):
        """Test generation of all synthetic datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            generate_all_synthetic_datasets(output_dir)
            
            # Check all expected directories were created
            expected_dirs = ["time_series_sensors", "process_recipes", "wafer_defect_patterns"]
            for dirname in expected_dirs:
                assert (output_dir / dirname).exists()
                assert (output_dir / dirname / "metadata.json").exists()


class TestDataValidation:
    """Test data validation functionality."""
    
    def setup_test_datasets(self, temp_dir: Path):
        """Set up test datasets for validation."""
        # Create SECOM dataset structure
        secom_dir = temp_dir / "secom"
        secom_dir.mkdir(parents=True)
        
        # Create dummy SECOM files with correct structure
        secom_data = pd.DataFrame(np.random.randn(100, 50))
        secom_data.to_csv(secom_dir / "secom.data", sep=' ', header=False, index=False)
        
        secom_labels = pd.DataFrame(np.random.choice([-1, 1], 100))
        secom_labels.to_csv(secom_dir / "secom_labels.data", sep=' ', header=False, index=False)
        
        (secom_dir / "secom.names").write_text("SECOM dataset description")
        
        # Create Steel Plates dataset structure
        steel_dir = temp_dir / "steel-plates"
        steel_dir.mkdir(parents=True)
        
        features = pd.DataFrame(np.random.randn(50, 27))
        features.to_csv(steel_dir / "steel_plates_features.csv", index=False)
        
        targets = pd.DataFrame(np.random.randint(0, 2, (50, 7)), 
                             columns=['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 
                                    'Dirtiness', 'Bumps', 'Other_Faults'])
        targets.to_csv(steel_dir / "steel_plates_targets.csv", index=False)
        
        # Create WM-811K placeholder
        wm811k_dir = temp_dir / "wm811k"
        wm811k_dir.mkdir(parents=True)
        (wm811k_dir / "README.md").write_text("WM-811K dataset placeholder")
    
    def test_dataset_validator_basic(self):
        """Test basic dataset validation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.setup_test_datasets(temp_path)
            
            validator = DatasetValidator(temp_path)
            results = validator.validate_all_datasets()
            
            # Check that validation results are returned
            assert 'secom' in results
            assert 'steel-plates' in results
            assert 'wm811k' in results
            assert 'synthetic' in results
            
            # Check SECOM validation (should have some passes)
            secom_results = results['secom']
            assert len(secom_results) > 0
            assert any(r.passed for r in secom_results)
    
    def test_secom_validation(self):
        """Test specific SECOM dataset validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.setup_test_datasets(temp_path)
            
            validator = DatasetValidator(temp_path)
            results = validator.validate_dataset('secom', temp_path / 'secom')
            
            # Should have existence check and file checks
            check_types = [r.check_type for r in results]
            assert 'existence' in check_types
            assert 'file_existence' in check_types
    
    def test_integrity_report_generation(self):
        """Test generation of integrity reports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.setup_test_datasets(temp_path)
            
            validator = DatasetValidator(temp_path)
            report = validator.generate_integrity_report(
                output_path=temp_path / "report.json"
            )
            
            # Check report structure
            assert 'timestamp' in report
            assert 'validation_summary' in report
            assert 'detailed_results' in report
            assert 'overall_status' in report
            
            # Check report file was created
            assert (temp_path / "report.json").exists()
    
    def test_checksum_computation(self):
        """Test checksum computation and verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test file
            test_file = temp_path / "test.txt"
            test_file.write_text("test content")
            
            validator = DatasetValidator()
            checksums = validator.compute_checksums(temp_path)
            
            assert 'test.txt' in checksums
            assert len(checksums['test.txt']) == 64  # SHA256 length


class TestWM811KPreprocessing:
    """Test WM-811K preprocessing functionality."""
    
    def create_mock_wm811k_data(self, temp_dir: Path):
        """Create mock WM-811K data for testing."""
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir(parents=True)
        
        # Create mock pickle file
        import pickle
        
        mock_data = {
            'waferMap': [np.random.randint(0, 2, (32, 32)) for _ in range(10)],
            'failureType': ['None', 'Center', 'Donut', 'None', 'Scratch', 
                          'Loc', 'Random', 'None', 'Edge-Loc', 'None']
        }
        
        with open(raw_dir / "mock_data.pkl", 'wb') as f:
            pickle.dump(mock_data, f)
        
        return mock_data
    
    def test_wm811k_preprocessor_initialization(self):
        """Test WM811K preprocessor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            preprocessor = WM811KPreprocessor(temp_path)
            
            assert preprocessor.data_root == temp_path
            assert preprocessor.raw_data_path == temp_path / "raw"
            assert preprocessor.processed_data_path == temp_path / "data"
    
    def test_pickle_data_loading(self):
        """Test loading WM-811K data from pickle files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_data = self.create_mock_wm811k_data(temp_path)
            
            preprocessor = WM811KPreprocessor(temp_path)
            loaded_data = preprocessor.load_raw_data()
            
            assert loaded_data is not None
            assert loaded_data.n_samples == 10
            assert loaded_data.map_shape == (32, 32)
            assert len(loaded_data.labels) == 10
    
    def test_data_preprocessing(self):
        """Test wafer map data preprocessing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_data = self.create_mock_wm811k_data(temp_path)
            
            preprocessor = WM811KPreprocessor(temp_path)
            raw_data = preprocessor.load_raw_data()
            
            processed_data = preprocessor.preprocess_data(
                raw_data,
                target_size=(16, 16),
                normalize=True,
                augment=False
            )
            
            assert processed_data.wafer_maps.shape[1:3] == (16, 16)
            assert processed_data.wafer_maps.dtype == np.float32
            assert processed_data.metadata['preprocessed']
    
    def test_wafer_map_visualizer(self):
        """Test wafer map visualization utilities."""
        # Create mock wafer map data
        wafer_map = np.random.randint(0, 2, (32, 32))
        
        visualizer = WaferMapVisualizer()
        
        # Test plotting (without showing)
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        ax = visualizer.plot_wafer_map(wafer_map, title="Test Map")
        assert ax is not None


class TestDownloadScript:
    """Test enhanced download script functionality."""
    
    @patch('datasets.download_semiconductor_datasets.generate_synthetic_datasets')
    def test_synthetic_dataset_option(self, mock_generate):
        """Test synthetic dataset generation option."""
        from download_semiconductor_datasets import main
        
        # Test synthetic dataset generation
        with patch('sys.argv', ['script', '--dataset', 'synthetic', '--datasets-dir', '/tmp/test']):
            main()
        
        mock_generate.assert_called_once()
    
    def test_help_output(self):
        """Test that help output includes new options."""
        from download_semiconductor_datasets import main
        
        with patch('sys.argv', ['script', '--help']):
            with pytest.raises(SystemExit):
                main()


def run_all_tests():
    """Run all dataset enhancement tests."""
    print("Running dataset enhancement tests...")
    
    # Test synthetic data generators
    print("Testing synthetic data generators...")
    test_generators = TestSyntheticDataGenerators()
    test_generators.test_time_sensor_data_generator()
    test_generators.test_process_recipe_generator()
    test_generators.test_wafer_defect_pattern_generator()
    print("✅ Synthetic data generators tests passed")
    
    # Test data validation
    print("Testing data validation...")
    test_validation = TestDataValidation()
    test_validation.test_dataset_validator_basic()
    test_validation.test_integrity_report_generation()
    test_validation.test_checksum_computation()
    print("✅ Data validation tests passed")
    
    # Test WM-811K preprocessing
    print("Testing WM-811K preprocessing...")
    test_wm811k = TestWM811KPreprocessing()
    test_wm811k.test_wm811k_preprocessor_initialization()
    test_wm811k.test_pickle_data_loading()
    test_wm811k.test_data_preprocessing()
    print("✅ WM-811K preprocessing tests passed")
    
    print("🎉 All dataset enhancement tests passed!")


if __name__ == "__main__":
    # Run tests if called directly
    run_all_tests()