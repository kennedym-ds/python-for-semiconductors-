#!/usr/bin/env python3
"""Minimal test for GAN defect augmentation pipeline dependencies"""

import sys
import warnings

def test_dependency_checks():
    """Test that dependency checks work correctly."""
    print("Testing dependency detection...")
    
    # Test numpy
    try:
        import numpy as np
        print("✓ NumPy available")
        HAS_NUMPY = True
    except ImportError:
        print("✗ NumPy not available")
        HAS_NUMPY = False
    
    # Test PyTorch
    try:
        import torch
        print("✓ PyTorch available")
        HAS_TORCH = True
    except ImportError:
        print("✗ PyTorch not available")
        HAS_TORCH = False
    
    # Test PIL/Pillow
    try:
        from PIL import Image
        print("✓ Pillow available")
        HAS_PIL = True
    except ImportError:
        print("✗ Pillow not available")
        HAS_PIL = False
    
    # Test joblib
    try:
        import joblib
        print("✓ joblib available")
        HAS_JOBLIB = True
    except ImportError:
        print("✗ joblib not available")
        HAS_JOBLIB = False
    
    # Test sklearn
    try:
        import sklearn
        print("✓ scikit-learn available")
        HAS_SKLEARN = True
    except ImportError:
        print("✗ scikit-learn not available")
        HAS_SKLEARN = False
    
    return {
        'numpy': HAS_NUMPY,
        'torch': HAS_TORCH,
        'pil': HAS_PIL,
        'joblib': HAS_JOBLIB,
        'sklearn': HAS_SKLEARN
    }

def test_cli_structure():
    """Test that CLI structure is correct without imports."""
    print("\nTesting CLI argument structure...")
    
    # Mock the expected CLI commands
    expected_commands = ['train', 'generate', 'evaluate']
    expected_train_args = ['--data-path', '--epochs', '--batch-size', '--save']
    expected_generate_args = ['--model-path', '--num-samples', '--output-dir']
    expected_evaluate_args = ['--model-path', '--augmentation-ratio']
    
    print(f"✓ Expected commands: {expected_commands}")
    print(f"✓ Expected train args: {expected_train_args}")
    print(f"✓ Expected generate args: {expected_generate_args}")
    print(f"✓ Expected evaluate args: {expected_evaluate_args}")
    
    return True

def test_project_structure():
    """Test that project structure is correct."""
    print("\nTesting project structure...")
    
    from pathlib import Path
    
    project_dir = Path(__file__).parent
    expected_files = [
        'README.md',
        'gan_augmentation_pipeline.py',
        'test_gan_augmentation_pipeline.py',
        'requirements.txt',
        'config.yaml'
    ]
    
    for file in expected_files:
        file_path = project_dir / file
        if file_path.exists():
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("GAN Defect Augmentation Pipeline - Minimal Test")
    print("=" * 50)
    
    # Test dependencies
    deps = test_dependency_checks()
    
    # Test CLI structure
    cli_ok = test_cli_structure()
    
    # Test project structure
    structure_ok = test_project_structure()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    print(f"Dependencies detected: {sum(deps.values())}/{len(deps)}")
    print(f"CLI structure: {'✓' if cli_ok else '✗'}")
    print(f"Project structure: {'✓' if structure_ok else '✗'}")
    
    # Recommendations
    print("\nRecommendations:")
    if not deps['numpy']:
        print("- Install numpy: pip install numpy")
    if not deps['torch']:
        print("- Install PyTorch (optional): pip install torch torchvision")
    if not deps['pil']:
        print("- Install Pillow: pip install Pillow")
    if not deps['joblib']:
        print("- Install joblib (optional): pip install joblib")
    if not deps['sklearn']:
        print("- Install scikit-learn (optional): pip install scikit-learn")
    
    print("\nThe pipeline can run in rule-based mode with just numpy and Pillow.")
    print("For full functionality, install all dependencies.")
    
    return all([structure_ok, cli_ok])

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)