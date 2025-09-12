"""Production Model Deployment Pipeline Script for Module 9.1

Provides a CLI to export trained models in a format compatible with the FastAPI deployment service.
Demonstrates model packaging, metadata generation, and artifact preparation for deployment.

Features:
- Export existing models to deployment-ready format
- Generate model metadata with versioning and schema information
- Validate model artifacts for API compatibility
- JSON-only output for automation integration
- Compatible with FastAPI service model loading

Example usage:
    python 9.1-model-deployment-pipeline.py export --model-path ../../../temp_models/regression_model.joblib --output-dir ./deployment_artifacts
    python 9.1-model-deployment-pipeline.py validate --deployment-dir ./deployment_artifacts
    python 9.1-model-deployment-pipeline.py package --model-path ../../../temp_models/regression_model.joblib --version 1.0.0
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import hashlib
import datetime

import joblib
import numpy as np
import pandas as pd

RANDOM_SEED = 42


@dataclass
class ModelDeploymentMetadata:
    """Metadata for deployed models."""
    model_name: str
    version: str
    created_at: str
    model_type: str
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    model_hash: str
    performance_metrics: Dict[str, float]
    deployment_config: Dict[str, Any]


class ModelDeploymentPipeline:
    """Pipeline for preparing models for deployment."""
    
    def __init__(self):
        self.metadata: Optional[ModelDeploymentMetadata] = None
    
    def export_model(self, model_path: Path, output_dir: Path, version: str = "1.0.0") -> Dict[str, Any]:
        """Export a trained model to deployment-ready format."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the model to extract metadata
        try:
            model_obj = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Extract model information
        model_info = self._extract_model_info(model_obj)
        
        # Copy model to deployment directory
        deployment_model_path = output_dir / "model.joblib"
        joblib.dump(model_obj, deployment_model_path)
        
        # Calculate model hash for integrity checking (on the copied model)
        model_hash = self._calculate_file_hash(deployment_model_path)
        
        # Generate metadata
        self.metadata = ModelDeploymentMetadata(
            model_name=model_info.get('name', 'unknown'),
            version=version,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model_type=model_info.get('type', 'unknown'),
            input_schema=model_info.get('input_schema', {}),
            output_schema=model_info.get('output_schema', {}),
            model_hash=model_hash,
            performance_metrics=model_info.get('metrics', {}),
            deployment_config={
                'api_version': 'v1',
                'max_batch_size': 100,
                'timeout_seconds': 30
            }
        )
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(self.metadata), f, indent=2)
        
        return {
            'status': 'exported',
            'output_dir': str(output_dir),
            'model_path': str(deployment_model_path),
            'metadata_path': str(metadata_path),
            'version': version,
            'model_hash': model_hash
        }
    
    def validate_deployment(self, deployment_dir: Path) -> Dict[str, Any]:
        """Validate a deployment artifact directory."""
        if not deployment_dir.exists():
            raise FileNotFoundError(f"Deployment directory not found: {deployment_dir}")
        
        model_path = deployment_dir / "model.joblib"
        metadata_path = deployment_dir / "metadata.json"
        
        issues = []
        
        # Check required files
        if not model_path.exists():
            issues.append("Missing model.joblib file")
        if not metadata_path.exists():
            issues.append("Missing metadata.json file")
        
        if issues:
            return {
                'status': 'invalid',
                'issues': issues
            }
        
        # Validate model can be loaded
        try:
            model_obj = joblib.load(model_path)
        except Exception as e:
            issues.append(f"Model cannot be loaded: {e}")
        
        # Validate metadata format
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            required_fields = ['model_name', 'version', 'model_type', 'model_hash']
            for field in required_fields:
                if field not in metadata:
                    issues.append(f"Missing metadata field: {field}")
        except Exception as e:
            issues.append(f"Invalid metadata format: {e}")
        
        # Validate model hash
        if not issues:
            expected_hash = metadata.get('model_hash')
            actual_hash = self._calculate_file_hash(model_path)
            if expected_hash != actual_hash:
                issues.append(f"Model hash mismatch: expected {expected_hash}, got {actual_hash}")
        
        return {
            'status': 'valid' if not issues else 'invalid',
            'issues': issues,
            'deployment_dir': str(deployment_dir)
        }
    
    def package_model(self, model_path: Path, version: str = "1.0.0", output_name: Optional[str] = None) -> Dict[str, Any]:
        """Package model for deployment with automatic naming."""
        if output_name is None:
            output_name = f"deployment_v{version.replace('.', '_')}"
        
        output_dir = model_path.parent / output_name
        return self.export_model(model_path, output_dir, version)
    
    def _extract_model_info(self, model_obj: Any) -> Dict[str, Any]:
        """Extract information from a loaded model object."""
        info = {
            'name': 'model',
            'type': 'unknown',
            'input_schema': {},
            'output_schema': {},
            'metrics': {}
        }
        
        # Try to extract information from common model types
        if hasattr(model_obj, '__class__'):
            class_name = model_obj.__class__.__name__
            info['type'] = class_name
            info['name'] = class_name.lower()
        
        # If it's a pipeline or has metadata attribute
        if hasattr(model_obj, 'metadata'):
            metadata = model_obj.metadata
            if hasattr(metadata, '__dict__'):
                metadata_dict = asdict(metadata) if hasattr(metadata, '__dataclass_fields__') else metadata.__dict__
                if 'model' in metadata_dict:
                    info['type'] = metadata_dict['model']
                if 'trained_at' in metadata_dict:
                    info['trained_at'] = metadata_dict['trained_at']
        
        # Default schema for semiconductor data
        info['input_schema'] = {
            'temperature': 'float',
            'pressure': 'float',
            'flow': 'float',
            'time': 'float'
        }
        info['output_schema'] = {
            'prediction': 'float',
            'confidence': 'float'
        }
        
        return info
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


# CLI Functions
def action_export(args):
    """Export model for deployment."""
    try:
        pipeline = ModelDeploymentPipeline()
        result = pipeline.export_model(
            model_path=Path(args.model_path),
            output_dir=Path(args.output_dir),
            version=args.version
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        sys.exit(1)


def action_validate(args):
    """Validate deployment artifacts."""
    try:
        pipeline = ModelDeploymentPipeline()
        result = pipeline.validate_deployment(Path(args.deployment_dir))
        print(json.dumps(result, indent=2))
        if result['status'] == 'invalid':
            sys.exit(1)
    except Exception as e:
        print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        sys.exit(1)


def action_package(args):
    """Package model with automatic naming."""
    try:
        pipeline = ModelDeploymentPipeline()
        result = pipeline.package_model(
            model_path=Path(args.model_path),
            version=args.version,
            output_name=args.output_name
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        sys.exit(1)


def build_parser():
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description='Module 9.1 - Model Deployment Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export model for deployment
  python 9.1-model-deployment-pipeline.py export --model-path model.joblib --output-dir ./deployment --version 1.0.0
  
  # Validate deployment artifacts
  python 9.1-model-deployment-pipeline.py validate --deployment-dir ./deployment
  
  # Package model with automatic naming
  python 9.1-model-deployment-pipeline.py package --model-path model.joblib --version 1.2.0
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')
    
    # Export subcommand
    parser_export = subparsers.add_parser('export', help='Export model for deployment')
    parser_export.add_argument('--model-path', required=True, help='Path to trained model file')
    parser_export.add_argument('--output-dir', required=True, help='Output directory for deployment artifacts')
    parser_export.add_argument('--version', default='1.0.0', help='Model version (default: 1.0.0)')
    parser_export.set_defaults(func=action_export)
    
    # Validate subcommand
    parser_validate = subparsers.add_parser('validate', help='Validate deployment artifacts')
    parser_validate.add_argument('--deployment-dir', required=True, help='Directory containing deployment artifacts')
    parser_validate.set_defaults(func=action_validate)
    
    # Package subcommand
    parser_package = subparsers.add_parser('package', help='Package model with automatic naming')
    parser_package.add_argument('--model-path', required=True, help='Path to trained model file')
    parser_package.add_argument('--version', default='1.0.0', help='Model version (default: 1.0.0)')
    parser_package.add_argument('--output-name', help='Output directory name (auto-generated if not provided)')
    parser_package.set_defaults(func=action_package)
    
    return parser


def main(argv: Optional[List[str]] = None):
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()