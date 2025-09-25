"""Data validation and integrity checking utilities for semiconductor datasets.

This module provides comprehensive data validation, integrity checking, and quality
assessment tools for all datasets in the repository. It ensures data consistency,
validates formats, and provides quality metrics for both real and synthetic datasets.

Features:
- Dataset integrity validation with checksums
- Data quality assessment and metrics
- Schema validation for structured data
- Cross-dataset compatibility checking
- Automated testing integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a dataset validation check."""
    dataset_name: str
    check_type: str
    passed: bool
    message: str
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics."""
    completeness: float  # % of non-null values
    consistency: float   # Schema/format consistency score
    validity: float      # Data validation score  
    accuracy: float      # Accuracy assessment (if ground truth available)
    uniqueness: float    # Uniqueness of records
    timeliness: float    # Data freshness score
    overall_score: float # Composite quality score
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetValidator:
    """Comprehensive dataset validation and integrity checking."""
    
    def __init__(self, datasets_root: Path = Path("datasets")):
        self.datasets_root = Path(datasets_root)
        self.validation_results: List[ValidationResult] = []
        
    def validate_all_datasets(self) -> Dict[str, List[ValidationResult]]:
        """Validate all datasets in the repository."""
        results = {}
        
        # Check for expected dataset directories
        expected_datasets = ["secom", "steel-plates", "wm811k", "synthetic"]
        
        for dataset_name in expected_datasets:
            dataset_path = self.datasets_root / dataset_name
            results[dataset_name] = self.validate_dataset(dataset_name, dataset_path)
        
        return results
    
    def validate_dataset(self, name: str, path: Path) -> List[ValidationResult]:
        """Validate a single dataset."""
        results = []
        
        # Check if dataset directory exists
        if not path.exists():
            results.append(ValidationResult(
                dataset_name=name,
                check_type="existence",
                passed=False,
                message=f"Dataset directory {path} does not exist"
            ))
            return results
        
        results.append(ValidationResult(
            dataset_name=name,
            check_type="existence",
            passed=True,
            message=f"Dataset directory found at {path}"
        ))
        
        # Validate based on dataset type
        if name == "secom":
            results.extend(self._validate_secom(path))
        elif name == "steel-plates":
            results.extend(self._validate_steel_plates(path))
        elif name == "wm811k":
            results.extend(self._validate_wm811k(path))
        elif name == "synthetic":
            results.extend(self._validate_synthetic(path))
        
        return results
    
    def _validate_secom(self, path: Path) -> List[ValidationResult]:
        """Validate SECOM dataset."""
        results = []
        
        # Check required files
        required_files = ["secom.data", "secom_labels.data", "secom.names"]
        
        for filename in required_files:
            filepath = path / filename
            if filepath.exists():
                results.append(ValidationResult(
                    dataset_name="secom",
                    check_type="file_existence",
                    passed=True,
                    message=f"Found required file: {filename}"
                ))
                
                # Validate file format and content
                if filename == "secom.data":
                    results.extend(self._validate_secom_data_file(filepath))
                elif filename == "secom_labels.data":
                    results.extend(self._validate_secom_labels_file(filepath))
                    
            else:
                results.append(ValidationResult(
                    dataset_name="secom",
                    check_type="file_existence",
                    passed=False,
                    message=f"Missing required file: {filename}"
                ))
        
        return results
    
    def _validate_secom_data_file(self, filepath: Path) -> List[ValidationResult]:
        """Validate SECOM data file format and content."""
        results = []
        
        try:
            # Read data and validate dimensions
            data = pd.read_csv(filepath, sep=r'\s+', header=None)
            
            expected_shape = (1567, 590)  # Known SECOM dimensions
            if data.shape == expected_shape:
                results.append(ValidationResult(
                    dataset_name="secom",
                    check_type="data_format",
                    passed=True,
                    message=f"SECOM data has correct shape: {data.shape}",
                    details={"shape": data.shape, "expected_shape": expected_shape}
                ))
            else:
                results.append(ValidationResult(
                    dataset_name="secom",
                    check_type="data_format",
                    passed=False,
                    message=f"SECOM data has incorrect shape: {data.shape}, expected: {expected_shape}",
                    details={"shape": data.shape, "expected_shape": expected_shape}
                ))
            
            # Check for missing values
            missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            results.append(ValidationResult(
                dataset_name="secom",
                check_type="data_quality",
                passed=True,
                message=f"SECOM data missing values: {missing_pct:.2f}%",
                details={"missing_percentage": missing_pct}
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                dataset_name="secom",
                check_type="data_format",
                passed=False,
                message=f"Failed to validate SECOM data file: {str(e)}"
            ))
        
        return results
    
    def _validate_secom_labels_file(self, filepath: Path) -> List[ValidationResult]:
        """Validate SECOM labels file."""
        results = []
        
        try:
            labels = pd.read_csv(filepath, sep=r'\s+', header=None)
            
            # Check dimensions
            expected_length = 1567
            if len(labels) == expected_length:
                results.append(ValidationResult(
                    dataset_name="secom",
                    check_type="labels_format",
                    passed=True,
                    message=f"SECOM labels have correct length: {len(labels)}"
                ))
            else:
                results.append(ValidationResult(
                    dataset_name="secom",
                    check_type="labels_format",
                    passed=False,
                    message=f"SECOM labels have incorrect length: {len(labels)}, expected: {expected_length}"
                ))
            
            # Check label values
            unique_labels = labels.iloc[:, 0].unique()
            expected_labels = [-1, 1]  # Pass/Fail labels
            if set(unique_labels).issubset(set(expected_labels)):
                results.append(ValidationResult(
                    dataset_name="secom",
                    check_type="labels_values",
                    passed=True,
                    message=f"SECOM labels have valid values: {sorted(unique_labels)}"
                ))
            else:
                results.append(ValidationResult(
                    dataset_name="secom",
                    check_type="labels_values",
                    passed=False,
                    message=f"SECOM labels have unexpected values: {sorted(unique_labels)}"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                dataset_name="secom",
                check_type="labels_format",
                passed=False,
                message=f"Failed to validate SECOM labels file: {str(e)}"
            ))
        
        return results
    
    def _validate_steel_plates(self, path: Path) -> List[ValidationResult]:
        """Validate Steel Plates dataset."""
        results = []
        
        # Check for CSV files
        features_file = path / "steel_plates_features.csv"
        targets_file = path / "steel_plates_targets.csv"
        
        if features_file.exists():
            try:
                features = pd.read_csv(features_file)
                results.append(ValidationResult(
                    dataset_name="steel-plates",
                    check_type="features_format",
                    passed=True,
                    message=f"Steel plates features loaded: {features.shape}",
                    details={"shape": features.shape}
                ))
            except Exception as e:
                results.append(ValidationResult(
                    dataset_name="steel-plates",
                    check_type="features_format",
                    passed=False,
                    message=f"Failed to load steel plates features: {str(e)}"
                ))
        else:
            results.append(ValidationResult(
                dataset_name="steel-plates",
                check_type="file_existence",
                passed=False,
                message="Missing steel_plates_features.csv file"
            ))
        
        if targets_file.exists():
            try:
                targets = pd.read_csv(targets_file)
                results.append(ValidationResult(
                    dataset_name="steel-plates",
                    check_type="targets_format",
                    passed=True,
                    message=f"Steel plates targets loaded: {targets.shape}",
                    details={"shape": targets.shape}
                ))
            except Exception as e:
                results.append(ValidationResult(
                    dataset_name="steel-plates",
                    check_type="targets_format",
                    passed=False,
                    message=f"Failed to load steel plates targets: {str(e)}"
                ))
        else:
            results.append(ValidationResult(
                dataset_name="steel-plates",
                check_type="file_existence",
                passed=False,
                message="Missing steel_plates_targets.csv file"
            ))
        
        return results
    
    def _validate_wm811k(self, path: Path) -> List[ValidationResult]:
        """Validate WM-811K dataset (may be placeholder)."""
        results = []
        
        # Check if data directory exists (actual data)
        data_dir = path / "data"
        raw_dir = path / "raw"
        
        if data_dir.exists() and list(data_dir.iterdir()):
            results.append(ValidationResult(
                dataset_name="wm811k",
                check_type="data_availability",
                passed=True,
                message="WM-811K data directory found with content"
            ))
        elif raw_dir.exists() and list(raw_dir.iterdir()):
            results.append(ValidationResult(
                dataset_name="wm811k",
                check_type="data_availability",
                passed=True,
                message="WM-811K raw data found (needs preprocessing)"
            ))
        else:
            results.append(ValidationResult(
                dataset_name="wm811k",
                check_type="data_availability",
                passed=False,
                message="WM-811K data not available (placeholder only)"
            ))
        
        # Check for README files
        readme_files = list(path.glob("README*"))
        if readme_files:
            results.append(ValidationResult(
                dataset_name="wm811k",
                check_type="documentation",
                passed=True,
                message=f"Found {len(readme_files)} documentation files"
            ))
        
        return results
    
    def _validate_synthetic(self, path: Path) -> List[ValidationResult]:
        """Validate synthetic datasets."""
        results = []
        
        expected_subdirs = ["time_series_sensors", "process_recipes", "wafer_defect_patterns"]
        
        for subdir_name in expected_subdirs:
            subdir_path = path / subdir_name
            if subdir_path.exists():
                results.append(ValidationResult(
                    dataset_name="synthetic",
                    check_type="synthetic_data",
                    passed=True,
                    message=f"Found synthetic dataset: {subdir_name}"
                ))
                
                # Check for metadata file
                metadata_file = subdir_path / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        results.append(ValidationResult(
                            dataset_name="synthetic",
                            check_type="metadata",
                            passed=True,
                            message=f"Valid metadata for {subdir_name}",
                            details=metadata
                        ))
                    except Exception as e:
                        results.append(ValidationResult(
                            dataset_name="synthetic",
                            check_type="metadata",
                            passed=False,
                            message=f"Invalid metadata for {subdir_name}: {str(e)}"
                        ))
            else:
                results.append(ValidationResult(
                    dataset_name="synthetic",
                    check_type="synthetic_data", 
                    passed=False,
                    message=f"Missing synthetic dataset: {subdir_name}"
                ))
        
        return results
    
    def calculate_data_quality_metrics(self, dataset_name: str, data: pd.DataFrame) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics."""
        
        # Completeness: percentage of non-null values
        total_cells = data.shape[0] * data.shape[1]
        non_null_cells = data.count().sum()
        completeness = (non_null_cells / total_cells) * 100
        
        # Consistency: schema and format consistency
        # Check for consistent data types in each column
        consistency_score = 100.0  # Start with perfect score
        for col in data.columns:
            try:
                # Check if numeric columns have consistent numeric types
                if data[col].dtype in ['int64', 'float64']:
                    pd.to_numeric(data[col], errors='raise')
            except:
                consistency_score -= (100 / len(data.columns))
        
        # Validity: data validation against expected ranges/formats
        validity_score = 100.0
        # Basic validity checks (customize per dataset)
        for col in data.select_dtypes(include=[np.number]).columns:
            # Check for extreme outliers (beyond 5 std devs)
            if len(data[col].dropna()) > 0:
                mean_val = data[col].mean()
                std_val = data[col].std()
                if std_val > 0:
                    outliers = np.abs((data[col] - mean_val) / std_val) > 5
                    outlier_rate = outliers.sum() / len(data[col])
                    validity_score -= outlier_rate * 10  # Penalize outliers
        
        # Accuracy: cannot be determined without ground truth
        accuracy = 100.0  # Assume perfect for synthetic data
        
        # Uniqueness: percentage of unique records
        if len(data) > 0:
            unique_rows = len(data.drop_duplicates())
            uniqueness = (unique_rows / len(data)) * 100
        else:
            uniqueness = 100.0
        
        # Timeliness: assume current for synthetic data
        timeliness = 100.0
        
        # Overall score: weighted average
        weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'validity': 0.20,
            'accuracy': 0.15,
            'uniqueness': 0.10,
            'timeliness': 0.10
        }
        
        overall_score = (
            completeness * weights['completeness'] +
            consistency_score * weights['consistency'] +
            validity_score * weights['validity'] +
            accuracy * weights['accuracy'] +
            uniqueness * weights['uniqueness'] +
            timeliness * weights['timeliness']
        )
        
        return DataQualityMetrics(
            completeness=completeness,
            consistency=consistency_score,
            validity=validity_score,
            accuracy=accuracy,
            uniqueness=uniqueness,
            timeliness=timeliness,
            overall_score=overall_score
        )
    
    def generate_integrity_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive dataset integrity report."""
        
        validation_results = self.validate_all_datasets()
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "datasets_root": str(self.datasets_root),
            "validation_summary": {},
            "detailed_results": validation_results,
            "overall_status": "PASS"
        }
        
        # Generate summary
        for dataset_name, results in validation_results.items():
            passed_checks = sum(1 for r in results if r.passed)
            total_checks = len(results)
            
            report["validation_summary"][dataset_name] = {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "success_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0,
                "status": "PASS" if passed_checks == total_checks else "FAIL"
            }
            
            if passed_checks < total_checks:
                report["overall_status"] = "FAIL"
        
        # Save report if path specified
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Integrity report saved to {output_path}")
        
        return report
    
    def compute_checksums(self, dataset_path: Path) -> Dict[str, str]:
        """Compute SHA256 checksums for all files in a dataset."""
        checksums = {}
        
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.sha256()
                        for chunk in iter(lambda: f.read(8192), b""):
                            file_hash.update(chunk)
                    
                    # Store relative path as key
                    rel_path = file_path.relative_to(dataset_path)
                    checksums[str(rel_path)] = file_hash.hexdigest()
                except Exception as e:
                    logger.warning(f"Could not compute checksum for {file_path}: {e}")
        
        return checksums
    
    def verify_checksums(self, dataset_path: Path, expected_checksums: Dict[str, str]) -> List[ValidationResult]:
        """Verify file checksums against expected values."""
        results = []
        current_checksums = self.compute_checksums(dataset_path)
        
        for file_path, expected_checksum in expected_checksums.items():
            if file_path in current_checksums:
                if current_checksums[file_path] == expected_checksum:
                    results.append(ValidationResult(
                        dataset_name=dataset_path.name,
                        check_type="checksum",
                        passed=True,
                        message=f"Checksum verified for {file_path}"
                    ))
                else:
                    results.append(ValidationResult(
                        dataset_name=dataset_path.name,
                        check_type="checksum",
                        passed=False,
                        message=f"Checksum mismatch for {file_path}"
                    ))
            else:
                results.append(ValidationResult(
                    dataset_name=dataset_path.name,
                    check_type="checksum",
                    passed=False,
                    message=f"Missing file for checksum verification: {file_path}"
                ))
        
        return results


def run_dataset_validation(datasets_root: Path = Path("datasets"), 
                         output_report: bool = True) -> Dict[str, Any]:
    """Run comprehensive dataset validation and return results."""
    
    validator = DatasetValidator(datasets_root)
    
    print("Running comprehensive dataset validation...")
    report = validator.generate_integrity_report(
        output_path=datasets_root / "validation_report.json" if output_report else None
    )
    
    # Print summary
    print("\nValidation Summary:")
    print("-" * 50)
    for dataset_name, summary in report["validation_summary"].items():
        status_icon = "✅" if summary["status"] == "PASS" else "❌"
        print(f"{status_icon} {dataset_name}: {summary['passed_checks']}/{summary['total_checks']} checks passed "
              f"({summary['success_rate']:.1f}%)")
    
    overall_icon = "✅" if report["overall_status"] == "PASS" else "❌"
    print(f"\n{overall_icon} Overall Status: {report['overall_status']}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate dataset integrity and quality")
    parser.add_argument("--datasets-root", default="datasets", 
                       help="Root directory containing datasets")
    parser.add_argument("--output-report", action="store_true", default=True,
                       help="Generate detailed JSON report")
    parser.add_argument("--dataset", help="Validate specific dataset only")
    
    args = parser.parse_args()
    
    datasets_root = Path(args.datasets_root)
    
    if args.dataset:
        # Validate specific dataset
        validator = DatasetValidator(datasets_root)
        results = validator.validate_dataset(args.dataset, datasets_root / args.dataset)
        
        print(f"\nValidation results for {args.dataset}:")
        for result in results:
            status_icon = "✅" if result.passed else "❌"
            print(f"{status_icon} {result.check_type}: {result.message}")
    else:
        # Validate all datasets
        run_dataset_validation(datasets_root, args.output_report)