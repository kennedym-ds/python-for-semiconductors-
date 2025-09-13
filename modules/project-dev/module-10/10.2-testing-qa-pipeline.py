"""Production Testing & QA Pipeline Script for Module 10.2

Provides a comprehensive CLI-based testing and quality assurance framework
for semiconductor manufacturing ML pipelines. Demonstrates testing patterns,
coverage measurement, and quality gates.

Features:
- Test execution with coverage reporting
- Code quality checks (flake8, black)
- Dataset path validation
- Pipeline smoke testing
- Performance benchmarking
- Manufacturing-specific metric validation

Example usage:
    python 10.2-testing-qa-pipeline.py train --test-suite unit --coverage-threshold 80
    python 10.2-testing-qa-pipeline.py evaluate --target-module "modules.foundation"
    python 10.2-testing-qa-pipeline.py predict --check-type lint --fix-issues
"""

from __future__ import annotations
import argparse
import json
import sys
import subprocess
import time
import importlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------- QA Pipeline Core Classes -------------------- #


@dataclass
class QAMetrics:
    """Quality assurance metrics for ML pipelines."""

    test_pass_rate: float
    coverage_percentage: float
    lint_score: float
    performance_score: float
    total_tests: int
    failed_tests: int
    execution_time: float
    timestamp: str


@dataclass
class QAMetadata:
    """Metadata for QA pipeline execution."""

    python_version: str
    test_framework: str
    coverage_tool: str
    lint_tools: List[str]
    target_modules: List[str]
    quality_gates: Dict[str, float]
    random_seed: int = RANDOM_SEED


class TestingQAPipeline:
    """Production-grade testing and quality assurance pipeline."""

    def __init__(
        self,
        coverage_threshold: float = 80.0,
        lint_threshold: float = 90.0,
        performance_threshold: float = 30.0,
    ):
        self.coverage_threshold = coverage_threshold
        self.lint_threshold = lint_threshold
        self.performance_threshold = performance_threshold
        self.metadata = None
        self.last_results = None

    def run_tests(self, test_suite: str = "all", target_modules: Optional[List[str]] = None) -> QAMetrics:
        """Execute test suite with coverage measurement.

        Args:
            test_suite: Type of tests to run ('unit', 'integration', 'smoke', 'all')
            target_modules: Specific modules to test (None for all)

        Returns:
            QAMetrics with test results and coverage data
        """
        start_time = time.time()

        if target_modules is None:
            target_modules = ["modules/"]

        # Initialize metadata
        self.metadata = QAMetadata(
            python_version=sys.version.split()[0],
            test_framework="pytest",
            coverage_tool="pytest-cov",
            lint_tools=["flake8", "black"],
            target_modules=target_modules,
            quality_gates={
                "coverage": self.coverage_threshold,
                "lint": self.lint_threshold,
                "performance": self.performance_threshold,
            },
        )

        try:
            # Run tests based on suite type
            if test_suite == "unit":
                test_results = self._run_unit_tests(target_modules)
            elif test_suite == "integration":
                test_results = self._run_integration_tests(target_modules)
            elif test_suite == "smoke":
                test_results = self._run_smoke_tests(target_modules)
            elif test_suite == "all":
                test_results = self._run_all_tests(target_modules)
            else:
                raise ValueError(f"Unknown test suite type: {test_suite}")

            # Calculate metrics
            execution_time = time.time() - start_time
            pass_rate = ((test_results["total"] - test_results["failed"]) / max(test_results["total"], 1)) * 100

            metrics = QAMetrics(
                test_pass_rate=pass_rate,
                coverage_percentage=test_results.get("coverage", 0.0),
                lint_score=100.0,  # Will be updated by lint check
                performance_score=max(0, 100 - (execution_time / self.performance_threshold) * 100),
                total_tests=test_results["total"],
                failed_tests=test_results["failed"],
                execution_time=execution_time,
                timestamp=pd.Timestamp.now().isoformat(),
            )

            self.last_results = metrics
            return metrics

        except Exception:
            # Return failure metrics
            execution_time = time.time() - start_time
            return QAMetrics(
                test_pass_rate=0.0,
                coverage_percentage=0.0,
                lint_score=0.0,
                performance_score=0.0,
                total_tests=0,
                failed_tests=1,
                execution_time=execution_time,
                timestamp=pd.Timestamp.now().isoformat(),
            )

    def check_code_quality(self, target_path: str = ".", fix_issues: bool = False) -> Dict[str, Any]:
        """Run code quality checks using flake8 and black.

        Args:
            target_path: Path to check (default: current directory)
            fix_issues: Whether to automatically fix formatting issues

        Returns:
            Dict with lint results and scores
        """
        results = {
            "flake8": self._run_flake8(target_path),
            "black": self._run_black(target_path, fix=fix_issues),
            "overall_score": 0.0,
        }

        # Calculate overall lint score
        flake8_score = max(0, 100 - results["flake8"]["violations"] * 5)
        black_score = 100.0 if results["black"]["compliant"] else 70.0
        results["overall_score"] = (flake8_score + black_score) / 2

        return results

    def validate_dataset_paths(self, module_level: int = 3) -> Dict[str, Any]:
        """Validate that dataset paths are accessible from pipeline locations.

        Args:
            module_level: Number of directory levels to go up for datasets

        Returns:
            Dict with validation results for each dataset
        """
        # Resolve dataset directory
        script_dir = Path(__file__).parent
        data_dir = script_dir / ("../" * module_level) / "datasets"
        data_dir = data_dir.resolve()

        # Standard datasets to check
        datasets = {
            "secom": "secom/secom.data",
            "steel_plates": "steel-plates/faults.csv",
            "wm811k": "wm811k/README.md",  # Check for instructions
            "synthetic": None,  # Generated on-demand
        }

        results = {"dataset_root": str(data_dir), "datasets": {}, "all_valid": True}

        for name, path in datasets.items():
            if path is None:
                # Synthetic datasets are always "valid"
                results["datasets"][name] = {
                    "path": "synthetic",
                    "exists": True,
                    "type": "generated",
                }
            else:
                full_path = data_dir / path
                exists = full_path.exists()
                results["datasets"][name] = {
                    "path": str(full_path),
                    "exists": exists,
                    "type": "file",
                }
                if not exists:
                    results["all_valid"] = False

        return results

    def run_smoke_tests(self, target_modules: List[str]) -> Dict[str, Any]:
        """Run basic smoke tests to ensure system functionality.

        Args:
            target_modules: List of module paths to test

        Returns:
            Dict with smoke test results
        """
        results = {"import_tests": {}, "basic_functionality": {}, "all_passed": True}

        # Test critical imports
        critical_imports = [
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("scikit-learn", "sklearn"),
            ("matplotlib", "matplotlib"),
            ("jupyter", "jupyter"),
            ("pytest", "pytest"),
            ("black", "black"),
            ("flake8", "flake8"),
        ]

        for package_name, import_name in critical_imports:
            try:
                importlib.import_module(import_name)
                results["import_tests"][package_name] = {
                    "status": "pass",
                    "error": None,
                }
            except ImportError as e:
                results["import_tests"][package_name] = {
                    "status": "fail",
                    "error": str(e),
                }
                results["all_passed"] = False

        # Test basic pipeline functionality (if available)
        try:
            test_data = self._generate_test_data()
            results["basic_functionality"]["data_generation"] = {
                "status": "pass",
                "shape": test_data.shape,
            }
        except Exception as e:
            results["basic_functionality"]["data_generation"] = {
                "status": "fail",
                "error": str(e),
            }
            results["all_passed"] = False

        return results

    def benchmark_performance(self, target_operations: List[str]) -> Dict[str, Any]:
        """Benchmark performance of key pipeline operations.

        Args:
            target_operations: List of operations to benchmark

        Returns:
            Dict with performance metrics
        """
        results = {"benchmarks": {}, "overall_score": 0.0}

        for operation in target_operations:
            if operation == "data_generation":
                timing = self._benchmark_data_generation()
            elif operation == "model_training":
                timing = self._benchmark_model_training()
            elif operation == "prediction":
                timing = self._benchmark_prediction()
            else:
                timing = {"execution_time": 0.0, "status": "skipped"}

            results["benchmarks"][operation] = timing

        # Calculate overall performance score
        total_time = sum(b.get("execution_time", 0) for b in results["benchmarks"].values())
        results["overall_score"] = max(0, 100 - (total_time / self.performance_threshold) * 100)

        return results

    # -------------------- Private Helper Methods -------------------- #

    def _run_unit_tests(self, target_modules: List[str]) -> Dict[str, Any]:
        """Run unit tests for specified modules."""
        return self._execute_pytest(target_modules, "-k", "test_", "--tb=short")

    def _run_integration_tests(self, target_modules: List[str]) -> Dict[str, Any]:
        """Run integration tests for specified modules."""
        return self._execute_pytest(target_modules, "-k", "integration", "--tb=short")

    def _run_smoke_tests(self, target_modules: List[str]) -> Dict[str, Any]:
        """Run smoke tests for specified modules."""
        return self._execute_pytest(target_modules, "-k", "smoke", "--tb=short")

    def _run_all_tests(self, target_modules: List[str]) -> Dict[str, Any]:
        """Run all available tests."""
        return self._execute_pytest(target_modules, "--tb=short")

    def _execute_pytest(self, target_modules: List[str], *args) -> Dict[str, Any]:
        """Execute pytest with coverage and return parsed results."""
        try:
            # Build pytest command
            cmd = (
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    f"--cov={':'.join(target_modules)}",
                    "--cov-report=term-missing",
                    "--cov-report=json:/tmp/coverage.json",
                    "-v",
                ]
                + list(args)
                + target_modules
            )

            # Run pytest
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Parse results
            return self._parse_pytest_output(result.stdout, result.stderr, result.returncode)

        except subprocess.TimeoutExpired:
            return {
                "total": 0,
                "failed": 1,
                "coverage": 0.0,
                "error": "Test execution timeout",
            }
        except Exception as e:
            return {"total": 0, "failed": 1, "coverage": 0.0, "error": str(e)}

    def _parse_pytest_output(self, stdout: str, stderr: str, returncode: int) -> Dict[str, Any]:
        """Parse pytest output to extract test metrics."""
        results = {"total": 0, "failed": 0, "coverage": 0.0}

        # Parse test summary from stdout
        lines = stdout.split("\n")
        for line in lines:
            if "failed" in line and "passed" in line:
                # Example: "1 failed, 5 passed in 2.34s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "failed," and i > 0:
                        results["failed"] = int(parts[i - 1])
                    elif part == "passed" and i > 0:
                        passed = int(parts[i - 1])
                        results["total"] = results["failed"] + passed
            elif "passed in" in line and "failed" not in line:
                # Example: "5 passed in 2.34s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        results["total"] = int(parts[i - 1])

        # Try to read coverage from JSON report
        try:
            coverage_file = Path("/tmp/coverage.json")
            if coverage_file.exists():
                import json

                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    results["coverage"] = coverage_data.get("totals", {}).get("percent_covered", 0.0)
        except Exception:
            pass  # Coverage data not available

        return results

    def _run_flake8(self, target_path: str) -> Dict[str, Any]:
        """Run flake8 linting and return results."""
        try:
            cmd = [
                sys.executable,
                "-m",
                "flake8",
                target_path,
                "--count",
                "--statistics",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Count violations from output
            violations = 0
            if result.stdout:
                lines = result.stdout.strip().split("\n")
                if lines and lines[-1].isdigit():
                    violations = int(lines[-1])

            return {
                "violations": violations,
                "output": result.stdout,
                "errors": result.stderr,
                "compliant": violations == 0,
            }
        except Exception as e:
            return {
                "violations": 999,
                "output": "",
                "errors": str(e),
                "compliant": False,
            }

    def _run_black(self, target_path: str, fix: bool = False) -> Dict[str, Any]:
        """Run black formatting check/fix and return results."""
        try:
            cmd = [sys.executable, "-m", "black"]
            if not fix:
                cmd.append("--check")
            cmd.append(target_path)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            return {
                "compliant": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "action": "fixed" if fix else "checked",
            }
        except Exception as e:
            return {
                "compliant": False,
                "output": "",
                "errors": str(e),
                "action": "error",
            }

    def _generate_test_data(self) -> pd.DataFrame:
        """Generate synthetic test data for validation."""
        np.random.seed(RANDOM_SEED)
        return pd.DataFrame(
            {
                "temperature": np.random.normal(450, 15, 100),
                "pressure": np.random.normal(2.5, 0.3, 100),
                "flow": np.random.normal(120, 10, 100),
                "time": np.random.normal(60, 5, 100),
                "target": np.random.normal(85, 8, 100),
            }
        )

    def _benchmark_data_generation(self) -> Dict[str, Any]:
        """Benchmark data generation performance."""
        start_time = time.time()
        try:
            data = self._generate_test_data()
            execution_time = time.time() - start_time
            return {
                "execution_time": execution_time,
                "status": "success",
                "data_shape": data.shape,
            }
        except Exception as e:
            return {
                "execution_time": time.time() - start_time,
                "status": "error",
                "error": str(e),
            }

    def _benchmark_model_training(self) -> Dict[str, Any]:
        """Benchmark simple model training performance."""
        start_time = time.time()
        try:
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import train_test_split

            data = self._generate_test_data()
            X = data.drop("target", axis=1)
            y = data["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

            model = Ridge(random_state=RANDOM_SEED)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            execution_time = time.time() - start_time
            return {
                "execution_time": execution_time,
                "status": "success",
                "r2_score": score,
            }
        except Exception as e:
            return {
                "execution_time": time.time() - start_time,
                "status": "error",
                "error": str(e),
            }

    def _benchmark_prediction(self) -> Dict[str, Any]:
        """Benchmark prediction performance."""
        start_time = time.time()
        try:
            from sklearn.linear_model import Ridge

            data = self._generate_test_data()
            X = data.drop("target", axis=1)
            y = data["target"]

            model = Ridge(random_state=RANDOM_SEED)
            model.fit(X, y)

            # Single prediction benchmark
            pred_start = time.time()
            prediction = model.predict(X.iloc[[0]])
            pred_time = time.time() - pred_start

            total_time = time.time() - start_time
            return {
                "execution_time": total_time,
                "prediction_latency": pred_time,
                "status": "success",
                "prediction": float(prediction[0]),
            }
        except Exception as e:
            return {
                "execution_time": time.time() - start_time,
                "status": "error",
                "error": str(e),
            }

    # -------------------- Persistence Methods -------------------- #

    def save(self, path: Path) -> None:
        """Save QA pipeline configuration and results."""
        save_data = {
            "metadata": asdict(self.metadata) if self.metadata else None,
            "last_results": asdict(self.last_results) if self.last_results else None,
            "config": {
                "coverage_threshold": self.coverage_threshold,
                "lint_threshold": self.lint_threshold,
                "performance_threshold": self.performance_threshold,
            },
        }

        import joblib

        joblib.dump(save_data, path)

    @staticmethod
    def load(path: Path) -> "TestingQAPipeline":
        """Load QA pipeline from saved state."""
        import joblib

        save_data = joblib.load(path)

        config = save_data.get("config", {})
        pipeline = TestingQAPipeline(
            coverage_threshold=config.get("coverage_threshold", 80.0),
            lint_threshold=config.get("lint_threshold", 90.0),
            performance_threshold=config.get("performance_threshold", 30.0),
        )

        # Restore metadata and results if available
        if save_data.get("metadata"):
            pipeline.metadata = QAMetadata(**save_data["metadata"])
        if save_data.get("last_results"):
            pipeline.last_results = QAMetrics(**save_data["last_results"])

        return pipeline


# -------------------- CLI Actions -------------------- #


def action_train(args):
    """Train (setup and run) the QA pipeline with specified test suite."""
    pipeline = TestingQAPipeline(
        coverage_threshold=args.coverage_threshold,
        lint_threshold=args.lint_threshold,
        performance_threshold=args.performance_threshold,
    )

    target_modules = args.target_modules or ["modules/"]
    metrics = pipeline.run_tests(args.test_suite, target_modules)

    if args.save:
        pipeline.save(Path(args.save))

    # Format output
    result = {
        "status": "trained",
        "metrics": asdict(metrics),
        "metadata": asdict(pipeline.metadata) if pipeline.metadata else None,
        "quality_gates": {
            "coverage_passed": metrics.coverage_percentage >= args.coverage_threshold,
            "performance_passed": metrics.execution_time <= args.performance_threshold,
            "tests_passed": metrics.test_pass_rate >= 95.0,
        },
    }

    print(json.dumps(result, indent=2))


def action_evaluate(args):
    """Evaluate code quality and run various QA checks."""
    if args.qa_pipeline_path:
        pipeline = TestingQAPipeline.load(Path(args.qa_pipeline_path))
    else:
        pipeline = TestingQAPipeline()

    results = {}

    # Code quality check
    if args.check_type in ["lint", "all"]:
        results["code_quality"] = pipeline.check_code_quality(target_path=args.target_path, fix_issues=args.fix_issues)

    # Dataset path validation
    if args.check_type in ["paths", "all"]:
        results["dataset_paths"] = pipeline.validate_dataset_paths()

    # Smoke tests
    if args.check_type in ["smoke", "all"]:
        target_modules = args.target_modules or ["modules/"]
        results["smoke_tests"] = pipeline.run_smoke_tests(target_modules)

    # Performance benchmarks
    if args.check_type in ["performance", "all"]:
        operations = ["data_generation", "model_training", "prediction"]
        results["performance"] = pipeline.benchmark_performance(operations)

    result = {
        "status": "evaluated",
        "checks": results,
        "summary": {
            "total_checks": len(results),
            "all_passed": all(r.get("all_passed", r.get("overall_score", 0) > 70) for r in results.values()),
        },
    }

    print(json.dumps(result, indent=2))


def action_predict(args):
    """Predict (validate) the health of a specific module or code component."""
    pipeline = TestingQAPipeline()

    if args.target_module:
        # Validate specific module
        try:
            module = importlib.import_module(args.target_module)
            module_path = Path(module.__file__).parent if hasattr(module, "__file__") else None

            validation_results = {
                "module_import": {"status": "success", "module": args.target_module},
                "module_path": str(module_path) if module_path else None,
            }

            # Run targeted checks
            if module_path:
                lint_results = pipeline.check_code_quality(str(module_path))
                validation_results["code_quality"] = lint_results

        except ImportError as e:
            validation_results = {
                "module_import": {"status": "failed", "error": str(e)},
                "module_path": None,
            }
    else:
        # General system health check
        validation_results = pipeline.run_smoke_tests(["modules/"])

    result = {
        "status": "predicted",
        "target": args.target_module or "system",
        "health_check": validation_results,
        "recommendation": ("healthy" if validation_results.get("all_passed", True) else "needs_attention"),
    }

    print(json.dumps(result, indent=2))


# -------------------- Argument Parsing -------------------- #


def build_parser():
    parser = argparse.ArgumentParser(description="Module 10.2 Testing & QA Pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # Train subcommand (setup and run tests)
    p_train = sub.add_parser("train", help="Setup and run QA test suite")
    p_train.add_argument(
        "--test-suite",
        default="all",
        choices=["unit", "integration", "smoke", "all"],
        help="Type of test suite to run",
    )
    p_train.add_argument("--target-modules", nargs="+", help="Specific modules to test")
    p_train.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Minimum coverage percentage required",
    )
    p_train.add_argument("--lint-threshold", type=float, default=90.0, help="Minimum lint score required")
    p_train.add_argument(
        "--performance-threshold",
        type=float,
        default=30.0,
        help="Maximum execution time in seconds",
    )
    p_train.add_argument("--save", help="Path to save QA pipeline state")
    p_train.set_defaults(func=action_train)

    # Evaluate subcommand (run quality checks)
    p_eval = sub.add_parser("evaluate", help="Run code quality and validation checks")
    p_eval.add_argument("--qa-pipeline-path", help="Path to saved QA pipeline")
    p_eval.add_argument(
        "--check-type",
        default="all",
        choices=["lint", "paths", "smoke", "performance", "all"],
        help="Type of quality check to run",
    )
    p_eval.add_argument("--target-path", default=".", help="Path to check for code quality")
    p_eval.add_argument("--target-modules", nargs="+", help="Specific modules to check")
    p_eval.add_argument("--fix-issues", action="store_true", help="Automatically fix formatting issues")
    p_eval.set_defaults(func=action_evaluate)

    # Predict subcommand (validate module health)
    p_pred = sub.add_parser("predict", help="Validate health of specific module or system")
    p_pred.add_argument(
        "--target-module",
        help='Specific module to validate (e.g., "modules.foundation.module_3")',
    )
    p_pred.set_defaults(func=action_predict)

    return parser


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        error_result = {"status": "error", "command": args.command, "error": str(e)}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
