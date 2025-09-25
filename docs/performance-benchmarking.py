#!/usr/bin/env python3
"""
Performance Benchmarking System for Python for Semiconductors Learning Series

This module provides comprehensive performance benchmarking for ML models,
educational progress tracking, and system performance monitoring.

Usage Examples:
    # Benchmark ML model performance
    python performance-benchmarking.py benchmark-model --model RandomForest --dataset secom
    
    # Generate performance report
    python performance-benchmarking.py report --output benchmarks.html
    
    # Monitor system performance
    python performance-benchmarking.py monitor --duration 60
"""

import argparse
import json
import time
import psutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Check for optional dependencies
try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas/numpy not available. Some benchmarking features disabled.")

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.datasets import make_classification, make_regression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. ML benchmarking disabled.")

RANDOM_SEED = 42

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    benchmark_id: str
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    training_time: float
    prediction_time: float
    memory_usage_mb: float
    timestamp: str
    parameters: Dict[str, Any]
    system_info: Dict[str, str]

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    timestamp: str

@dataclass
class LearningMetrics:
    """Educational learning metrics."""
    student_id: str
    module_completion_time: float
    assessment_scores: List[float]
    interaction_count: int
    help_requests: int
    timestamp: str

class ModelBenchmarker:
    """Benchmarks ML model performance."""
    
    def __init__(self, output_dir: Path = Path('./benchmarks')):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
        if not HAS_SKLEARN or not HAS_PANDAS:
            raise ImportError("Required dependencies not available for ML benchmarking")
    
    def generate_synthetic_semiconductor_data(self, 
                                            problem_type: str = 'classification',
                                            n_samples: int = 1000,
                                            n_features: int = 20) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic semiconductor manufacturing data."""
        np.random.seed(RANDOM_SEED)
        
        if problem_type == 'classification':
            # Simulate wafer pass/fail classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.7),
                n_redundant=int(n_features * 0.2),
                n_clusters_per_class=2,
                random_state=RANDOM_SEED,
                class_sep=0.8
            )
            
            # Create realistic feature names
            feature_names = []
            process_params = ['Temperature', 'Pressure', 'FlowRate', 'Power', 'Time']
            measurement_types = ['Uniformity', 'Thickness', 'Resistance', 'Defect_Count']
            
            for i in range(n_features):
                if i < len(process_params):
                    feature_names.append(f'Process_{process_params[i]}')
                elif i < len(process_params) + len(measurement_types):
                    feature_names.append(f'Measure_{measurement_types[i - len(process_params)]}')
                else:
                    feature_names.append(f'Param_{i+1}')
            
            df = pd.DataFrame(X, columns=feature_names)
            return df, y
        
        else:  # regression
            # Simulate yield prediction
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=10,
                random_state=RANDOM_SEED
            )
            
            # Scale y to realistic yield percentages (70-99%)
            y_scaled = ((y - y.min()) / (y.max() - y.min())) * 29 + 70
            
            feature_names = [f'Process_Param_{i+1}' for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            return df, y_scaled
    
    def benchmark_model(self, 
                       model_class,
                       model_params: Dict[str, Any],
                       X: pd.DataFrame,
                       y: np.ndarray,
                       problem_type: str) -> BenchmarkResult:
        """Benchmark a single model."""
        
        # Initialize model
        model = model_class(**model_params)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Measure training time and memory
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # Measure prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {}
        if problem_type == 'classification':
            metrics.update({
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            })
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()
            
        else:  # regression
            metrics.update({
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            })
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
        
        # Create result
        result = BenchmarkResult(
            benchmark_id=f"{model_class.__name__}_{int(time.time())}",
            model_name=model_class.__name__,
            dataset_name="synthetic_semiconductor",
            metrics=metrics,
            training_time=training_time,
            prediction_time=prediction_time,
            memory_usage_mb=max(0, memory_usage),
            timestamp=datetime.now(timezone.utc).isoformat(),
            parameters=model_params,
            system_info=self._get_system_info()
        )
        
        self.results.append(result)
        return result
    
    def benchmark_suite(self, problem_type: str = 'classification') -> List[BenchmarkResult]:
        """Run comprehensive benchmark suite."""
        print(f"Running {problem_type} benchmark suite...")
        
        # Generate data
        X, y = self.generate_synthetic_semiconductor_data(problem_type)
        print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Define models to benchmark
        if problem_type == 'classification':
            models = [
                (RandomForestClassifier, {'n_estimators': 100, 'random_state': RANDOM_SEED}),
                (RandomForestClassifier, {'n_estimators': 200, 'random_state': RANDOM_SEED}),
                (LogisticRegression, {'random_state': RANDOM_SEED, 'max_iter': 1000}),
                (SVC, {'random_state': RANDOM_SEED, 'probability': True})
            ]
        else:
            models = [
                (RandomForestRegressor, {'n_estimators': 100, 'random_state': RANDOM_SEED}),
                (RandomForestRegressor, {'n_estimators': 200, 'random_state': RANDOM_SEED}),
                (LinearRegression, {}),
                (SVR, {})
            ]
        
        suite_results = []
        for i, (model_class, params) in enumerate(models, 1):
            print(f"Benchmarking {model_class.__name__} ({i}/{len(models)})...")
            try:
                result = self.benchmark_model(model_class, params, X, y, problem_type)
                suite_results.append(result)
                
                # Print key metrics
                if problem_type == 'classification':
                    print(f"  Accuracy: {result.metrics['accuracy']:.4f}")
                else:
                    print(f"  R²: {result.metrics['r2']:.4f}")
                print(f"  Training time: {result.training_time:.2f}s")
                print(f"  Memory usage: {result.memory_usage_mb:.1f}MB")
                
            except Exception as e:
                print(f"  Error benchmarking {model_class.__name__}: {e}")
        
        return suite_results
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get current system information."""
        return {
            'python_version': sys.version.split()[0],
            'platform': sys.platform,
            'cpu_count': str(psutil.cpu_count()),
            'memory_total_gb': f"{psutil.virtual_memory().total / (1024**3):.1f}",
            'sklearn_version': getattr(__import__('sklearn'), '__version__', 'unknown') if HAS_SKLEARN else 'not_available'
        }
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save benchmark results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        results_data = {
            'benchmark_run': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_benchmarks': len(self.results),
                'system_info': self._get_system_info()
            },
            'results': [asdict(result) for result in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def generate_report(self, output_format: str = 'html') -> str:
        """Generate benchmark report."""
        if not self.results:
            return "No benchmark results available"
        
        if output_format == 'html':
            return self._generate_html_report()
        else:
            return self._generate_text_report()
    
    def _generate_html_report(self) -> str:
        """Generate HTML benchmark report."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Model Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #4CAF50; color: white; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }
        .results { margin: 20px 0; }
        .model-result { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .metric { background: #f0f0f0; padding: 10px; border-radius: 3px; text-align: center; }
        .metric-value { font-size: 1.2em; font-weight: bold; color: #2196F3; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ML Model Benchmark Report</h1>
        <p>Semiconductor Manufacturing ML Performance Analysis</p>
    </div>
    
    <div class="summary">
        <h2>Benchmark Summary</h2>
        <p><strong>Total Models Tested:</strong> """ + str(len(self.results)) + """</p>
        <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
    
    <div class="results">
        <h2>Individual Model Results</h2>
"""
        
        for result in self.results:
            html += f"""
        <div class="model-result">
            <h3>{result.model_name}</h3>
            <div class="metrics">
"""
            
            # Add key metrics
            for metric_name, value in result.metrics.items():
                if isinstance(value, float):
                    html += f"""
                <div class="metric">
                    <div class="metric-value">{value:.4f}</div>
                    <div>{metric_name.replace('_', ' ').title()}</div>
                </div>
"""
            
            html += f"""
                <div class="metric">
                    <div class="metric-value">{result.training_time:.2f}s</div>
                    <div>Training Time</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result.memory_usage_mb:.1f}MB</div>
                    <div>Memory Usage</div>
                </div>
            </div>
            
            <h4>Parameters</h4>
            <table>
"""
            
            for param, value in result.parameters.items():
                html += f"<tr><td>{param}</td><td>{value}</td></tr>"
            
            html += """
            </table>
        </div>
"""
        
        html += """
    </div>
    
    <div class="summary">
        <h2>Performance Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Primary Metric</th>
                <th>Training Time</th>
                <th>Memory Usage</th>
            </tr>
"""
        
        for result in self.results:
            primary_metric = 'accuracy' if 'accuracy' in result.metrics else 'r2'
            metric_value = result.metrics.get(primary_metric, 0)
            
            html += f"""
            <tr>
                <td>{result.model_name}</td>
                <td>{metric_value:.4f}</td>
                <td>{result.training_time:.2f}s</td>
                <td>{result.memory_usage_mb:.1f}MB</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_text_report(self) -> str:
        """Generate text benchmark report."""
        report = "ML Model Benchmark Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Total Models Tested: {len(self.results)}\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for result in self.results:
            report += f"Model: {result.model_name}\n"
            report += "-" * 30 + "\n"
            
            for metric_name, value in result.metrics.items():
                if isinstance(value, float):
                    report += f"  {metric_name}: {value:.4f}\n"
            
            report += f"  Training Time: {result.training_time:.2f}s\n"
            report += f"  Memory Usage: {result.memory_usage_mb:.1f}MB\n\n"
        
        return report

class SystemMonitor:
    """Monitor system performance during learning activities."""
    
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
    
    def capture_metrics(self) -> SystemMetrics:
        """Capture current system metrics."""
        metrics = SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            memory_available_gb=psutil.virtual_memory().available / (1024**3),
            disk_usage_percent=psutil.disk_usage('/').percent,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def monitor_session(self, duration_seconds: int = 60) -> List[SystemMetrics]:
        """Monitor system for specified duration."""
        print(f"Monitoring system performance for {duration_seconds} seconds...")
        
        interval = min(5, duration_seconds // 10)  # Sample every 5 seconds or 10 samples
        samples = []
        
        for i in range(duration_seconds // interval):
            metrics = self.capture_metrics()
            samples.append(metrics)
            
            print(f"Sample {i+1}: CPU {metrics.cpu_percent:.1f}%, "
                  f"Memory {metrics.memory_percent:.1f}%, "
                  f"Available {metrics.memory_available_gb:.1f}GB")
            
            if i < (duration_seconds // interval) - 1:
                time.sleep(interval)
        
        return samples
    
    def generate_summary(self) -> Dict[str, float]:
        """Generate summary statistics from monitoring session."""
        if not self.metrics_history:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'max_memory_percent': max(memory_values),
            'min_available_memory_gb': min(m.memory_available_gb for m in self.metrics_history),
            'sample_count': len(self.metrics_history)
        }

def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Performance Benchmarking System for Python for Semiconductors"
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Benchmark model command
    bench_parser = subparsers.add_parser('benchmark-model', help='Benchmark ML models')
    bench_parser.add_argument('--problem-type', choices=['classification', 'regression'],
                             default='classification', help='Problem type to benchmark')
    bench_parser.add_argument('--output', default='./benchmarks', help='Output directory')
    bench_parser.set_defaults(func=action_benchmark_model)
    
    # Generate report command
    report_parser = subparsers.add_parser('report', help='Generate benchmark report')
    report_parser.add_argument('--input', help='Input benchmark results JSON')
    report_parser.add_argument('--output', default='benchmark_report.html', help='Output file')
    report_parser.add_argument('--format', choices=['html', 'text'], default='html', 
                              help='Report format')
    report_parser.set_defaults(func=action_generate_report)
    
    # Monitor system command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor system performance')
    monitor_parser.add_argument('--duration', type=int, default=60, 
                               help='Monitoring duration in seconds')
    monitor_parser.set_defaults(func=action_monitor_system)
    
    return parser

def action_benchmark_model(args) -> None:
    """Handle benchmark-model command."""
    try:
        if not HAS_SKLEARN or not HAS_PANDAS:
            print("❌ Required dependencies not available for ML benchmarking")
            print("Install with: pip install scikit-learn pandas numpy")
            return
        
        benchmarker = ModelBenchmarker(Path(args.output))
        results = benchmarker.benchmark_suite(args.problem_type)
        
        # Save results
        results_file = benchmarker.save_results()
        
        # Generate and save HTML report
        html_report = benchmarker.generate_report('html')
        report_file = benchmarker.output_dir / 'benchmark_report.html'
        with open(report_file, 'w') as f:
            f.write(html_report)
        
        print(f"\n✅ Benchmark completed!")
        print(f"   Results: {results_file}")
        print(f"   Report: {report_file}")
        print(f"   Models tested: {len(results)}")
        
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()

def action_generate_report(args) -> None:
    """Handle report command."""
    try:
        # This is a simplified version - in practice would load from file
        print("Report generation from saved results not implemented in demo")
        print("Use 'benchmark-model' command which generates reports automatically")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")

def action_monitor_system(args) -> None:
    """Handle monitor command."""
    try:
        monitor = SystemMonitor()
        samples = monitor.monitor_session(args.duration)
        summary = monitor.generate_summary()
        
        print(f"\n📊 System Monitoring Summary:")
        print(f"   Average CPU: {summary['avg_cpu_percent']:.1f}%")
        print(f"   Peak CPU: {summary['max_cpu_percent']:.1f}%")
        print(f"   Average Memory: {summary['avg_memory_percent']:.1f}%")
        print(f"   Peak Memory: {summary['max_memory_percent']:.1f}%")
        print(f"   Min Available Memory: {summary['min_available_memory_gb']:.1f}GB")
        print(f"   Samples Collected: {summary['sample_count']}")
        
    except Exception as e:
        print(f"❌ Error monitoring system: {e}")

def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()