"""
Health Check Script for Monitoring Infrastructure

This script validates that all monitoring components are functioning correctly.
"""
import sys
import requests
import subprocess
import json
from pathlib import Path


def check_mlflow_server(host="localhost", port=5000):
    """Check if MLflow server is running and accessible."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        if response.status_code == 200:
            print("✅ MLflow server is healthy")
            return True
        else:
            print(f"❌ MLflow server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ MLflow server not accessible: {e}")
        return False


def check_monitoring_pipeline():
    """Check if the monitoring pipeline script is working."""
    try:
        script_path = (
            Path(__file__).parent.parent.parent
            / "modules"
            / "cutting-edge"
            / "module-9"
            / "9.2-monitoring-maintenance-pipeline.py"
        )

        if not script_path.exists():
            print(f"❌ Pipeline script not found: {script_path}")
            return False

        # Test help command
        result = subprocess.run([sys.executable, str(script_path), "--help"], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Monitoring pipeline script is functional")
            return True
        else:
            print(f"❌ Pipeline script failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error checking pipeline: {e}")
        return False


def check_python_dependencies():
    """Check if required Python packages are installed."""
    required_packages = ["mlflow", "scikit-learn", "pandas", "numpy", "scipy"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)

    return len(missing_packages) == 0


def main():
    """Run comprehensive health check."""
    print("🔍 Running Monitoring Infrastructure Health Check")
    print("=" * 50)

    checks = [
        ("Python Dependencies", check_python_dependencies),
        ("Monitoring Pipeline", check_monitoring_pipeline),
        ("MLflow Server", check_mlflow_server),
    ]

    passed = 0
    total = len(checks)

    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        if check_func():
            passed += 1
        else:
            print(f"   💡 See documentation for fixing {check_name}")

    print("\n" + "=" * 50)
    print(f"📊 Health Check Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All systems operational!")
        return 0
    else:
        print("⚠️  Some issues detected. Please address them before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
