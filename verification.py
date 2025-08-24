# verification.py
import sys
import importlib

required_packages = [
    'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
    'jupyter', 'torch', 'tensorflow', 'xgboost', 'lightgbm',
    'cv2', 'plotly', 'dash', 'streamlit'
]

def verify_environment():
    """Verify that all required packages are installed and working."""
    print("=" * 60)
    print("🔍 Machine Learning for Semiconductor Engineers")
    print("   Environment Verification Script")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("\nChecking required packages:")
    print("-" * 60)
    
    all_packages_installed = True
    
    for package in required_packages:
        # Handle special cases for package imports
        import_name = package
        if package == 'cv2':
            import_name = 'cv2'
            package_display = 'opencv-python'
        else:
            package_display = package
            import_name = package.replace('-', '_')
        
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package_display:<20}: {version}")
        except ImportError:
            print(f"❌ {package_display:<20}: Not installed")
            all_packages_installed = False
    
    print("-" * 60)
    
    if all_packages_installed:
        print("🎉 All packages are installed correctly!")
        print("\nNext steps:")
        print("1. Navigate to modules/foundation/module-1/")
        print("2. Open the first Jupyter notebook")
        print("3. Start your ML journey!")
    else:
        print("⚠️  Some packages are missing.")
        print("\nTo install missing packages:")
        print("pip install -r requirements.txt")
        print("\nOr use the setup guide in docs/setup-guide.md")
    
    print("=" * 60)
    return all_packages_installed

if __name__ == "__main__":
    verify_environment()
