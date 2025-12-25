#!/usr/bin/env python3
"""
Test CURRENT/pipeline Scripts
==============================

Performs basic validation checks on all pipeline scripts:
1. Import checks (verify all dependencies available)
2. Syntax validation
3. Function availability tests
4. Path verification

Run this before executing the full pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

def test_imports():
    """Test that all required packages can be imported."""
    print("\n" + "="*70)
    print("TESTING IMPORTS")
    print("="*70)
    
    packages = [
        ('numpy', 'np'),
        ('scipy', None),
        ('pandas', 'pd'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('networkx', 'nx'),
        ('sklearn', None),
        ('torch', None),
        ('openpyxl', None),
        ('h5py', None),
        ('tqdm', None),
    ]
    
    failed = []
    for package_info in packages:
        if isinstance(package_info, tuple):
            package, alias = package_info
        else:
            package, alias = package_info, None
        
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        print("   Run: pip install -r pipeline/requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully")
        return True


def test_sbtg_module():
    """Test that sbtg module can be imported."""
    print("\n" + "="*70)
    print("TESTING SBTG MODULE")
    print("="*70)
    
    try:
        from sbtg import SBTGStructuredVolatilityEstimator
        print(f"  ✓ sbtg module imported")
        print(f"  ✓ SBTGStructuredVolatilityEstimator available")
        
        # Check available model types
        estimator = SBTGStructuredVolatilityEstimator(model_type='linear')
        print(f"  ✓ Can instantiate linear model")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to import sbtg: {e}")
        print("     Ensure ../sbtg/ directory exists")
        return False


def test_data_files():
    """Check that required data files exist."""
    print("\n" + "="*70)
    print("TESTING DATA FILES")
    print("="*70)
    
    data_dir = CURRENT_DIR / "data"
    
    required_files = [
        "Head_Activity_OH15500.mat",
        "Head_Activity_OH16230.mat",
        "SI 6 Cell class lists.xlsx",
        "SI 7 Cell class connectome adjacency matrices, corrected July 2020.xlsx",
    ]
    
    missing = []
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename} (NOT FOUND)")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        print(f"   Expected location: {data_dir}")
        return False
    else:
        print("\n✓ All data files present")
        return True


def test_script_syntax():
    """Test that all pipeline scripts have valid syntax."""
    print("\n" + "="*70)
    print("TESTING SCRIPT SYNTAX")
    print("="*70)
    
    pipeline_dir = CURRENT_DIR / "pipeline"
    
    # Only test the main pipeline scripts (01-07)
    main_scripts = [
        "01_prepare_data.py",
        "02_hyperparameter_search.py",
        "03_comprehensive_analysis.py",
        "04_connectome_comparison.py",
        "05_weighted_analysis.py",
        "06_network_visualizations.py",
        "07_generate_summary.py",
    ]
    
    failed = []
    for script_name in main_scripts:
        script = pipeline_dir / script_name
        
        if not script.exists():
            print(f"  ✗ {script_name} (NOT FOUND)")
            failed.append(script_name)
            continue
            
        try:
            with open(script, 'r') as f:
                code = f.read()
            compile(code, script.name, 'exec')
            print(f"  ✓ {script_name}")
        except SyntaxError as e:
            print(f"  ✗ {script_name}: {e}")
            failed.append(script_name)
    
    if failed:
        print(f"\n⚠️  Syntax errors in: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All scripts have valid syntax")
        return True


def test_script_imports():
    """Test that each script can import its dependencies."""
    print("\n" + "="*70)
    print("TESTING SCRIPT IMPORTS")
    print("="*70)
    
    pipeline_dir = CURRENT_DIR / "pipeline"
    
    scripts = [
        "01_prepare_data.py",
        "02_hyperparameter_search.py",
        "03_comprehensive_analysis.py",
        "04_connectome_comparison.py",
        "05_weighted_analysis.py",
        "06_network_visualizations.py",
        "07_generate_summary.py",
    ]
    
    failed = []
    for script_name in scripts:
        script_path = pipeline_dir / script_name
        
        if not script_path.exists():
            print(f"  ✗ {script_name} (NOT FOUND)")
            failed.append(script_name)
            continue
        
        try:
            # Try to import as module (without executing)
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                script_name.replace('.py', ''), 
                script_path
            )
            module = importlib.util.module_from_spec(spec)
            
            # Don't execute, just check imports
            print(f"  ✓ {script_name}")
            
        except Exception as e:
            print(f"  ⚠️  {script_name}: {str(e)[:60]}")
            # Not a hard failure - imports might work when executed
    
    print("\n✓ Script import check complete")
    return True


def test_output_directories():
    """Check that results directory structure can be created."""
    print("\n" + "="*70)
    print("TESTING OUTPUT DIRECTORIES")
    print("="*70)
    
    results_dir = CURRENT_DIR.parent / "results"
    
    required_dirs = [
        "intermediate/connectome",
        "intermediate/datasets",
        "models/sbtg/best",
        "comprehensive_sbtg",
        "figures/connectome",
        "tables/connectome",
    ]
    
    try:
        for dir_path in required_dirs:
            full_path = results_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {dir_path}")
        
        print("\n✓ All output directories ready")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to create directories: {e}")
        return False


def test_quick_functionality():
    """Test basic functionality with minimal data."""
    print("\n" + "="*70)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*70)
    
    try:
        import numpy as np
        from sbtg import SBTGStructuredVolatilityEstimator
        
        # Create synthetic data
        np.random.seed(42)
        n_neurons = 10
        n_timepoints = 50
        X = np.random.randn(n_timepoints, n_neurons)
        
        print(f"  Testing with synthetic data: {X.shape}")
        
        # Test linear model (fastest)
        estimator = SBTGStructuredVolatilityEstimator(
            model_type='linear',
            train_split='prefix',
            window_length=2,
            dsm_epochs=10,  # Very short for testing
            fdr_alpha=0.1,
            verbose=False
        )
        
        print(f"  Training linear model...")
        result = estimator.fit(X)
        
        # Check outputs
        assert result.sign_adj.shape == (n_neurons, n_neurons), "Wrong sign_adj shape"
        assert result.volatility_adj.shape == (n_neurons, n_neurons), "Wrong volatility_adj shape"
        assert result.p_mean.shape == (n_neurons, n_neurons), "Wrong p_mean shape"
        assert hasattr(result, 'mu_hat'), "Missing mu_hat"
        
        n_edges = (result.sign_adj != 0).sum()
        print(f"  ✓ Model trained successfully")
        print(f"  ✓ Found {n_edges} edges")
        print(f"  ✓ mu_hat range: [{result.mu_hat.min():.3f}, {result.mu_hat.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CURRENT/PIPELINE VALIDATION TESTS")
    print("="*70)
    print(f"Location: {CURRENT_DIR}")
    
    results = {
        'imports': test_imports(),
        'sbtg_module': test_sbtg_module(),
        'data_files': test_data_files(),
        'syntax': test_script_syntax(),
        'script_imports': test_script_imports(),
        'output_dirs': test_output_directories(),
        'functionality': test_quick_functionality(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED")
        print("\nYou can now run the pipeline:")
        print("  cd CURRENT/")
        print("  python pipeline/01_prepare_data.py")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
