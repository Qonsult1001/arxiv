"""
DEPRECATED: This benchmark suite has been replaced

The comprehensive evaluation suite is now located at:
    LAM-base-v1/evaluation/

New Location:
=============
    cd LAM-base-v1/evaluation
    python run_all_tests.py --model ../LAM-base-v1 --baseline all-MiniLM-L6-v2

The new suite includes:
========================
✅ test_pearson_score.py - STS-B validation with bootstrap CI
✅ test_linear_scaling.py - O(n) complexity proof with regression analysis
✅ test_long_context.py - 32K to 1M token processing tests
✅ test_ablation_study.py - Component contribution analysis
✅ run_all_tests.py - Master runner for all tests

Benefits of New Suite:
======================
✓ Uses LAMEncoder to properly load lam_base.bin + lam_tweak.pt
✓ 24+ publication-quality visualizations (300 DPI)
✓ Statistical rigor: bootstrap CI, regression analysis
✓ Comprehensive JSON outputs for reproducibility
✓ Complete documentation in LAM-base-v1/evaluation/README.md

To run the new comprehensive evaluation:
=========================================
    cd LAM-base-v1/evaluation
    pip install -r requirements.txt
    python run_all_tests.py --model ../LAM-base-v1 --baseline all-MiniLM-L6-v2

For details, see: LAM-base-v1/evaluation/README.md
"""

import sys
from pathlib import Path

print("="*80)
print("⚠️  DEPRECATED: This benchmark script has been replaced")
print("="*80)
print()
print("The new comprehensive evaluation suite is located at:")
print("  📁 LAM-base-v1/evaluation/")
print()
print("To run the new tests:")
print("  cd LAM-base-v1/evaluation")
print("  pip install -r requirements.txt")
print("  python run_all_tests.py --model ../LAM-base-v1 --baseline all-MiniLM-L6-v2")
print()
print("The new suite includes:")
print("  ✅ Test 1: Pearson Score Validation (STS-B with bootstrap CI)")
print("  ✅ Test 2: Linear Scaling Validation (O(n) complexity proof)")
print("  ✅ Test 3: Long Context Processing (32K-1M tokens)")
print("  ✅ Test 4: Ablation Study (component analysis)")
print()
print("For documentation:")
print("  📖 LAM-base-v1/evaluation/README.md")
print()
print("="*80)
print()

# Check if user wants to be redirected
response = input("Would you like to navigate to the new evaluation suite? (y/n): ")

if response.lower() in ['y', 'yes']:
    print()
    print("Navigating to LAM-base-v1/evaluation/...")
    print()
    import os
    os.chdir(Path(__file__).parent / "LAM-base-v1" / "evaluation")
    print(f"Current directory: {os.getcwd()}")
    print()
    print("Run the following command to start the evaluation:")
    print("  python run_all_tests.py --model ../LAM-base-v1 --baseline all-MiniLM-L6-v2")
else:
    print()
    print("Exiting. Please use the new evaluation suite at LAM-base-v1/evaluation/")
    print()

sys.exit(1)
