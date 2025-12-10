#!/usr/bin/env python3
"""
LAM Package Builder
===================
Quick build script that compiles the package using setup.py.

Usage:
    python3 build.py [--backend cython|jax] [--protect-jax]
    
Options:
    --backend cython|jax    Choose backend to build (default: cython)
    --protect-jax          Protect JAX source code using PyArmor (JAX only, requires: pip install pyarmor)
    
Examples:
    python3 build.py                    # Build Cython version (default)
    python3 build.py --backend cython   # Build Cython version
    python3 build.py --backend jax      # Build JAX version
    python3 build.py --backend jax --protect-jax  # Build JAX version with protection
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Set default backend if not specified
if 'LAM_BACKEND' not in os.environ:
    os.environ['LAM_BACKEND'] = 'cython'

def protect_jax_code(script_dir):
    """Protect JAX source code using PyArmor."""
    protect_script = script_dir / "protect_jax_code.py"
    
    if not protect_script.exists():
        print(f"⚠️  Protection script not found: {protect_script}")
        return False
    
    print("\n" + "="*60)
    print("🔒 PROTECTING JAX SOURCE CODE")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(protect_script)],
            check=True,
            cwd=str(script_dir)
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ JAX protection failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ PyArmor not found. Install with: pip install pyarmor")
        return False

def main():
    parser = argparse.ArgumentParser(description='Build LAM package')
    parser.add_argument('--backend', choices=['cython', 'jax'], default='cython',
                       help='Backend to build: cython (default) or jax')
    parser.add_argument('--protect-jax', action='store_true',
                       help='Protect JAX source code using PyArmor (JAX only)')
    args = parser.parse_args()
    
    # Validate protect-jax only with JAX backend
    if args.protect_jax and args.backend != 'jax':
        print("⚠️  --protect-jax only works with --backend jax")
        print(f"   Current backend: {args.backend}")
        return False
    
    print("="*60)
    print("🔧 LAM PACKAGE BUILDER")
    print("="*60)
    print(f"\n📦 Building {args.backend.upper()} backend version")
    print("="*60)
    
    # Get script directory
    script_dir = Path(__file__).parent
    build_dir = script_dir / "build"
    setup_py = build_dir / "setup.py"
    
    # Protect JAX code if requested
    if args.protect_jax and args.backend == 'jax':
        if not protect_jax_code(script_dir):
            print("\n⚠️  JAX protection failed, but continuing with build...")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
    
    # Set environment variable for backend selection
    os.environ['LAM_BACKEND'] = args.backend
    
    # Check if setup.py exists
    if not setup_py.exists():
        print(f"❌ setup.py not found: {setup_py}")
        print("   Expected location: build/setup.py")
        return False
    
    print(f"\n📋 Building from: {setup_py}")
    print(f"   Output directory: {script_dir / 'lam'}")
    
    # Change to build directory
    os.chdir(build_dir)
    
    # Run setup.py build_ext --inplace
    print("\n🔨 Running: python3 setup.py build_ext --inplace")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            check=True,
            capture_output=False
        )
        
        print("-" * 60)
        print("\n✅ BUILD COMPLETE!")
        print("="*60)
        
        # Check for backend-specific files
        lam_dir = script_dir / "lam"
        
        if args.backend == "cython":
            # Check for compiled binaries
            core_so = list(lam_dir.glob("_core*.so"))
            secrets_so = list(lam_dir.glob("_secrets*.so"))
            
            print(f"\n📦 Compiled binaries:")
            if core_so:
                size = core_so[0].stat().st_size / 1024
                print(f"   ✅ {core_so[0].name} ({size:.1f} KB)")
            else:
                print("   ❌ _core.so not found")
            
            if secrets_so:
                size = secrets_so[0].stat().st_size / 1024
                print(f"   ✅ {secrets_so[0].name} ({size:.1f} KB)")
            else:
                print("   ❌ _secrets.so not found")
        else:
            # Check for JAX files
            jax_files = [
                lam_dir / "_jax_core.py",
                lam_dir / "_jax_model_optimized.py",
                lam_dir / "_jax_model.py",
            ]
            
            print(f"\n📦 JAX files:")
            for jax_file in jax_files:
                if jax_file.exists():
                    size = jax_file.stat().st_size / 1024
                    print(f"   ✅ {jax_file.name} ({size:.1f} KB)")
                else:
                    print(f"   ❌ {jax_file.name} not found")
        
        print("\n" + "="*60)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Build error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

