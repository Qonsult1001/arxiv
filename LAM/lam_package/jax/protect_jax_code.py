#!/usr/bin/env python3
"""
Protect JAX Source Code
=======================

This script obfuscates JAX source files to protect proprietary code.
Uses PyArmor to obfuscate the code while maintaining JAX JIT compatibility.

Usage:
    python protect_jax_code.py

Requirements:
    pip install pyarmor
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# JAX files to protect
JAX_FILES = [
    'lam/_jax_core.py',
    'lam/_jax_model_optimized.py',
    'lam/_jax_model.py',
]

# Backup directory
BACKUP_DIR = Path('lam/_jax_backup')
OBFUSCATED_DIR = Path('lam/_jax_obfuscated')


def check_pyarmor():
    """Check if PyArmor is installed."""
    try:
        import pyarmor
        return True
    except ImportError:
        print("❌ PyArmor not installed. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyarmor'])
        return True


def backup_files():
    """Backup original JAX files."""
    print("📦 Backing up original JAX files...")
    BACKUP_DIR.mkdir(exist_ok=True)
    
    for file_path in JAX_FILES:
        src = Path(file_path)
        if src.exists():
            dst = BACKUP_DIR / src.name
            shutil.copy2(src, dst)
            print(f"   ✅ Backed up {src} -> {dst}")
    
    print("✅ Backup complete\n")


def obfuscate_files():
    """Obfuscate JAX files using PyArmor."""
    print("🔒 Obfuscating JAX files with PyArmor...")
    
    # Create obfuscated directory
    OBFUSCATED_DIR.mkdir(exist_ok=True)
    
    for file_path in JAX_FILES:
        src = Path(file_path)
        if not src.exists():
            print(f"   ⚠️  File not found: {src}")
            continue
        
        print(f"   🔒 Obfuscating {src}...")
        
        # PyArmor obfuscation command
        # --recursive: obfuscate recursively
        # --restrict: restrict mode (more secure)
        # --enable-rft: enable runtime protection
        cmd = [
            'pyarmor', 'gen',
            '--recursive',
            '--restrict',  # More secure, but may have compatibility issues
            '--enable-rft',  # Runtime protection
            '--output', str(OBFUSCATED_DIR),
            str(src)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✅ Obfuscated {src.name}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to obfuscate {src}: {e}")
            print(f"   Error output: {e.stderr}")
            # Try without --restrict if it fails
            print(f"   🔄 Retrying without --restrict...")
            cmd = [
                'pyarmor', 'gen',
                '--recursive',
                '--output', str(OBFUSCATED_DIR),
                str(src)
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"   ✅ Obfuscated {src.name} (without --restrict)")
            except subprocess.CalledProcessError as e2:
                print(f"   ❌ Failed again: {e2.stderr}")
                return False
    
    print("✅ Obfuscation complete\n")
    return True


def replace_files():
    """Replace original files with obfuscated versions."""
    print("🔄 Replacing original files with obfuscated versions...")
    
    for file_path in JAX_FILES:
        src = Path(file_path)
        obf_file = OBFUSCATED_DIR / src.name
        
        if not obf_file.exists():
            # PyArmor creates files in a subdirectory
            obf_file = OBFUSCATED_DIR / src.parent.name / src.name
            if not obf_file.exists():
                print(f"   ⚠️  Obfuscated file not found: {obf_file}")
                continue
        
        # Backup original
        backup_file = BACKUP_DIR / src.name
        if not backup_file.exists():
            shutil.copy2(src, backup_file)
        
        # Replace with obfuscated
        shutil.copy2(obf_file, src)
        print(f"   ✅ Replaced {src} with obfuscated version")
    
    print("✅ File replacement complete\n")


def create_protection_wrapper():
    """Create a wrapper that handles PyArmor runtime."""
    wrapper_content = '''"""
PyArmor Runtime Wrapper for JAX
================================

This module ensures PyArmor runtime is available for obfuscated JAX code.
"""
import os
import sys
from pathlib import Path

# Add PyArmor runtime to path if needed
_pyarmor_runtime = Path(__file__).parent / 'pyarmor_runtime'
if _pyarmor_runtime.exists():
    sys.path.insert(0, str(_pyarmor_runtime))

# Import obfuscated modules
try:
    from . import _jax_core
    from . import _jax_model_optimized
    from . import _jax_model
except ImportError as e:
    # Fallback to original if obfuscated version fails
    import warnings
    warnings.warn(f"Failed to import obfuscated JAX modules: {e}. Using original.")
    # Original imports would go here if we keep originals
'''
    
    wrapper_path = Path('lam/_jax_wrapper.py')
    wrapper_path.write_text(wrapper_content)
    print(f"✅ Created protection wrapper: {wrapper_path}")


def main():
    """Main protection process."""
    print("=" * 70)
    print("🔒 JAX Source Code Protection")
    print("=" * 70)
    print()
    
    # Check PyArmor
    if not check_pyarmor():
        print("❌ Failed to install PyArmor. Exiting.")
        return 1
    
    # Backup files
    backup_files()
    
    # Obfuscate files
    if not obfuscate_files():
        print("❌ Obfuscation failed. Original files preserved in backup.")
        return 1
    
    # Replace files
    replace_files()
    
    # Create wrapper
    create_protection_wrapper()
    
    print("=" * 70)
    print("✅ JAX Code Protection Complete!")
    print("=" * 70)
    print()
    print("📝 Notes:")
    print("   - Original files backed up in: lam/_jax_backup/")
    print("   - Obfuscated files in: lam/_jax_obfuscated/")
    print("   - Test your code to ensure JAX JIT still works correctly")
    print("   - If issues occur, restore from backup:")
    print("     cp lam/_jax_backup/*.py lam/")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


