#!/usr/bin/env python3
"""
LAM (Linear Attention Model) - Installation Package

Cython-only production build.
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path
import os

# ============================================================================
# HYBRID STRATEGY: Compile Proprietary Secret (_secrets.pyx)
# ============================================================================
# This ensures the position interpolation formula is compiled into a binary
# on the user's machine, ensuring it works on Windows, Mac, and Linux.

class BuildExtWithCleanup(build_ext):
    """Custom build_ext that cleans up intermediate .c files after compilation."""
    
    def run(self):
        """Run build_ext and clean up intermediate files."""
        super().run()
        self.cleanup_intermediate_files()
    
    def cleanup_intermediate_files(self):
        """Remove intermediate .c files and temporary build directories after compilation."""
        import shutil
        
        # Clean .c files from both build/ and lam/ directories
        build_dir = Path(__file__).parent.absolute()  # build/ folder
        lam_dir = build_dir.parent / "lam"  # lam/ folder
        
        cleaned = []
        removed_dirs = []
        
        # Clean from build/ folder first (where .c files are generated)
        if build_dir.exists():
            # Remove .c files
            for c_file in build_dir.glob("*.c"):
                try:
                    c_file.unlink()
                    cleaned.append(f"build/{c_file.name}")
                except Exception:
                    pass
            
            # Remove temporary build/ directory created by setuptools
            temp_build_dir = build_dir / "build"
            if temp_build_dir.exists() and temp_build_dir.is_dir():
                try:
                    shutil.rmtree(temp_build_dir)
                    removed_dirs.append("build/build/")
                except Exception:
                    pass
        
        # Also clean from lam/ folder (just in case)
        if lam_dir.exists():
            for c_file in lam_dir.glob("*.c"):
                try:
                    c_file.unlink()
                    cleaned.append(f"lam/{c_file.name}")
                except Exception:
                    pass
        
        # Report cleanup
        if cleaned or removed_dirs:
            if cleaned:
                print(f"\n✅ Cleaned up intermediate files: {', '.join(cleaned)}")
            if removed_dirs:
                print(f"✅ Removed temporary directories: {', '.join(removed_dirs)}")

try:
    from Cython.Build import cythonize
    import numpy
    from pathlib import Path
    import shutil
    
    # Paths - build files are in build/ folder, output goes to lam/
    build_dir = Path(__file__).parent.absolute()  # build/ folder
    lam_dir = build_dir.parent / "lam"  # lam/ folder (output)
    formula_src = build_dir.parent.parent / "final_solution_formula_final.py"
    
    # Ensure lam/ directory exists
    lam_dir.mkdir(exist_ok=True)
    
    # Check if _core.py exists in build/, if not copy from formula
    core_py = build_dir / "_core.py"
    if not core_py.exists() and formula_src.exists():
        print(f"📋 Copying formula to build/_core.py...")
        shutil.copy(formula_src, core_py)
        print(f"   ✅ Copied from {formula_src}")
    
    # Define extensions to compile
    extensions = []
    
    # Compile _core.py from build/ folder, output to lam/
    if core_py.exists():
        extensions.append(
            Extension(
                name="lam._core",
                sources=[str(core_py)],  # Source in build/
                include_dirs=[numpy.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        )
    
    # Compile _secrets.pyx from build/ folder, output to lam/
    secrets_pyx = build_dir / "_secrets.pyx"
    if secrets_pyx.exists():
        extensions.append(
            Extension(
                name="lam._secrets",
                sources=[str(secrets_pyx)],  # Source in build/
                include_dirs=[numpy.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        )
    
    # Compile _license.pyx from build/ folder, output to lam/
    license_pyx = build_dir / "_license.pyx"
    if license_pyx.exists():
        extensions.append(
            Extension(
                name="lam._license",
                sources=[str(license_pyx)],  # Source in build/
                include_dirs=[numpy.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        )
    
    # Build the extensions (Cython only)
    if extensions:
        ext_modules = cythonize(extensions, compiler_directives={'language_level': "3"})
    else:
        ext_modules = None
        print("⚠️  No source files found to compile (_core.py or _secrets.pyx)")
    
except ImportError:
    # Fallback if Cython is not installed (e.g., just installing from a Wheel)
    # Note: Pre-built wheels should include the compiled .so file
    ext_modules = None
    BuildExtWithCleanup = build_ext  # Use standard build_ext if Cython not available
    print("⚠️  Cython not found. If installing from source, install Cython first:")
    print("   pip install Cython numpy")

# ============================================================================
# SETUP CONFIGURATION
# ============================================================================

readme_file = Path(__file__).parent.parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8")
    if readme_file.exists()
    else "Linear Attention Model (LAM): Infinite context (32k+) with O(N) complexity."
)

# ============================================================================
# BUILD CONFIGURATION
# ============================================================================
print(f"\n{'='*70}")
print(f"🔧 Building LAM package (Cython backend)")
print(f"{'='*70}\n")

setup(
    # Core metadata
    name="lam-attn",  # ✅ The name on PyPI
    version="0.1.0-beta",
    
    # Users will type: pip install lam-attn
    # Users will write: import lam
    packages=find_packages(where=str(Path(__file__).parent.parent)),
    package_dir={"": str(Path(__file__).parent.parent)},
    
    description="Linear Attention Model (LAM): Infinite context (32k+) with O(N) complexity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    author="Willie Olivier",
    author_email="willie@saidhome.com",
    license="Apache-2.0",
    license_files=["LICENSE_GUIDE.md"],
    url="https://github.com/said-research/lam",
    project_urls={
        "Homepage": "https://github.com/said-research/lam",
        "Documentation": "https://github.com/said-research/lam",
        "Repository": "https://github.com/said-research/lam",
        "Issues": "https://github.com/said-research/lam/issues",
    },
    
    # Dependencies
    install_requires=[
        "torch>=2.0",
        "numpy",
        "tokenizers",
    ],
    
    # Your Hybrid Strategy (Compiled Secret)
    # Ensure you have your MANIFEST.in file ready!
    ext_modules=ext_modules,
    
    # Custom build command to clean up intermediate .c files
    cmdclass={'build_ext': BuildExtWithCleanup},
    
    # Include compiled binaries
    package_data={
        "lam": ["*.so", "*.pyd", "*.dylib"],  # Compiled Cython binaries only
    },
    include_package_data=True,
    
    # Exclude source files from distribution (keep proprietary)
    exclude_package_data={
        "": [
            "*.pyx", "*.c",  # Don't include Cython source or C files
        ],
    },
    
    # Must be False because we have C-extensions
    zip_safe=False,
)
