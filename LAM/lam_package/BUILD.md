# LAM Package - Build & Distribution Guide

## Overview

LAM is a standalone package with minimal dependencies. The package includes two compiled binaries:
- **`_core.so`** - Core LAM algorithm (compiled from formula)
- **`_secrets.so`** - Proprietary position embedding interpolation (compiled from `_secrets.pyx`)

## Latest Changes

- ✅ **Standalone package** - Uses bundled tokenizer and model weights
- ✅ **Core formula compiled** - Entire algorithm in protected binary
- ✅ **Proprietary interpolation compiled** - `_secrets.pyx` compiled to `_secrets.so` for 32k token support

## Prerequisites

### 1. Install Cython

Cython is required to compile the Python formula into a binary:

```bash
pip install Cython
```

### 2. Required Python Packages

```bash
pip install torch>=2.0 tokenizers>=0.13.0 numpy
```

## Building the Package

### Quick Rebuild (After Formula Changes)

```bash
cd /workspace/LAM/lam_package
python3 setup.py build_ext --inplace
```

### Full Build Process

The build script performs these steps:

1. **Copies formula source** - `final_solution_formula_final.py` → `lam/_core.py`
2. **Compiles _core.py** - Uses Cython to create `_core.cpython-312-x86_64-linux-gnu.so`
3. **Compiles _secrets.pyx** - Uses Cython to create `_secrets.cpython-312-x86_64-linux-gnu.so` (proprietary interpolation for 32k tokens)
4. **Strips debug symbols** - Reduces binary size for both .so files
5. **Cleans up** - Removes source files (`.py`, `.pyx`, `.c`), leaving only the protected binaries

### Step-by-Step

#### 1. Make Changes to Formula

Edit `/workspace/LAM/final_solution_formula_final.py` with your formula updates.

**Important**: The formula only pure PyTorch.

#### 2. Install Build Dependencies

```bash
# Install Cython and numpy (required for compilation)
pip install Cython numpy

# Or using pip3
pip3 install Cython numpy
```

#### 3. Rebuild the Package

```bash
cd /workspace/LAM/lam_package
python3 setup.py build_ext --inplace
```

Expected output:
```
============================================================
🔧 LAM PACKAGE BUILDER
============================================================

📋 Step 1: Checking formula...
   ✅ Found: /workspace/LAM/final_solution_formula_final.py

📋 Step 2: Copying formula...
   ✅ Copied to: /workspace/LAM/lam_package/lam/_core.py

📋 Step 3: Compiling to binary...
   📦 Found _secrets.pyx - will compile proprietary interpolation
   ✅ Compiled: _core.cpython-312-x86_64-linux-gnu.so
   ✅ Compiled: _secrets.cpython-312-x86_64-linux-gnu.so (proprietary interpolation)

📋 Step 4: Stripping debug symbols...
   ✅ Stripped _core: 502 KB
   ✅ Stripped _secrets: 260 KB

📋 Step 5: Cleaning up...
   ✅ Removed _core.py (source)
   ✅ Removed _secrets.pyx (source - proprietary)
   ✅ Removed _core.c
   ✅ Removed _secrets.c
   ✅ Removed build/
   ✅ Cleaned up intermediate .c files automatically
   ✅ Removed __pycache__/

============================================================
✅ BUILD COMPLETE!
============================================================

Protected binaries:
  • lam/_core.cpython-312-x86_64-linux-gnu.so (502 KB)
  • lam/_secrets.cpython-312-x86_64-linux-gnu.so (260 KB) - Proprietary interpolation

To test: python3 test.py
============================================================
```

#### 4. Test the Build

```bash
cd /workspace/LAM/lam_package
python3 test.py
```

## Package Structure

After building, the package contains:

```
lam_package/
├── lam/
│   ├── __init__.py          # Main LAM class
│   ├── _core.cpython-312-x86_64-linux-gnu.so  # Compiled core formula (protected)
│   ├── _secrets.cpython-312-x86_64-linux-gnu.so  # Compiled interpolation (proprietary)
│   ├── _license.py          # License management
│   └── ...                  # Other module files
├── setup.py                 # Package installer (includes Cython build)
├── test.py                  # Test script
└── production_test.py       # Production test suite
```

**Note:** Source files (`_core.py`, `_secrets.pyx`, `_secrets.c`, `_core.c`) are removed after compilation to protect proprietary code.

## Distribution

### What to Distribute

Give users:
- `lam_package/` folder (contains protected binary)
- `LAM-base-v1/` folder (contains model weights, config, tokenizer)

### User Installation

Users install with:

```bash
pip install path/to/lam_package/
```

### User Usage

```python
from lam import LAM

model = LAM('LAM-base-v1')
embeddings = model.encode(['Hello world', 'How are you?'])
```

## Troubleshooting

### Cython Not Found

If you see `ModuleNotFoundError: No module named 'Cython'`:

```bash
pip install Cython
```

### Compilation Errors

If compilation fails:
1. Check that `final_solution_formula_final.py` has no syntax errors
2. Ensure all imports in the formula are available
3. Verify no Triton/ONNX code remains in the formula

### Binary Not Found

If the `.so` files aren't created:
1. Check Cython is installed: `pip list | grep Cython`
2. Check numpy is installed: `pip list | grep numpy`
3. Check Python version matches (3.12 for current binary)
4. Review build output for errors
5. Try manual compilation:
   ```bash
   python3 setup.py build_ext --inplace
   ```

### _secrets.so Not Found

If `_secrets.so` is missing (causes 512 token limit instead of 32k):
1. Ensure `_secrets.pyx` exists in `lam/` directory
2. Install Cython and numpy: `pip install Cython numpy`
3. Compile manually:
   ```bash
   python3 setup.py build_ext --inplace
   ```
4. Verify: `ls lam/_secrets*.so`
5. If still missing, check build output for Cython errors
6. Test import: `python3 -c "from lam import _secrets; print('OK')"`

## Notes

### Compiled Binaries

- **`_core.so`**: Contains the entire LAM core formula - no source code is exposed
- **`_secrets.so`**: Contains proprietary position embedding interpolation for 32k token support
- Both binaries are platform-specific (Linux x86_64 for current build)
- Rebuild required after any changes to `final_solution_formula_final.py` or `_secrets.pyx`
- Binary sizes: ~500 KB (`_core.so`) and ~260 KB (`_secrets.so`) after stripping

### Build Methods

**Using setup.py (Recommended)**
```bash
cd lam_package
pip install Cython numpy
python3 setup.py build_ext --inplace
```
- Automatically handles both `_core.py` and `_secrets.pyx`
- Cleans up intermediate `.c` files after compilation
- Uses standard setuptools build process

### Testing After Build

```bash
# Run internal tests
python3 test.py

# Run production tests (verifies README examples)
python3 production_test.py
```

### Distribution

When distributing the package:
- ✅ Include compiled `.so` files
- ❌ Do NOT include `_secrets.pyx` (proprietary source)
- ❌ Do NOT include `_core.py` (source removed after build)
- ✅ Include `_license.py` (license management)
- ✅ Include all other Python files
