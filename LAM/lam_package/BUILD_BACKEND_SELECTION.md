# Build-Time Backend Selection

## Overview

LAM supports building **EITHER** Cython **OR** JAX version at build time. The publisher (you) chooses which backend to include in the published package. End users get the version you built - they don't choose.

## Build Options

### Option 1: Build Cython Version (Default) ⭐

**Recommended for maximum security** - Binary compiled code.

```bash
# Build Cython version
python build.py --backend cython

# Or directly
cd build
LAM_BACKEND=cython python setup.py build_ext --inplace
```

**What's included:**
- ✅ Compiled `.so` files (`_core.so`, `_secrets.so`, `_license.so`)
- ✅ Dependencies: `torch`, `numpy`, `tokenizers`
- ❌ No JAX files
- ❌ No JAX dependencies

**Security:** High (binary compiled code)

### Option 2: Build JAX Version 🚀

**Recommended for maximum speed** - 6-8x faster.

```bash
# Build JAX version
python build.py --backend jax

# With protection (obfuscated)
python build.py --backend jax --protect-jax

# Or directly
cd build
LAM_BACKEND=jax python setup.py install
```

**What's included:**
- ✅ JAX Python files (`_jax_core.py`, `_jax_model_optimized.py`, `_jax_model.py`)
- ✅ Dependencies: `torch`, `numpy`, `tokenizers`, `jax`, `jaxlib`
- ❌ No Cython binaries
- ❌ No `.so` files

**Security:** Medium-High (obfuscated Python, if using `--protect-jax`)

## Build Commands

### Using build.py (Recommended)

```bash
# Cython version (default)
python build.py --backend cython

# JAX version
python build.py --backend jax

# JAX version with protection
python build.py --backend jax --protect-jax
```

### Using setup.py directly

```bash
cd build

# Cython version
LAM_BACKEND=cython python setup.py build_ext --inplace

# JAX version
LAM_BACKEND=jax python setup.py install
```

## Publishing to PyPI

### Publish Cython Version

```bash
# Build Cython version
python build.py --backend cython

# Create distribution
cd build
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

### Publish JAX Version

```bash
# Build JAX version (with protection recommended)
python build.py --backend jax --protect-jax

# Create distribution
cd build
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

## Backend Comparison

| Aspect | Cython | JAX |
|--------|--------|-----|
| **Build Command** | `--backend cython` | `--backend jax` |
| **Speed** | Baseline | 6-8x faster |
| **Security** | High (binary) | Medium-High (obfuscated) |
| **Dependencies** | torch, numpy, tokenizers | + jax, jaxlib |
| **File Types** | `.so` binaries | `.py` files |
| **Warmup** | None | First call per shape |
| **Recommended For** | Production, security-critical | Performance-critical |

## Decision Guide

### Choose Cython if:
- ✅ Security is top priority (binary compiled code)
- ✅ You want smaller package size (no JAX dependencies)
- ✅ You don't need the extra speed
- ✅ You're publishing to production

### Choose JAX if:
- ✅ Performance is top priority (6-8x faster)
- ✅ You're okay with obfuscated (not binary) code
- ✅ You can use `--protect-jax` for better security
- ✅ You're doing batch processing or benchmarks

## Protection Options

### Cython Version
- ✅ Already protected (compiled to binary)
- ✅ No additional steps needed

### JAX Version
- ⚠️ Source code is visible (Python files)
- ✅ Use `--protect-jax` to obfuscate
- ✅ Requires: `pip install pyarmor`

## Example Workflow

### Publishing Cython Version

```bash
# 1. Build Cython version
python build.py --backend cython

# 2. Test
python test.py --backend cython

# 3. Create distribution
cd build
python setup.py sdist bdist_wheel

# 4. Upload
twine upload dist/*
```

### Publishing JAX Version

```bash
# 1. Build JAX version with protection
python build.py --backend jax --protect-jax

# 2. Test
python test.py --backend jax

# 3. Create distribution
cd build
python setup.py sdist bdist_wheel

# 4. Upload
twine upload dist/*
```

## Important Notes

1. **Mutually Exclusive**: You build EITHER Cython OR JAX, not both
2. **Publisher Choice**: You (the publisher) decide which version to build
3. **End User**: End users get the version you built - they don't choose
4. **Runtime**: The built version determines which backend is available
5. **Protection**: JAX version should use `--protect-jax` for better security

## Summary

- **Cython**: `python build.py --backend cython` → Binary, secure, baseline speed
- **JAX**: `python build.py --backend jax --protect-jax` → Obfuscated, fast, 6-8x speedup

Choose based on your priorities: **security** (Cython) or **speed** (JAX).


