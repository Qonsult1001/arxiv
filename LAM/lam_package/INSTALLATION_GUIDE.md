# LAM Installation Guide

## Installation Options

LAM supports two backends: **Cython** (default) and **JAX** (optional, faster). You can install either or both.

### Option 1: Cython Only (Default) ⭐

**Recommended for most users** - Stable, protected binary code.

```bash
pip install lam-attn
```

**Includes:**
- ✅ Cython backend (compiled `.so` files)
- ✅ Full functionality
- ✅ No additional dependencies

**Usage:**
```python
from lam import LAM

model = LAM('path/to/model', backend='cython')  # or just backend='cython' (default)
```

### Option 2: JAX Backend (Faster) 🚀

**Recommended for performance** - 6-8x faster than Cython after warmup.

```bash
pip install lam-attn[jax]
```

**Includes:**
- ✅ Cython backend (default)
- ✅ JAX backend (optional, faster)
- ✅ JAX dependencies (`jax`, `jaxlib`)

**Usage:**
```python
from lam import LAM

# Use JAX backend (faster)
model = LAM('path/to/model', backend='jax')

# Or use Cython backend (default)
model = LAM('path/to/model', backend='cython')
```

### Option 3: All Features

Install everything (same as JAX option):

```bash
pip install lam-attn[all]
```

## Backend Comparison

| Feature | Cython | JAX |
|---------|--------|-----|
| **Speed** | Baseline | 6-8x faster |
| **Protection** | Binary (`.so`) | Obfuscated Python |
| **Dependencies** | None (included) | `jax`, `jaxlib` |
| **Warmup** | None | First call per shape |
| **GPU Support** | ✅ Yes | ✅ Yes |
| **CPU Support** | ✅ Yes | ✅ Yes |

## Installation Examples

### Basic Installation (Cython only)
```bash
pip install lam-attn
```

### With JAX Support
```bash
pip install lam-attn[jax]
```

### From Source
```bash
git clone https://github.com/said-research/lam
cd lam/lam_package
pip install -e .  # Cython only
# or
pip install -e .[jax]  # With JAX
```

### Development Installation
```bash
git clone https://github.com/said-research/lam
cd lam/lam_package
pip install -e ".[jax,dev]"  # If dev extras exist
```

## Runtime Backend Selection

You can choose the backend at runtime, regardless of installation:

```python
from lam import LAM

# Use Cython (always available)
model_cython = LAM('path/to/model', backend='cython')

# Use JAX (if installed)
try:
    model_jax = LAM('path/to/model', backend='jax')
except ImportError:
    print("JAX backend not installed. Install with: pip install lam-attn[jax]")
```

## Requirements

### Minimum Requirements (Cython only)
- Python >= 3.8
- PyTorch >= 2.0
- NumPy
- tokenizers

### JAX Requirements (if using JAX backend)
- All of the above, plus:
- JAX >= 0.4.0
- jaxlib >= 0.4.0

## Verification

After installation, verify your setup:

```python
from lam import LAM

# Test Cython backend
model = LAM('path/to/model', backend='cython')
embeddings = model.encode(['Hello world'])
print(f"Cython backend works! Shape: {embeddings.shape}")

# Test JAX backend (if installed)
try:
    model_jax = LAM('path/to/model', backend='jax')
    embeddings_jax = model_jax.encode(['Hello world'])
    print(f"JAX backend works! Shape: {embeddings_jax.shape}")
except ImportError as e:
    print(f"JAX backend not available: {e}")
    print("Install with: pip install lam-attn[jax]")
```

## Troubleshooting

### JAX Backend Not Available

**Error:** `ImportError: JAX backend requires jax and jaxlib packages`

**Solution:**
```bash
pip install lam-attn[jax]
```

### CUDA/GPU Issues

**JAX GPU Support:**
```bash
# For CUDA 11.x
pip install jax[cuda11_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 12.x
pip install jax[cuda12_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Cython GPU Support:**
- Automatically uses CUDA if PyTorch was installed with CUDA support
- No additional setup needed

### Performance Issues

**JAX is slow on first call:**
- This is normal! JAX compiles functions on first use
- Subsequent calls with the same input shape are fast
- Warm up your model before benchmarking

**Example warmup:**
```python
model = LAM('path/to/model', backend='jax')
# Warm up
_ = model.encode(['warmup sentence'])
# Now fast
embeddings = model.encode(['actual sentence'])
```

## Which Backend Should I Use?

### Use Cython if:
- ✅ You want maximum code protection (binary)
- ✅ You don't need the extra speed
- ✅ You want simpler dependencies
- ✅ You're deploying to production

### Use JAX if:
- ✅ You need maximum performance (6-8x faster)
- ✅ You're doing batch processing
- ✅ You're running benchmarks
- ✅ You're okay with obfuscated (not binary) code

### Use Both:
- ✅ Install both: `pip install lam-attn[jax]`
- ✅ Choose at runtime based on your needs
- ✅ Best of both worlds!

## Summary

```bash
# Cython only (default, recommended)
pip install lam-attn

# With JAX support (faster, optional)
pip install lam-attn[jax]

# Both backends available at runtime
from lam import LAM
model = LAM('path/to/model', backend='jax')  # or 'cython'
```


