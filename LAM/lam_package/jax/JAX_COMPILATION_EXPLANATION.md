# JAX Compilation vs Cython Compilation

## Overview

**Cython** and **JAX** use fundamentally different compilation approaches:

### Cython (Ahead-of-Time Compilation)
- **Build Time**: `.pyx` files → compiled to `.so` (shared object) files
- **Result**: Binary `.so` files (e.g., `_core.cpython-312-x86_64-linux-gnu.so`)
- **Location**: In `lam/` directory alongside Python files
- **How it works**:
  1. Write code in `.pyx` (Cython syntax)
  2. Run `python setup.py build_ext` or `pip install`
  3. Cython compiler generates C code
  4. C compiler generates `.so` binary
  5. Python imports the `.so` file at runtime

**Example:**
```bash
# Build time
lam/_core.pyx → lam/_core.c → lam/_core.cpython-312-x86_64-linux-gnu.so

# Runtime
import lam._core  # Loads the .so file
```

### JAX (Just-In-Time Compilation)
- **Build Time**: No compilation - just Python `.py` files
- **Runtime**: Python code → XLA/HLO → GPU/CPU code (compiled on first call)
- **Result**: Cached compiled functions in memory
- **Location**: Python `.py` files (e.g., `_jax_core.py`, `_jax_model_optimized.py`)
- **How it works**:
  1. Write code in Python with `@jax.jit` decorator
  2. On first function call, JAX:
     - Traces the function execution
     - Converts to XLA (Accelerated Linear Algebra) IR
     - Compiles to GPU/CPU code
     - Caches the compiled function
  3. Subsequent calls use the cached compiled function

**Example:**
```python
# File: lam/_jax_core.py (just Python, no .so file)
@jax.jit
def hierarchical_delta_rule_jax(...):
    # Python code
    ...

# First call: compiles (slow, ~3-7 seconds)
result = hierarchical_delta_rule_jax(...)  # Compilation happens here

# Subsequent calls: uses cached compilation (fast, ~2-8ms)
result = hierarchical_delta_rule_jax(...)  # Fast!
```

## Key Differences

| Aspect | Cython | JAX |
|--------|--------|-----|
| **Compilation Time** | Build time (once) | Runtime (first call per shape) |
| **Output Files** | `.so` binary files | Python `.py` files |
| **Compilation Target** | C → machine code | Python → XLA → GPU/CPU code |
| **Caching** | N/A (always compiled) | In-memory cache per shape |
| **Portability** | Platform-specific (`.so` per OS/arch) | Cross-platform (Python is portable) |
| **Speed** | Fast (pre-compiled) | Fast after warmup (JIT-compiled) |
| **Debugging** | Harder (compiled binary) | Easier (Python source) |

## Why No `.so` File for JAX?

JAX doesn't create `.so` files because:
1. **JIT Compilation**: Code is compiled at runtime, not build time
2. **Shape-Specific**: Each different input shape gets its own compiled function
3. **Dynamic**: Can recompile for different devices (CPU/GPU) without rebuilding
4. **Python Native**: Stays as Python until execution, making it easier to debug

## Where Are JAX Files?

JAX implementation files are in `lam/` directory:
- `lam/_jax_core.py` - Core hierarchical delta rule
- `lam/_jax_model_optimized.py` - Full model forward pass
- `lam/_jax_model.py` - Alternative model implementation

These are **regular Python files** - no compilation needed!

## Compilation Cache

JAX caches compiled functions in memory. You can see this in action:

```python
import jax
from lam import LAM

model = LAM('path/to/model', backend='jax')

# First call: compiles (slow)
emb1 = model.encode(['Hello'])  # ~3-7 seconds

# Second call: uses cache (fast)
emb2 = model.encode(['Hello'])  # ~2-8ms

# Different shape: compiles again (slow)
emb3 = model.encode(['Hello world'])  # ~3-7 seconds

# Same shape again: uses cache (fast)
emb4 = model.encode(['Hello world'])  # ~2-8ms
```

## Summary

- **Cython**: Pre-compiled `.so` files → fast, but platform-specific
- **JAX**: Python files with JIT compilation → fast after warmup, cross-platform
- **Both**: Produce fast, optimized code, just at different stages

The JAX approach is more flexible (can recompile for different devices) but requires warmup for each input shape.


