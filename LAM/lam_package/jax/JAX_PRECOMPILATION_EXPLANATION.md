# JAX Pre-Compilation at Initialization

## The Problem

**JAX compiles at runtime** (JIT - Just-In-Time), which means:
- First call is slow (compilation happens)
- Each unique input shape needs compilation
- **Too slow for production** - users experience delays

**Cython is pre-compiled** at build time:
- Compiled to `.so` binary files
- Fast from the start
- No runtime compilation

## The Solution: Pre-Compile at Initialization

JAX now **pre-compiles functions at model initialization** (like Cython is pre-compiled):

```python
model = LAM("LAM-base-v1", backend='jax')
# ↑ Compilation happens HERE (during initialization)
# Functions are now ready for production use

embeddings = model.encode(sentences)
# ↑ Fast! Already compiled, no delay
```

## How It Works

1. **Model Initialization**: When you create `LAM(..., backend='jax')`
   - JAX pre-compiles functions for common input shapes
   - Compilation happens once, upfront
   - Functions are cached and ready to use

2. **First Use**: When you call `model.encode()`
   - **Fast!** Functions are already compiled
   - No compilation delay
   - Production-ready performance

## Pre-Compiled Shapes

JAX pre-compiles for common shapes:
- Single sentences: (1, 16), (1, 32), (1, 64), (1, 128), (1, 256), (1, 512)
- Small batches: (2-8, 32-128)
- Medium batches: (16-32, 32-128) - most common in production
- Larger batches: (64, 128)

**Total: ~22 common shapes pre-compiled**

## Comparison

| Aspect | Cython | JAX (Before) | JAX (Now) |
|--------|--------|--------------|-----------|
| **Compilation Time** | Build time | First use | **Initialization** |
| **First Call Speed** | Fast | Slow (compiles) | **Fast (pre-compiled)** |
| **Production Ready** | ✅ Yes | ❌ No | **✅ Yes** |
| **User Experience** | Fast | Slow first call | **Fast from start** |

## Important Notes

### Shape-Specific Compilation

JAX compiles per unique input shape. If you use a shape that wasn't pre-compiled:
- First call with that shape will compile (slow)
- Subsequent calls with same shape are fast (cached)

**Solution**: Pre-compilation covers most common shapes. For edge cases, first call compiles, then it's fast.

### Initialization Time

JAX initialization is slower than Cython because:
- Cython: Already compiled (just loads `.so` file)
- JAX: Compiles ~22 shapes during initialization

**Trade-off**: Slower initialization, but fast from first use (production-ready).

## Usage

```python
from lam import LAM

# Initialization (compilation happens here)
model = LAM("LAM-base-v1", backend='jax')
# ↑ Takes ~30-40s (one-time compilation cost)

# First use (fast! already compiled)
embeddings = model.encode(sentences)
# ↑ Fast! No compilation delay

# Subsequent uses (fast! cached)
embeddings2 = model.encode(more_sentences)
# ↑ Fast! Uses cached compilation
```

## Recommendation

**For Production:**
- ✅ **Cython**: Fastest initialization, pre-compiled, production-ready
- ✅ **JAX**: Now also production-ready (pre-compiled at init), but slower initialization

**Choose based on:**
- **Cython**: If you want fastest initialization and maximum code protection (binary)
- **JAX**: If you want maximum speed after initialization (6-8x faster than Cython)

Both are now production-ready! 🎉

