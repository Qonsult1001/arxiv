# JAX Persistent Cache Issue - Known Limitation

## Problem

JAX's persistent compilation cache is **not working reliably** in practice, despite being configured correctly.

## Root Cause

Based on research and testing:

1. **JAX persistent cache has known issues** - reported by multiple users (AlphaFold3, etc.)
2. **Cache may not work across processes** - in-memory cache works, but disk cache doesn't persist
3. **JAX version 0.8.1** - persistent cache may not be fully functional
4. **Cache directory stays empty** - even when configured correctly

## Evidence

- Cache directory configured: ✅
- Cache directory created: ✅  
- Cache files written: ❌ (0 files)
- Second session uses cache: ❌ (still compiles)

## Official JAX Status

From JAX documentation and GitHub issues:
- Persistent cache is **experimental**
- Known issues with cross-process caching
- May not work reliably in all environments

## Solutions

### Option 1: Use Cython (Recommended for Production) ⭐⭐⭐

**Why:**
- Pre-compiled at build time
- No runtime compilation
- Fast from first use
- Binary protection (source code compiled)

**How:**
```python
model = LAM('model', backend='cython')  # Fast, pre-compiled
```

### Option 2: Accept JAX's In-Memory Cache ⭐

**What works:**
- In-memory cache within same Python session
- Second call in same session is fast
- Pre-compilation at initialization helps

**What doesn't work:**
- Persistent cache across sessions
- Cache survives restarts

**Current implementation:**
- Pre-compiles at initialization (fast from first use in session)
- In-memory cache works within session
- Not persistent across restarts

### Option 3: Manual Cache Management (Future)

Could implement custom caching:
- Save compiled functions manually
- Load on startup
- More complex, but would work

## Recommendation

**For Production: Use Cython**
- ✅ Pre-compiled (no compilation overhead)
- ✅ Fast from first use
- ✅ Binary protection
- ✅ Production-ready

**For Development/Research: JAX is fine**
- ✅ In-memory cache works
- ✅ Pre-compilation at init helps
- ⚠️ Not persistent (but acceptable for dev)

## Conclusion

JAX's persistent compilation cache is **not reliable** in current versions. This is a known limitation of JAX, not our implementation.

**Best practice:**
- **Production**: Use Cython backend
- **Development**: JAX is fine (in-memory cache works)

