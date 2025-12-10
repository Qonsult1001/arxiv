# Protecting JAX Source Code

## Overview

Unlike Cython code which is compiled to `.so` binary files, JAX code remains as readable Python source. This guide explains how to protect your JAX implementation.

## Protection Options

### Option 1: PyArmor (Recommended for JAX)

**PyArmor** obfuscates Python code while maintaining compatibility with JAX JIT compilation.

#### Installation
```bash
pip install pyarmor
```

#### Usage
```bash
# Run the protection script
python protect_jax_code.py
```

#### What it does:
1. Backs up original JAX files to `lam/_jax_backup/`
2. Obfuscates files using PyArmor
3. Replaces original files with obfuscated versions
4. Maintains JAX JIT compatibility

#### Pros:
- ✅ Works with JAX JIT compilation
- ✅ Maintains Python import system
- ✅ Can be integrated into build process
- ✅ Free and open source

#### Cons:
- ⚠️ Not as secure as compiled binaries (determined attackers can still reverse engineer)
- ⚠️ Adds small runtime overhead
- ⚠️ Requires PyArmor runtime library

### Option 2: Convert to Cython

Convert JAX code to Cython and compile to `.so` files (like the Cython backend).

#### Challenges:
- JAX uses `@jax.jit` decorators which don't work in Cython
- Would need to rewrite JAX code to use NumPy/C operations
- JAX's JIT compilation wouldn't work
- Significant code rewrite required

#### Pros:
- ✅ Binary protection (like Cython backend)
- ✅ No runtime overhead
- ✅ Same protection level as existing Cython code

#### Cons:
- ❌ Requires major code rewrite
- ❌ Loses JAX JIT benefits
- ❌ More complex to maintain

### Option 3: Nuitka (Compile to Binary)

**Nuitka** compiles Python to C++ and then to binary executables.

#### Installation
```bash
pip install nuitka
```

#### Usage
```bash
# Compile JAX module to binary
python -m nuitka --module lam/_jax_core.py
```

#### Pros:
- ✅ True binary compilation
- ✅ Better protection than obfuscation
- ✅ Can create standalone executables

#### Cons:
- ⚠️ May break JAX JIT compilation
- ⚠️ Larger file sizes
- ⚠️ Platform-specific binaries
- ⚠️ More complex build process

### Option 4: Hybrid Approach (Recommended)

**Best Practice**: Use PyArmor for JAX code, keep Cython for core logic.

1. **Cython backend**: Already protected (`.so` files)
2. **JAX backend**: Protect with PyArmor
3. **User choice**: Let users choose which backend to use

## Implementation

### Step 1: Protect JAX Files

```bash
# Install PyArmor
pip install pyarmor

# Run protection script
python protect_jax_code.py
```

### Step 2: Update setup.py

Add PyArmor runtime to your package:

```python
# In setup.py
setup(
    ...
    package_data={
        'lam': [
            'pyarmor_runtime/*',  # PyArmor runtime files
            ...
        ],
    },
    ...
)
```

### Step 3: Test Protected Code

```bash
# Test that JAX still works
python -c "from lam import LAM; model = LAM('path/to/model', backend='jax'); print(model.encode(['test']))"
```

### Step 4: Build Package

```bash
# Build with protected code
python setup.py sdist bdist_wheel
```

## Security Levels

| Method | Security Level | JAX Compatible | Effort |
|--------|---------------|----------------|--------|
| **PyArmor** | Medium-High | ✅ Yes | Low |
| **Cython** | High | ❌ No (requires rewrite) | High |
| **Nuitka** | High | ⚠️ Maybe | Medium |
| **No Protection** | None | ✅ Yes | None |

## Recommendations

1. **For Production**: Use PyArmor to protect JAX code
2. **For Maximum Security**: Consider converting critical parts to Cython
3. **For Development**: Keep original code, protect only for releases

## Restoring Original Code

If you need to restore original files:

```bash
# Restore from backup
cp lam/_jax_backup/*.py lam/
```

## License Considerations

- PyArmor is free for open-source projects
- Commercial use may require PyArmor Pro license
- Check PyArmor license for your use case

## Testing Protected Code

After protection, test thoroughly:

```python
# Test JAX backend
from lam import LAM

model = LAM('path/to/model', backend='jax')

# Test encoding
embeddings = model.encode(['Hello world', 'Test sentence'])
print(f"Embeddings shape: {embeddings.shape}")

# Test similarity
from scipy.stats import spearmanr
# ... run your tests
```

## Notes

- **JAX JIT**: PyArmor maintains JAX JIT compilation compatibility
- **Performance**: Minimal overhead from obfuscation
- **Distribution**: Include PyArmor runtime in your package
- **Updates**: Re-protect code after each update

## Alternative: Keep JAX Code Open

If protection isn't critical, you can:
- Keep JAX code as reference implementation
- Use Cython backend for production (already protected)
- Let users choose which backend to use

This is acceptable if:
- Core algorithms are in Cython (protected)
- JAX is just an alternative implementation
- Performance is the main concern, not code secrecy


