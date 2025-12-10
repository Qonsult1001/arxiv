# Build Directory

This directory contains all build-related files for the LAM package.

## Files

- **`setup.py`** - Main build script (run from this directory)
- **`_core.py`** - Core LAM algorithm source (copied from formula)
- **`_secrets.pyx`** - Proprietary position interpolation source

## Building

Run the build from this directory:

```bash
cd build
python3 setup.py build_ext --inplace
```

**Important:** Always use `build_ext --inplace` command. Running just `python3 setup.py` will show an error.

The build will:
1. Compile `_core.py` → `../lam/_core.so`
2. Compile `_secrets.pyx` → `../lam/_secrets.so`
3. Automatically clean up intermediate `.c` files from `build/` folder

This will:
1. Compile `_core.py` → `../lam/_core.so`
2. Compile `_secrets.pyx` → `../lam/_secrets.so`
3. Clean up intermediate `.c` files automatically

## Output

Compiled binaries (`.so` files) are placed in the `../lam/` directory, which is the final package location.

