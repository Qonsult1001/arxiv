# JAX Implementation (Archived)

This folder contains the JAX implementation of LAM, which has been moved out of the main production package.

## Status

**Not in production use** - The JAX backend has been removed from the main LAM package.

## Why Moved?

1. **Persistent cache issues**: JAX's persistent compilation cache doesn't work reliably across sessions
2. **Production focus**: Main package is now Cython-only for production stability
3. **Future work**: JAX implementation preserved here for future development

## Contents

- `_jax_core.py` - Core hierarchical delta rule implementation
- `_jax_model_optimized.py` - Optimized full model forward pass
- `_jax_model.py` - Alternative model implementation
- Documentation files (JAX_*.md)
- Test/benchmark files

## Notes

- JAX implementation was fully functional and accurate (1:1 with Cython)
- In-memory cache works within same Python session
- Persistent cache across sessions does not work (JAX limitation)
- For production, use the Cython backend in the main package

## Future

This code is preserved for potential future integration once JAX's persistent cache issues are resolved.

