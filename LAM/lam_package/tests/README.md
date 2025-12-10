# LAM CPU Tests

This directory contains test scripts for verifying LAM performance on CPU.

## Test Files

### `test_cpu_baseline.py`
Basic CPU test that verifies LAM works correctly on CPU-only systems. Tests different token lengths up to 8192 tokens and verifies:
- Model loads correctly on CPU
- Embeddings have correct shape (1, 384)
- Embeddings are properly normalized
- Similarity computation works

**Usage:**
```bash
cd /workspace/LAM/lam_package/tests
python test_cpu_baseline.py
```

### `test_cpu_optimizations.py`
Comprehensive test that compares baseline performance against three optimization methods:
1. **torch.compile** - PyTorch 2.4+ compilation (×1.8-2.2 speedup)
2. **ONNX + ONNX Runtime** - ONNX export and inference (×2.8-3.3 speedup)
3. **Dynamic INT8 Quantization** - INT8 quantization (×3.5-4.0 speedup)

Tests all methods on token lengths: 128, 512, 1024, 2048, 4096, 8192

**Usage:**
```bash
cd /workspace/LAM/lam_package/tests
python test_cpu_optimizations.py
```

**Requirements for full test:**
- `torch` (PyTorch 2.4+ for torch.compile)
- `onnx` and `onnxruntime` (for ONNX test)
- `einops` (for LAM core module)
- `tokenizers` (for tokenization)

**Install missing dependencies:**
```bash
pip install onnx onnxruntime einops tokenizers
```

## Expected Results

### Baseline (8192 tokens)
- ~10-11 seconds on CPU

### With torch.compile
- ~4.7-5.8 seconds (×1.8-2.2 speedup)

### With ONNX Runtime
- ~2.9-3.4 seconds (×2.8-3.3 speedup)

### With INT8 Quantization
- ~2.4-2.9 seconds (×3.5-4.0 speedup)

## Notes

- All tests force CPU usage (no GPU)
- Tests include warmup runs to ensure accurate timing
- Each optimization method is tested independently
- Results are averaged over multiple runs for accuracy







