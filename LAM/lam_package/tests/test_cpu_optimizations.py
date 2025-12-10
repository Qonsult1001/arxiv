#!/usr/bin/env python3
"""
LAM CPU ONNX Optimization Test Script
======================================
Tests LAM model on CPU with ONNX + ONNX Runtime optimization.
Uses fixed-length export (8192 tokens) with INT8 quantization for optimal performance.
Note: Fixed length is required because position embedding interpolation doesn't support dynamic shapes in ONNX.
"""

import torch
import time
import sys
from pathlib import Path
import numpy as np
import multiprocessing

# Add parent directory to path to import lam
lam_package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(lam_package_dir))

print("="*80)
print("🧪 LAM CPU ONNX OPTIMIZATION TEST")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print()

# Import LAM
try:
    from lam import LAM
    print("✅ LAM imported successfully")
except Exception as e:
    print(f"❌ Failed to import LAM: {e}")
    sys.exit(1)

# Load model on CPU
print("\n📦 Loading LAM model on CPU...")
try:
    # Try to find LAM-base-v1 model
    model_path = Path(__file__).parent.parent.parent / "LAM-base-v1"
    if not model_path.exists():
        model_path = Path("/workspace/LAM/LAM-base-v1")
        if not model_path.exists():
            raise FileNotFoundError("LAM-base-v1 directory not found")
    
    model = LAM(str(model_path), device='cpu')
    print(f"✅ Model loaded from: {model_path}")
    print(f"   Device: {model.device}")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Configuration
MAX_SEQ_LEN = 8192  # Fixed export length (required for position embeddings)
test_dir = Path(__file__).parent
fp32_path = test_dir / f"lam_fixed_{MAX_SEQ_LEN}_fp32.onnx"
int8_path = test_dir / f"lam_fixed_{MAX_SEQ_LEN}_int8.onnx"

print(f"\n📋 Configuration:")
print(f"   Fixed-length export: {MAX_SEQ_LEN} tokens (required for position embeddings)")
print(f"   FP32 model: {fp32_path}")
print(f"   INT8 model: {int8_path}")
print(f"   Note: All inputs will be padded to {MAX_SEQ_LEN} tokens")
print(f"   This is required because position embedding interpolation doesn't support dynamic shapes in ONNX")

# ============================================================================
# Fixed-Length ONNX Export + Quantization
# ============================================================================
print("\n" + "="*80)
print("⚡ Fixed-Length ONNX Export + Quantization")
print("="*80)

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print("   ✅ ONNX and ONNX Runtime available")
    
    # Check if INT8 model already exists
    if int8_path.exists():
        print(f"   ✅ INT8 model already exists: {int8_path}")
        print(f"   Size: {int8_path.stat().st_size / 1024 / 1024:.1f} MB")
        print("   Skipping export (delete file to re-export)")
    else:
        # Create wrapper for export
        class ONNXExportWrapper(torch.nn.Module):
            def __init__(self, lam_model):
                super().__init__()
                self.inner = lam_model._model
            
            def forward(self, input_ids, attention_mask):
                return self.inner.get_sentence_embeddings(input_ids, attention_mask)
        
        wrapper = ONNXExportWrapper(model)
        wrapper.eval()
        
        # 1. Export FP32 model with FIXED length (position embeddings require this)
        if not fp32_path.exists():
            print("   [1/2] Exporting FP32 model with fixed length...")
            print(f"   Using fixed length {MAX_SEQ_LEN} (position embeddings don't support dynamic shapes in ONNX)")
            
            # Use FIXED length dummy input (no dynamic axes)
            print(f"   Exporting with fixed shape [1, {MAX_SEQ_LEN}]...")
            dummy_ids = torch.randint(0, 1000, (1, MAX_SEQ_LEN), dtype=torch.long, device='cpu')
            dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long, device='cpu')
            
            export_start = time.time()
            with torch.no_grad():
                # Try new dynamo exporter first (PyTorch 2.9+)
                try:
                    torch.onnx.export(
                        wrapper,
                        (dummy_ids, dummy_mask),
                        str(fp32_path),
                        opset_version=17,
                        input_names=['input_ids', 'attention_mask'],
                        output_names=['embeddings'],
                        # NO dynamic_axes - fixed shape required for position embeddings
                        export_params=True,
                        do_constant_folding=True,
                        verbose=False,
                        dynamo=True,  # Use new exporter
                    )
                    print("   ✅ Used new dynamo exporter")
                except Exception as e:
                    print(f"   ⚠️  New exporter failed: {e}")
                    print("   Falling back to legacy exporter...")
                    # Fallback to legacy exporter
                    torch.onnx.export(
                        wrapper,
                        (dummy_ids, dummy_mask),
                        str(fp32_path),
                        opset_version=17,
                        input_names=['input_ids', 'attention_mask'],
                        output_names=['embeddings'],
                        # NO dynamic_axes - fixed shape
                        export_params=True,
                        do_constant_folding=True,
                        verbose=False,
                    )
            
            export_time = time.time() - export_start
            file_size = fp32_path.stat().st_size / 1024 / 1024
            print(f"   ✅ FP32 export complete in {export_time:.1f}s ({file_size:.1f} MB)")
        else:
            print(f"   ✅ FP32 model already exists: {fp32_path}")
        
        # 2. Apply INT8 Quantization
        print("   [2/2] Quantizing to INT8 (this is where the speed comes from)...")
        print("   This compresses the model and uses CPU instructions optimized for integers.")
        
        quantize_start = time.time()
        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QUInt8
        )
        
        quantize_time = time.time() - quantize_start
        file_size = int8_path.stat().st_size / 1024 / 1024
        print(f"   ✅ INT8 quantization complete in {quantize_time:.1f}s ({file_size:.1f} MB)")
    
    # ============================================================================
    # Create ONNX Runtime Session (using INT8 model)
    # ============================================================================
    print("\n   Creating ONNX Runtime session (using INT8 model)...")
    
    num_cores = min(multiprocessing.cpu_count(), 16)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = num_cores
    sess_opts.inter_op_num_threads = 2
    sess_opts.enable_cpu_mem_arena = True
    sess_opts.enable_mem_pattern = True
    
    # Use the quantized INT8 model for inference
    sess = ort.InferenceSession(
        str(int8_path), 
        sess_options=sess_opts, 
        providers=['CPUExecutionProvider']
    )
    
    print(f"   ✅ Session created (using {num_cores} threads)")
    for inp in sess.get_inputs():
        print(f"   Input: {inp.name} {inp.shape} (fixed)")
    
    # ============================================================================
    # Inference Function (No Padding - Uses Actual Token Length)
    # ============================================================================
    def encode_onnx(text, max_tokens):
        """Encode text using ONNX model. Pads to MAX_SEQ_LEN (required for fixed export)."""
        # Tokenize
        model.tokenizer.enable_truncation(max_length=max_tokens)
        enc = model.tokenizer.encode(text)
        
        # Get actual tokenized length
        ids = enc.ids
        mask = enc.attention_mask
        actual_len = len(ids)
        
        # PAD TO FIXED LENGTH (required because export uses fixed shape)
        pad_len = MAX_SEQ_LEN - actual_len
        ids_padded = ids + [0] * pad_len
        mask_padded = mask + [0] * pad_len
        
        # Convert to numpy arrays with fixed shape [1, MAX_SEQ_LEN]
        input_ids = np.array([ids_padded], dtype=np.int64)
        attention_mask = np.array([mask_padded], dtype=np.int64)
        
        # Verify shapes match expected fixed shape
        assert input_ids.shape == (1, MAX_SEQ_LEN), \
            f"Shape mismatch: expected (1, {MAX_SEQ_LEN}), got {input_ids.shape}"
        assert attention_mask.shape == (1, MAX_SEQ_LEN), \
            f"Shape mismatch: expected (1, {MAX_SEQ_LEN}), got {attention_mask.shape}"
        
        # Run inference with fixed shape
        out = sess.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})
        emb = out[0]
        
        # Normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        
        return emb, actual_len
    
    # ============================================================================
    # Benchmark
    # ============================================================================
    print("\n" + "="*80)
    print("🚀 Performance Benchmark")
    print("="*80)
    
    test_text = "The quick brown fox jumps over the lazy dog. " * 500
    test_lengths = [128, 512, 1024, 2048, 4096, 8192]
    
    print(f"\n{'Tokens':<10} {'Actual':<10} {'Time (s)':<15} {'Tokens/sec':<15}")
    print("-" * 60)
    
    onnx_results = {}
    for max_tok in test_lengths:
        try:
            # Warmup
            for _ in range(2):
                _ = encode_onnx(test_text, max_tok)
            
            # Benchmark
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                emb, actual = encode_onnx(test_text, max_tok)
                times.append(time.perf_counter() - t0)
            
            avg_time = sum(times) / len(times)
            # Tokens/sec based on ACTUAL tokens (padded to MAX_SEQ_LEN for fixed export)
            tps = actual / avg_time if avg_time > 0 else 0
            onnx_results[max_tok] = avg_time
            
            print(f"{max_tok:<10} {actual:<10} {avg_time:<15.3f} {tps:<15.0f}")
            
        except Exception as e:
            print(f"{max_tok:<10} {'FAILED':<10} {str(e)[:40]}")
            break
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY")
    print("="*80)
    
    if 8192 in onnx_results:
        onnx_8192 = onnx_results[8192]
        tok_per_sec = 8192 / onnx_8192 if onnx_8192 > 0 else 0
        print(f"\n🎯 8192 tokens: {onnx_8192:.2f}s ({tok_per_sec:.0f} tok/s)")
    
    print(f"\n💡 Key improvements:")
    print("   ✅ INT8 quantization: Faster inference with optimized CPU instructions")
    print("   ✅ Fixed-length export: Required for position embedding compatibility")
    print(f"   ⚠️  Note: All inputs padded to {MAX_SEQ_LEN} tokens (compute cost is constant)")
    print("   This is a limitation of ONNX export with position embedding interpolation.")
    
except ImportError:
    print("   ⚠️  ONNX or ONNX Runtime not installed.")
    print("   Install with: pip install onnx onnxruntime")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ ONNX test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ ONNX OPTIMIZATION TEST COMPLETE!")
print("="*80)
