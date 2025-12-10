"""
Benchmark: Chunk Loop Operations from final_solution_formula_final.py

This benchmarks the ACTUAL operations in the current formula:
1. State decay + normalize (for readout)
2. Matrix multiplications (readout operations)
3. Token-level output blending
4. State updates + cross-interaction + normalize

The chunk LOOP itself cannot be parallelized (sequential dependency),
but we can benchmark the operations WITHIN each iteration.
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from pathlib import Path
import sys
import os
import shutil

# Enable TensorFloat32 (TF32) for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

sys.path.insert(0, str(Path(__file__).parent))

from final_solution_formula_final import create_enhanced_semantic_model


def clear_torch_cache():
    """Clear torch inductor cache to free disk space."""
    cache_dirs = [
        '/tmp/torchinductor_root',
        '/tmp/torch_compile_cache',
        '/tmp/.torch_cache',
    ]
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
            except:
                pass  # Ignore errors if cache is locked
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_chunk_loop_operations():
    """Benchmark individual operations from the actual chunk loop."""
    print("=" * 80)
    print("CHUNK LOOP OPERATIONS BENCHMARK")
    print("=" * 80)
    print("\nBenchmarking operations from final_solution_formula_final.py")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32 if device == 'cuda' else 'N/A'}")
    
    if device == 'cpu':
        print("\n⚠️  WARNING: Running on CPU. GPU recommended for accurate benchmarks.\n")
        return
    
    # Ensure all operations stay on GPU
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    # Test configurations (simulating different batch/head/dimension combinations)
    # Typical values: b=2-4, h=12, d_k=64, d_v=64, chunk_size=32-64
    configs = [
        (2, 12, 64, 64, "Small (b=2, h=12, d=64)"),
        (4, 12, 64, 64, "Medium (b=4, h=12, d=64)"),
        (8, 12, 64, 64, "Large (b=8, h=12, d=64)"),
    ]
    
    print("\n" + "=" * 80)
    print("1. STATE DECAY + NORMALIZE (for readout)")
    print("=" * 80)
    print("Current formula: S = S * decay; S_read = S / norm")
    print("Operations: 2 tensor ops (decay + normalize)")
    
    for b, h, d_k, d_v, desc in configs:
        S_fast = torch.randn(b, h, d_k, d_v, device=device)
        S_slow = torch.randn(b, h, d_k, d_v, device=device)
        fast_decay_modulated = torch.rand(b, h, 1, 1, device=device) * 0.5 + 0.5
        slow_decay_modulated = torch.rand(b, h, 1, 1, device=device) * 0.5 + 0.5
        
        # Warmup
        for _ in range(10):
            S_fast_decayed = S_fast * fast_decay_modulated
            S_slow_decayed = S_slow * slow_decay_modulated
            S_fast_read = S_fast_decayed / (S_fast_decayed.norm(dim=(-2, -1), keepdim=True) + 1e-8)
            S_slow_read = S_slow_decayed / (S_slow_decayed.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(100):
            if device == 'cuda':
            torch.cuda.synchronize()
            start = time.perf_counter()
            S_fast_decayed = S_fast * fast_decay_modulated
            S_slow_decayed = S_slow * slow_decay_modulated
            S_fast_read = S_fast_decayed / (S_fast_decayed.norm(dim=(-2, -1), keepdim=True) + 1e-8)
            S_slow_read = S_slow_decayed / (S_slow_decayed.norm(dim=(-2, -1), keepdim=True) + 1e-8)
            if device == 'cuda':
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\n  {desc}:")
        print(f"    Time: {mean_time:.4f} ms ± {std_time:.4f} ms")
    
    print("\n" + "=" * 80)
    print("2. READOUT MATRIX MULTIPLICATIONS")
    print("=" * 80)
    print("Current formula: u_i = u - w @ S_read; o = q @ S_read + attn @ u_i")
    print("Operations: 4 matrix multiplications per chunk (fast + slow paths)")
    
    for b, h, d_k, d_v, desc in configs:
        chunk_size = 32
        q_i = torch.randn(b, h, chunk_size, d_k, device=device)
        k_i = torch.randn(b, h, chunk_size, d_k, device=device)
        u_i = torch.randn(b, h, chunk_size, d_v, device=device)
        w_i = torch.randn(b, h, chunk_size, d_k, device=device)
        attn = torch.randn(b, h, chunk_size, chunk_size, device=device)
        S_fast_read = torch.randn(b, h, d_k, d_v, device=device)
        S_slow_read = torch.randn(b, h, d_k, d_v, device=device)
        # Normalize states
        S_fast_read = S_fast_read / (S_fast_read.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        S_slow_read = S_slow_read / (S_slow_read.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        
        # Warmup
        for _ in range(10):
            u_i_fast = u_i - w_i @ S_fast_read
            o_inter_fast = q_i @ S_fast_read
            o_fast = o_inter_fast + attn @ u_i_fast
            u_i_slow = u_i - w_i @ S_slow_read
            o_inter_slow = q_i @ S_slow_read
            o_slow = o_inter_slow + attn @ u_i_slow
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(100):
            if device == 'cuda':
            torch.cuda.synchronize()
            start = time.perf_counter()
            u_i_fast = u_i - w_i @ S_fast_read
            o_inter_fast = q_i @ S_fast_read
            o_fast = o_inter_fast + attn @ u_i_fast
            u_i_slow = u_i - w_i @ S_slow_read
            o_inter_slow = q_i @ S_slow_read
            o_slow = o_inter_slow + attn @ u_i_slow
            if device == 'cuda':
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\n  {desc}:")
        print(f"    Time: {mean_time:.4f} ms ± {std_time:.4f} ms")
    
    print("\n" + "=" * 80)
    print("3. TOKEN-LEVEL OUTPUT BLENDING")
    print("=" * 80)
    print("Current formula: alpha = 0.5 + 0.3 * token_flux_i; o = alpha * o_fast + (1-alpha) * o_slow")
    print("Operations: Element-wise operations with token-level granularity")
    
    for b, h, c, d_v, desc in [(2, 12, 32, 64, "chunk=32"), (2, 12, 64, 64, "chunk=64")]:
        o_fast = torch.randn(b, h, c, d_v, device=device)
        o_slow = torch.randn(b, h, c, d_v, device=device)
        token_flux_i = torch.rand(b, h, c, 1, device=device) * 0.5 + 0.25  # [0.25, 0.75]
        
        # Warmup
        for _ in range(10):
            alpha = 0.5 + 0.3 * token_flux_i
            beta_weight = 1.0 - alpha
            o_chunk = alpha * o_fast + beta_weight * o_slow
        if device == 'cuda':
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(100):
            if device == 'cuda':
            torch.cuda.synchronize()
            start = time.perf_counter()
            alpha = 0.5 + 0.3 * token_flux_i
            beta_weight = 1.0 - alpha
            o_chunk = alpha * o_fast + beta_weight * o_slow
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\n  {desc}:")
        print(f"    Time: {mean_time:.4f} ms ± {std_time:.4f} ms")
    
    print("\n" + "=" * 80)
    print("4. STATE UPDATE + CROSS-INTERACTION + NORMALIZE")
    print("=" * 80)
    print("Current formula: S = S + update; cross-interaction; normalize")
    print("Operations: 6 tensor operations (add, add, cross, cross, norm, norm)")
    
    for b, h, d_k, d_v, desc in configs:
        update_fast = torch.randn(b, h, d_k, d_v, device=device) * 0.1
        update_slow = torch.randn(b, h, d_k, d_v, device=device) * 0.1
        psi_i = torch.rand(b, h, device=device) * 0.5 + 0.5
        psi_expanded = psi_i.unsqueeze(-1).unsqueeze(-1)
        
        # Warmup
        for _ in range(10):
            S_fast = torch.randn(b, h, d_k, d_v, device=device)
            S_slow = torch.randn(b, h, d_k, d_v, device=device)
            S_fast = S_fast + update_fast
            S_slow = S_slow + update_slow
            cross_influence = 0.05 + 0.1 * psi_i.mean()
            cross_update_fast = cross_influence * psi_expanded * S_slow
            cross_update_slow = cross_influence * (1 - psi_expanded) * S_fast
            S_fast = S_fast + cross_update_fast
            S_slow = S_slow + cross_update_slow
            S_fast_norm = S_fast.norm(dim=(-2, -1), keepdim=True) + 1e-8
            S_slow_norm = S_slow.norm(dim=(-2, -1), keepdim=True) + 1e-8
            S_fast = S_fast / S_fast_norm
            S_slow = S_slow / S_slow_norm
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(100):
            S_fast = torch.randn(b, h, d_k, d_v, device=device)
            S_slow = torch.randn(b, h, d_k, d_v, device=device)
            if device == 'cuda':
            torch.cuda.synchronize()
            start = time.perf_counter()
            S_fast = S_fast + update_fast
            S_slow = S_slow + update_slow
            cross_influence = 0.05 + 0.1 * psi_i.mean()
            cross_update_fast = cross_influence * psi_expanded * S_slow
            cross_update_slow = cross_influence * (1 - psi_expanded) * S_fast
            S_fast = S_fast + cross_update_fast
            S_slow = S_slow + cross_update_slow
            S_fast_norm = S_fast.norm(dim=(-2, -1), keepdim=True) + 1e-8
            S_slow_norm = S_slow.norm(dim=(-2, -1), keepdim=True) + 1e-8
            S_fast = S_fast / S_fast_norm
            S_slow = S_slow / S_slow_norm
            if device == 'cuda':
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\n  {desc}:")
        print(f"    Time: {mean_time:.4f} ms ± {std_time:.4f} ms")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Per-chunk iteration operations (from final_solution_formula_final.py):
- State decay + normalize (readout): 4 ops (fast + slow)
- Readout matrix multiplications: 4 matmuls (fast + slow paths)
- Token-level output blending: 3 ops (alpha, beta, blend)
- State update + cross-interaction + normalize: 6 ops

Total: ~17 operations per chunk iteration

The chunk loop itself is sequential (unavoidable recurrence),
but torch.compile can fuse operations within each iteration.
""")


def benchmark_full_model():
    """Benchmark DeltaNet vs all-MiniLM-L6-v2."""
    print("\n" + "=" * 80)
    print("FULL MODEL BENCHMARK - DeltaNet vs all-MiniLM-L6-v2")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠️  Skipping - requires GPU")
        return
    
    print(f"\nDevice: {device}")
    print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    
    # Load all-MiniLM-L6-v2 for comparison
    print("\nLoading all-MiniLM-L6-v2...")
    from transformers import AutoModel, AutoTokenizer
    
    try:
        minilm_path = "/workspace/LAM/all-MiniLM-L6-v2"
        if not Path(minilm_path).exists():
            minilm_path = "sentence-transformers/all-MiniLM-L6-v2"
        
        minilm_model = AutoModel.from_pretrained(minilm_path).to(device)
        minilm_tokenizer = AutoTokenizer.from_pretrained(minilm_path)
        minilm_model.eval()
        print("✅ all-MiniLM-L6-v2 loaded")
    except Exception as e:
        print(f"⚠️  Could not load all-MiniLM-L6-v2: {e}")
        minilm_model = None
        minilm_tokenizer = None
    
    # Clear cache before loading model
    print("Clearing torch cache...")
    clear_torch_cache()
    
    # Load DeltaNet
    print("Loading DeltaNet...")
    deltanet_model = create_enhanced_semantic_model(hidden_size=384, num_heads=12).to(device)
    deltanet_model.eval()
    
    # ⚡ OPTIMIZE: Use torch.compile with "reduce-overhead" mode and CUDA Graphs
    # This unrolls the sequential chunk loop into a single GPU graph execution plan
    # Expected speedup: 10-20ms overhead reduction
    # Note: For very long sequences (8K+), we use fullgraph=False to avoid huge graphs
    print("Compiling DeltaNet with CUDA Graphs (reduce-overhead mode)...")
    try:
        # Use fullgraph=False for long sequences to avoid disk space issues
        # fullgraph=True generates huge graphs for 8K tokens
        deltanet_model = torch.compile(
            deltanet_model, 
            mode="reduce-overhead",  # Enables CUDA Graphs for sequential loops
            fullgraph=False  # Allow graph breaks to reduce cache size (was True)
        )
        print("✅ DeltaNet compiled with CUDA Graphs")
    except Exception as e:
        print(f"⚠️  torch.compile failed, using uncompiled model: {e}")
        print("✅ DeltaNet loaded (uncompiled)")
        clear_torch_cache()  # Clear cache on failure
    
    test_cases = [
        (1, 512, "Short (512 tokens)"),
        (1, 2048, "Medium (2K tokens)"),
        (1, 8192, "Long (8K tokens)"),
    ]
    
    print(f"\n{'Model':<25} {'Sequence':<20} {'Time (ms)':<15} {'Tokens/sec':<15}")
    print("-" * 75)
    
    for batch, seq_len, desc in test_cases:
        # Clear cache before each test case (especially important for long sequences)
        if seq_len >= 2048:
            print(f"  Clearing cache before {desc}...", end=" ", flush=True)
            clear_torch_cache()
            print("done")
        
        # DeltaNet benchmark
        x_deltanet = torch.randn(batch, seq_len, 384, device=device)
        
        # Warmup - torch.compile needs extra iterations to capture CUDA Graphs
        # The first few calls compile the graph, subsequent calls use the cached graph
        print(f"  Warming up DeltaNet for {desc}...", end=" ", flush=True)
        try:
            for _ in range(10):  # More warmup iterations for compiled model
                with torch.no_grad():
                    _ = deltanet_model(x_deltanet)
            torch.cuda.synchronize()
            print("done")
        except Exception as e:
            print(f"failed: {e}")
            # Clear cache and try uncompiled
            clear_torch_cache()
            print("  Retrying with cache cleared...")
            for _ in range(5):
            with torch.no_grad():
                    _ = deltanet_model(x_deltanet)
        torch.cuda.synchronize()
        
        # Benchmark DeltaNet
        times_deltanet = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = deltanet_model(x_deltanet)
            torch.cuda.synchronize()
            times_deltanet.append((time.perf_counter() - start) * 1000)
        
        mean_time_deltanet = np.mean(times_deltanet)
        std_time_deltanet = np.std(times_deltanet)
        tokens_per_sec_deltanet = (seq_len * batch) / (mean_time_deltanet / 1000)
        
        print(f"{'DeltaNet':<25} {desc:<20} {mean_time_deltanet:>8.2f} ms     {tokens_per_sec_deltanet:>12,.0f}")
        
        # all-MiniLM benchmark (if available) - only for sequences <= 512 tokens
        # all-MiniLM-L6-v2 has max_seq_length=512, so skip longer sequences
        if minilm_model is not None and seq_len <= 512:
            # Clear cached buffers in BERT model (token_type_ids buffer causes issues)
            # Access the embedding layer and clear its buffer cache
            if hasattr(minilm_model, 'embeddings') and hasattr(minilm_model.embeddings, '_token_type_ids'):
                if minilm_model.embeddings._token_type_ids is not None:
                    del minilm_model.embeddings._token_type_ids
                    minilm_model.embeddings._token_type_ids = None
            
            vocab_size = minilm_tokenizer.vocab_size
            
            # Create token_type_ids explicitly (zeros for single sequence)
            token_type_ids = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
            
            # Warmup - create fresh inputs each time
            for _ in range(5):
                dummy_input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device, dtype=torch.long)
                dummy_attention_mask = torch.ones_like(dummy_input_ids)
                # Explicitly pass token_type_ids to prevent caching issues
                token_type_ids = torch.zeros_like(dummy_input_ids)
                with torch.no_grad():
                    _ = minilm_model(
                        input_ids=dummy_input_ids, 
                        attention_mask=dummy_attention_mask,
                        token_type_ids=token_type_ids
                    )
            torch.cuda.synchronize()
            
            # Benchmark all-MiniLM - create fresh inputs each iteration
            times_minilm = []
            for _ in range(20):
                # Create fresh inputs to avoid buffer caching issues
                dummy_input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device, dtype=torch.long)
                dummy_attention_mask = torch.ones_like(dummy_input_ids)
                # Explicitly create token_type_ids with correct shape
                token_type_ids = torch.zeros_like(dummy_input_ids)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    _ = minilm_model(
                        input_ids=dummy_input_ids, 
                        attention_mask=dummy_attention_mask,
                        token_type_ids=token_type_ids
                    )
                torch.cuda.synchronize()
                times_minilm.append((time.perf_counter() - start) * 1000)
            
            mean_time_minilm = np.mean(times_minilm)
            std_time_minilm = np.std(times_minilm)
            tokens_per_sec_minilm = (seq_len * batch) / (mean_time_minilm / 1000)
            
            speedup = mean_time_minilm / mean_time_deltanet
            
            print(f"{'all-MiniLM-L6-v2':<25} {desc:<20} {mean_time_minilm:>8.2f} ms     {tokens_per_sec_minilm:>12,.0f}")
            print(f"{'Speedup (DeltaNet)':<25} {desc:<20} {speedup:>8.2f}x")
        elif minilm_model is not None and seq_len > 512:
            # all-MiniLM doesn't support sequences > 512 tokens
            print(f"{'all-MiniLM-L6-v2':<25} {desc:<20} {'N/A (max 512)':<15} {'N/A':<15}")
            print(f"{'Speedup (DeltaNet)':<25} {desc:<20} {'N/A':<15}")
        
        print()  # Blank line between test cases
    
    print("\n✅ Benchmark complete!")
    
    # Cleanup
    print("Cleaning up...")
    del deltanet_model
    if minilm_model is not None:
        del minilm_model
    torch.cuda.empty_cache()
    clear_torch_cache()  # Final cache clear
    print("✅ Cleanup complete")


if __name__ == "__main__":
    # Clear cache at start
    print("🧹 Clearing torch cache at startup...")
    clear_torch_cache()
    print("✅ Cache cleared\n")
    
    benchmark_chunk_loop_operations()
    benchmark_full_model()

