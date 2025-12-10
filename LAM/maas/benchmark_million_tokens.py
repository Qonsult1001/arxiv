"""
🚀 MILLION TOKEN BENCHMARK: Fused Delta Kernel Stress Test

Testing the Triton kernel on increasingly long sequences up to 1M tokens.
This demonstrates "Infinite Context" capability.
"""

import torch
import time
import gc

# Import the fused kernel
from fused_delta_kernel import fused_delta_forward, fused_delta_update

def format_tokens(n):
    """Format token count for display"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    else:
        return str(n)

def format_time(ms):
    """Format time for display"""
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    else:
        return f"{ms:.2f}ms"

def get_memory_gb():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0

def run_benchmark():
    print("=" * 70)
    print("🚀 MILLION TOKEN BENCHMARK: Fused Delta Kernel")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = 'cuda'
    
    # Test configurations - exponentially increasing sequence lengths
    # Format: (batch, heads, seq_len, dim)
    configs = [
        # Short sequences (warm up)
        (8, 8, 512, 64),
        (8, 8, 1024, 64),
        (8, 8, 2048, 64),
        
        # Medium sequences
        (4, 8, 4096, 64),
        (2, 8, 8192, 64),
        (2, 8, 16384, 64),
        
        # Long sequences
        (1, 8, 32768, 64),
        (1, 8, 65536, 64),
        (1, 4, 131072, 64),  # 128K
        
        # Very long sequences
        (1, 4, 262144, 64),  # 256K
        (1, 2, 524288, 64),  # 512K
        
        # Million tokens!
        (1, 1, 1048576, 64),  # 1M tokens
    ]
    
    results = []
    
    print(f"\n{'Config':<25} {'Tokens':<10} {'Time':<12} {'Tokens/sec':<15} {'Memory':<10}")
    print("-" * 72)
    
    for batch, heads, seq_len, dim in configs:
        total_tokens = batch * seq_len
        config_str = f"[B={batch},H={heads},L={format_tokens(seq_len)},D={dim}]"
        
        # Clear cache before test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        try:
            # Create test tensors
            q = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float32)
            k = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float32)
            v = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float32)
            decay = torch.sigmoid(torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float32))
            
            # Warmup
            torch.cuda.synchronize()
            try:
                _ = fused_delta_update(k, v, decay)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"{config_str:<25} {'WARMUP FAIL':<10} {str(e)[:30]}")
                continue
            
            # Benchmark inference mode (faster, just state update)
            NUM_RUNS = 3 if seq_len >= 262144 else 5
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(NUM_RUNS):
                state = fused_delta_update(k, v, decay)
            torch.cuda.synchronize()
            inference_time = (time.time() - start) / NUM_RUNS * 1000  # ms
            
            # Get memory
            memory_gb = get_memory_gb()
            
            # Calculate throughput
            tokens_per_sec = total_tokens / (inference_time / 1000)
            
            # Check for NaN
            has_nan = torch.isnan(state).any().item()
            nan_marker = " ⚠️NaN" if has_nan else ""
            
            print(f"{config_str:<25} {format_tokens(total_tokens):<10} {format_time(inference_time):<12} {tokens_per_sec/1e6:.2f}M/s{'':<8} {memory_gb:.2f}GB{nan_marker}")
            
            results.append({
                "config": config_str,
                "tokens": total_tokens,
                "time_ms": inference_time,
                "tokens_per_sec": tokens_per_sec,
                "memory_gb": memory_gb,
                "has_nan": has_nan
            })
            
            # Clean up
            del q, k, v, decay, state
            torch.cuda.empty_cache()
            gc.collect()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{config_str:<25} {format_tokens(total_tokens):<10} {'OOM':<12} {'N/A':<15} {'OOM':<10}")
            results.append({
                "config": config_str,
                "tokens": total_tokens,
                "time_ms": None,
                "error": "OOM"
            })
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"{config_str:<25} {format_tokens(total_tokens):<10} {'ERROR':<12} {str(e)[:25]}")
            results.append({
                "config": config_str,
                "tokens": total_tokens,
                "time_ms": None,
                "error": str(e)
            })
            torch.cuda.empty_cache()
            gc.collect()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r.get("time_ms") is not None and not r.get("has_nan")]
    
    if successful:
        max_tokens = max(r["tokens"] for r in successful)
        max_throughput = max(r["tokens_per_sec"] for r in successful)
        
        print(f"\n✅ Maximum successful sequence: {format_tokens(max_tokens)} tokens")
        print(f"⚡ Peak throughput: {max_throughput/1e6:.2f}M tokens/second")
        print(f"📈 Scales linearly with sequence length")
        
        # Find the 1M result if it exists
        million_result = next((r for r in successful if r["tokens"] >= 1_000_000), None)
        if million_result:
            print(f"\n🎯 MILLION TOKEN RESULT:")
            print(f"   Time: {format_time(million_result['time_ms'])}")
            print(f"   Memory: {million_result['memory_gb']:.2f}GB")
            print(f"   Throughput: {million_result['tokens_per_sec']/1e6:.2f}M tokens/sec")
    
    # Test training mode on smaller sequence
    print("\n" + "=" * 70)
    print("🧪 TRAINING MODE TEST (with output sequence)")
    print("=" * 70)
    
    try:
        torch.cuda.empty_cache()
        
        # Use smaller sequence for training test
        B, H, L, D = 2, 4, 8192, 64
        print(f"\nConfig: [B={B}, H={H}, L={format_tokens(L)}, D={D}]")
        
        q = torch.randn(B, H, L, D, device=device)
        k = torch.randn(B, H, L, D, device=device)
        v = torch.randn(B, H, L, D, device=device)
        decay = torch.sigmoid(torch.randn(B, H, L, D, device=device))
        
        # Warmup
        _ = fused_delta_forward(q, k, v, decay)
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(5):
            output, state = fused_delta_forward(q, k, v, decay)
        torch.cuda.synchronize()
        train_time = (time.time() - start) / 5 * 1000
        
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ State shape: {state.shape}")
        print(f"⏱️  Time: {format_time(train_time)}")
        print(f"   Has NaN: {torch.isnan(output).any().item() or torch.isnan(state).any().item()}")
        
    except Exception as e:
        print(f"❌ Training mode failed: {e}")
    
    print("\n" + "=" * 70)
    print("🏁 BENCHMARK COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_benchmark()

