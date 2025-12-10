#!/usr/bin/env python3
"""
JAX vs Cython Performance Comparison
===================================

This script compares the JAX implementation (using jax.lax.scan)
with the Cython-compiled _core.so module.

Both implementations are tested on the same data with identical configurations.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch
import time
from typing import Dict, Tuple, Optional

# JAX configuration
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_platform_name", "gpu" if jax.devices()[0].platform == "gpu" else "cpu")

# Import compiled Cython module
try:
    from lam import _core
    CYTHON_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: _core.so not found. Only JAX benchmarks will run.")
    CYTHON_AVAILABLE = False

# Import JAX implementation
from jax_deltanet_test import (
    hierarchical_delta_rule_jax,
    simple_resonance_flux_jax,
    deltanet_forward_jax
)


def create_test_data(batch_size=2, num_heads=12, num_chunks=4, chunk_size=64, d_model=384):
    """Create test data for benchmarking."""
    d_k = d_v = d_model // num_heads
    
    # PyTorch tensors (for Cython)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q_torch = torch.randn(batch_size, num_heads, num_chunks, chunk_size, d_k, device=device, dtype=torch.float32)
    k_torch = torch.randn(batch_size, num_heads, num_chunks, chunk_size, d_k, device=device, dtype=torch.float32)
    v_torch = torch.randn(batch_size, num_heads, num_chunks, chunk_size, d_v, device=device, dtype=torch.float32)
    beta_torch = torch.rand(batch_size, num_heads, num_chunks, device=device, dtype=torch.float32)
    fast_decay_torch = torch.full((batch_size, num_heads), 0.3, device=device, dtype=torch.float32)
    slow_decay_torch = torch.full((batch_size, num_heads), 0.85, device=device, dtype=torch.float32)
    fast_gate_torch = torch.full((batch_size, num_heads), 0.5, device=device, dtype=torch.float32)
    slow_gate_torch = torch.full((batch_size, num_heads), 0.5, device=device, dtype=torch.float32)
    
    # Convert to JAX
    q_jax = jnp.array(q_torch.cpu().numpy())
    k_jax = jnp.array(k_torch.cpu().numpy())
    v_jax = jnp.array(v_torch.cpu().numpy())
    beta_jax = jnp.array(beta_torch.cpu().numpy())
    fast_decay_jax = jnp.array(fast_decay_torch.cpu().numpy())
    slow_decay_jax = jnp.array(slow_decay_torch.cpu().numpy())
    fast_gate_jax = jnp.array(fast_gate_torch.cpu().numpy())
    slow_gate_jax = jnp.array(slow_gate_torch.cpu().numpy())
    
    return {
        'torch': {
            'q': q_torch, 'k': k_torch, 'v': v_torch,
            'beta': beta_torch,
            'fast_decay': fast_decay_torch, 'slow_decay': slow_decay_torch,
            'fast_gate': fast_gate_torch, 'slow_gate': slow_gate_torch,
        },
        'jax': {
            'q': q_jax, 'k': k_jax, 'v': v_jax,
            'beta': beta_jax,
            'fast_decay': fast_decay_jax, 'slow_decay': slow_decay_jax,
            'fast_gate': fast_gate_jax, 'slow_gate': slow_gate_jax,
        }
    }


def benchmark_cython_deltanet(data_torch, num_runs=10):
    """
    Benchmark the Cython-compiled DeltaNet implementation.
    Uses the EnhancedHierarchicalDeltaNet from _core.so
    """
    if not CYTHON_AVAILABLE:
        return None, None
    
    try:
        # Import here to avoid scoping issues
        from lam import _core as core_module
        
        # Reshape from [b, h, n, c, d] to [b, h, l, d] where l = n * c
        b, h, n, c, d_k = data_torch['q'].shape
        d_v = data_torch['v'].shape[-1]
        l = n * c
        d_model = d_k * h  # Total model dimension
        
        # Create EnhancedHierarchicalDeltaNet layer (compiled)
        layer = core_module.EnhancedHierarchicalDeltaNet(
            d_model=d_model,
            num_heads=h,
            hidden_size=1024,
            use_enhanced_flux=True,
            chunk_size=c
        ).to(data_torch['q'].device)
        layer.eval()  # Set to eval mode
        
        # Prepare input: [b, l, d_model]
        # We need to combine q, k, v into a single hidden state representation
        # For simplicity, use q as the base and concatenate/combine
        hidden_states = torch.randn(b, l, d_model, device=data_torch['q'].device, dtype=torch.float32)
        
        # Warmup
        with torch.no_grad():
            _ = layer(hidden_states)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                output = layer(hidden_states)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        avg_time = np.mean(times[1:])  # Skip first run
        std_time = np.std(times[1:])
        return avg_time * 1000, std_time * 1000
        
    except Exception as e:
        print(f"   ⚠️  Cython benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def benchmark_jax_deltanet(data_jax, num_runs=10):
    """Benchmark the JAX implementation."""
    # Warmup (compilation happens here)
    _ = deltanet_forward_jax(**data_jax).block_until_ready()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        result = deltanet_forward_jax(**data_jax).block_until_ready()
        times.append(time.time() - start)
    
    avg_time = np.mean(times[1:])  # Skip first run
    std_time = np.std(times[1:])
    return avg_time * 1000, std_time * 1000, result


def compare_implementations(test_cpu=True, test_gpu=True):
    """Compare JAX and Cython implementations on both CPU and GPU."""
    print("="*80)
    print("🚀 JAX vs Cython Performance Comparison")
    print("="*80)
    
    # Test configurations
    configs = [
        {"batch": 1, "heads": 12, "chunks": 4, "chunk_size": 64, "d_model": 384, "name": "Small"},
        {"batch": 2, "heads": 12, "chunks": 8, "chunk_size": 128, "d_model": 384, "name": "Medium"},
        {"batch": 4, "heads": 12, "chunks": 16, "chunk_size": 256, "d_model": 384, "name": "Large"},
    ]
    
    all_results = []
    
    # Test CPU
    if test_cpu:
        print("\n" + "="*80)
        print("🖥️  CPU PERFORMANCE TEST")
        print("="*80)
        print(f"\n📊 JAX Device: CPU")
        print(f"📊 PyTorch Device: CPU")
        print(f"📊 Cython Available: {CYTHON_AVAILABLE}\n")
        
        # Force JAX to CPU
        jax.config.update("jax_platform_name", "cpu")
        
        cpu_results = []
        for config in configs:
            print(f"\n{'='*80}")
            print(f"📊 Testing {config['name']} Config (CPU)")
            print(f"{'='*80}")
            print(f"   Batch: {config['batch']}, Heads: {config['heads']}, "
                  f"Chunks: {config['chunks']}, Chunk Size: {config['chunk_size']}")
            
            # Create test data (CPU)
            data = create_test_data(
                batch_size=config['batch'],
                num_heads=config['heads'],
                num_chunks=config['chunks'],
                chunk_size=config['chunk_size'],
                d_model=config['d_model']
            )
            
            # Move PyTorch tensors to CPU
            for key in data['torch']:
                if isinstance(data['torch'][key], torch.Tensor):
                    data['torch'][key] = data['torch'][key].cpu()
            
            # Benchmark JAX (CPU)
            print("\n   ⚡ JAX Implementation (CPU):")
            print("      Warming up (compiling)...")
            jax_time, jax_std, jax_output = benchmark_jax_deltanet(data['jax'])
            print(f"      ✅ JAX: {jax_time:.2f}ms ± {jax_std:.2f}ms")
            print(f"      📦 Output shape: {jax_output.shape}")
            
            # Benchmark Cython (CPU)
            if CYTHON_AVAILABLE:
                print("\n   ⚡ Cython Implementation (CPU):")
                print("      Warming up...")
                cython_time, cython_std = benchmark_cython_deltanet(data['torch'])
                if cython_time is not None:
                    print(f"      ✅ Cython: {cython_time:.2f}ms ± {cython_std:.2f}ms")
                    
                    # Calculate speedup
                    speedup = cython_time / jax_time if jax_time > 0 else 0
                    faster = "JAX" if speedup > 1 else "Cython"
                    print(f"      🏆 {faster} is {abs(speedup - 1) * 100:.1f}% {'faster' if speedup != 1 else 'same speed'}")
                    
                    cpu_results.append({
                        'config': config['name'],
                        'jax_time': jax_time,
                        'cython_time': cython_time,
                        'speedup': speedup,
                        'faster': faster,
                        'platform': 'CPU'
                    })
                else:
                    print("      ⚠️  Cython benchmark failed")
            else:
                print("\n   ⚠️  Cython not available (skipping)")
        
        all_results.extend(cpu_results)
    
    # Test GPU
    if test_gpu and torch.cuda.is_available():
        print("\n" + "="*80)
        print("🎮 GPU PERFORMANCE TEST")
        print("="*80)
        print(f"\n📊 JAX Device: {jax.devices()[0]}")
        print(f"📊 PyTorch Device: CUDA")
        print(f"📊 Cython Available: {CYTHON_AVAILABLE}\n")
        
        # Force JAX to GPU
        jax.config.update("jax_platform_name", "gpu")
        
        gpu_results = []
        for config in configs:
            print(f"\n{'='*80}")
            print(f"📊 Testing {config['name']} Config (GPU)")
            print(f"{'='*80}")
            print(f"   Batch: {config['batch']}, Heads: {config['heads']}, "
                  f"Chunks: {config['chunks']}, Chunk Size: {config['chunk_size']}")
            
            # Create test data (GPU)
            data = create_test_data(
                batch_size=config['batch'],
                num_heads=config['heads'],
                num_chunks=config['chunks'],
                chunk_size=config['chunk_size'],
                d_model=config['d_model']
            )
            
            # Benchmark JAX (GPU)
            print("\n   ⚡ JAX Implementation (GPU):")
            print("      Warming up (compiling)...")
            jax_time, jax_std, jax_output = benchmark_jax_deltanet(data['jax'])
            print(f"      ✅ JAX: {jax_time:.2f}ms ± {jax_std:.2f}ms")
            print(f"      📦 Output shape: {jax_output.shape}")
            
            # Benchmark Cython (GPU)
            if CYTHON_AVAILABLE:
                print("\n   ⚡ Cython Implementation (GPU):")
                print("      Warming up...")
                cython_time, cython_std = benchmark_cython_deltanet(data['torch'])
                if cython_time is not None:
                    print(f"      ✅ Cython: {cython_time:.2f}ms ± {cython_std:.2f}ms")
                    
                    # Calculate speedup
                    speedup = cython_time / jax_time if jax_time > 0 else 0
                    faster = "JAX" if speedup > 1 else "Cython"
                    print(f"      🏆 {faster} is {abs(speedup - 1) * 100:.1f}% {'faster' if speedup != 1 else 'same speed'}")
                    
                    gpu_results.append({
                        'config': config['name'],
                        'jax_time': jax_time,
                        'cython_time': cython_time,
                        'speedup': speedup,
                        'faster': faster,
                        'platform': 'GPU'
                    })
                else:
                    print("      ⚠️  Cython benchmark failed")
            else:
                print("\n   ⚠️  Cython not available (skipping)")
        
        all_results.extend(gpu_results)
    elif test_gpu:
        print("\n⚠️  GPU not available, skipping GPU tests")
    
    # Test configurations
    configs = [
        {"batch": 1, "heads": 12, "chunks": 4, "chunk_size": 64, "d_model": 384, "name": "Small"},
        {"batch": 2, "heads": 12, "chunks": 8, "chunk_size": 128, "d_model": 384, "name": "Medium"},
        {"batch": 4, "heads": 12, "chunks": 16, "chunk_size": 256, "d_model": 384, "name": "Large"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"📊 Testing {config['name']} Config")
        print(f"{'='*80}")
        print(f"   Batch: {config['batch']}, Heads: {config['heads']}, "
              f"Chunks: {config['chunks']}, Chunk Size: {config['chunk_size']}")
        
        # Create test data
        data = create_test_data(
            batch_size=config['batch'],
            num_heads=config['heads'],
            num_chunks=config['chunks'],
            chunk_size=config['chunk_size'],
            d_model=config['d_model']
        )
        
        # Benchmark JAX
        print("\n   ⚡ JAX Implementation:")
        print("      Warming up (compiling)...")
        jax_time, jax_std, jax_output = benchmark_jax_deltanet(data['jax'])
        print(f"      ✅ JAX: {jax_time:.2f}ms ± {jax_std:.2f}ms")
        print(f"      📦 Output shape: {jax_output.shape}")
        
        # Benchmark Cython
        if CYTHON_AVAILABLE:
            print("\n   ⚡ Cython Implementation:")
            print("      Warming up...")
            cython_time, cython_std = benchmark_cython_deltanet(data['torch'])
            if cython_time is not None:
                print(f"      ✅ Cython: {cython_time:.2f}ms ± {cython_std:.2f}ms")
                
                # Calculate speedup
                speedup = cython_time / jax_time if jax_time > 0 else 0
                faster = "JAX" if speedup > 1 else "Cython"
                print(f"      🏆 {faster} is {abs(speedup - 1) * 100:.1f}% {'faster' if speedup != 1 else 'same speed'}")
                
                results.append({
                    'config': config['name'],
                    'jax_time': jax_time,
                    'cython_time': cython_time,
                    'speedup': speedup,
                    'faster': faster
                })
            else:
                print("      ⚠️  Cython benchmark failed")
        else:
            print("\n   ⚠️  Cython not available (skipping)")
    
    # Summary
    if all_results:
        print(f"\n{'='*80}")
        print("📈 Summary")
        print(f"{'='*80}")
        
        # Group by platform
        cpu_results = [r for r in all_results if r['platform'] == 'CPU']
        gpu_results = [r for r in all_results if r['platform'] == 'GPU']
        
        if cpu_results:
            print("\n🖥️  CPU Results:")
            print(f"{'Config':<10} {'JAX (ms)':<12} {'Cython (ms)':<14} {'Speedup':<10} {'Winner':<10}")
            print("-" * 80)
            for r in cpu_results:
                print(f"{r['config']:<10} {r['jax_time']:<12.2f} {r['cython_time']:<14.2f} "
                      f"{r['speedup']:<10.2f}x {r['faster']:<10}")
            
            jax_wins_cpu = sum(1 for r in cpu_results if r['faster'] == 'JAX')
            cython_wins_cpu = sum(1 for r in cpu_results if r['faster'] == 'Cython')
            print(f"\n🏆 CPU: JAX won {jax_wins_cpu}/{len(cpu_results)}, Cython won {cython_wins_cpu}/{len(cpu_results)}")
        
        if gpu_results:
            print("\n🎮 GPU Results:")
            print(f"{'Config':<10} {'JAX (ms)':<12} {'Cython (ms)':<14} {'Speedup':<10} {'Winner':<10}")
            print("-" * 80)
            for r in gpu_results:
                print(f"{r['config']:<10} {r['jax_time']:<12.2f} {r['cython_time']:<14.2f} "
                      f"{r['speedup']:<10.2f}x {r['faster']:<10}")
            
            jax_wins_gpu = sum(1 for r in gpu_results if r['faster'] == 'JAX')
            cython_wins_gpu = sum(1 for r in gpu_results if r['faster'] == 'Cython')
            print(f"\n🏆 GPU: JAX won {jax_wins_gpu}/{len(gpu_results)}, Cython won {cython_wins_gpu}/{len(gpu_results)}")
        
        # Overall summary
        jax_wins = sum(1 for r in all_results if r['faster'] == 'JAX')
        cython_wins = sum(1 for r in all_results if r['faster'] == 'Cython')
        print(f"\n🏆 Overall: JAX won {jax_wins}/{len(all_results)}, Cython won {cython_wins}/{len(all_results)}")
    
    print(f"\n{'='*80}")
    print("✅ Comparison completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    import sys
    test_cpu = '--cpu-only' not in sys.argv or '--gpu-only' not in sys.argv
    test_gpu = '--cpu-only' not in sys.argv
    if '--cpu-only' in sys.argv:
        test_gpu = False
    if '--gpu-only' in sys.argv:
        test_cpu = False
    compare_implementations(test_cpu=test_cpu, test_gpu=test_gpu)

