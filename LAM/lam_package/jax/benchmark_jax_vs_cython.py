#!/usr/bin/env python3
"""
End-to-end benchmark comparing JAX and Cython implementations.
Tests accuracy and speed for both backends.
"""

import torch
import jax
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path
import sys
from typing import List, Tuple, Dict
import statistics

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'build'))

from lam import LAM
from einops import rearrange

def compare_outputs(cython_output: torch.Tensor, jax_output: np.ndarray, name: str = "Output") -> Dict:
    """Compare Cython and JAX outputs and return statistics."""
    if isinstance(cython_output, torch.Tensor):
        cython_np = cython_output.detach().cpu().numpy()
    else:
        cython_np = cython_output
    
    if isinstance(jax_output, jnp.ndarray):
        jax_np = np.array(jax_output)
    else:
        jax_np = jax_output
    
    # Ensure shapes match
    if cython_np.shape != jax_np.shape:
        print(f"⚠️  Shape mismatch: Cython {cython_np.shape} vs JAX {jax_np.shape}")
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(cython_np.shape, jax_np.shape))
        cython_np = cython_np[tuple(slice(0, s) for s in min_shape)]
        jax_np = jax_np[tuple(slice(0, s) for s in min_shape)]
    
    diff = np.abs(cython_np - jax_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    median_diff = np.median(diff)
    std_diff = np.std(diff)
    
    # Cosine similarity
    cython_flat = cython_np.flatten()
    jax_flat = jax_np.flatten()
    cos_sim = np.dot(cython_flat, jax_flat) / (np.linalg.norm(cython_flat) * np.linalg.norm(jax_flat) + 1e-9)
    
    # Relative error
    rel_error = np.abs(diff) / (np.abs(cython_np) + 1e-9)
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error)
    
    # Percentiles
    diff_sorted = np.sort(diff.flatten())
    p95_diff = np.percentile(diff_sorted, 95)
    p99_diff = np.percentile(diff_sorted, 99)
    
    # Count of elements above thresholds
    num_elements = diff.size
    num_above_1e3 = np.sum(diff > 1e-3)
    num_above_1e2 = np.sum(diff > 1e-2)
    
    return {
        'name': name,
        'shape': cython_np.shape,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'std_diff': std_diff,
        'cosine_sim': cos_sim,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'p95_diff': p95_diff,
        'p99_diff': p99_diff,
        'num_elements': num_elements,
        'num_above_1e3': num_above_1e3,
        'num_above_1e2': num_above_1e2,
        'cython_first_5': cython_np.flatten()[:5],
        'jax_first_5': jax_np.flatten()[:5],
    }

def benchmark_backend(model: LAM, sentences: List[str], backend: str, num_warmup: int = 3, num_runs: int = 10) -> Tuple[List[Dict], List[float]]:
    """Benchmark a backend and return results."""
    print(f"\n{'='*80}")
    print(f"Benchmarking {backend.upper()} backend")
    print(f"{'='*80}")
    
    results = []
    times = []
    
    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        for sentence in sentences[:1]:  # Just first sentence for warmup
            _ = model.generate(sentence, max_length=len(sentence.split()) + 10, backend=backend)
    
    # Actual benchmark
    print(f"Running benchmark ({num_runs} runs)...")
    for run_idx in range(num_runs):
        run_times = []
        for sentence in sentences:
            start_time = time.time()
            output = model.generate(sentence, max_length=len(sentence.split()) + 10, backend=backend, return_hidden_states=True)
            elapsed = time.time() - start_time
            run_times.append(elapsed)
            
            if run_idx == 0:  # Only compare outputs on first run
                if backend == 'cython':
                    cython_output = output['last_hidden_state']
                else:
                    jax_output = output['last_hidden_state']
        
        times.append(sum(run_times))
    
    return results, times

def benchmark_forward_pass(model_path: str, sentences: List[str], num_runs: int = 10) -> Dict:
    """Benchmark forward pass only (no generation)."""
    print(f"\n{'='*80}")
    print("Benchmarking Forward Pass Only")
    print(f"{'='*80}")
    
    results = {}
    
    # Create separate model instances for each backend
    print("\nCreating Cython model...")
    model_cython = LAM(model_path, backend='cython')
    print("✅ Cython model created")
    
    print("\nCreating JAX model...")
    try:
        model_jax = LAM(model_path, backend='jax')
        print("✅ JAX model created")
        jax_available = True
    except Exception as e:
        print(f"⚠️  JAX model creation failed: {e}")
        jax_available = False
    
    for backend_name, model in [('cython', model_cython), ('jax', model_jax)]:
        if backend_name == 'jax' and not jax_available:
            continue
            
        print(f"\nTesting {backend_name.upper()} backend...")
        times = []
        outputs = []
        
        # Warmup - CRITICAL: For JAX, warm up ALL different input shapes to avoid recompilation
        print(f"  Warming up...")
        if backend_name == 'jax':
            # JAX needs to compile for each different input shape
            # Warm up with ALL sentences multiple times to ensure full compilation
            # More warmup rounds = better consistency (ensures all shapes fully compiled)
            print(f"    Warming up JAX with all {len(sentences)} sentence shapes (this may take a moment)...")
            for warmup_round in range(5):  # Increased from 3 to 5 for better consistency
                for sentence in sentences:
                    _ = model.encode([sentence])
                if warmup_round == 0:
                    print(f"    First warmup round complete (initial compilation done)")
                elif warmup_round == 4:
                    print(f"    Warmup complete (all shapes fully compiled)")
        else:
            # Cython doesn't need shape-specific warmup
            for _ in range(3):
                for sentence in sentences[:1]:
                    _ = model.encode([sentence])
        
        # Actual benchmark - all shapes should now be compiled/cached
        print(f"  Running {num_runs} benchmark runs (all shapes pre-compiled)...")
        for run_idx in range(num_runs):
            run_times = []
            run_outputs = []
            for sentence in sentences:
                start_time = time.time()
                output = model.encode([sentence])
                elapsed = time.time() - start_time
                
                run_times.append(elapsed)
                if run_idx == 0:
                    # Convert to numpy for comparison
                    if isinstance(output, torch.Tensor):
                        run_outputs.append(output.detach().cpu().numpy())
                    elif isinstance(output, np.ndarray):
                        run_outputs.append(output)
                    else:
                        # If it's a list, take first element
                        if isinstance(output, list) and len(output) > 0:
                            if isinstance(output[0], torch.Tensor):
                                run_outputs.append(output[0].detach().cpu().numpy())
                            else:
                                run_outputs.append(np.array(output[0]))
            
            times.append(sum(run_times))
            if run_idx == 0:
                outputs = run_outputs
        
        results[backend_name] = {
            'times': times,
            'outputs': outputs,
            'mean_time': statistics.mean(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
        }
    
    # Compare outputs
    if 'cython' in results and 'jax' in results:
        if len(results['cython']['outputs']) > 0 and len(results['jax']['outputs']) > 0:
            # Compare first sentence output
            comparison = compare_outputs(
                results['cython']['outputs'][0],
                results['jax']['outputs'][0],
                "Forward Pass Output"
            )
            results['comparison'] = comparison
    
    return results

def print_results(results: Dict):
    """Print benchmark results in a formatted way."""
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}\n")
    
    # Speed comparison
    print("SPEED COMPARISON:")
    print("-" * 80)
    for backend in ['cython', 'jax']:
        if backend in results:
            stats = results[backend]
            print(f"{backend.upper()}:")
            print(f"  Mean time: {stats['mean_time']*1000:.2f} ms ± {stats['std_time']*1000:.2f} ms")
            print(f"  Min time:  {stats['min_time']*1000:.2f} ms")
            print(f"  Max time:  {stats['max_time']*1000:.2f} ms")
            print()
    
    if 'cython' in results and 'jax' in results:
        speedup = results['cython']['mean_time'] / results['jax']['mean_time']
        print(f"Speedup (JAX vs Cython): {speedup:.2f}x")
        if speedup > 1:
            print(f"  ✅ JAX is {speedup:.2f}x faster")
        else:
            print(f"  ⚠️  Cython is {1/speedup:.2f}x faster")
        print()
    
    # Accuracy comparison
    if 'comparison' in results:
        comp = results['comparison']
        print("ACCURACY COMPARISON:")
        print("-" * 80)
        print(f"Cosine Similarity: {comp['cosine_sim']:.10f}")
        if comp['cosine_sim'] > 0.9999:
            print("  ✅ Excellent alignment (>0.9999)")
        elif comp['cosine_sim'] > 0.99:
            print("  ✅ Good alignment (>0.99)")
        elif comp['cosine_sim'] > 0.95:
            print("  ⚠️  Moderate alignment (>0.95)")
        else:
            print("  ❌ Poor alignment (<0.95)")
        print()
        
        print(f"Max Difference: {comp['max_diff']:.10e}")
        print(f"Mean Difference: {comp['mean_diff']:.10e}")
        print(f"Median Difference: {comp['median_diff']:.10e}")
        print(f"Std Difference: {comp['std_diff']:.10e}")
        print()
        
        print(f"95th Percentile Diff: {comp['p95_diff']:.10e}")
        print(f"99th Percentile Diff: {comp['p99_diff']:.10e}")
        print()
        
        print(f"Max Relative Error: {comp['max_rel_error']:.10e}")
        print(f"Mean Relative Error: {comp['mean_rel_error']:.10e}")
        print()
        
        print(f"Elements > 1e-3: {comp['num_above_1e3']} / {comp['num_elements']} ({100*comp['num_above_1e3']/comp['num_elements']:.2f}%)")
        print(f"Elements > 1e-2: {comp['num_above_1e2']} / {comp['num_elements']} ({100*comp['num_above_1e2']/comp['num_elements']:.2f}%)")
        print()
        
        print("First 5 values comparison:")
        print(f"  Cython: {comp['cython_first_5']}")
        print(f"  JAX:    {comp['jax_first_5']}")
        print()
        
        # Overall assessment
        print("OVERALL ASSESSMENT:")
        print("-" * 80)
        if comp['cosine_sim'] > 0.9999 and comp['max_diff'] < 1e-2:
            print("  ✅ EXCELLENT: JAX and Cython are highly aligned")
        elif comp['cosine_sim'] > 0.99 and comp['max_diff'] < 1e-1:
            print("  ✅ GOOD: JAX and Cython are well aligned")
        elif comp['cosine_sim'] > 0.95:
            print("  ⚠️  MODERATE: Some differences exist")
        else:
            print("  ❌ POOR: Significant differences detected")
        print()

def main():
    """Main benchmark function."""
    print("="*80)
    print("JAX vs Cython End-to-End Benchmark")
    print("="*80)
    
    # Model path
    model_path = "../LAM-base-v1"
    
    # Test sentences of varying lengths
    test_sentences = [
        "Hello",
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.",
        "The sun was shining on the sea, shining with all his might: He did his very best to make the billows smooth and bright— And this was odd, because it was the middle of the night.",
    ]
    
    print(f"\nTest sentences ({len(test_sentences)}):")
    for i, sent in enumerate(test_sentences, 1):
        print(f"  {i}. {sent[:60]}{'...' if len(sent) > 60 else ''} ({len(sent.split())} words)")
    
    # Run forward pass benchmark
    results = benchmark_forward_pass(model_path, test_sentences, num_runs=10)
    
    # Print results
    print_results(results)
    
    print("="*80)
    print("Benchmark complete!")
    print("="*80)

if __name__ == "__main__":
    main()

