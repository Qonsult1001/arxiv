#!/usr/bin/env python3
"""
Accuracy and Speed Comparison: JAX vs Cython
============================================

This script tests both JAX and Cython implementations side-by-side to:
1. Verify accuracy (embeddings should be similar)
2. Compare speed (JAX should be faster)
3. Test various input sizes
"""

import numpy as np
import torch
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lam import LAM


def compare_embeddings(emb1, emb2, name="Embeddings"):
    """Compare two embedding arrays."""
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
    
    # Cosine similarity
    dot_product = np.dot(emb1.flatten(), emb2.flatten())
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    cosine_sim = dot_product / (norm1 * norm2 + 1e-9)
    
    # L2 distance
    l2_dist = np.linalg.norm(emb1 - emb2)
    
    # Mean absolute error
    mae = np.mean(np.abs(emb1 - emb2))
    
    print(f"  {name}:")
    print(f"    Cosine Similarity: {cosine_sim:.6f}")
    print(f"    L2 Distance: {l2_dist:.6f}")
    print(f"    Mean Absolute Error: {mae:.6f}")
    
    return cosine_sim, l2_dist, mae


def benchmark_encode(model, sentences, num_runs=10, name="Model"):
    """Benchmark encoding speed."""
    times = []
    for _ in range(num_runs):
        start = time.time()
        embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)
        if torch.cuda.is_available() and model.device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)
    
    avg_time = np.mean(times[1:])  # Skip first run
    std_time = np.std(times[1:])
    
    print(f"  {name} Speed: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    return avg_time, embeddings


def test_accuracy_and_speed():
    """Test accuracy and speed comparison."""
    print("="*80)
    print("🔬 JAX vs Cython: Accuracy and Speed Comparison")
    print("="*80)
    
    model_path = Path(__file__).parent.parent / "LAM-base-v1"
    if not model_path.exists():
        print(f"❌ Model path not found: {model_path}")
        return
    
    # Test sentences
    test_cases = [
        {
            'name': 'Short',
            'sentences': ['Hello world', 'How are you?', 'This is a test']
        },
        {
            'name': 'Medium',
            'sentences': [
                'The quick brown fox jumps over the lazy dog.',
                'Machine learning is a subset of artificial intelligence.',
                'Natural language processing enables computers to understand human language.'
            ]
        },
        {
            'name': 'Long',
            'sentences': [
                ' '.join(['This is a longer sentence.'] * 10),
                ' '.join(['Another long sentence for testing.'] * 10),
            ]
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"📊 Testing: {test_case['name']} ({len(test_case['sentences'])} sentences)")
        print(f"{'='*80}")
        
        # Load both models
        print("\n⚡ Loading models...")
        try:
            model_cython = LAM(str(model_path), backend='cython')
            print("✅ Cython model loaded")
        except Exception as e:
            print(f"❌ Cython model failed: {e}")
            continue
        
        try:
            model_jax = LAM(str(model_path), backend='jax')
            print("✅ JAX model loaded")
        except Exception as e:
            print(f"❌ JAX model failed: {e}")
            print("⚠️  Skipping JAX comparison")
            continue
        
        # Get embeddings
        print("\n🧪 Computing embeddings...")
        try:
            emb_cython = model_cython.encode(test_case['sentences'], batch_size=32, show_progress_bar=False)
        except Exception as e:
            print(f"❌ Cython encoding failed: {e}")
            continue
        
        try:
            emb_jax = model_jax.encode(test_case['sentences'], batch_size=32, show_progress_bar=False)
        except Exception as e:
            print(f"❌ JAX encoding failed: {e}")
            continue
        
        # Compare accuracy
        print("\n📊 Accuracy Comparison:")
        cosine_sim, l2_dist, mae = compare_embeddings(emb_cython, emb_jax, "Embeddings")
        
        # Benchmark speed
        print("\n⚡ Speed Comparison:")
        time_cython, _ = benchmark_encode(model_cython, test_case['sentences'], name="Cython")
        time_jax, _ = benchmark_encode(model_jax, test_case['sentences'], name="JAX")
        
        speedup = time_cython / time_jax if time_jax > 0 else 0
        print(f"  Speedup: {speedup:.2f}x ({'JAX faster' if speedup > 1 else 'Cython faster'})")
        
        results.append({
            'name': test_case['name'],
            'cosine_sim': cosine_sim,
            'l2_dist': l2_dist,
            'mae': mae,
            'time_cython': time_cython,
            'time_jax': time_jax,
            'speedup': speedup
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("📈 Summary")
    print(f"{'='*80}")
    print(f"{'Test':<10} {'Cosine Sim':<12} {'L2 Dist':<12} {'MAE':<12} {'Cython (ms)':<14} {'JAX (ms)':<12} {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<10} {r['cosine_sim']:<12.6f} {r['l2_dist']:<12.6f} {r['mae']:<12.6f} "
              f"{r['time_cython']*1000:<14.2f} {r['time_jax']*1000:<12.2f} {r['speedup']:<10.2f}x")
    
    print(f"\n{'='*80}")
    print("✅ Comparison completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_accuracy_and_speed()





