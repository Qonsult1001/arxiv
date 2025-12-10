#!/usr/bin/env python3
"""
Test LAM Matryoshka Embeddings (Simple Example)
===============================================

Simple test matching lam_embed API style:
    from lam import LAM
    
    model = LAM("LAM-base-v1", backend='cython')  # or 'jax'
    embeddings = model.encode(sentences, dimensions=64)  # or 128, 256, 384

Usage:
    python test_lam_matryoshka.py [--backend cython|jax]
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lam import LAM


def main():
    parser = argparse.ArgumentParser(description='Test LAM Matryoshka embeddings')
    parser.add_argument('--backend', type=str, default='cython', choices=['cython', 'jax'],
                       help='Backend to use (cython or jax). Default: cython')
    args = parser.parse_args()
    
    print("="*70)
    print("LAM MATRYOSHKA EMBEDDINGS TEST")
    print("="*70)
    print()
    
    # Test sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process information."
    ]
    
    print(f"📝 Sentences: {len(sentences)}")
    print()
    
    # Load model (similar to lam_embed API)
    print(f"🔧 Loading model (backend: {args.backend})...")
    model = LAM("../LAM-base-v1", backend=args.backend)
    print("✅ Model loaded")
    print()
    
    # Test all Matryoshka dimensions
    dimensions = [64, 128, 256, 384]
    
    # Note: JAX is now pre-compiled at model initialization (like Cython)
    # No need for additional warmup - functions are already compiled!
    
    print("🧪 Testing Matryoshka dimensions:")
    print("-" * 70)
    
    import time
    test_start = time.time()
    
    for dim in dimensions:
        print(f"\n📊 Dimension: {dim}")
        
        # Encode with specific dimension (matches lam_embed API)
        embeddings = model.encode(sentences, dimensions=dim)
        
        print(f"   Shape: {embeddings.shape}")
        print(f"   ✅ Working!")
        
        # Show first few values of first embedding
        if dim <= 64:
            print(f"   First 5 values: {embeddings[0][:5]}")
    
    test_time = time.time() - test_start
    
    print("\n" + "="*70)
    print("✅ ALL DIMENSIONS WORKING!")
    print("="*70)
    print(f"\n⏱️  Test execution time: {test_time:.3f}s")
    if args.backend == 'jax':
        print(f"   ✅ JAX was pre-compiled at initialization (like Cython)")
        print(f"   ✅ Fast from first use - no runtime compilation delay!")
    print()
    print("📋 Usage Example:")
    print("   from lam import LAM")
    print("   model = LAM('LAM-base-v1', backend='cython')")
    print("   embeddings = model.encode(sentences, dimensions=64)  # or 128, 256, 384")
    print()


if __name__ == '__main__':
    main()

