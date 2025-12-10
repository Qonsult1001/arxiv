#!/usr/bin/env python3
"""
Test Semantic Similarity with Matryoshka Embeddings
===================================================

This test verifies that LAM embeddings preserve semantic meaning across
different Matryoshka dimensions using STS-B dataset.

Tests:
- Semantic similarity correlation (Spearman/Pearson) for each dimension
- Verifies that lower dimensions still preserve semantic meaning
- Compares performance across 64, 128, 256, 384 dimensions

Usage:
    python test_semantic_matryoshka.py [--subset N] [--model-path PATH] [--dimension DIM] [--quick]
    
    --dimension: Test specific dimension (64, 128, 256, 384). Default: all dimensions
    --quick: Quick inference test with 10 sentences (no STS-B dataset)
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from lam import LAM
except ImportError:
    print("❌ ERROR: Could not import LAM")
    print("   Make sure you've installed the package: pip install -e .")
    sys.exit(1)


def test_quick_inference(model_path=None, dimension=64):
    """Quick inference test with sample sentences."""
    print("="*80)
    print(f"QUICK INFERENCE TEST - Dimension: {dimension}")
    print("="*80)
    print()
    
    # Sample sentences for testing
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast animal leaps over a sleeping canine.",
        "Machine learning is transforming artificial intelligence.",
        "AI and ML are revolutionizing technology.",
        "Python is a popular programming language.",
        "Many developers use Python for data science.",
        "The weather today is sunny and warm.",
        "It's a beautiful day with clear skies.",
        "Cooking requires patience and practice.",
        "Good chefs spend years perfecting their skills."
    ]
    
    # Determine model path
    if model_path is None:
        possible_paths = [
            Path('../best'),
            Path('../LAM-base-v1'),
            Path('best'),
            Path('LAM-base-v1'),
        ]
        for path in possible_paths:
            if path.exists() and (path / 'pytorch_model.bin').exists():
                model_path = path
                break
        
        if model_path is None:
            print("❌ ERROR: Could not find model directory")
            return False
    
    # Load model
    print(f"🔧 Loading LAM model from: {model_path}")
    model = LAM(str(model_path), backend='cython')
    print("✅ Model loaded")
    print()
    
    # Test encoding
    print(f"🧪 Testing inference with {len(test_sentences)} sentences at dimension {dimension}...")
    print("-" * 80)
    
    import time
    start = time.time()
    embeddings = model.encode(test_sentences, dimensions=dimension, convert_to_numpy=True)
    elapsed = time.time() - start
    
    print(f"✅ Encoded {len(test_sentences)} sentences in {elapsed:.3f}s")
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Expected shape: ({len(test_sentences)}, {dimension})")
    
    if embeddings.shape == (len(test_sentences), dimension):
        print("   ✅ Shape matches expected!")
    else:
        print(f"   ❌ Shape mismatch! Expected ({len(test_sentences)}, {dimension}), got {embeddings.shape}")
        return False
    
    # Test similarity between first two sentences
    sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    print(f"\n📊 Semantic similarity (sentence 0 vs 1): {sim:.4f}")
    print(f"   Sentence 0: {test_sentences[0][:50]}...")
    print(f"   Sentence 1: {test_sentences[1][:50]}...")
    
    print("\n" + "="*80)
    print("✅ QUICK INFERENCE TEST COMPLETE")
    print("="*80)
    print()
    
    return True


def test_semantic_similarity(model_path=None, subset=None, dimensions=None):
    """Test semantic similarity using STS-B dataset across Matryoshka dimensions."""
    print("="*80)
    print("SEMANTIC SIMILARITY TEST (STS-B) - Cython Backend")
    print("="*80)
    print()
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ ERROR: datasets library not found")
        print("   Install with: pip install datasets")
        sys.exit(1)
    
    # Load STS-B test set
    print("📊 Loading STS-B test set...")
    try:
        ds_test = load_dataset("sentence-transformers/stsb", split="test")
    except Exception as e:
        print(f"❌ Failed to load STS-B: {e}")
        print("   Trying alternative dataset...")
        try:
            ds_test = load_dataset("glue", "stsb", split="test")
        except Exception as e2:
            print(f"❌ Failed to load alternative: {e2}")
            return False
    
    s1_test = list(ds_test["sentence1"])
    s2_test = list(ds_test["sentence2"])
    
    # Get labels (score or label column)
    if "score" in ds_test.column_names:
        labels_test = np.array(ds_test["score"], dtype=float)
    elif "label" in ds_test.column_names:
        labels_test = np.array(ds_test["label"], dtype=float)
    else:
        print("❌ No score/label column found in dataset")
        return False
    
    full_size = len(s1_test)
    
    # Handle subset or full test
    if subset:
        if subset == 1500 or subset == 1370:
            # Full STS-B test sizes
            if subset <= full_size:
                s1_test = s1_test[:subset]
                s2_test = s2_test[:subset]
                labels_test = labels_test[:subset]
                print(f"   ✅ Using FULL STS-B test set: {subset} pairs")
            else:
                print(f"   ⚠️  Requested {subset} pairs but only {full_size} available, using all")
        else:
            # Custom subset
            if subset < full_size:
                s1_test = s1_test[:subset]
                s2_test = s2_test[:subset]
                labels_test = labels_test[:subset]
                print(f"   Using subset: {subset} pairs (for faster testing)")
            else:
                print(f"   ⚠️  Requested {subset} pairs but only {full_size} available, using all")
    else:
        # Default: use all available
        print(f"   ✅ Using FULL dataset: {full_size} pairs")
    
    print(f"   ✅ Loaded {len(s1_test)} sentence pairs")
    print(f"   Score range: {labels_test.min():.2f} - {labels_test.max():.2f}")
    print()
    
    # Determine model path
    if model_path is None:
        # Try to find best model
        possible_paths = [
            Path('../best'),  # Best model directory
            Path('../LAM-base-v1'),  # Fallback
            Path('best'),  # Relative to current directory
            Path('LAM-base-v1'),  # Relative to current directory
        ]
        model_path = None
        for path in possible_paths:
            if path.exists() and (path / 'pytorch_model.bin').exists():
                model_path = path
                break
        
        if model_path is None:
            print("❌ ERROR: Could not find model directory")
            print("   Please specify --model-path or ensure 'best/' or 'LAM-base-v1/' exists")
            return False
    
    # Load model
    print(f"🔧 Loading LAM model from: {model_path}")
    model = LAM(str(model_path), backend='cython')
    print("✅ Model loaded")
    print()
    
    # Test dimensions (use provided or default to all)
    if dimensions is None:
        test_dimensions = [64, 128, 256, 384]
    else:
        test_dimensions = [dimensions] if isinstance(dimensions, int) else dimensions
    
    results = {}
    
    # Test each dimension
    print("🧪 Testing semantic similarity for each dimension...")
    print("-" * 80)
    
    for dim in test_dimensions:
        print(f"\n📊 Testing dimension: {dim}")
        print(f"   Encoding {len(s1_test)} sentence pairs...")
        
        
        # Encode sentences in batches
        batch_size = 32
        emb1_list = []
        emb2_list = []
        
        for i in range(0, len(s1_test), batch_size):
            batch_s1 = s1_test[i:i+batch_size]
            batch_s2 = s2_test[i:i+batch_size]
            
            # Encode with specific dimension
            emb1_batch = model.encode(batch_s1, dimensions=dim, batch_size=batch_size, convert_to_numpy=True)
            emb2_batch = model.encode(batch_s2, dimensions=dim, batch_size=batch_size, convert_to_numpy=True)
            
            emb1_list.append(emb1_batch)
            emb2_list.append(emb2_batch)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Progress: {min(i + batch_size, len(s1_test))}/{len(s1_test)} pairs")
        
        # Stack embeddings
        emb1 = np.vstack(emb1_list)
        emb2 = np.vstack(emb2_list)
        
        print(f"   ✅ Embedding shapes: {emb1.shape}, {emb2.shape}")
        
        # Compute cosine similarities
        sims = np.array([
            np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            for e1, e2 in zip(emb1, emb2)
        ])
        
        # Show sample sentence pairs and similarities (human-readable)
        print()
        print("   📊 Sample sentence pairs and similarities:")
        for idx in range(min(10, len(sims))):
            sim = sims[idx]
            label = labels_test[idx]
            print(f"      Pair {idx+1}: sim={sim:.4f}, label={label:.4f}")
            print(f"         S1: {s1_test[idx][:70]}{'...' if len(s1_test[idx]) > 70 else ''}")
            print(f"         S2: {s2_test[idx][:70]}{'...' if len(s2_test[idx]) > 70 else ''}")
        if len(sims) > 10:
            print(f"      ... and {len(sims) - 10} more pairs")
        print()
        
        # Compute correlations
        spearman = spearmanr(sims, labels_test)[0]
        
        # Expected score ranges (based on dimension)
        if dim == 384:
            expected_min = 0.77
            expected_max = 0.85
            expected_range = "High (0.77-0.85)"
        elif dim == 256:
            expected_min = 0.73
            expected_max = 0.82
            expected_range = "High (0.73-0.82)"
        elif dim == 128:
            expected_min = 0.68
            expected_max = 0.78
            expected_range = "Medium-High (0.68-0.78)"
        else:  # 64
            expected_min = 0.63
            expected_max = 0.73
            expected_range = "Medium (0.63-0.73)"
        
        # For dimension 64, only compute Spearman (as requested)
        if dim == 64:
            pearson = None
            print(f"   ✅ Spearman correlation: {spearman:.4f}")
            print(f"   📊 Expected range: {expected_range}")
            if spearman >= expected_min:
                if spearman >= expected_max:
                    print(f"   ✅ Score: EXCELLENT (above expected max)")
                else:
                    print(f"   ✅ Score: GOOD (within expected range)")
            else:
                print(f"   ⚠️  Score: BELOW EXPECTED MINIMUM ({expected_min:.2f})")
            print(f"   ℹ️  Pearson skipped for dimension 64")
        else:
            pearson = pearsonr(sims, labels_test)[0]
            print(f"   ✅ Spearman correlation: {spearman:.4f}")
            print(f"   ✅ Pearson correlation:  {pearson:.4f}")
            print(f"   📊 Expected range: {expected_range}")
            if spearman >= expected_min:
                if spearman >= expected_max:
                    print(f"   ✅ Score: EXCELLENT (above expected max)")
                else:
                    print(f"   ✅ Score: GOOD (within expected range)")
            else:
                print(f"   ⚠️  Score: BELOW EXPECTED MINIMUM ({expected_min:.2f})")
        
        results[dim] = {
            'spearman': spearman,
            'pearson': pearson,
            'sims': sims
        }
        
        # Expected ranges (based on typical Matryoshka performance)
        if dim == 384:
            expected_min = 0.77
        elif dim == 256:
            expected_min = 0.73
        elif dim == 128:
            expected_min = 0.68
        else:  # 64
            expected_min = 0.63
        
        if spearman >= expected_min:
            print(f"   ✅ Meets expected minimum ({expected_min:.2f})")
        else:
            print(f"   ⚠️  Below expected minimum ({expected_min:.2f})")
    
    # Summary (aligned with test.py format)
    print("\n" + "="*80)
    print("📊 SEMANTIC SIMILARITY RESULTS SUMMARY")
    print("="*80)
    print()
    print(f"{'Dimension':<12} {'Spearman':<12} {'Pearson':<12}")
    print("-" * 80)
    
    for dim in sorted(test_dimensions):
        spearman = results[dim]['spearman']
        pearson = results[dim]['pearson']
        pearson_str = f"{pearson:.4f}" if pearson is not None else "N/A"
        print(f"{dim:<12} {spearman:<12.4f} {pearson_str:<12}")
    
    print()
    print("="*80)
    print("✅ SEMANTIC MEANING TEST COMPLETE")
    print("="*80)
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test semantic similarity with Matryoshka embeddings')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model directory (default: auto-detect best/ or LAM-base-v1/)')
    parser.add_argument('--subset', type=int, default=None,
                       help='Use subset of STS-B pairs (e.g., 64, 1500, 1370). Use 1500 or 1370 for full test. Default: all')
    parser.add_argument('--dimension', type=int, default=None, choices=[64, 128, 256, 384],
                       help='Test specific dimension only (64, 128, 256, 384). Default: all dimensions')
    parser.add_argument('--quick', action='store_true',
                       help='Quick inference test with 10 sentences (no STS-B dataset)')
    parser.add_argument('--full', action='store_true',
                       help='Run full STS-B test (1500 pairs) - production validation')
    args = parser.parse_args()
    
    # Handle --full flag
    if args.full:
        args.subset = 1500
    
    if args.quick:
        # Quick inference test
        dimension = args.dimension if args.dimension else 64
        success = test_quick_inference(model_path=args.model_path, dimension=dimension)
    else:
        # Full STS-B semantic similarity test
        success = test_semantic_similarity(
            model_path=args.model_path, 
            subset=args.subset,
            dimensions=args.dimension
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

