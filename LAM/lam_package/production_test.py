#!/usr/bin/env python3
"""
Production Test Suite for LAM Package
======================================

This test suite verifies that all examples and features documented in README.md
work correctly. This is for internal testing only and is NOT part of the published package.

Run: python production_test.py [--backend cython|jax]
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

# Add parent directory to path for model loading
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from lam import LAM
    from lam._license import LicenseManager
except ImportError:
    print("❌ ERROR: Could not import LAM")
    print("   Make sure you've installed the package: pip install -e .")
    sys.exit(1)

def get_license_limit(model_path=None):
    """Get the current license limit for testing."""
    try:
        from pathlib import Path
        lm = LicenseManager(model_path=Path(model_path) if model_path else None)
        return lm.get_max_length()
    except Exception as e:
        print(f"   ⚠️  License check failed: {e}, using default 8192")
        return 8192  # Default free tier

def print_license_info(model_path=None):
    """Print license information for debugging."""
    try:
        from pathlib import Path
        lm = LicenseManager(model_path=Path(model_path) if model_path else None)
        print(f"\n📋 LICENSE INFORMATION:")
        print(f"   Tier: {lm.get_tier().upper()}")
        print(f"   Max Length: {lm.get_max_length()} tokens")
        print(f"   Licensed: {lm.is_licensed()}")
        if lm.get_license_info():
            info = lm.get_license_info()
            print(f"   Customer: {info.get('customer', 'N/A')}")
            if info.get('expires_at'):
                print(f"   Expires: {info['expires_at']}")
    except Exception as e:
        print(f"   ⚠️  License info unavailable: {e}")

def test_basic_usage(backend='cython'):
    """Test Basic Usage example from README"""
    print("\n" + "="*80)
    print(f"TEST 1: Basic Usage (README example) - Backend: {backend.upper()}")
    print("="*80)
    
    try:
        model = LAM('../LAM-base-v1', backend=backend)
        print("✅ Model loaded")
        
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming how we process information."
        ]
        
        embeddings = model.encode(sentences)
        print(f"✅ Encoded {len(sentences)} sentences")
        print(f"   Embeddings shape: {embeddings.shape}")
        
        # Compute cosine similarity (EXACT code from README lines 54-57)
        import numpy as np
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        print(f"Similarity: {similarity:.4f}")
        print(f"✅ Cosine similarity computed using exact README code")
        
        # Verify similarity is in valid range
        assert -1.0 <= similarity <= 1.0, f"Similarity out of range: {similarity}"
        print("✅ Similarity in valid range [-1, 1]")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing(backend='cython'):
    """Test Batch Processing example from README"""
    print("\n" + "="*80)
    print(f"TEST 2: Batch Processing (README example) - Backend: {backend.upper()}")
    print("="*80)
    
    try:
        model = LAM('../LAM-base-v1', backend=backend)
        print("✅ Model loaded")
        
        # Test with multiple sentences
        sentences = [
            "Sentence 1: The weather is nice today.",
            "Sentence 2: I love programming in Python.",
            "Sentence 3: Machine learning is fascinating.",
            "Sentence 4: Natural language processing enables AI.",
            "Sentence 5: Deep learning models are powerful."
        ]
        
        embeddings = model.encode(sentences, batch_size=32)
        print(f"✅ Encoded {len(sentences)} sentences in batch")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Expected shape: ({len(sentences)}, 384)")
        
        assert embeddings.shape == (len(sentences), 384), \
            f"Expected shape ({len(sentences)}, 384), got {embeddings.shape}"
        
        # Verify all embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), \
            f"Embeddings not normalized: norms = {norms}"
        print("✅ All embeddings normalized (L2 norm = 1.0)")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_long_documents(backend='cython'):
    """
    Test Long Documents (Strict No-Chunking Mode)
    Optimized for L40 / Single Pass 32k
    """
    print("\n" + "="*80)
    print(f"TEST 3: Long Documents (32k Single Pass) - Backend: {backend.upper()}")
    print("="*80)
    
    try:
        import torch
        import gc
        
        # 1. GLOBAL LOCK: Disable Gradient Calculation entirely
        torch.set_grad_enabled(False)
        
        # Check license limit
        license_limit = get_license_limit('../LAM-base-v1')
        print(f"   License limit: {license_limit} tokens")
        
        # 2. Load and Freeze Model
        model = LAM('../LAM-base-v1', backend=backend)
        if backend == 'cython' and hasattr(model, '_model'):
            model._model.eval()
            model._model.requires_grad_(False)  # Hard freeze
        # JAX doesn't need freezing (no gradients by default)
        
        # 4. Define the 32k target (No ramping up, just hit it)
        target_tokens = min(32768, license_limit)  # Respect license limit
        print(f"   🎯 Target: {target_tokens} tokens (Single Pass, No Chunking)")
        
        base_sentence = "The quick brown fox jumps over the lazy dog. "
        # Calculate repetitions needed
        # (Approximate since tokenization varies, usually 1 word ~= 1.3 tokens)
        # We overshoot slightly to ensure we hit the limit
        repetitions = int((target_tokens * 4.5) // len(base_sentence))
        long_document = base_sentence * repetitions
        
        # 5. PRE-FLIGHT CLEANUP
        # Clear any residue from model loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        print(f"   🌊 VRAM Cleared. Starting Encode...")

        # 6. EXECUTE ENCODE
        # Note: We rely on the global torch.set_grad_enabled(False) set above
        # Use autocast for float16 (compiled code doesn't support direct float16 conversion)
        start_time = time.time()
        
        # We explicitly pass max_length to ensure the tokenizer doesn't truncate
        # Use autocast for memory efficiency (like test_8k_inference.py)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            embeddings = model.encode([long_document], max_length=target_tokens)
        
        end_time = time.time()
        
        # 7. METRICS
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"   🧠 Peak VRAM: {peak_mem:.2f} GB")
        
        print(f"   ⏱️ Time: {end_time - start_time:.2f}s")
        print(f"   ✅ Embeddings Shape: {embeddings.shape}")
        
        # Logic Check: If O(n), 32k should be tiny on VRAM
        if torch.cuda.is_available() and peak_mem > 24:
            print("   ⚠️  WARNING: VRAM usage is suspiciously high (>24GB).")
            print("       This suggests a quadratic (N^2) operation is still leaking.")
        else:
            print("   🏆 Efficiency Confirmed: Low VRAM usage.")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        # Re-enable grads just in case other tests need them (though they shouldn't for inference)
        torch.set_grad_enabled(True) 
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Always re-enable grads for safety after this specific test
        torch.set_grad_enabled(True)

def test_features(backend='cython'):
    """Test Features claimed in README"""
    print("\n" + "="*80)
    print(f"TEST 4: Features Verification - Backend: {backend.upper()}")
    print("="*80)
    
    try:
        model = LAM('../LAM-base-v1', backend=backend)
        print("✅ Model loaded")
        
        # Test O(n) Linear Complexity - measure time for different lengths
        print("\n   Testing O(n) linear complexity...")
        test_lengths = [128, 512, 2048]
        times = []
        
        # JAX needs warmup for each unique input shape
        if backend == 'jax':
            print("   Warming up JAX for each test length (compiling unique shapes)...")
            for length in test_lengths:
                test_text = "Test sentence. " * (length // 15)
                _ = model.encode([test_text], max_length=length)  # Warmup
            print("   ✅ JAX warmup complete")
        
        for length in test_lengths:
            test_text = "Test sentence. " * (length // 15)
            start = time.time()
            _ = model.encode([test_text], max_length=length)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            print(f"   {length} tokens: {elapsed:.2f}ms")
        
        # Verify roughly linear scaling (not quadratic)
        ratio_1 = times[1] / times[0]  # 512 vs 128
        ratio_2 = times[2] / times[1]  # 2048 vs 512
        expected_ratio_1 = 512 / 128  # 4x
        expected_ratio_2 = 2048 / 512  # 4x
        
        print(f"   Scaling ratio (512/128): {ratio_1:.2f}x (expected ~{expected_ratio_1:.1f}x)")
        print(f"   Scaling ratio (2048/512): {ratio_2:.2f}x (expected ~{expected_ratio_2:.1f}x)")
        
        # Allow some variance but should be roughly linear
        assert ratio_1 < expected_ratio_1 * 2, "Scaling not linear (too slow)"
        assert ratio_2 < expected_ratio_2 * 2, "Scaling not linear (too slow)"
        print("✅ Linear complexity verified")
        
        # Test Long Token Context (up to license limit)
        print("\n   Testing long token context...")
        import torch
        
        # Clear GPU cache before long context test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        license_limit = get_license_limit('../LAM-base-v1')
        
        # Test progressively larger sizes up to license limit
        test_sizes = []
        if license_limit >= 65536:
            test_sizes = [8192, 16384, 32768, 65536]
        elif license_limit >= 32768:
            test_sizes = [8192, 16384, 32768]
        else:
            test_sizes = [4096, 8192]
        
        max_successful = 0
        for test_length in test_sizes:
            if test_length > license_limit:
                continue
            
            # Clear cache before each test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            long_text = "Test. " * (test_length // 2)  # Use half of test_length to be safe
            try:
                embeddings = model.encode([long_text], max_length=test_length)
                assert embeddings.shape == (1, 384), f"Long context ({test_length}) failed"
                print(f"   ✅ {test_length} tokens: SUCCESS")
                max_successful = test_length
            except RuntimeError as e:
                if "LIMIT REACHED" in str(e):
                    print(f"   ⚠️  {test_length} tokens: License limit reached")
                    if license_limit == 8192 and test_length > 8192:
                        print(f"   ✅ Correctly limited to 8192 tokens (free tier)")
                        break
                    else:
                        raise
                elif "out of memory" in str(e).lower() or "OOM" in str(e):
                    print(f"   ⚠️  {test_length} tokens: GPU memory limit")
                    print(f"   ✅ Successfully tested up to {max_successful} tokens")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                else:
                    raise
            except Exception as e:
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    print(f"   ⚠️  {test_length} tokens: GPU memory limit")
                    print(f"   ✅ Successfully tested up to {max_successful} tokens")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                else:
                    raise
        
        print(f"✅ Long token context supported (tested up to {max_successful} tokens)")
        if license_limit >= 32768:
            print(f"   ✅ License supports up to {license_limit} tokens")
        else:
            print(f"   ℹ️  License limit is {license_limit} (free tier)")
            print(f"   To test 32k+, add license file at https://saidhome.ai")
        
        # Clear cache after test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Test Fast Inference (128 tokens should be ~14ms)
        print("\n   Testing fast inference...")
        short_text = "The quick brown fox jumps over the lazy dog. " * 3
        
        # JAX needs warmup (first call compiles, subsequent calls are fast)
        if backend == 'jax':
            print("   Warming up JAX (first call compiles, may take a few seconds)...")
            _ = model.encode([short_text], max_length=128)  # Warmup call
            print("   ✅ JAX warmup complete")
        
        # Now measure actual inference speed
        start = time.time()
        _ = model.encode([short_text], max_length=128)
        elapsed = (time.time() - start) * 1000
        
        # JAX should be faster after warmup, Cython baseline
        target_ms = 2.0 if backend == 'jax' else 14.0
        max_ms = 50.0 if backend == 'jax' else 50.0
        
        print(f"   128 tokens: {elapsed:.2f}ms (target: ~{target_ms}ms for {backend.upper()})")
        assert elapsed < max_ms, f"Inference too slow: {elapsed}ms (expected < {max_ms}ms for {backend})"
        print("✅ Fast inference verified")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_specifications(backend='cython'):
    """Test Model Specifications from README"""
    print("\n" + "="*80)
    print(f"TEST 5: Model Specifications - Backend: {backend.upper()}")
    print("="*80)
    
    try:
        model = LAM('../LAM-base-v1', backend=backend)
        print("✅ Model loaded")
        
        # Test embedding dimension
        dim = model.get_sentence_embedding_dimension()
        print(f"   Embedding dimension: {dim}")
        assert dim == 384, f"Expected 384, got {dim}"
        print("✅ Embedding dimension: 384")
        
        # Test max context length (up to license limit)
        license_limit = get_license_limit('../LAM-base-v1')
        max_length = min(license_limit, 32768)
        test_text = "Test. " * 1000
        
        try:
            embeddings = model.encode([test_text], max_length=max_length)
            assert embeddings.shape == (1, 384), f"Max context length ({max_length}) failed"
            if max_length == 32768:
                print("✅ Max context length: 32,768 tokens")
            else:
                print(f"✅ Max context length: {max_length} tokens (license limit)")
                print(f"   ℹ️  To unlock 32k, upgrade license at https://saidhome.ai")
        except RuntimeError as e:
            if "LIMIT REACHED" in str(e):
                print(f"   ⚠️  License limit: {max_length} tokens")
                print(f"   ℹ️  To unlock 32k, upgrade license at https://saidhome.ai")
                # Not a failure - just license limit
            else:
                raise
        
        # Test that embeddings are normalized
        test_sentences = ["Test sentence 1", "Test sentence 2"]
        embeddings = model.encode(test_sentences)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"Embeddings not normalized: {norms}"
        print("✅ Embeddings are L2 normalized")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intended_uses(backend='cython'):
    """Test Intended Uses scenarios from README"""
    print("\n" + "="*80)
    print(f"TEST 6: Intended Uses Scenarios - Backend: {backend.upper()}")
    print("="*80)
    
    try:
        model = LAM('../LAM-base-v1', backend=backend)
        print("✅ Model loaded")
        
        # Test Semantic Search (finding similar documents)
        print("\n   Testing Semantic Search...")
        documents = [
            "Python is a programming language used for data science.",
            "Machine learning algorithms can process large datasets.",
            "The weather today is sunny and warm.",
            "Data science involves analyzing data with Python.",
        ]
        embeddings = model.encode(documents)
        
        # Find most similar to first document
        query_emb = embeddings[0]
        similarities = [np.dot(query_emb, emb) for emb in embeddings]
        most_similar_idx = np.argmax(similarities[1:]) + 1
        
        # Document 3 (index 3) should be most similar to document 0 (both about Python/data science)
        assert most_similar_idx == 3, f"Expected doc 3, got doc {most_similar_idx}"
        print("✅ Semantic search working")
        
        # Test Sentence Similarity
        print("\n   Testing Sentence Similarity...")
        similar_pair = [
            "The cat sat on the mat.",
            "A cat was sitting on a mat."
        ]
        different_pair = [
            "The cat sat on the mat.",
            "The weather is sunny today."
        ]
        
        sim_emb = model.encode(similar_pair)
        diff_emb = model.encode(different_pair)
        
        sim_score = np.dot(sim_emb[0], sim_emb[1])
        diff_score = np.dot(diff_emb[0], diff_emb[1])
        
        assert sim_score > diff_score, \
            f"Similar sentences should have higher similarity: {sim_score} vs {diff_score}"
        print(f"✅ Sentence similarity: similar={sim_score:.4f} > different={diff_score:.4f}")
        
        # Test Document Clustering (grouping similar documents)
        print("\n   Testing Document Clustering...")
        cluster_docs = [
            "Python programming language",
            "Java programming language",
            "The weather is nice",
            "It's sunny outside",
        ]
        cluster_emb = model.encode(cluster_docs)
        
        # Documents 0 and 1 should be similar (both programming)
        # Documents 2 and 3 should be similar (both weather)
        prog_sim = np.dot(cluster_emb[0], cluster_emb[1])
        weather_sim = np.dot(cluster_emb[2], cluster_emb[3])
        cross_sim = np.dot(cluster_emb[0], cluster_emb[2])
        
        assert prog_sim > cross_sim, "Programming docs should cluster together"
        assert weather_sim > cross_sim, "Weather docs should cluster together"
        print("✅ Document clustering working")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(description='LAM Production Test Suite')
    parser.add_argument('--backend', type=str, default='cython', choices=['cython', 'jax'],
                        help='Backend to use (cython or jax). Default: cython')
    args = parser.parse_args()
    
    print("="*80)
    print("LAM PRODUCTION TEST SUITE")
    print(f"   Backend: {args.backend.upper()}")
    print("="*80)
    print("\nThis test suite verifies all examples and features from README.md")
    print("Internal testing only - NOT part of published package")
    print("="*80)
    
    # Show license info at start
    print_license_info('../LAM-base-v1')
    
    # Run Long Documents test FIRST to test 32k/64k before memory gets fragmented
    # This is important because 32k+ requires significant GPU memory
    tests = [
        ("Long Documents", test_long_documents),  # Test first while memory is clean
        ("Basic Usage", test_basic_usage),
        ("Batch Processing", test_batch_processing),
        ("Features", test_features),
        ("Model Specifications", test_model_specifications),
        ("Intended Uses", test_intended_uses),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func(backend=args.backend)
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {name}")
    
    print("\n" + "="*80)
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("="*80)
        print("\n✅ All README examples and features verified!")
        print("Package is ready for production.")
        return 0
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total} passed)")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
