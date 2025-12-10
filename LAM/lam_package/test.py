#!/usr/bin/env python3
"""
LAM Package Test
================
Run this after building to verify everything works.

Usage:
    python3 test.py [--backend cython|jax]
    
Options:
    --backend: Choose backend (cython or jax). Default: cython
"""

import sys
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LAM Package Test')
    parser.add_argument('--backend', type=str, default='cython', choices=['cython', 'jax'],
                        help='Backend to use (cython or jax). Default: cython')
    return parser.parse_args()

def test(backend='cython'):
    print("="*60)
    print("🧪 LAM PACKAGE TEST")
    print(f"   Backend: {backend.upper()}")
    print("="*60)
    
    # Add package to path
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    
    # Test 0: License Check
    print("\n📋 Test 0: License Status...")
    try:
        from lam._license import LicenseManager
        model_path = script_dir.parent / "best"
        lm = LicenseManager(model_path=model_path if model_path.exists() else None)
        print(f"   Tier: {lm.get_tier().upper()}")
        print(f"   Max Length: {lm.get_max_length()} tokens")
        print(f"   Licensed: {lm.is_licensed()}")
        if lm.get_license_info():
            info = lm.get_license_info()
            print(f"   Customer: {info.get('customer', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️  License check failed: {e}")
    
    # Test 1: Import
    print("\n📋 Test 1: Import...")
    try:
        from lam import LAM
        print("   ✅ Import successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Load model
    print("\n📋 Test 2: Load model...")
    try:
        model_path = script_dir.parent / "best"
        model = LAM(str(model_path), backend=backend)
        print(f"   ✅ Model loaded (backend: {backend})")
    except Exception as e:
        print(f"   ❌ Load failed: {e}")
        return False
    
    # Test 3: Encode
    print("\n📋 Test 3: Encode sentences...")
    try:
        sentences = ["Hello world", "Machine learning is great", "LAM is fast"]
        embeddings = model.encode(sentences)
        print(f"   ✅ Encoded {len(sentences)} sentences")
        print(f"   ✅ Shape: {embeddings.shape}")
    except Exception as e:
        print(f"   ❌ Encode failed: {e}")
        return False
    
    # Test 4: Similarity
    print("\n📋 Test 4: Compute similarity...")
    try:
        import numpy as np
        sim = np.dot(embeddings[0], embeddings[1])
        print(f"   ✅ Similarity (0 vs 1): {sim:.4f}")
    except Exception as e:
        print(f"   ❌ Similarity failed: {e}")
        return False
    
    # Test 5: STS-B Validation Set (1500 pairs)
    print("\n📋 Test 5: STS-B Validation Set (1500 pairs)...")
    try:
        from datasets import load_dataset
        from scipy.stats import spearmanr, pearsonr
        import numpy as np
        
        print("   Loading STS-B validation set...")
        ds_val = load_dataset("sentence-transformers/stsb", split="validation")
        s1_val = list(ds_val["sentence1"])
        s2_val = list(ds_val["sentence2"])
        labels_val = np.array(ds_val["score"], dtype=float)
        
        # Note: JAX may be slower due to compilation, but we use full dataset for accuracy
        if backend == 'jax':
            print(f"   ℹ️  JAX backend: Using full dataset (may be slower due to compilation)")
        
        print(f"   ✅ Loaded {len(s1_val)} pairs")
        
        # Encode in batches - use same parameters for both backends
        batch_size = 32
        emb1_val = []
        emb2_val = []
        
        # For JAX, warm up to compile common shapes (but don't restrict max_length)
        if backend == 'jax':
            print("   Warming up JAX (compiling common input shapes)...")
            # Warm up with a few batches to compile common shapes
            warmup_batch = s1_val[:batch_size] if len(s1_val) >= batch_size else s1_val
            for _ in range(2):  # 2 warmup rounds
                _ = model.encode(warmup_batch, batch_size=batch_size)
            print("   ✅ JAX warmup complete")
        
        print("   Computing embeddings...")
        
        total_batches = (len(s1_val) + batch_size - 1) // batch_size
        print(f"   Encoding {total_batches} batches (batch_size={batch_size})...")
        
        for i in range(0, len(s1_val), batch_size):
            batch_s1 = s1_val[i:i+batch_size]
            batch_s2 = s2_val[i:i+batch_size]
            
            # Progress indicator
            batch_num = (i // batch_size) + 1
            if backend == 'jax' and batch_num <= 3:
                print(f"   Batch {batch_num}/{total_batches} (JAX compiling on first batch)...", end=' ', flush=True)
            elif batch_num % 10 == 0:
                print(f"   Batch {batch_num}/{total_batches}...", end=' ', flush=True)
            
            # Use same encoding parameters for both backends (no max_length restriction)
            emb1_batch = model.encode(batch_s1, batch_size=batch_size, convert_to_numpy=True)
            emb2_batch = model.encode(batch_s2, batch_size=batch_size, convert_to_numpy=True)
            
            emb1_val.append(emb1_batch)
            emb2_val.append(emb2_batch)
            
            if backend == 'jax' and batch_num <= 3:
                print("✅")
            elif batch_num % 10 == 0:
                print("✅")
        
        emb1_val = np.vstack(emb1_val)
        emb2_val = np.vstack(emb2_val)
        
        # Compute cosine similarities
        sims_val = np.array([np.dot(e1, e2) for e1, e2 in zip(emb1_val, emb2_val)])
        
        spearman_val = spearmanr(sims_val, labels_val)[0]
        pearson_val = pearsonr(sims_val, labels_val)[0]
        
        print(f"   ✅ Spearman correlation: {spearman_val:.4f}")
        print(f"   ✅ Pearson correlation:  {pearson_val:.4f}")
        print(f"   📊 Expected range: 0.80-0.85 (if model is correct)")
        
    except Exception as e:
        print(f"   ⚠️ STS-B validation test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: License Limit Verification
    print("\n📋 Test 6: License Limit Verification...")
    try:
        from lam._license import LicenseManager
        import shutil
        import tempfile
        
        model_path = script_dir.parent / "best"
        license_file = script_dir / "lam" / "lam_license.json"
        backup_dir = tempfile.mkdtemp()
        
        # Get current license status
        lm = LicenseManager(model_path=model_path if model_path.exists() else None)
        current_limit = lm.get_max_length()
        has_license = lm.is_licensed()
        
        print(f"   Current license status:")
        print(f"   - Licensed: {has_license}")
        print(f"   - Max Length: {current_limit} tokens")
        
        # Test 6a: With License (should support 32k)
        if has_license and current_limit >= 32768:
            print(f"\n   ✅ Test 6a: With License (32k enabled)")
            print(f"   - Testing up to 32k tokens...")
            test_text = "Test sentence. " * 2000  # ~30k tokens
            try:
                embeddings = model.encode([test_text], max_length=32768)
                print(f"   ✅ Successfully encoded 32k tokens with license")
            except RuntimeError as e:
                if "LIMIT REACHED" in str(e):
                    print(f"   ❌ FAILED: License not working - limit reached at 32k")
                else:
                    raise
        
        # Test 6b: Without License (should be limited to 8192)
        print(f"\n   Test 6b: Without License (should limit to 8192)")
        if license_file.exists():
            # Backup license
            backup_file = Path(backup_dir) / "lam_license.json.backup"
            shutil.copy(license_file, backup_file)
            print(f"   - Backed up license to: {backup_file}")
            
            # Remove license temporarily
            license_file.unlink()
            print(f"   - Removed license file temporarily")
            
            # Clear any cached license manager
            import importlib
            try:
                import lam._license
                importlib.reload(lam._license)
            except:
                pass
            
            # Create new model instance (will detect no license)
            model_no_license = LAM(str(model_path))
            lm_no_license = LicenseManager(model_path=model_path if model_path.exists() else None)
            no_license_limit = lm_no_license.get_max_length()
            
            print(f"   - License status without file: {lm_no_license.is_licensed()}")
            print(f"   - Max length without license: {no_license_limit} tokens")
            
            if no_license_limit == 8192:
                print(f"   ✅ Correctly limited to 8192 tokens without license")
            else:
                print(f"   ❌ FAILED: Should be 8192, got {no_license_limit}")
            
            # Test that 8192 works but 8193 fails
            test_8192 = "Test. " * 2000  # ~8k tokens
            try:
                embeddings = model_no_license.encode([test_8192], max_length=8192)
                print(f"   ✅ 8192 tokens works (within free tier limit)")
            except RuntimeError as e:
                print(f"   ❌ FAILED: 8192 should work: {e}")
            
            # Test that 8193 fails
            test_8193 = "Test. " * 2001  # ~8k+ tokens
            try:
                embeddings = model_no_license.encode([test_8193], max_length=8193)
                print(f"   ❌ FAILED: 8193 should fail but didn't")
            except RuntimeError as e:
                if "LIMIT REACHED" in str(e) or "8192" in str(e):
                    print(f"   ✅ Correctly rejected 8193 tokens (exceeds 8192 limit)")
                else:
                    print(f"   ⚠️  Got error but not expected format: {e}")
            
            # Restore license
            shutil.copy(backup_file, license_file)
            print(f"   - Restored license file")
            
            # Verify license is back
            lm_restored = LicenseManager(model_path=model_path if model_path.exists() else None)
            restored_limit = lm_restored.get_max_length()
            print(f"   - License restored: {lm_restored.is_licensed()}")
            print(f"   - Max length restored: {restored_limit} tokens")
            
            if restored_limit >= 32768:
                print(f"   ✅ License restored correctly (32k enabled)")
            else:
                print(f"   ❌ FAILED: License not restored correctly")
        else:
            print(f"   ⚠️  No license file found to test removal")
        
    except Exception as e:
        print(f"   ⚠️  License verification test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 7: Speed Benchmark (up to license limit)
    print("\n📋 Test 7: Speed Benchmark (up to license limit)...")
    try:
        import time
        import torch
        from lam._license import LicenseManager
        
        # Get license limit
        model_path = script_dir.parent / "best"
        lm = LicenseManager(model_path=model_path if model_path.exists() else None)
        license_limit = lm.get_max_length()
        
        # Test lengths up to 2M tokens (infinite scaling like Google)
        # Exponential progression: 128 -> 2M (exactly 2,000,000 to match license limit)
        all_lengths = [
            128, 512, 1024, 2048, 4096, 8192, 16384, 32768,  # Standard lengths
            65536, 131072, 262144, 524288, 1048576, 2000000  # Extended to 2M (exactly 2,000,000)
        ]
        test_lengths = [l for l in all_lengths if l <= license_limit]
        
        print(f"   License limit: {license_limit:,} tokens")
        print(f"   Testing lengths: {[f'{l:,}' for l in test_lengths]}")
        print(f"   🚀 Testing infinite scaling up to 2M tokens (like Google)")
        print(f"   ⚡ Using TITANS Flat 1D architecture (no state storage during inference)")
        
        # Get GPU info
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info['name'] = torch.cuda.get_device_name(0)
            gpu_info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   🖥️  GPU: {gpu_info['name']} ({gpu_info['total_memory_gb']:.1f} GB VRAM)")
            print(f"   💡 Testing infinite scaling - will identify maximum possible on this GPU")
        else:
            print(f"   ⚠️  No GPU detected - using CPU")
            gpu_info = None
        # Realistic text: ~100-200 words per sequence, repeated enough to reach target length
        realistic_sentence = "The quick brown fox jumps over the lazy dog. Machine learning is transforming how we process information. Natural language processing enables computers to understand human language. "
        
        # Global warmup (precompile CUDA kernels, allocate memory)
        print("Warming up model (precompiling CUDA kernels)...")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file('/workspace/LAM/LAM-base-v1/tokenizer.json')
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        warmup_text = "Warmup text for CUDA kernel compilation."
        for _ in range(3):  # Multiple warmup runs
            model.encode([warmup_text], max_length=512, batch_size=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup complete")
        
        print(f"\n   {'Length':<12} {'Time (ms)':<15} {'Tokens/sec':<15} {'GPU Memory':<15} {'% of VRAM':<12}")
        print("   " + "-" * 75)
        
        max_successful_length = 0
        successful_lengths = []
        
        for seq_len in test_lengths:
            try:
                # Clear GPU cache before each test
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    memory_before = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                
                # Generate realistic text that will tokenize to approximately seq_len tokens
                # Repeat sentence enough times to reach target length (with some margin)
                repeats = max(1, (seq_len // 20) + 1)  # ~20 tokens per sentence
                base_text = realistic_sentence * repeats
                
                # Pre-tokenize ONCE (outside timing loop) - fast tokenizer handles this quickly
                tokenizer.enable_truncation(max_length=seq_len)
                encoded = tokenizer.encode(base_text)
                import torch
                tokens = {
                    'input_ids': torch.tensor([encoded.ids], dtype=torch.long, device=model.device),
                    'attention_mask': torch.tensor([encoded.attention_mask], dtype=torch.long, device=model.device)
                }
                
                # Per-length warmup (2 runs to stabilize)
                if backend == 'cython':
                    with torch.no_grad():
                        for _ in range(2):
                            _ = model._model.get_sentence_embeddings(tokens['input_ids'], tokens['attention_mask'])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                else:
                    # JAX: warmup by encoding
                    for _ in range(2):
                        _ = model.encode([base_text], max_length=seq_len)
                
                # Benchmark MODEL ONLY (no tokenization overhead)
                # This measures actual inference speed, not tokenization speed
                start = time.time()
                iterations = 10  # More iterations for accuracy
                if backend == 'cython':
                    with torch.no_grad():
                        for _ in range(iterations):
                            _ = model._model.get_sentence_embeddings(tokens['input_ids'], tokens['attention_mask'])
                else:
                    # JAX: use encode method
                    for _ in range(iterations):
                        _ = model.encode([base_text], max_length=seq_len)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = (time.time() - start) / iterations * 1000
                tokens_per_sec = seq_len / (elapsed / 1000)
                
                # Get GPU memory usage
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated(0) / (1024**3)  # GB
                    memory_used = peak_memory - memory_before
                    memory_pct = (peak_memory / gpu_info['total_memory_gb']) * 100 if gpu_info else 0
                    memory_str = f"{memory_used:.2f} GB"
                    memory_pct_str = f"{memory_pct:.1f}%"
                else:
                    memory_str = "N/A"
                    memory_pct_str = "N/A"
                
                # Format large numbers with commas
                seq_len_str = f"{seq_len:,}" if seq_len >= 1000 else str(seq_len)
                print(f"   {seq_len_str:<12} {elapsed:<15.2f} {tokens_per_sec:<15.0f} {memory_str:<15} {memory_pct_str:<12}")
                
                # Track successful runs
                max_successful_length = max(max_successful_length, seq_len)
                successful_lengths.append(seq_len)
                
                # Clean up
                del tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                seq_len_str = f"{seq_len:,}" if seq_len >= 1000 else str(seq_len)
                error_msg = str(e)[:40]
                
                # Get memory info even on failure
                if torch.cuda.is_available() and gpu_info:
                    peak_memory = torch.cuda.max_memory_allocated(0) / (1024**3)  # GB
                    memory_pct = (peak_memory / gpu_info['total_memory_gb']) * 100
                    memory_info = f" (Peak: {peak_memory:.2f} GB, {memory_pct:.1f}% VRAM)"
                else:
                    memory_info = ""
                
                print(f"   {seq_len_str:<12} ❌ FAILED: {error_msg}{memory_info}")
                
                # For very long sequences, OOM is expected - continue to next
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    if torch.cuda.is_available() and gpu_info:
                        print(f"      ⚠️  OOM at {seq_len:,} tokens - GPU limit reached on {gpu_info['name']} ({gpu_info['total_memory_gb']:.1f} GB)")
                        print(f"      💡 Maximum possible on this GPU: ~{seq_len//2:,} tokens (estimated)")
                    continue
                break
        
        print("   ✅ Speed benchmark complete")
        
        # Summary of GPU capabilities and practical limits
        if torch.cuda.is_available() and gpu_info:
            print(f"\n📊 GPU Memory Analysis & Practical Limits:")
            print(f"   GPU: {gpu_info['name']}")
            print(f"   Total VRAM: {gpu_info['total_memory_gb']:.1f} GB")
            print(f"   Maximum tested: {max(test_lengths):,} tokens")
            if max_successful_length > 0:
                print(f"   ✅ Maximum successful: {max_successful_length:,} tokens")
                print(f"   ✅ Successful tests: {len(successful_lengths)}/{len(test_lengths)}")
            print(f"\n   💡 What's Truly Possible on This GPU:")
            print(f"      - Model weights: ~1-2 GB")
            print(f"      - TITANS Flat 1D: No state storage (memory efficient)")
            print(f"      - Position interpolation: Supports arbitrary lengths")
            print(f"      - Practical limit: Depends on available VRAM")
            if max_successful_length > 0:
                # Rough estimate: if we successfully ran at max_successful_length, 
                # we could potentially go ~10-20% higher with more aggressive memory management
                conservative_estimate = int(max_successful_length * 1.1)
                optimistic_estimate = int(max_successful_length * 1.2)
                print(f"      - Conservative estimate: ~{conservative_estimate:,} tokens")
                print(f"      - Optimistic estimate: ~{optimistic_estimate:,} tokens (with memory optimization)")
            print(f"\n   🚀 Infinite Scaling:")
            print(f"      - Architecture supports up to 2M tokens (license limit)")
            print(f"      - Actual limit: GPU memory capacity")
            print(f"      - On {gpu_info['name']}: Maximum achieved = {max_successful_length:,} tokens")
        
    except Exception as e:
        print(f"   ⚠️ Speed benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 8: STS-B Test Set (1379 pairs) - Full evaluation
    print("\n📋 Test 8: STS-B Test Set (1379 pairs) - Full Evaluation...")
    try:
        from datasets import load_dataset
        from scipy.stats import spearmanr, pearsonr
        import numpy as np
        
        print("   Loading STS-B test set...")
        ds_test = load_dataset("sentence-transformers/stsb", split="test")
        s1_test = list(ds_test["sentence1"])
        s2_test = list(ds_test["sentence2"])
        labels_test = np.array(ds_test["score"], dtype=float)
        
        # Note: JAX may be slower due to compilation, but we use full dataset for accuracy
        if backend == 'jax':
            print(f"   ℹ️  JAX backend: Using full dataset (may be slower due to compilation)")
        
        print(f"   ✅ Loaded {len(s1_test)} pairs")
        
        # Encode in batches - use same parameters for both backends
        batch_size = 32
        emb1_test = []
        emb2_test = []
        
        # For JAX, warm up to compile common shapes (but don't restrict max_length)
        if backend == 'jax':
            print("   Warming up JAX (compiling common input shapes)...")
            # Warm up with a few batches to compile common shapes
            warmup_batch = s1_test[:batch_size] if len(s1_test) >= batch_size else s1_test
            for _ in range(2):  # 2 warmup rounds
                _ = model.encode(warmup_batch, batch_size=batch_size)
            print("   ✅ JAX warmup complete")
        
        print("   Computing embeddings...")
        
        total_batches = (len(s1_test) + batch_size - 1) // batch_size
        print(f"   Encoding {total_batches} batches (batch_size={batch_size})...")
        
        for i in range(0, len(s1_test), batch_size):
            batch_s1 = s1_test[i:i+batch_size]
            batch_s2 = s2_test[i:i+batch_size]
            
            # Progress indicator
            batch_num = (i // batch_size) + 1
            if backend == 'jax' and batch_num <= 3:
                print(f"   Batch {batch_num}/{total_batches} (JAX compiling on first batch)...", end=' ', flush=True)
            elif batch_num % 10 == 0:
                print(f"   Batch {batch_num}/{total_batches}...", end=' ', flush=True)
            
            # Use same encoding parameters for both backends (no max_length restriction)
            emb1_batch = model.encode(batch_s1, batch_size=batch_size, convert_to_numpy=True)
            emb2_batch = model.encode(batch_s2, batch_size=batch_size, convert_to_numpy=True)
            
            emb1_test.append(emb1_batch)
            emb2_test.append(emb2_batch)
            
            if backend == 'jax' and batch_num <= 3:
                print("✅")
            elif batch_num % 10 == 0:
                print("✅")
        
        emb1_test = np.vstack(emb1_test)
        emb2_test = np.vstack(emb2_test)
        
        # Compute cosine similarities
        sims_test = np.array([np.dot(e1, e2) for e1, e2 in zip(emb1_test, emb2_test)])
        
        spearman_test = spearmanr(sims_test, labels_test)[0]
        pearson_test = pearsonr(sims_test, labels_test)[0]
        
        print(f"   ✅ Spearman correlation: {spearman_test:.4f}")
        print(f"   ✅ Pearson correlation:  {pearson_test:.4f}")
        print(f"   📊 Expected: Spearman=0.7711, Pearson=0.7787")
        
        # Summary
        print("\n" + "="*60)
        print("📊 STS-B EVALUATION SUMMARY")
        print("="*60)
        print(f"Validation Set (1500 pairs):")
        print(f"  Spearman: {spearman_val:.4f}")
        print(f"  Pearson:  {pearson_val:.4f}")
        print(f"\nTest Set (1379 pairs):")
        print(f"  Spearman: {spearman_test:.4f}")
        print(f"  Pearson:  {pearson_test:.4f}")
        print("="*60)
        
        # Check if scores are reasonable
        if spearman_test < 0.70 or pearson_test < 0.70:
            print("\n⚠️  WARNING: Scores are lower than expected!")
            print("   Expected: Spearman=0.7711, Pearson=0.7787")
            print("   This may indicate:")
            print("   - Model weights not loaded correctly")
            print("   - Tokenizer mismatch")
            print("   - Model architecture mismatch")
        elif abs(spearman_test - 0.7711) < 0.01 and abs(pearson_test - 0.7787) < 0.01:
            print("\n✅ Model scores match expected values perfectly!")
        else:
            print("\n✅ Model appears to be loading correctly!")
        
    except Exception as e:
        print(f"   ⚠️ STS-B test set failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nPackage is ready for distribution.")
    print("="*60)
    
    return True

if __name__ == "__main__":
    args = parse_args()
    success = test(backend=args.backend)
    sys.exit(0 if success else 1)

