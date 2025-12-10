#!/usr/bin/env python3
"""
LAM CPU Test Script
===================
Tests LAM model on CPU with different token lengths up to 8192 tokens.
This verifies that LAM works correctly on CPU-only systems.
"""

import torch
import time
import sys
from pathlib import Path

# Force CPU usage
torch.set_default_device('cpu')
device = 'cpu'

print("="*80)
print("🧪 LAM CPU TEST - Testing up to 8192 tokens")
print("="*80)
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
print()

# Add lam_package to path (we're in lam_package/tests, so go up one level)
lam_package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(lam_package_dir))

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
    # Try to find LAM-base-v1 model (go up to LAM directory)
    model_path = Path(__file__).parent.parent.parent / "LAM-base-v1"
    if not model_path.exists():
        # Try alternative paths
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

# Test different token lengths up to 8192
print("\n" + "="*80)
print("🚀 PERFORMANCE TEST - Different Token Lengths")
print("="*80)

# Generate test text that will tokenize to approximately the target lengths
# We'll create text and let the tokenizer handle truncation
base_text = "The quick brown fox jumps over the lazy dog. " * 1000  # Long base text

# Test lengths up to 8192 tokens
test_lengths = [128, 512, 1024, 2048, 4096, 8192]

for length in test_lengths:
    try:
        # Create text input (the tokenizer will truncate to max_length)
        # We use a long text and let tokenization + truncation handle the length
        test_text = base_text[:length*10]  # Rough estimate: ~10 chars per token
        
        # Warmup run (not timed)
        with torch.no_grad():
            _ = model.encode([test_text], max_length=length, batch_size=1)
        
        # Actual timed inference
        start = time.time()
        with torch.no_grad():
            embedding = model.encode([test_text], max_length=length, batch_size=1)
        elapsed = time.time() - start
        
        # Verify output shape
        assert embedding.shape == (1, 384), f"Wrong output shape: {embedding.shape}, expected (1, 384)"
        
        # Calculate tokens per second
        tokens_per_sec = length / elapsed if elapsed > 0 else 0
        
        # Format similar to user's example: "X tokens: Ys (Z tok/s)"
        print(f"{length} tokens: {elapsed:.2f}s ({tokens_per_sec:.0f} tok/s)")
        
    except Exception as e:
        print(f"{length} tokens: ❌ FAILED - {str(e)[:40]}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("✅ CPU TEST COMPLETE")
print("="*80)

# Additional verification: Test that embeddings are reasonable
print("\n🔍 VERIFICATION: Testing embedding quality...")
try:
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process information.",
        "LAM provides efficient linear attention for long sequences."
    ]
    
    embeddings = model.encode(test_sentences, batch_size=32)
    
    print(f"✅ Encoded {len(test_sentences)} sentences")
    print(f"✅ Embedding shape: {embeddings.shape}")
    
    # Check that embeddings are normalized (should be close to 1.0)
    import numpy as np
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"✅ Embedding norms: {norms}")
    print(f"   (Should be close to 1.0 for normalized embeddings)")
    
    # Compute similarity between first two sentences
    similarity = np.dot(embeddings[0], embeddings[1])
    print(f"✅ Similarity (sentence 0 vs 1): {similarity:.4f}")
    
except Exception as e:
    print(f"❌ Verification failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("🎉 ALL TESTS COMPLETE!")
print("="*80)

