#!/usr/bin/env python3
"""Quick test to verify Super-Vector dimension fix."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "lam_package"))

def test_super_vector_dimensions():
    """Test that Super-Vector produces correct dimensions for all document sizes."""
    print("="*70)
    print("TESTING SUPER-VECTOR DIMENSION FIX")
    print("="*70)
    
    model_path = "/workspace/LAM/best"
    if not Path(model_path).exists():
        print("⚠️  Model not found, skipping test")
        return False
    
    try:
        from lam_scientific_proof_suite import LAMForMTEB
        
        print("\n1. Creating model with Super-Vector enabled...")
        model = LAMForMTEB(model_path, device="cuda", use_super_vector=True)
        print(f"   ✅ Embedding dimension: {model.embedding_dim}")
        assert model.embedding_dim == 12288, f"Expected 12288, got {model.embedding_dim}"
        
        # Test with short documents (the problematic case)
        print("\n2. Testing short documents (previously caused broadcasting error)...")
        short_texts = [
            "This is a short document.",
            "Another short text.",
            "Short document number three."
        ]
        
        embs = model.encode(short_texts, batch_size=32)
        print(f"   ✅ Encoded {len(short_texts)} short documents")
        print(f"   ✅ Embedding shape: {embs.shape}")
        assert embs.shape == (len(short_texts), 12288), f"Expected ({len(short_texts)}, 12288), got {embs.shape}"
        print(f"   ✅ All embeddings have correct dimension (12,288)")
        
        # Test with mixed lengths
        print("\n3. Testing mixed document lengths...")
        mixed_texts = [
            "Short",
            "This is a medium length document that should work fine.",
            "A" * 10000  # Long document
        ]
        
        embs = model.encode(mixed_texts, batch_size=32)
        print(f"   ✅ Encoded {len(mixed_texts)} documents (mixed lengths)")
        print(f"   ✅ Embedding shape: {embs.shape}")
        assert embs.shape == (len(mixed_texts), 12288), f"Expected ({len(mixed_texts)}, 12288), got {embs.shape}"
        
        # Verify all embeddings are valid
        for i, emb in enumerate(embs):
            assert emb.shape == (12288,), f"Document {i} has wrong shape: {emb.shape}"
            assert np.isfinite(emb).all(), f"Document {i} has non-finite values"
            assert np.linalg.norm(emb) > 0, f"Document {i} has zero norm"
        
        print("   ✅ All embeddings are valid")
        
        # Test queries
        print("\n4. Testing query encoding...")
        queries = ["What is this about?", "Find relevant information"]
        query_embs = model.encode_queries(queries)
        print(f"   ✅ Encoded {len(queries)} queries")
        print(f"   ✅ Query embedding shape: {query_embs.shape}")
        assert query_embs.shape == (len(queries), 12288), f"Expected ({len(queries)}, 12288), got {query_embs.shape}"
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("   Super-Vector now correctly handles all document sizes")
        print("   No more broadcasting errors!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_super_vector_dimensions()
    sys.exit(0 if success else 1)

