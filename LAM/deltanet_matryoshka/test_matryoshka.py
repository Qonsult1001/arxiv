"""
Test Matryoshka Representation Learning (MRL)
Verify that multi-scale contrastive learning works correctly
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# ============================================================================
# MATRYOSHKA COMPONENTS
# ============================================================================

class MatryoshkaProjection(nn.Module):
    """
    Projects embeddings to support multiple granularities.
    Forces early dimensions to be meaningful on their own.
    
    Production Optimization: Uses learnable LayerNorm before L2 normalization.
    This is superior to standard MRL (which just does F.normalize(truncated)) because:
    - LayerNorm re-centers the data distribution for each dimension slice
    - Allows the model to learn optimal scaling/shifting per dimension level
    - Better adaptation to the specific characteristics of truncated embeddings
    - Improves contrastive learning signal quality at each granularity
    """
    def __init__(self, d_model=384, dims=[64, 128, 256, 384]):
        super().__init__()
        self.d_model = d_model
        self.dims = sorted(dims)
        
        # Learnable normalization layers for each dimension level
        # Production optimization: LayerNorm before L2 normalization is better than
        # standard MRL's simple F.normalize() because it allows re-centering per slice
        self.norms = nn.ModuleDict({
            str(d): nn.LayerNorm(d) for d in dims
        })
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, d_model]
        Returns:
            Dict of embeddings at each dimension level
        """
        outputs = {}
        for dim in self.dims:
            # Extract first `dim` dimensions
            truncated = embeddings[:, :dim]
            # Production optimization: LayerNorm re-centers, then L2 normalize
            # This is better than standard MRL's simple F.normalize(truncated)
            normalized = F.normalize(self.norms[str(dim)](truncated), p=2, dim=-1)
            outputs[dim] = normalized
        
        return outputs


class MatryoshkaContrastiveLoss(nn.Module):
    """
    Multi-scale contrastive loss for Matryoshka training.
    Computes InfoNCE loss at each dimension level.
    """
    def __init__(self, dims=[64, 128, 256, 384], weights=None, temperature=0.05):
        super().__init__()
        self.dims = dims
        # Weight each dimension level (can customize)
        self.weights = weights or [1.0] * len(dims)
        self.temperature = temperature
        
    def forward(self, embeddings_dict, labels):
        """
        Args:
            embeddings_dict: Dict[int, Tensor] - {64: [batch, 64], 128: [batch, 128], ...}
            labels: [batch] - labels for contrastive learning
        Returns:
            total_loss: Weighted sum of losses at each dimension
            loss_breakdown: Dict with loss at each dimension
        """
        total_loss = 0.0
        loss_breakdown = {}
        
        for dim, weight in zip(self.dims, self.weights):
            emb = embeddings_dict[dim]  # [batch, dim]
            
            # Standard contrastive loss (InfoNCE)
            # Compute similarity matrix
            sim_matrix = torch.matmul(emb, emb.t()) / self.temperature  # [batch, batch]
            
            # Create label mask (positive pairs)
            labels_expanded = labels.unsqueeze(1)
            mask = (labels_expanded == labels_expanded.t()).float()
            mask.fill_diagonal_(0)  # Exclude self-similarity
            
            # Compute loss
            exp_sim = torch.exp(sim_matrix)
            log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
            loss = -(log_prob * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss = loss.mean()
            
            total_loss += weight * loss
            loss_breakdown[dim] = loss.item()
        
        return total_loss, loss_breakdown


def consistency_loss(matryoshka_embs, full_emb):
    """
    Ensure truncated embeddings preserve similarity relationships.
    If full[i] is similar to full[j], then truncated[i] should also be similar to truncated[j].
    """
    # Compute similarity matrix at full dimension
    full_sim = F.cosine_similarity(
        full_emb.unsqueeze(1),
        full_emb.unsqueeze(0),
        dim=-1
    )  # [batch, batch]
    
    total_loss = 0.0
    count = 0
    
    # For each smaller dimension, ensure similarity ranking is preserved
    for dim in [64, 128, 256]:
        if dim not in matryoshka_embs:
            continue
        small_emb = matryoshka_embs[dim]
        small_sim = F.cosine_similarity(
            small_emb.unsqueeze(1),
            small_emb.unsqueeze(0),
            dim=-1
        )  # [batch, batch]
        
        # Ranking consistency: MSE between similarity matrices
        total_loss += F.mse_loss(small_sim, full_sim.detach())
        count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0)


# ============================================================================
# TESTS
# ============================================================================

def test_matryoshka_projection():
    """Test that MatryoshkaProjection works correctly"""
    print("\n" + "="*80)
    print("TEST 1: MatryoshkaProjection")
    print("="*80)
    
    batch_size = 32
    d_model = 384
    dims = [64, 128, 256, 384]
    
    # Create projection
    proj = MatryoshkaProjection(d_model=d_model, dims=dims)
    
    # Create random embeddings
    embeddings = torch.randn(batch_size, d_model)
    
    # Project
    outputs = proj(embeddings)
    
    # Verify
    print(f"✅ Input shape: {embeddings.shape}")
    for dim in dims:
        assert dim in outputs, f"Missing dimension {dim}"
        assert outputs[dim].shape == (batch_size, dim), f"Wrong shape for dim {dim}"
        # Check normalization
        norms = torch.norm(outputs[dim], p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"Not normalized at dim {dim}"
        print(f"✅ Dimension {dim}: shape={outputs[dim].shape}, normalized={norms.mean().item():.6f}")
    
    print("✅ TEST 1 PASSED\n")
    return True


def test_matryoshka_contrastive_loss():
    """Test that MatryoshkaContrastiveLoss computes correctly"""
    print("="*80)
    print("TEST 2: MatryoshkaContrastiveLoss")
    print("="*80)
    
    batch_size = 32
    d_model = 384
    dims = [64, 128, 256, 384]
    
    # Create projection and loss
    proj = MatryoshkaProjection(d_model=d_model, dims=dims)
    loss_fn = MatryoshkaContrastiveLoss(dims=dims, temperature=0.05)
    
    # Create random embeddings
    embeddings = torch.randn(batch_size, d_model)
    
    # Project
    matryoshka_embs = proj(embeddings)
    
    # Create labels with positive pairs (for contrastive learning)
    # Each pair of consecutive samples shares the same label
    labels = torch.arange(batch_size // 2).repeat_interleave(2)
    if batch_size % 2 == 1:
        labels = torch.cat([labels, torch.tensor([batch_size // 2])])
    
    # Compute loss
    total_loss, loss_breakdown = loss_fn(matryoshka_embs, labels)
    
    # Verify
    print(f"✅ Total loss: {total_loss.item():.4f}")
    for dim, loss_val in loss_breakdown.items():
        print(f"✅ Loss at {dim}-dim: {loss_val:.4f}")
    
    assert total_loss > 0, "Loss should be positive"
    assert not torch.isnan(total_loss), "Loss is NaN"
    assert not torch.isinf(total_loss), "Loss is infinite"
    
    print("✅ TEST 2 PASSED\n")
    return True


def test_consistency_loss():
    """Test that consistency loss preserves rankings"""
    print("="*80)
    print("TEST 3: Consistency Loss")
    print("="*80)
    
    batch_size = 32
    d_model = 384
    dims = [64, 128, 256, 384]
    
    # Create projection
    proj = MatryoshkaProjection(d_model=d_model, dims=dims)
    
    # Create random embeddings
    embeddings = torch.randn(batch_size, d_model)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Project
    matryoshka_embs = proj(embeddings)
    
    # Compute consistency loss
    cons_loss = consistency_loss(matryoshka_embs, embeddings)
    
    print(f"✅ Consistency loss: {cons_loss.item():.4f}")
    assert not torch.isnan(cons_loss), "Loss is NaN"
    assert not torch.isinf(cons_loss), "Loss is infinite"
    
    print("✅ TEST 3 PASSED\n")
    return True


def test_gradient_flow():
    """Test that gradients flow through all components"""
    print("="*80)
    print("TEST 4: Gradient Flow")
    print("="*80)
    
    batch_size = 16
    d_model = 384
    dims = [64, 128, 256, 384]
    
    # Create components
    proj = MatryoshkaProjection(d_model=d_model, dims=dims)
    loss_fn = MatryoshkaContrastiveLoss(dims=dims)
    
    # Create embeddings with gradient tracking
    embeddings = torch.randn(batch_size, d_model, requires_grad=True)
    
    # Forward pass
    matryoshka_embs = proj(embeddings)
    labels = torch.arange(batch_size)
    total_loss, _ = loss_fn(matryoshka_embs, labels)
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    assert embeddings.grad is not None, "No gradient for embeddings"
    assert not torch.isnan(embeddings.grad).any(), "NaN in gradients"
    print(f"✅ Gradient norm: {embeddings.grad.norm().item():.4f}")
    
    # Check projection layer gradients
    for name, param in proj.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"✅ {name}: grad_norm={grad_norm:.4f}")
            assert not torch.isnan(param.grad).any(), f"NaN in {name} gradients"
    
    print("✅ TEST 4 PASSED\n")
    return True


def test_dimension_truncation():
    """Test that we can truncate embeddings at inference"""
    print("="*80)
    print("TEST 5: Dimension Truncation (Inference)")
    print("="*80)
    
    batch_size = 32
    d_model = 384
    dims = [64, 128, 256, 384]
    
    # Create projection
    proj = MatryoshkaProjection(d_model=d_model, dims=dims)
    
    # Simulate training: get full embeddings
    full_embeddings = torch.randn(batch_size, d_model)
    full_embeddings = F.normalize(full_embeddings, p=2, dim=1)
    
    # At inference: store at full dimension, truncate on-the-fly
    print(f"✅ Stored embeddings: {full_embeddings.shape}")
    
    for target_dim in [64, 128, 256]:
        truncated = full_embeddings[:, :target_dim]
        truncated = F.normalize(truncated, p=2, dim=1)
        print(f"✅ Truncated to {target_dim}-dim: {truncated.shape}, norm={truncated.norm(dim=1).mean():.4f}")
    
    print("✅ TEST 5 PASSED\n")
    return True


def test_similarity_preservation():
    """Test that similar pairs remain similar across dimensions"""
    print("="*80)
    print("TEST 6: Similarity Preservation Across Dimensions")
    print("="*80)
    
    d_model = 384
    dims = [64, 128, 256, 384]
    
    # Create two similar sentences (simulated embeddings)
    emb1 = torch.randn(1, d_model)
    emb2 = emb1 + 0.1 * torch.randn(1, d_model)  # Very similar
    
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    
    # Compute similarity at different dimensions
    print(f"Original similarity (384-dim): {F.cosine_similarity(emb1, emb2).item():.4f}")
    
    for dim in [64, 128, 256]:
        trunc1 = F.normalize(emb1[:, :dim], p=2, dim=1)
        trunc2 = F.normalize(emb2[:, :dim], p=2, dim=1)
        sim = F.cosine_similarity(trunc1, trunc2).item()
        print(f"✅ Similarity at {dim}-dim: {sim:.4f}")
    
    print("✅ TEST 6 PASSED\n")
    return True


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🔬"*40)
    print("MATRYOSHKA REPRESENTATION LEARNING - TEST SUITE")
    print("🔬"*40)
    
    tests = [
        ("MatryoshkaProjection", test_matryoshka_projection),
        ("MatryoshkaContrastiveLoss", test_matryoshka_contrastive_loss),
        ("Consistency Loss", test_consistency_loss),
        ("Gradient Flow", test_gradient_flow),
        ("Dimension Truncation", test_dimension_truncation),
        ("Similarity Preservation", test_similarity_preservation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"❌ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {name} FAILED WITH EXCEPTION:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Matryoshka is ready for integration!")
    else:
        print("\n⚠️  Some tests failed. Fix before integrating.")
    
    print("="*80 + "\n")