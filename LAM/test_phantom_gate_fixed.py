#!/usr/bin/env python3
"""
🚀 PHANTOM GATE TEST (FIXED KERNEL)
====================================
Using the FIXED kernel that matches original algorithm:
- Original kernel: o = q * diag(S)
- FIXED kernel:    o = q @ S (full matrix multiplication)

This should restore 0.8190 score!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from einops import rearrange
import time
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# USE FIXED KERNEL!
from fused_delta_kernel_fixed import fused_delta_forward_fixed
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

device = 'cuda'

# =============================================================================
# PHANTOM GATE with FIXED kernel
# =============================================================================

def phantom_gate_delta_rule(q, k, v, beta, fast_decay, slow_decay, 
                             fast_gate, slow_gate, resonance_flux):
    """
    PHANTOM GATE with FIXED kernel + Cross-timescale Coupling.
    Uses full matrix multiplication (o = q @ S) to match original.
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Apply beta
    beta_exp = beta.unsqueeze(-1)
    v = v * beta_exp
    k = k * beta_exp
    
    # Compute Flux for decay modulation and cross-coupling
    try:
        k_reshaped = rearrange(k, "b h l d -> b l h d")
        v_reshaped = rearrange(v, "b h l d -> b l h d")
        with torch.no_grad():
            psi = resonance_flux.compute_token_flux(k_reshaped, v_reshaped)
            psi = rearrange(psi, "b l h 1 -> b h l 1")
    except:
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        psi = (k_norm * v_norm).sum(dim=-1, keepdim=True).abs()
        psi = torch.sigmoid(psi * 3)
    
    # Modulate DECAY (safe)
    fast_decay_exp = fast_decay.unsqueeze(-1)
    slow_decay_exp = slow_decay.unsqueeze(-1)
    
    fast_decay_mod = fast_decay_exp * (1 - 0.1 * psi)
    slow_decay_mod = slow_decay_exp * (1 - 0.05 * psi)
    
    fast_decay_mod = fast_decay_mod.expand_as(v)
    slow_decay_mod = slow_decay_mod.expand_as(v)
    
    # DUAL-PASS with FIXED kernel (full signal, no gating)
    o_fast, s_fast = fused_delta_forward_fixed(q, k, v, fast_decay_mod)
    o_slow, s_slow = fused_delta_forward_fixed(q, k, v, slow_decay_mod)
    
    # CROSS-TIMESCALE COUPLING (the missing ingredient!)
    # This is what makes dual-state meaningful
    cross_coupling = 0.05 * psi.mean()  # Data-dependent coupling strength
    
    # Apply coupling: fast learns from slow, slow learns from fast
    # This happens in output space (simpler than state space coupling)
    o_coupled_fast = o_fast + cross_coupling * o_slow
    o_coupled_slow = o_slow + cross_coupling * o_fast
    
    # Apply output gates and mix
    o_coupled_fast = o_coupled_fast * fast_gate
    o_coupled_slow = o_coupled_slow * slow_gate
    
    alpha = 0.5 + 0.3 * psi
    o = alpha * o_coupled_fast + (1 - alpha) * o_coupled_slow
    
    return o, None


# =============================================================================
# PATCHED LAYER
# =============================================================================

class PhantomGateLayer(nn.Module):
    """DeltaNet layer with Phantom Gate (FIXED kernel)"""
    
    def __init__(self, original_layer):
        super().__init__()
        self.hidden_size = original_layer.hidden_size
        self.num_heads = original_layer.num_heads
        self.head_k_dim = original_layer.head_k_dim
        self.head_v_dim = original_layer.head_v_dim
        
        self.q_proj = original_layer.q_proj
        self.k_proj = original_layer.k_proj
        self.v_proj = original_layer.v_proj
        self.o_proj = original_layer.o_proj
        
        self.use_short_conv = original_layer.use_short_conv
        if self.use_short_conv:
            self.q_conv1d = original_layer.q_conv1d
            self.k_conv1d = original_layer.k_conv1d
            self.v_conv1d = original_layer.v_conv1d
        
        self.use_beta = original_layer.use_beta
        if self.use_beta:
            self.b_proj = original_layer.b_proj
        
        self.use_hierarchical_decay = original_layer.use_hierarchical_decay
        if self.use_hierarchical_decay:
            self.fast_decay_proj = original_layer.fast_decay_proj
            self.fast_decay_bias = original_layer.fast_decay_bias
            self.slow_decay_proj = original_layer.slow_decay_proj
            self.slow_decay_bias = original_layer.slow_decay_bias
        
        self.use_output_gate = original_layer.use_output_gate
        if self.use_output_gate:
            self.fast_gate_proj = original_layer.fast_gate_proj
            self.slow_gate_proj = original_layer.slow_gate_proj
        
        self.resonance_flux = original_layer.resonance_flux
        
        self.use_gate = original_layer.use_gate
        if self.use_gate:
            self.g_proj = original_layer.g_proj
            self.o_norm = original_layer.o_norm
        
        self.use_delta_rule = original_layer.use_delta_rule
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        
        if self.use_short_conv:
            q, _ = self.q_conv1d(self.q_proj(hidden_states))
            k, _ = self.k_conv1d(self.k_proj(hidden_states))
            v, _ = self.v_conv1d(self.v_proj(hidden_states))
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))
        
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)
        
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
            beta = rearrange(beta, "b l h -> b h l")
        else:
            beta = torch.ones(batch_size, self.num_heads, seq_len, device=hidden_states.device)
        
        if self.use_hierarchical_decay:
            fast_decay = torch.sigmoid(self.fast_decay_proj(hidden_states) + self.fast_decay_bias)
            slow_decay = torch.sigmoid(self.slow_decay_proj(hidden_states) + self.slow_decay_bias)
            fast_decay = rearrange(fast_decay, "b l h -> b h l")
            slow_decay = rearrange(slow_decay, "b l h -> b h l")
        else:
            fast_decay = torch.ones(batch_size, self.num_heads, seq_len, device=hidden_states.device) * 0.3
            slow_decay = torch.ones(batch_size, self.num_heads, seq_len, device=hidden_states.device) * 0.9
        
        if self.use_output_gate:
            fast_gate = torch.sigmoid(self.fast_gate_proj(hidden_states))
            slow_gate = torch.sigmoid(self.slow_gate_proj(hidden_states))
            fast_gate = rearrange(fast_gate, "b l h -> b h l 1")
            slow_gate = rearrange(slow_gate, "b l h -> b h l 1")
        else:
            fast_gate = torch.ones(batch_size, self.num_heads, seq_len, 1, device=hidden_states.device)
            slow_gate = torch.ones(batch_size, self.num_heads, seq_len, 1, device=hidden_states.device)
        
        # PHANTOM GATE with FIXED kernel
        o, _ = phantom_gate_delta_rule(
            q=q, k=k, v=v, beta=beta,
            fast_decay=fast_decay, slow_decay=slow_decay,
            fast_gate=fast_gate, slow_gate=slow_gate,
            resonance_flux=self.resonance_flux
        )
        
        o = rearrange(o, "b h l d -> b l h d")
        
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g)
        
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        
        return o, None, None, None


# =============================================================================
# TEST
# =============================================================================

def main():
    print("="*60)
    print("🚀 PHANTOM GATE TEST (FIXED KERNEL)")
    print("="*60)
    print("""
   FIX: Changed kernel from o = q * diag(S) to o = q @ S
   This matches the original algorithm exactly!
   
   Expected:
   - Speed: ~2x faster (Triton)
   - Score: ~0.8190 (RESTORED!)
    """)
    
    # First verify the fixed kernel works
    print("📦 Testing fixed kernel...")
    from fused_delta_kernel_fixed import fused_delta_forward_fixed
    
    B, H, L, D = 1, 12, 128, 32
    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    v = torch.randn(B, H, L, D, device=device)
    w = torch.sigmoid(torch.randn(B, H, L, D, device=device))
    
    output, state = fused_delta_forward_fixed(q, k, v, w)
    print(f"   Output shape: {output.shape} ✅")
    
    # Load models
    print("\n📦 Loading models...")
    teacher = AutoModel.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2")
    
    original_layers = nn.ModuleList([
        EnhancedHierarchicalDeltaNet(d_model=384, num_heads=12,
                                      use_hierarchical_decay=True, use_enhanced_flux=True)
        for _ in range(6)
    ]).to(device)
    
    state_dict = torch.load("/workspace/LAM/best/pytorch_model.bin", map_location=device, weights_only=False)
    layer_dict = {k.replace('deltanet_layers.', ''): v for k, v in state_dict.items() if 'deltanet_layers' in k}
    for i in range(6):
        layer_state = {k[2:]: v for k, v in layer_dict.items() if k.startswith(f'{i}.')}
        if layer_state:
            original_layers[i].load_state_dict(layer_state, strict=False)
    
    phantom_layers = nn.ModuleList([
        PhantomGateLayer(layer) for layer in original_layers
    ]).to(device)
    
    original_layers.eval()
    phantom_layers.eval()
    
    print("✅ Models loaded!")
    
    # ==========================================================================
    # STS-B TEST
    # ==========================================================================
    print("\n" + "="*60)
    print("📊 STS-B EVALUATION")
    print("="*60)
    
    ds = load_dataset("sentence-transformers/stsb", split="test")
    s1, s2 = list(ds["sentence1"]), list(ds["sentence2"])
    labels = np.array(ds["score"] if "score" in ds.column_names else ds["label"])
    
    norms = [teacher.encoder.layer[i].attention.output.LayerNorm for i in range(6)]
    ffns = [teacher.encoder.layer[i].intermediate for i in range(6)]
    output_denses = [teacher.encoder.layer[i].output.dense for i in range(6)]
    ffn_norms = [teacher.encoder.layer[i].output.LayerNorm for i in range(6)]
    
    def get_embeddings(layers, input_ids, mask):
        x = teacher.embeddings(input_ids)
        for i in range(6):
            residual = x
            x_attn, _, _, _ = layers[i](x, mask)
            x = norms[i](residual + x_attn)
            residual = x
            x_ffn = F.gelu(ffns[i](x))
            x_ffn = output_denses[i](x_ffn)
            x = ffn_norms[i](residual + x_ffn)
        mask_exp = mask.unsqueeze(-1).float()
        emb = (x * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=1)
    
    print("\n   Evaluating Original (0.8190 target)...")
    orig_sims = []
    with torch.no_grad():
        for i in range(0, len(s1), 32):
            t1 = tokenizer(s1[i:i+32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2 = tokenizer(s2[i:i+32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            e1 = get_embeddings(original_layers, t1['input_ids'], t1['attention_mask'])
            e2 = get_embeddings(original_layers, t2['input_ids'], t2['attention_mask'])
            orig_sims.extend(F.cosine_similarity(e1, e2).cpu().numpy())
    
    orig_score = spearmanr(orig_sims, labels)[0]
    print(f"   Original: {orig_score:.4f}")
    
    print("\n   Evaluating FIXED Phantom Gate...")
    phantom_sims = []
    with torch.no_grad():
        for i in range(0, len(s1), 32):
            t1 = tokenizer(s1[i:i+32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2 = tokenizer(s2[i:i+32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            e1 = get_embeddings(phantom_layers, t1['input_ids'], t1['attention_mask'])
            e2 = get_embeddings(phantom_layers, t2['input_ids'], t2['attention_mask'])
            phantom_sims.extend(F.cosine_similarity(e1, e2).cpu().numpy())
    
    phantom_score = spearmanr(phantom_sims, labels)[0]
    print(f"   Phantom (FIXED): {phantom_score:.4f}")
    
    # ==========================================================================
    # SPEED
    # ==========================================================================
    print("\n" + "="*60)
    print("⚡ SPEED BENCHMARK")
    print("="*60)
    
    text = "The quick brown fox " * 30
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt').to(device)
    
    def benchmark(layers, iterations=20):
        with torch.no_grad():
            for _ in range(3):
                _ = get_embeddings(layers, tokens['input_ids'], tokens['attention_mask'])
        torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = get_embeddings(layers, tokens['input_ids'], tokens['attention_mask'])
        torch.cuda.synchronize()
        return (time.time() - start) / iterations * 1000
    
    orig_time = benchmark(original_layers)
    phantom_time = benchmark(phantom_layers)
    
    print(f"\n   Original: {orig_time:.1f} ms")
    print(f"   Phantom:  {phantom_time:.1f} ms")
    print(f"   Speedup:  {orig_time/phantom_time:.2f}x")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "="*60)
    print("🎯 SUMMARY")
    print("="*60)
    
    score_diff = phantom_score - orig_score
    speedup = orig_time / phantom_time
    
    print(f"""
   Original:     {orig_score:.4f} @ {orig_time:.1f}ms
   FIXED Phantom: {phantom_score:.4f} @ {phantom_time:.1f}ms
   
   Score Change: {score_diff:+.4f} ({score_diff/orig_score*100:+.1f}%)
   Speedup:      {speedup:.2f}x
    """)
    
    if abs(score_diff) < 0.01:
        print("🎉 SUCCESS! Score RESTORED with speedup!")
    elif phantom_score >= 0.80:
        print("✅ Score close enough (≥0.80)")
    else:
        print("⚠️ Score still needs work")


if __name__ == "__main__":
    main()

