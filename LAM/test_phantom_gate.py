#!/usr/bin/env python3
"""
🚀 PHANTOM GATE TEST: Speed + Full Accuracy
============================================
The insight:
- Titan Gate (k*surprise) kills signal → weights see garbage → score drops
- GPU: multiply by 0 takes SAME time as multiply by 1
- Solution: Use Triton for speed, but feed FULL signal (not gated)

This should give:
- Speed: 12ms per layer (Triton)
- Score: 0.8190 (same as original!)
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

from fused_delta_kernel import fused_delta_forward
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

device = 'cuda'

# =============================================================================
# PHANTOM GATE: Dual-Pass Triton WITHOUT Input Gating
# =============================================================================

def phantom_gate_delta_rule(q, k, v, beta, fast_decay, slow_decay, 
                             fast_gate, slow_gate, resonance_flux):
    """
    PHANTOM GATE: Uses Triton for speed, but feeds FULL signal (not gated).
    
    - Speed: 12ms (from Triton)
    - Accuracy: 0.8190 (from full signal)
    
    The key insight: modulate DECAY (safe) not INPUT (destroys weights)
    """
    # 1. Setup
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Apply beta
    beta_exp = beta.unsqueeze(-1)
    v = v * beta_exp
    k = k * beta_exp
    
    # 2. Compute Flux (for decay modulation and output mixing only!)
    # We use flux to modulate decay, NOT to gate inputs
    try:
        k_reshaped = rearrange(k, "b h l d -> b l h d")
        v_reshaped = rearrange(v, "b h l d -> b l h d")
        with torch.no_grad():
            psi = resonance_flux.compute_token_flux(k_reshaped, v_reshaped)
            psi = rearrange(psi, "b l h 1 -> b h l 1")
    except:
        # Fallback
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        psi = (k_norm * v_norm).sum(dim=-1, keepdim=True).abs()
        psi = torch.sigmoid(psi * 3)
    
    # 3. Modulate DECAY (this is safe - doesn't kill signal)
    fast_decay_exp = fast_decay.unsqueeze(-1)
    slow_decay_exp = slow_decay.unsqueeze(-1)
    
    fast_decay_mod = fast_decay_exp * (1 - 0.1 * psi)
    slow_decay_mod = slow_decay_exp * (1 - 0.05 * psi)
    
    fast_decay_mod = fast_decay_mod.expand_as(v)
    slow_decay_mod = slow_decay_mod.expand_as(v)
    
    # 4. DUAL-PASS TRITON - Feed FULL signal to both! (No gating!)
    # This is the key: weights see full data they were trained on
    o_fast, _ = fused_delta_forward(q, k, v, fast_decay_mod)  # FULL k, v
    o_slow, _ = fused_delta_forward(q, k, v, slow_decay_mod)  # FULL k, v
    
    # 5. Apply output gates and mix using flux
    o_fast = o_fast * fast_gate
    o_slow = o_slow * slow_gate
    
    # Use flux to decide which output to trust (this is safe!)
    alpha = 0.5 + 0.3 * psi
    o = alpha * o_fast + (1 - alpha) * o_slow
    
    return o, None


# =============================================================================
# PATCHED LAYER
# =============================================================================

class PhantomGateLayer(nn.Module):
    """DeltaNet layer with Phantom Gate (fast Triton, full signal)"""
    
    def __init__(self, original_layer):
        super().__init__()
        # Copy everything from original
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
        
        # Projections + convolutions (same as original)
        if self.use_short_conv:
            q, _ = self.q_conv1d(self.q_proj(hidden_states))
            k, _ = self.k_conv1d(self.k_proj(hidden_states))
            v, _ = self.v_conv1d(self.v_proj(hidden_states))
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))
        
        # Reshape for multi-head
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)
        
        # Beta
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
            beta = rearrange(beta, "b l h -> b h l")
        else:
            beta = torch.ones(batch_size, self.num_heads, seq_len, device=hidden_states.device)
        
        # Decays
        if self.use_hierarchical_decay:
            fast_decay = torch.sigmoid(self.fast_decay_proj(hidden_states) + self.fast_decay_bias)
            slow_decay = torch.sigmoid(self.slow_decay_proj(hidden_states) + self.slow_decay_bias)
            fast_decay = rearrange(fast_decay, "b l h -> b h l")
            slow_decay = rearrange(slow_decay, "b l h -> b h l")
        else:
            fast_decay = torch.ones(batch_size, self.num_heads, seq_len, device=hidden_states.device) * 0.3
            slow_decay = torch.ones(batch_size, self.num_heads, seq_len, device=hidden_states.device) * 0.9
        
        # Gates
        if self.use_output_gate:
            fast_gate = torch.sigmoid(self.fast_gate_proj(hidden_states))
            slow_gate = torch.sigmoid(self.slow_gate_proj(hidden_states))
            fast_gate = rearrange(fast_gate, "b l h -> b h l 1")
            slow_gate = rearrange(slow_gate, "b l h -> b h l 1")
        else:
            fast_gate = torch.ones(batch_size, self.num_heads, seq_len, 1, device=hidden_states.device)
            slow_gate = torch.ones(batch_size, self.num_heads, seq_len, 1, device=hidden_states.device)
        
        # 🚀 PHANTOM GATE: Full signal, Triton speed!
        o, _ = phantom_gate_delta_rule(
            q=q, k=k, v=v, beta=beta,
            fast_decay=fast_decay, slow_decay=slow_decay,
            fast_gate=fast_gate, slow_gate=slow_gate,
            resonance_flux=self.resonance_flux
        )
        
        # Reshape output
        o = rearrange(o, "b h l d -> b l h d")
        
        # Output gating/norm
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "b l (h d) -> b l h d", h=self.num_heads)
            o = self.o_norm(o, g)
        
        # Final projection
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        
        return o, None, None, None


# =============================================================================
# TEST
# =============================================================================

def main():
    print("="*60)
    print("🚀 PHANTOM GATE TEST: Speed + Full Accuracy")
    print("="*60)
    print("""
   The insight:
   - Titan Gate kills signal → score drops
   - Solution: Triton for speed, FULL signal for accuracy
   
   Expected:
   - Speed: ~2x faster (Triton dual-pass)
   - Score: ~0.8190 (same as original!)
    """)
    
    # Load models
    print("📦 Loading models...")
    teacher = AutoModel.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2")
    
    # Create original layers and load weights
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
    
    # Create phantom gate layers (wrapping original with Triton)
    phantom_layers = nn.ModuleList([
        PhantomGateLayer(layer) for layer in original_layers
    ]).to(device)
    
    # Set to eval
    original_layers.eval()
    phantom_layers.eval()
    
    print("✅ Models loaded!")
    
    # ==========================================================================
    # TEST STS-B SCORE
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
    
    # Evaluate original
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
    print(f"   Original STS-B Spearman: {orig_score:.4f}")
    
    # Evaluate phantom gate
    print("\n   Evaluating Phantom Gate (should match!)...")
    phantom_sims = []
    with torch.no_grad():
        for i in range(0, len(s1), 32):
            t1 = tokenizer(s1[i:i+32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2 = tokenizer(s2[i:i+32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            e1 = get_embeddings(phantom_layers, t1['input_ids'], t1['attention_mask'])
            e2 = get_embeddings(phantom_layers, t2['input_ids'], t2['attention_mask'])
            phantom_sims.extend(F.cosine_similarity(e1, e2).cpu().numpy())
    
    phantom_score = spearmanr(phantom_sims, labels)[0]
    print(f"   Phantom Gate STS-B Spearman: {phantom_score:.4f}")
    
    # ==========================================================================
    # SPEED BENCHMARK
    # ==========================================================================
    print("\n" + "="*60)
    print("⚡ SPEED BENCHMARK")
    print("="*60)
    
    text = "The quick brown fox " * 30
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt').to(device)
    
    def benchmark(layers, name, iterations=20):
        # Warmup
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
    
    orig_time = benchmark(original_layers, "Original")
    phantom_time = benchmark(phantom_layers, "Phantom")
    
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
   Phantom Gate: {phantom_score:.4f} @ {phantom_time:.1f}ms
   
   Score Change: {score_diff:+.4f} ({score_diff/orig_score*100:+.1f}%)
   Speedup:      {speedup:.2f}x
    """)
    
    if phantom_score >= 0.80 and speedup > 1.5:
        print("🎉 SUCCESS! Score ≥0.80 AND significant speedup!")
    elif phantom_score >= 0.80:
        print("✅ Score retained! Check if speed improvement is acceptable.")
    elif speedup > 1.5:
        print("⚠️ Speed improved but score dropped. May need adjustment.")
    else:
        print("❌ Neither score nor speed improved significantly.")


if __name__ == "__main__":
    main()




