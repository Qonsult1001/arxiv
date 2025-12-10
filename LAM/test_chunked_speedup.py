#!/usr/bin/env python3
"""
🚀 CHUNKED SPEEDUP TEST: Keep EXACT algorithm, just bigger chunks
=================================================================
The profiling showed:
- Token-by-token: 712ms
- Chunk-512: 2.14ms (316x faster!)

This keeps the EXACT algorithm (including all cross-timescale coupling,
state normalization, intra-chunk attention) but with larger chunks.

Expected:
- Speed: 5-10x faster
- Score: EXACTLY 0.8190 (no loss!)
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
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

device = 'cuda'


# =============================================================================
# PATCHED _enhanced_hierarchical_delta_rule_impl with larger chunks
# =============================================================================

def patched_delta_impl(
    q, k, v, beta, fast_decay, slow_decay, fast_gate, slow_gate, resonance_flux,
    chunk_size=256,  # BIGGER CHUNKS for speed!
    training=False, use_delta_rule=True
):
    """
    PATCHED: Same algorithm, but with chunk_size=256 for speed.
    
    This preserves ALL the original logic:
    - Dual-state (S_fast, S_slow)
    - Cross-timescale coupling
    - State normalization
    - Intra-chunk attention
    
    Just processes in bigger chunks = fewer iterations = faster!
    """
    # Import the original implementation
    from final_solution_formula_final import _enhanced_hierarchical_delta_rule_impl
    
    # Call original with bigger chunk size
    return _enhanced_hierarchical_delta_rule_impl(
        q, k, v, beta, fast_decay, slow_decay, fast_gate, slow_gate, resonance_flux,
        chunk_size=chunk_size,  # INCREASED!
        training=training, use_delta_rule=use_delta_rule
    )


class ChunkedLayer(nn.Module):
    """Wrapper that uses larger chunk size"""
    
    def __init__(self, original_layer, chunk_size=256):
        super().__init__()
        self.layer = original_layer
        self.chunk_size = chunk_size
        
        # Store reference to attributes
        self.hidden_size = original_layer.hidden_size
        self.num_heads = original_layer.num_heads
        self.head_k_dim = original_layer.head_k_dim
        self.head_v_dim = original_layer.head_v_dim
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Use original layer's projections
        layer = self.layer
        
        if layer.use_short_conv:
            q, _ = layer.q_conv1d(layer.q_proj(hidden_states))
            k, _ = layer.k_conv1d(layer.k_proj(hidden_states))
            v, _ = layer.v_conv1d(layer.v_proj(hidden_states))
        else:
            q = F.silu(layer.q_proj(hidden_states))
            k = F.silu(layer.k_proj(hidden_states))
            v = F.silu(layer.v_proj(hidden_states))
        
        q = rearrange(q, "b l (h d) -> b h l d", h=layer.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=layer.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=layer.num_heads)
        
        if layer.use_beta:
            beta = layer.b_proj(hidden_states).sigmoid()
            beta = rearrange(beta, "b l h -> b h l")
        else:
            beta = torch.ones(batch_size, layer.num_heads, seq_len, device=hidden_states.device)
        
        if layer.use_hierarchical_decay:
            fast_decay = torch.sigmoid(layer.fast_decay_proj(hidden_states) + layer.fast_decay_bias)
            slow_decay = torch.sigmoid(layer.slow_decay_proj(hidden_states) + layer.slow_decay_bias)
            fast_decay = rearrange(fast_decay, "b l h -> b h l")
            slow_decay = rearrange(slow_decay, "b l h -> b h l")
        else:
            fast_decay = torch.ones(batch_size, layer.num_heads, seq_len, device=hidden_states.device) * 0.3
            slow_decay = torch.ones(batch_size, layer.num_heads, seq_len, device=hidden_states.device) * 0.9
        
        if layer.use_output_gate:
            # fast_gate_proj outputs [B, L, num_heads], add unsqueeze for [B, L, H, 1]
            fast_gate = torch.sigmoid(layer.fast_gate_proj(hidden_states)).unsqueeze(-1)
            slow_gate = torch.sigmoid(layer.slow_gate_proj(hidden_states)).unsqueeze(-1)
            # Rearrange to [B, H, L, 1]
            fast_gate = rearrange(fast_gate, "b l h 1 -> b h l 1")
            slow_gate = rearrange(slow_gate, "b l h 1 -> b h l 1")
        else:
            fast_gate = torch.ones(batch_size, layer.num_heads, seq_len, 1, device=hidden_states.device)
            slow_gate = torch.ones(batch_size, layer.num_heads, seq_len, 1, device=hidden_states.device)
        
        # Call with LARGER chunk size!
        o, _ = patched_delta_impl(
            q=q, k=k, v=v, beta=beta,
            fast_decay=fast_decay, slow_decay=slow_decay,
            fast_gate=fast_gate, slow_gate=slow_gate,
            resonance_flux=layer.resonance_flux,
            chunk_size=self.chunk_size,  # BIGGER!
            training=layer.training, use_delta_rule=layer.use_delta_rule
        )
        
        o = rearrange(o, "b h l d -> b l h d")
        
        if layer.use_gate:
            g = rearrange(layer.g_proj(hidden_states), "b l (h d) -> b l h d", h=layer.num_heads)
            o = layer.o_norm(o, g)
        
        o = rearrange(o, "b l h d -> b l (h d)")
        o = layer.o_proj(o)
        
        return o, None, None, None


# =============================================================================
# TEST
# =============================================================================

def main():
    print("="*60)
    print("🚀 CHUNKED SPEEDUP TEST")
    print("="*60)
    print("""
   Strategy: Keep EXACT algorithm, just use bigger chunks
   
   This should give:
   - Speed: 2-5x faster (bigger chunks = fewer iterations)
   - Score: EXACTLY 0.8190 (no algorithm change!)
    """)
    
    # Load models
    print("📦 Loading models...")
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
    
    # Create chunked layers with bigger chunks
    chunked_layers = nn.ModuleList([
        ChunkedLayer(layer, chunk_size=256) for layer in original_layers
    ]).to(device)
    
    original_layers.eval()
    chunked_layers.eval()
    
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
    
    print("\n   Evaluating Original (chunk=32)...")
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
    
    print("\n   Evaluating Chunked (chunk=256)...")
    chunked_sims = []
    with torch.no_grad():
        for i in range(0, len(s1), 32):
            t1 = tokenizer(s1[i:i+32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2 = tokenizer(s2[i:i+32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            e1 = get_embeddings(chunked_layers, t1['input_ids'], t1['attention_mask'])
            e2 = get_embeddings(chunked_layers, t2['input_ids'], t2['attention_mask'])
            chunked_sims.extend(F.cosine_similarity(e1, e2).cpu().numpy())
    
    chunked_score = spearmanr(chunked_sims, labels)[0]
    print(f"   Chunked:  {chunked_score:.4f}")
    
    # ==========================================================================
    # SPEED BENCHMARK
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
    chunked_time = benchmark(chunked_layers)
    
    print(f"\n   Original (chunk=32):  {orig_time:.1f} ms")
    print(f"   Chunked (chunk=256):  {chunked_time:.1f} ms")
    print(f"   Speedup: {orig_time/chunked_time:.2f}x")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "="*60)
    print("🎯 SUMMARY")
    print("="*60)
    
    score_diff = chunked_score - orig_score
    speedup = orig_time / chunked_time
    
    print(f"""
   Original (chunk=32):  {orig_score:.4f} @ {orig_time:.1f}ms
   Chunked (chunk=256):  {chunked_score:.4f} @ {chunked_time:.1f}ms
   
   Score Change: {score_diff:+.4f} ({score_diff/orig_score*100:+.2f}%)
   Speedup:      {speedup:.2f}x
    """)
    
    if abs(score_diff) < 0.005:
        print("🎉 SUCCESS! Score preserved with speedup!")
    elif speedup > 1.5 and chunked_score >= 0.80:
        print("✅ Good enough! Minor score change with speedup.")
    else:
        print("⚠️ Need to investigate further.")


if __name__ == "__main__":
    main()

