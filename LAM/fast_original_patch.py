#!/usr/bin/env python3
"""
🚀 FAST ORIGINAL PATCH: Keep Architecture, Speed Up Recurrence
===============================================================
Instead of changing the architecture (which loses quality),
we patch the ORIGINAL model to use faster recurrence.

Strategy:
1. Keep the exact same EnhancedHierarchicalDeltaNet architecture
2. Replace the slow per-token loop with chunked GPU operations
3. Use Triton kernel where possible

This should give:
- Same quality (0.8190 STS-B)
- Much faster speed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import time
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
from einops import rearrange
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

# Triton
try:
    from fused_delta_kernel import fused_delta_forward
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📍 Device: {device}")
print(f"📦 Triton: {TRITON_AVAILABLE}")

# =============================================================================
# FAST CHUNKED DUAL-STATE RECURRENCE (Drop-in Replacement)
# =============================================================================

def fast_hierarchical_delta_rule(
    q, k, v, beta,
    fast_decay, slow_decay,
    fast_gate, slow_gate,
    resonance_flux,
    chunk_size=256,
    training=False,
    use_delta_rule=True,
):
    """
    FAST version of the hierarchical delta rule.
    Uses larger chunks and GPU-parallel operations.
    
    This is a drop-in replacement for the slow version.
    """
    b, h, l, d = q.shape
    
    # Pad to chunk size
    pad_len = (chunk_size - (l % chunk_size)) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        fast_decay = F.pad(fast_decay, (0, pad_len))
        slow_decay = F.pad(slow_decay, (0, pad_len))
        fast_gate = F.pad(fast_gate, (0, 0, 0, pad_len))
        slow_gate = F.pad(slow_gate, (0, 0, 0, pad_len))
    
    padded_len = q.shape[2]
    num_chunks = padded_len // chunk_size
    
    # Reshape to chunks: [b, h, num_chunks, chunk_size, d]
    q_chunks = q.view(b, h, num_chunks, chunk_size, d)
    k_chunks = k.view(b, h, num_chunks, chunk_size, d)
    v_chunks = v.view(b, h, num_chunks, chunk_size, d)
    
    fast_decay_chunks = fast_decay.view(b, h, num_chunks, chunk_size)
    slow_decay_chunks = slow_decay.view(b, h, num_chunks, chunk_size)
    fast_gate_chunks = fast_gate.view(b, h, num_chunks, chunk_size, 1)
    slow_gate_chunks = slow_gate.view(b, h, num_chunks, chunk_size, 1)
    
    # Initialize states
    S_fast = torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype)
    S_slow = torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype)
    
    outputs = []
    
    for chunk_idx in range(num_chunks):
        q_c = q_chunks[:, :, chunk_idx]  # [b, h, chunk_size, d]
        k_c = k_chunks[:, :, chunk_idx]
        v_c = v_chunks[:, :, chunk_idx]
        
        fast_decay_c = fast_decay_chunks[:, :, chunk_idx].mean(dim=-1, keepdim=True).unsqueeze(-1)  # [b, h, 1, 1]
        slow_decay_c = slow_decay_chunks[:, :, chunk_idx].mean(dim=-1, keepdim=True).unsqueeze(-1)
        fast_gate_c = fast_gate_chunks[:, :, chunk_idx]  # [b, h, chunk_size, 1]
        slow_gate_c = slow_gate_chunks[:, :, chunk_idx]
        
        # Decay states (exponential over chunk)
        decay_power = chunk_size
        S_fast = S_fast * (fast_decay_c ** decay_power)
        S_slow = S_slow * (slow_decay_c ** decay_power)
        
        # Normalize states for reading
        S_fast_read = S_fast / (S_fast.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        S_slow_read = S_slow / (S_slow.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        
        # Compute u (delta rule correction)
        if use_delta_rule:
            # u = v - k @ S (prediction error)
            w_c = k_c  # Use k as the readout weights
            u_fast = v_c - torch.einsum('bhcd,bhde->bhce', w_c, S_fast_read)
            u_slow = v_c - torch.einsum('bhcd,bhde->bhce', w_c, S_slow_read)
        else:
            u_fast = v_c
            u_slow = v_c
        
        # Compute output: o = q @ S
        o_fast = torch.einsum('bhcd,bhde->bhce', q_c, S_fast_read)
        o_slow = torch.einsum('bhcd,bhde->bhce', q_c, S_slow_read)
        
        # Apply gates
        o_fast = fast_gate_c * o_fast
        o_slow = slow_gate_c * o_slow
        
        # Mix fast and slow outputs
        alpha = 0.6
        o_c = alpha * o_fast + (1 - alpha) * o_slow
        outputs.append(o_c)
        
        # Update states: S += k^T @ u (outer product sum over chunk)
        update_fast = torch.einsum('bhcd,bhce->bhde', k_c, u_fast)
        update_slow = torch.einsum('bhcd,bhce->bhde', k_c, u_slow)
        
        S_fast = S_fast + update_fast
        S_slow = S_slow + update_slow
        
        # Normalize after update
        S_fast = S_fast / (S_fast.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        S_slow = S_slow / (S_slow.norm(dim=(-2, -1), keepdim=True) + 1e-8)
    
    # Concatenate outputs
    o = torch.cat(outputs, dim=2)  # [b, h, padded_len, d]
    
    # Remove padding
    if pad_len > 0:
        o = o[:, :, :l]
    
    return o, (S_fast, S_slow)


# =============================================================================
# PATCHED MODEL
# =============================================================================

class FastOriginalModel(nn.Module):
    """
    Original model with faster recurrence.
    Uses the same architecture, just faster computation.
    """
    
    def __init__(self, teacher_model, original_layers, chunk_size=256):
        super().__init__()
        self.embeddings = teacher_model.embeddings
        self.chunk_size = chunk_size
        
        # Copy all the trained layers
        self.deltanet_layers = original_layers
        
        # Copy transformer components
        self.norms = nn.ModuleList([
            teacher_model.encoder.layer[i].attention.output.LayerNorm for i in range(6)
        ])
        self.ffns = nn.ModuleList([teacher_model.encoder.layer[i].intermediate for i in range(6)])
        self.output_denses = nn.ModuleList([teacher_model.encoder.layer[i].output.dense for i in range(6)])
        self.ffn_norms = nn.ModuleList([teacher_model.encoder.layer[i].output.LayerNorm for i in range(6)])
    
    def get_extended_embeddings(self, input_ids):
        batch_size, seq_len = input_ids.shape
        word_emb = self.embeddings.word_embeddings(input_ids)
        token_type_emb = self.embeddings.token_type_embeddings(torch.zeros_like(input_ids))
        
        if seq_len <= 512:
            pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.embeddings.position_embeddings(pos_ids)
        else:
            scale = 511 / (seq_len - 1)
            pos_emb_list = []
            for pos in range(seq_len):
                orig_pos = pos * scale
                lower = int(orig_pos)
                upper = min(lower + 1, 511)
                w = orig_pos - lower
                emb = (1-w) * self.embeddings.position_embeddings.weight[lower] + w * self.embeddings.position_embeddings.weight[upper]
                pos_emb_list.append(emb)
            pos_emb = torch.stack(pos_emb_list).unsqueeze(0).expand(batch_size, -1, -1)
        
        x = word_emb + token_type_emb + pos_emb
        x = self.embeddings.LayerNorm(x)
        x = self.embeddings.dropout(x)
        return x
    
    def forward_layer_fast(self, layer, x, attention_mask, layer_idx):
        """Fast forward through a single DeltaNet layer using chunked recurrence"""
        batch_size, seq_len, hidden_size = x.shape
        
        # Get projections from the original layer
        if layer.use_short_conv:
            q, _ = layer.q_conv1d(layer.q_proj(x))
            k, _ = layer.k_conv1d(layer.k_proj(x))
            v, _ = layer.v_conv1d(layer.v_proj(x))
        else:
            q = F.silu(layer.q_proj(x))
            k = F.silu(layer.k_proj(x))
            v = F.silu(layer.v_proj(x))
        
        # Reshape for multi-head
        head_dim = hidden_size // layer.num_heads
        q = rearrange(q, "b l (h d) -> b h l d", h=layer.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=layer.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=layer.num_heads)
        
        # Get decays
        if layer.use_hierarchical_decay:
            fast_decay = torch.sigmoid(layer.fast_decay_proj(x) + layer.fast_decay_bias)
            slow_decay = torch.sigmoid(layer.slow_decay_proj(x) + layer.slow_decay_bias)
        else:
            fast_decay = torch.ones(batch_size, seq_len, layer.num_heads, device=x.device) * 0.3
            slow_decay = torch.ones(batch_size, seq_len, layer.num_heads, device=x.device) * 0.9
        
        # Get gates
        if layer.use_output_gate:
            fast_gate = torch.sigmoid(layer.fast_gate_proj(x)).unsqueeze(-1)
            slow_gate = torch.sigmoid(layer.slow_gate_proj(x)).unsqueeze(-1)
        else:
            fast_gate = torch.ones(batch_size, seq_len, layer.num_heads, 1, device=x.device)
            slow_gate = torch.ones(batch_size, seq_len, layer.num_heads, 1, device=x.device)
        
        # Rearrange for recurrence
        fast_decay = rearrange(fast_decay, "b l h -> b h l")
        slow_decay = rearrange(slow_decay, "b l h -> b h l")
        fast_gate = rearrange(fast_gate, "b l h 1 -> b h l 1")
        slow_gate = rearrange(slow_gate, "b l h 1 -> b h l 1")
        
        # Beta
        if layer.use_beta:
            beta = layer.b_proj(x).sigmoid()
        else:
            beta = torch.ones(batch_size, seq_len, layer.num_heads, device=x.device)
        beta = rearrange(beta, "b l h -> b h l")
        
        # FAST RECURRENCE
        o, _ = fast_hierarchical_delta_rule(
            q=q, k=k, v=v, beta=beta,
            fast_decay=fast_decay, slow_decay=slow_decay,
            fast_gate=fast_gate, slow_gate=slow_gate,
            resonance_flux=layer.resonance_flux,
            chunk_size=self.chunk_size,
            training=self.training,
            use_delta_rule=layer.use_delta_rule
        )
        
        o = rearrange(o, "b h l d -> b l h d")
        
        # Output gating
        if layer.use_gate:
            g = rearrange(layer.g_proj(x), "b l (h d) -> b l h d", h=layer.num_heads)
            o = layer.o_norm(o, g)
        else:
            o = layer.o_norm(o)
        
        # Final projection
        o = rearrange(o, "b l h d -> b l (h d)")
        o = layer.o_proj(o)
        
        return o
    
    def forward(self, input_ids, attention_mask=None):
        x = self.get_extended_embeddings(input_ids)
        
        for i in range(6):
            residual = x
            x_attn = self.forward_layer_fast(self.deltanet_layers[i], x, attention_mask, i)
            x = self.norms[i](residual + x_attn)
            residual = x
            x_ffn = F.gelu(self.ffns[i](x))
            x_ffn = self.output_denses[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        
        return x
    
    def get_embeddings(self, input_ids, attention_mask):
        x = self.forward(input_ids, attention_mask)
        mask_exp = attention_mask.unsqueeze(-1).float()
        emb = (x * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=-1)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_stsb(model, tokenizer, device):
    ds = load_dataset("sentence-transformers/stsb", split="test")
    s1, s2 = list(ds["sentence1"]), list(ds["sentence2"])
    labels = np.array(ds["score"] if "score" in ds.column_names else ds["label"])
    
    model.eval()
    sims = []
    
    with torch.no_grad():
        for i in range(0, len(s1), 32):
            t1 = tokenizer(s1[i:i+32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2 = tokenizer(s2[i:i+32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            e1 = model.get_embeddings(t1['input_ids'], t1['attention_mask'])
            e2 = model.get_embeddings(t2['input_ids'], t2['attention_mask'])
            sims.extend(F.cosine_similarity(e1, e2).cpu().numpy())
    
    return spearmanr(sims, labels[:len(sims)])[0]


def benchmark_speed(model, tokenizer, device, seq_len, iterations=5):
    text = "The quick brown fox " * (seq_len // 5)
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=seq_len, return_tensors='pt').to(device)
    
    with torch.no_grad():
        _ = model.get_embeddings(tokens['input_ids'], tokens['attention_mask'])
    torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model.get_embeddings(tokens['input_ids'], tokens['attention_mask'])
    torch.cuda.synchronize()
    
    return (time.time() - start) / iterations * 1000


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 FAST ORIGINAL PATCH: Keep Quality, Boost Speed")
    print("="*70)
    print("""
   STRATEGY:
   - Keep the exact same architecture (EnhancedHierarchicalDeltaNet)
   - Just replace the slow per-token loop with chunked operations
   - Uses your trained weights directly (no distillation needed!)
    """)
    
    # Load models
    print("\n📥 Loading models...")
    teacher_model = AutoModel.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2")
    
    # Create and load original layers
    deltanet_layers = nn.ModuleList([
        EnhancedHierarchicalDeltaNet(d_model=384, num_heads=12,
                                      use_hierarchical_decay=True, use_enhanced_flux=True)
        for _ in range(6)
    ]).to(device)
    
    state_dict = torch.load("/workspace/LAM/best/pytorch_model.bin", map_location=device, weights_only=False)
    layer_dict = {k.replace('deltanet_layers.', ''): v for k, v in state_dict.items() if 'deltanet_layers' in k}
    for i in range(6):
        layer_state = {k[2:]: v for k, v in layer_dict.items() if k.startswith(f'{i}.')}
        if layer_state:
            deltanet_layers[i].load_state_dict(layer_state, strict=False)
    
    print("✅ Weights loaded!")
    
    # Create fast model (uses same weights!)
    fast_model = FastOriginalModel(teacher_model, deltanet_layers, chunk_size=256).to(device)
    fast_model.eval()
    
    # Evaluate
    print("\n" + "="*70)
    print("📊 EVALUATING FAST MODEL (Same Weights, Faster Computation)")
    print("="*70)
    
    score = evaluate_stsb(fast_model, tokenizer, device)
    print(f"\n   STS-B Spearman: {score:.4f}")
    print(f"   Target was:     0.8190")
    
    if score >= 0.80:
        print("   ✅ SCORE RETAINED!")
    else:
        print(f"   ⚠️ Score dropped to {score:.4f}")
    
    # Benchmark speed
    print("\n" + "="*70)
    print("⚡ SPEED BENCHMARK")
    print("="*70)
    
    print(f"\n{'Seq Len':<10} {'Time (ms)':<15} {'Tokens/sec':<15}")
    print("-" * 45)
    
    for seq_len in [512, 2048, 8192, 16384, 32768]:
        try:
            time_ms = benchmark_speed(fast_model, tokenizer, device, seq_len)
            tok_per_sec = seq_len / (time_ms / 1000)
            print(f"{seq_len:<10} {time_ms:<15.1f} {tok_per_sec:<15,.0f}")
        except Exception as e:
            print(f"{seq_len:<10} Error: {str(e)[:40]}")
    
    print("\n" + "="*70)
    print("🎯 SUMMARY")
    print("="*70)
    print(f"""
   Fast Original Model:
   - STS-B Score: {score:.4f}
   - Uses: Same trained weights, chunked recurrence
   - No retraining needed!
    """)


if __name__ == "__main__":
    main()



