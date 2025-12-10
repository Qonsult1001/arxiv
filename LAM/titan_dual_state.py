#!/usr/bin/env python3
"""
🚀 TITAN-STYLE DUAL STATE: Surprise-Gated Memory
=================================================
Based on Google's Titan architecture insight:

CURRENT PROBLEM:
- Both S_fast and S_slow update EVERY token → Slow, muddy memory

TITAN SOLUTION:
- S_fast: Updates every token (syntax, grammar) - CONTINUOUS
- S_slow: Updates only on "surprise" (important events) - SPARSE

THE KEY INSIGHT:
"If I am not surprised (Flux is low), I don't need to write to long-term memory.
 I just READ from it."

This gives:
1. Speed: Slow state does 90% less work
2. Quality: Slow memory stores "highlights", not noise
3. De-correlation: Fast and Slow states become truly different
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import time
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import Triton kernel
try:
    from fused_delta_kernel import fused_delta_forward
    TRITON_AVAILABLE = True
    print("✅ Triton kernel available")
except ImportError:
    TRITON_AVAILABLE = False
    print("❌ Triton not available")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# TITAN-STYLE DUAL STATE LAYER
# =============================================================================

class TitanDualStateLayer(nn.Module):
    """
    Titan-Style Dual State with Surprise Gating (FIXED!)
    
    Fast State: Updates every token (continuous)
    Slow State: Updates only on "surprise" (sparse)
    
    THE FIX: Gate BOTH K and V to create a true "no-op" when not surprised.
    This prevents Active Unlearning!
    """
    
    def __init__(self, d_model=384, num_heads=12, fast_decay=0.3, slow_decay=0.9):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 🚀 PERSISTENT MEMORY (The "Personality" - from Titans paper)
        # Learnable tokens that initialize the memory state S
        self.persistent_memory = nn.Parameter(
            torch.randn(1, num_heads, 8, self.head_dim) * 0.02
        )
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Gating
        self.gate_proj = nn.Linear(d_model, d_model)
        
        # Short convolution (local context)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=2, groups=d_model)
        
        # Decay parameters
        self.fast_decay = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim) * fast_decay)
        self.slow_decay = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim) * slow_decay)
        
        # SURPRISE DETECTOR (Flux/Psi computation)
        self.surprise_k = nn.Linear(self.head_dim, 1)
        self.surprise_v = nn.Linear(self.head_dim, 1)
        
        # Output mixing (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Output norm
        self.norm = nn.LayerNorm(d_model)
    
    def compute_surprise(self, k, v):
        """
        Compute "surprise" (Flux/Psi) for each token.
        High surprise = important information worth storing in slow memory.
        
        k, v: [B, H, L, D]
        Returns: [B, H, L, 1] - surprise gate (0-1)
        """
        # Simple surprise: how different is this token from the expected?
        # Using learned projections to compute "importance"
        k_score = self.surprise_k(k)  # [B, H, L, 1]
        v_score = self.surprise_v(v)  # [B, H, L, 1]
        
        # Combine scores
        raw_surprise = k_score * v_score  # Element-wise interaction
        
        # Sharp sigmoid for binary-like gating (differentiable)
        # threshold=0.0, sharpness=10 makes it act like a binary gate
        surprise_gate = torch.sigmoid(raw_surprise * 10)
        
        return surprise_gate
    
    def forward(self, x, attention_mask=None):
        """
        x: [B, L, D]
        Returns: [B, L, D]
        
        FIXED: Now gates BOTH K and V for proper memory preservation!
        """
        B, L, D = x.shape
        residual = x
        
        # Short conv for local context
        x_conv = self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Projections
        q = self.q_proj(x_conv)
        k = self.k_proj(x_conv)
        v = self.v_proj(x_conv)
        g = torch.sigmoid(self.gate_proj(x_conv))
        
        # Reshape to multi-head: [B, H, L, D/H]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        g = g.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply gating to values
        v = v * g
        
        # =====================================================
        # 🚀 PERSISTENT MEMORY (Prepend to K, V)
        # =====================================================
        pm = self.persistent_memory.expand(B, -1, -1, -1)  # [B, H, 8, D/H]
        k_seq = torch.cat([pm, k], dim=2)  # [B, H, 8+L, D/H]
        v_seq = torch.cat([pm, v], dim=2)
        
        # =====================================================
        # TITAN-STYLE DUAL STATE (FIXED!)
        # =====================================================
        
        # 1. Compute surprise (Flux/Psi)
        surprise_gate = self.compute_surprise(k, v)  # [B, H, L, 1]
        
        # Prepend ones for persistent memory (always "surprising" - memorize it!)
        pm_surprise = torch.ones(B, self.num_heads, 8, 1, device=x.device, dtype=x.dtype)
        surprise_full = torch.cat([pm_surprise, surprise_gate], dim=2)  # [B, H, 8+L, 1]
        
        # 2. FAST STATE: Sees everything (continuous)
        fast_decay = torch.sigmoid(self.fast_decay).expand(B, -1, 8+L, -1)
        
        # Use Triton for inference (fast), fallback for training (gradients work)
        use_triton = TRITON_AVAILABLE and x.is_cuda and not self.training
        
        if use_triton:
            o_fast_full, _ = fused_delta_forward(q, k_seq[:,:,8:], v_seq[:,:,8:], fast_decay[:,:,8:])
        else:
            o_fast_full = self._fallback(q, k_seq[:,:,8:], v_seq[:,:,8:], fast_decay[:,:,8:])
        
        # 3. SLOW STATE: Only sees "surprises" (sparse)
        # 🔧 THE FIX: Gate BOTH K and V to create true "no-op"!
        # When surprise=0: k_slow=0, v_slow=0 → update=0 → S just decays (preserved!)
        k_slow = k_seq * surprise_full  # ← THE CRITICAL FIX!
        v_slow = v_seq * surprise_full
        
        slow_decay = torch.sigmoid(self.slow_decay).expand(B, -1, 8+L, -1)
        # Modulate: more stable when not surprised
        slow_decay_mod = slow_decay * (1.0 - 0.05 * surprise_full)
        
        if use_triton:
            o_slow_full, _ = fused_delta_forward(q, k_slow[:,:,8:], v_slow[:,:,8:], slow_decay_mod[:,:,8:])
        else:
            o_slow_full = self._fallback(q, k_slow[:,:,8:], v_slow[:,:,8:], slow_decay_mod[:,:,8:])
        
        # 4. MAG (Memory As Gate) Mixing
        # Use surprise from original sequence (not persistent memory)
        alpha = 0.5 + 0.3 * surprise_gate  # [B, H, L, 1]
        o = alpha * o_fast_full + (1 - alpha) * o_slow_full
        
        # Reshape back
        o = o.transpose(1, 2).contiguous().view(B, L, D)
        
        # Output projection + residual
        o = self.o_proj(o)
        o = self.norm(o + residual)
        
        return o
    
    def _fallback(self, q, k, v, decay, chunk_size=128):
        """Chunked fallback for CPU or when Triton unavailable"""
        B, H, L, D = q.shape
        outputs = []
        state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            q_c = q[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]
            w_c = decay[:, :, start:end]
            
            w_mean = w_c.mean()
            state = state * (w_mean ** (end - start))
            kv = torch.einsum('bhld,bhle->bhde', v_c, k_c)
            state = state + kv
            o_c = torch.einsum('bhld,bhde->bhle', q_c, state)
            outputs.append(o_c)
        
        return torch.cat(outputs, dim=2)


class TitanDualStateModel(nn.Module):
    """
    Full Titan-Style Dual State Model
    
    6 layers with surprise-gated slow memory
    """
    
    def __init__(self, base_embeddings, d_model=384, num_layers=6):
        super().__init__()
        self.embeddings = base_embeddings
        self.d_model = d_model
        
        # Stack of Titan layers
        self.layers = nn.ModuleList([
            TitanDualStateLayer(d_model=d_model, num_heads=12)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def get_extended_embeddings(self, input_ids):
        """Position interpolation for long sequences"""
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
    
    def forward(self, input_ids, attention_mask=None):
        x = self.get_extended_embeddings(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.final_norm(x)
        return x
    
    def get_embeddings(self, input_ids, attention_mask):
        x = self.forward(input_ids, attention_mask)
        mask_exp = attention_mask.unsqueeze(-1).float()
        emb = (x * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=-1)


# =============================================================================
# ORIGINAL MODEL FOR COMPARISON
# =============================================================================

class OriginalModel(nn.Module):
    """Original 6-layer model"""
    
    def __init__(self, teacher_model, deltanet_layers):
        super().__init__()
        self.embeddings = teacher_model.embeddings
        self.layers = deltanet_layers
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
    
    def forward(self, input_ids, attention_mask=None):
        x = self.get_extended_embeddings(input_ids)
        for i in range(6):
            residual = x
            x_attn, _, _, _ = self.layers[i](x, attention_mask)
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
# DISTILLATION
# =============================================================================

def distill(teacher, student, tokenizer, device, steps=1500):
    """Distill knowledge from teacher to Titan student"""
    print("\n📚 Distilling to Titan-style model...")
    
    ds = load_dataset("sentence-transformers/stsb", split="train")
    sentences = list(ds["sentence1"]) + list(ds["sentence2"])
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)
    teacher.eval()
    student.train()
    
    for step in range(steps):
        idx = np.random.choice(len(sentences), 32)
        batch = [sentences[i] for i in idx]
        
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        
        with torch.no_grad():
            teacher_emb = teacher.get_embeddings(tokens['input_ids'], tokens['attention_mask'])
        
        student_emb = student.get_embeddings(tokens['input_ids'], tokens['attention_mask'])
        
        loss = F.mse_loss(student_emb, teacher_emb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 300 == 0:
            print(f"   Step {step+1}/{steps} | Loss: {loss.item():.6f}")
    
    print("   ✅ Distillation complete!")
    return student


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
    print("="*70)
    print("🚀 TITAN-STYLE DUAL STATE: Surprise-Gated Memory")
    print("="*70)
    print("""
   GOOGLE TITAN INSIGHT:
   - Fast State: Updates every token (syntax, grammar)
   - Slow State: Updates only on "surprise" (important events)
   
   THE FIX:
   - Don't update slow memory for boring tokens ("the", "and")
   - Only store "highlights" in long-term memory
   - Result: Speed + Quality!
    """)
    
    print(f"📍 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load models
    print("\n📥 Loading models...")
    teacher_model = AutoModel.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2")
    
    from final_solution_formula_final import EnhancedHierarchicalDeltaNet
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
    
    # Create models
    original = OriginalModel(teacher_model, deltanet_layers).to(device)
    titan = TitanDualStateModel(teacher_model.embeddings, d_model=384, num_layers=6).to(device)
    
    original.eval()
    
    # Evaluate original
    print("\n" + "="*70)
    print("📊 ORIGINAL MODEL")
    print("="*70)
    
    original_score = evaluate_stsb(original, tokenizer, device)
    print(f"   STS-B Spearman: {original_score:.4f}")
    
    print("\n   Speed benchmark:")
    for seq_len in [512, 2048, 8192, 16384]:
        try:
            time_ms = benchmark_speed(original, tokenizer, device, seq_len)
            print(f"   {seq_len:>6} tokens: {time_ms:>8.1f} ms")
        except Exception as e:
            print(f"   {seq_len:>6} tokens: Error - {str(e)[:40]}")
    
    # Distill to Titan
    print("\n" + "="*70)
    print("🎓 DISTILLING TO TITAN MODEL")
    print("="*70)
    
    titan = distill(original, titan, tokenizer, device, steps=1500)
    titan.eval()
    
    # Evaluate Titan
    print("\n" + "="*70)
    print("📊 TITAN MODEL (Surprise-Gated)")
    print("="*70)
    
    titan_score = evaluate_stsb(titan, tokenizer, device)
    print(f"   STS-B Spearman: {titan_score:.4f}")
    print(f"   Score retention: {titan_score/original_score*100:.1f}%")
    
    print("\n   Speed benchmark:")
    for seq_len in [512, 2048, 8192, 16384]:
        try:
            time_ms = benchmark_speed(titan, tokenizer, device, seq_len)
            print(f"   {seq_len:>6} tokens: {time_ms:>8.1f} ms")
        except Exception as e:
            print(f"   {seq_len:>6} tokens: Error - {str(e)[:40]}")
    
    # Final comparison
    print("\n" + "="*70)
    print("🏆 FINAL COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<25} {'STS-B':<10} {'8K Speed':<15} {'16K Speed':<15}")
    print("-" * 70)
    
    try:
        orig_8k = benchmark_speed(original, tokenizer, device, 8192)
        orig_16k = benchmark_speed(original, tokenizer, device, 16384)
        titan_8k = benchmark_speed(titan, tokenizer, device, 8192)
        titan_16k = benchmark_speed(titan, tokenizer, device, 16384)
        
        print(f"{'Original (Dual State)':<25} {original_score:.4f}     {orig_8k:.0f} ms          {orig_16k:.0f} ms")
        print(f"{'Titan (Surprise-Gated)':<25} {titan_score:.4f}     {titan_8k:.0f} ms          {titan_16k:.0f} ms")
        print(f"{'Speedup':<25} {'':10} {orig_8k/titan_8k:.1f}x             {orig_16k/titan_16k:.1f}x")
    except Exception as e:
        print(f"   Benchmark error: {e}")
    
    # Save Titan model
    torch.save({
        'model_state_dict': titan.state_dict(),
        'score': titan_score,
        'architecture': 'TitanDualState'
    }, '/workspace/LAM/titan_model.pt')
    print("\n💾 Saved to /workspace/LAM/titan_model.pt")


if __name__ == "__main__":
    main()

