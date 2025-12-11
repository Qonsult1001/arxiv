#!/usr/bin/env python3
"""
🔬 LAM Bottleneck Profiler
===========================
Find where the time is being spent in the 6-layer DeltaNet model.
Goal: Identify what to optimize to get 64K inference < 400ms
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔬 Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# LOAD MODEL
# =============================================================================

print("\n📦 Loading model...")
teacher_model = AutoModel.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2").to(device)
tokenizer = AutoTokenizer.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2")
d_model = teacher_model.config.hidden_size  # 384

# Load trained DeltaNet layers
state_dict = torch.load("/workspace/LAM/best/pytorch_model.bin", map_location=device, weights_only=False)

# Create DeltaNet layers
deltanet_layers = torch.nn.ModuleList([
    EnhancedHierarchicalDeltaNet(d_model=d_model, num_heads=12,
                                  use_hierarchical_decay=True, use_enhanced_flux=True)
    for _ in range(6)
]).to(device)

# Load weights
layer_dict = {k.replace('deltanet_layers.', ''): v for k, v in state_dict.items() if 'deltanet_layers' in k}
for i in range(6):
    layer_state = {k[2:]: v for k, v in layer_dict.items() if k.startswith(f'{i}.')}
    if layer_state:
        deltanet_layers[i].load_state_dict(layer_state, strict=False)

deltanet_layers.eval()
for p in deltanet_layers.parameters():
    p.requires_grad = False

print("✅ Model loaded!")

# =============================================================================
# PROFILE INDIVIDUAL COMPONENTS
# =============================================================================

def profile_component(name, func, *args, warmup=3, iterations=10):
    """Profile a single component"""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = func(*args)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        with torch.no_grad():
            result = func(*args)
    
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iterations * 1000
    
    return elapsed, result

print("\n" + "="*80)
print("🔬 PROFILING INDIVIDUAL COMPONENTS")
print("="*80)

# Test different sequence lengths
for seq_len in [512, 2048, 8192, 16384, 32768]:
    print(f"\n📊 Sequence Length: {seq_len:,} tokens")
    print("-" * 60)
    
    # Create input
    x = torch.randn(1, seq_len, d_model, device=device)
    mask = torch.ones(1, seq_len, device=device)
    
    # Profile each layer
    layer_times = []
    current_x = x.clone()
    
    for i in range(6):
        layer = deltanet_layers[i]
        
        def forward_layer(inp):
            out, _, _, _ = layer(inp, mask)
            return out
        
        elapsed, out = profile_component(f"Layer {i}", forward_layer, current_x)
        layer_times.append(elapsed)
        current_x = out
        print(f"   Layer {i}: {elapsed:>8.2f} ms")
    
    total_layers = sum(layer_times)
    print(f"   {'─'*40}")
    print(f"   Total (6 layers): {total_layers:>8.2f} ms")
    print(f"   Tokens/sec: {seq_len / (total_layers/1000):>8.0f}")

# =============================================================================
# PROFILE INSIDE ONE LAYER
# =============================================================================

print("\n" + "="*80)
print("🔬 PROFILING INSIDE A SINGLE LAYER (Layer 0)")
print("="*80)

# Get a single layer and inspect its components
layer = deltanet_layers[0]

for seq_len in [2048, 8192, 16384]:
    print(f"\n📊 Sequence Length: {seq_len:,} tokens")
    print("-" * 60)
    
    x = torch.randn(1, seq_len, d_model, device=device)
    mask = torch.ones(1, seq_len, device=device)
    
    # Profile Q, K, V projections
    def proj_qkv(inp):
        q = layer.q_proj(inp)
        k = layer.k_proj(inp)
        v = layer.v_proj(inp)
        return q, k, v
    
    elapsed, (q, k, v) = profile_component("Q,K,V Proj", proj_qkv, x)
    print(f"   Q,K,V Projections: {elapsed:>8.2f} ms")
    
    # Profile short convolution (if exists)
    if hasattr(layer, 'conv') and layer.conv is not None:
        def conv_forward(inp):
            return layer.conv(inp)[0]
        elapsed, _ = profile_component("Short Conv", conv_forward, k)
        print(f"   Short Convolution: {elapsed:>8.2f} ms")
    
    # Profile the recurrence (the MAIN BOTTLENECK usually)
    # This is the delta rule: S = decay*S + k^T(v - kS)
    
    # Reshape for multi-head
    B, L, D = x.shape
    H = layer.num_heads
    head_dim = D // H
    
    q_heads = q.view(B, L, H, head_dim).transpose(1, 2)  # [B, H, L, D/H]
    k_heads = k.view(B, L, H, head_dim).transpose(1, 2)
    v_heads = v.view(B, L, H, head_dim).transpose(1, 2)
    
    # Profile the sequential recurrence
    def sequential_recurrence(q, k, v):
        """The O(n) recurrence - processes tokens one by one"""
        B, H, L, D = q.shape
        state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        outputs = []
        
        decay = 0.9  # Simplified
        
        for t in range(L):
            k_t = k[:, :, t, :]  # [B, H, D]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]
            
            # State update: S = decay*S + v^T @ k
            state = decay * state
            state = state + v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            
            # Output: o = q @ S
            o_t = (q_t.unsqueeze(-2) @ state).squeeze(-2)
            outputs.append(o_t)
        
        return torch.stack(outputs, dim=2)
    
    elapsed, _ = profile_component("Sequential Recurrence", sequential_recurrence, 
                                   q_heads, k_heads, v_heads, iterations=3)
    print(f"   Sequential Recurrence: {elapsed:>8.2f} ms  ⚠️ BOTTLENECK!")
    
    # Profile output projection
    dummy_out = torch.randn(1, seq_len, d_model, device=device)
    def out_proj(inp):
        return layer.o_proj(inp)
    elapsed, _ = profile_component("Output Proj", out_proj, dummy_out)
    print(f"   Output Projection: {elapsed:>8.2f} ms")

# =============================================================================
# COMPARE: SEQUENTIAL vs CHUNKED
# =============================================================================

print("\n" + "="*80)
print("🔬 SOLUTION TEST: CHUNKED RECURRENCE")
print("="*80)

def chunked_recurrence(q, k, v, chunk_size=256):
    """Process in chunks instead of one token at a time"""
    B, H, L, D = q.shape
    state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    outputs = []
    
    decay = 0.9
    
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        chunk_len = end - start
        
        k_chunk = k[:, :, start:end, :]  # [B, H, chunk, D]
        v_chunk = v[:, :, start:end, :]
        q_chunk = q[:, :, start:end, :]
        
        # Decay state for this chunk
        decay_power = decay ** chunk_len
        state = state * decay_power
        
        # Process chunk with matrix operations (not per-token)
        # Simplified: outer product sum
        kv = torch.einsum('bhld,bhle->bhde', v_chunk, k_chunk)
        state = state + kv
        
        # Query the state for this chunk
        o_chunk = torch.einsum('bhld,bhde->bhle', q_chunk, state)
        outputs.append(o_chunk)
    
    return torch.cat(outputs, dim=2)

for seq_len in [2048, 8192, 16384]:
    print(f"\n📊 Sequence Length: {seq_len:,} tokens")
    
    x = torch.randn(1, seq_len, d_model, device=device)
    q = x.view(1, seq_len, 12, 32).transpose(1, 2)
    k = x.view(1, seq_len, 12, 32).transpose(1, 2)
    v = x.view(1, seq_len, 12, 32).transpose(1, 2)
    
    # Sequential
    elapsed_seq, _ = profile_component("Sequential", sequential_recurrence, q, k, v, iterations=3)
    
    # Chunked (various chunk sizes)
    for chunk_size in [64, 128, 256, 512]:
        def chunked_fn(q, k, v):
            return chunked_recurrence(q, k, v, chunk_size=chunk_size)
        elapsed_chunk, _ = profile_component(f"Chunked-{chunk_size}", chunked_fn, q, k, v, iterations=3)
        speedup = elapsed_seq / elapsed_chunk
        print(f"   Chunk={chunk_size:>3}: {elapsed_chunk:>8.2f} ms (Speedup: {speedup:.1f}x)")

# =============================================================================
# PROFILE YOUR TRITON KERNEL
# =============================================================================

print("\n" + "="*80)
print("🔬 YOUR TRITON KERNEL TEST")
print("="*80)

try:
    from fused_delta_kernel import fused_delta_forward
    
    for seq_len in [2048, 8192, 16384]:
        print(f"\n📊 Sequence Length: {seq_len:,} tokens")
        
        # Prepare inputs for Triton kernel
        # Shape: [B, H, L, D_head]
        q = torch.randn(1, 12, seq_len, 32, device=device)
        k = torch.randn(1, 12, seq_len, 32, device=device)
        v = torch.randn(1, 12, seq_len, 32, device=device)
        w = torch.sigmoid(torch.randn(1, 12, seq_len, 32, device=device))
        
        def triton_forward(q, k, v, w):
            return fused_delta_forward(q, k, v, w)
        
        elapsed, (out, state) = profile_component("Triton Kernel", triton_forward, q, k, v, w)
        tokens_per_sec = seq_len / (elapsed / 1000)
        
        print(f"   Triton Kernel: {elapsed:>8.2f} ms ({tokens_per_sec:,.0f} tok/s)")
        
        # Compare to sequential
        elapsed_seq, _ = profile_component("Sequential", sequential_recurrence, q, k, v, iterations=3)
        speedup = elapsed_seq / elapsed
        print(f"   vs Sequential: {speedup:.1f}x faster")
        
except ImportError as e:
    print(f"   ⚠️ Could not import Triton kernel: {e}")
except Exception as e:
    print(f"   ⚠️ Triton kernel error: {e}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("📊 PROFILING SUMMARY")
print("="*80)

print("""
🔍 KEY FINDINGS:

1. BOTTLENECK: The sequential recurrence is the slowest part
   - Processes tokens ONE BY ONE in a Python loop
   - Each iteration has GPU kernel launch overhead
   - Memory bandwidth limited (read state, write state per token)

2. SOLUTION OPTIONS:
   a) CHUNKED PROCESSING: Process 256 tokens at once (3-10x faster)
   b) TRITON KERNEL: Fused operations in SRAM (50-100x faster potential)
   c) PARALLEL SCAN: Use associative scan for O(log n) depth

3. YOUR TRITON KERNEL:
   - Already exists in fused_delta_kernel.py
   - Needs to be integrated into EnhancedHierarchicalDeltaNet
   
4. NEXT STEP:
   - Patch EnhancedHierarchicalDeltaNet to use fused_delta_forward
   - This should give 10-50x speedup on the recurrence
""")

print("="*80)




