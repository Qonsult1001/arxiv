"""
TITANS DUAL CORE KERNEL - Both S_fast and S_slow in SRAM

Based on Google's Nested Learning paper:
https://abehrouz.github.io/files/NL.pdf

Key insight: Run BOTH cores entirely in SRAM (no HBM round-trips)
- Core 1: S_fast (attention) - parallel, vectorized
- Core 2: S_slow (memory) - recurrence in SRAM

This achieves maximum throughput by keeping all state in fast memory.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def titans_slow_core_kernel(
    # Inputs
    q_ptr, k_ptr, v_ptr, u_ptr, w_ptr,
    slow_decay_ptr, slow_gate_ptr, psi_ptr,
    attn_ptr,  # Pre-computed attention
    # Outputs
    o_slow_ptr,
    # State (in/out)
    S_slow_ptr,
    # Dimensions
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
    # Strides
    stride_b, stride_h, stride_n, stride_c, stride_d,
):
    """
    S_SLOW CORE: Linear memory with delta rule recurrence
    
    Runs entirely in SRAM - no HBM round-trips during recurrence!
    
    For each chunk i:
        S_slow = S_slow * decay
        S_slow_read = normalize(S_slow)
        u_i = u[i] - w[i] @ S_slow_read  (delta rule)
        o_inter = q[i] @ S_slow_read
        o_slow[i] = gate * (o_inter + attn @ u_i)
        S_slow = S_slow + k[i].T @ u_i
    """
    # Program ID
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Initialize S_slow in SRAM (D x D matrix)
    # Load from HBM once at start
    S_slow = tl.zeros((D, D), dtype=tl.float32)
    
    # Process all chunks sequentially (recurrence)
    for chunk_idx in range(N):
        # Compute offsets for this chunk
        base_offset = pid_b * stride_b + pid_h * stride_h + chunk_idx * stride_n
        
        # Load chunk data into SRAM
        q_block = tl.load(q_ptr + base_offset + tl.arange(0, C)[:, None] * stride_c + tl.arange(0, D)[None, :] * stride_d)
        k_block = tl.load(k_ptr + base_offset + tl.arange(0, C)[:, None] * stride_c + tl.arange(0, D)[None, :] * stride_d)
        v_block = tl.load(v_ptr + base_offset + tl.arange(0, C)[:, None] * stride_c + tl.arange(0, D)[None, :] * stride_d)
        u_block = tl.load(u_ptr + base_offset + tl.arange(0, C)[:, None] * stride_c + tl.arange(0, D)[None, :] * stride_d)
        w_block = tl.load(w_ptr + base_offset + tl.arange(0, C)[:, None] * stride_c + tl.arange(0, D)[None, :] * stride_d)
        
        # Load scalars
        decay_offset = pid_b * H * N + pid_h * N + chunk_idx
        decay = tl.load(slow_decay_ptr + decay_offset)
        gate = tl.load(slow_gate_ptr + decay_offset)
        psi = tl.load(psi_ptr + decay_offset)
        
        # Load attention matrix for this chunk
        attn_offset = pid_b * H * N * C * C + pid_h * N * C * C + chunk_idx * C * C
        attn_block = tl.load(attn_ptr + attn_offset + tl.arange(0, C)[:, None] * C + tl.arange(0, C)[None, :])
        
        # Decay modulation
        decay_mod = decay * (1.0 - 0.05 * psi)
        
        # Apply decay to S_slow (in SRAM!)
        S_slow = S_slow * decay_mod
        
        # Normalize S_slow for readout
        S_norm = tl.sqrt(tl.sum(S_slow * S_slow) + 1e-8)
        S_slow_read = S_slow / S_norm
        
        # Delta rule: u_i = u - w @ S_slow_read
        # u_i shape: [C, D]
        w_S = tl.dot(w_block, S_slow_read)  # [C, D]
        u_i = u_block - w_S
        
        # Output: o_inter = q @ S_slow_read
        o_inter = tl.dot(q_block, S_slow_read)  # [C, D]
        
        # Intra-chunk attention: attn @ u_i
        attn_u = tl.dot(attn_block, u_i)  # [C, D]
        
        # Gated output
        o_slow_chunk = gate * (o_inter + attn_u)
        
        # Store output
        out_offset = pid_b * stride_b + pid_h * stride_h + chunk_idx * stride_n
        tl.store(o_slow_ptr + out_offset + tl.arange(0, C)[:, None] * stride_c + tl.arange(0, D)[None, :] * stride_d, o_slow_chunk)
        
        # State update: S_slow = S_slow + k.T @ u_i
        k_T = tl.trans(k_block)  # [D, C]
        update = tl.dot(k_T, u_i)  # [D, D]
        S_slow = S_slow + update
    
    # Store final S_slow state
    S_offset = pid_b * H * D * D + pid_h * D * D
    tl.store(S_slow_ptr + S_offset + tl.arange(0, D)[:, None] * D + tl.arange(0, D)[None, :], S_slow)


def titans_dual_core_forward(
    q, k, v, u, w,
    slow_decay, slow_gate, psi,
    fast_gate, attn_all,
    chunk_size: int = 128,
):
    """
    TITANS DUAL CORE: Run both cores, merge outputs
    
    Core 1 (S_fast): Vectorized attention (PyTorch - already fast)
    Core 2 (S_slow): Triton kernel with SRAM state
    
    Returns:
        o: Combined output
        S_slow: Final memory state
    """
    B, H, N, C, D = q.shape
    
    # ═══════════════════════════════════════════════════════════════════════
    # CORE 1: S_FAST - Pure Attention (vectorized, no kernel needed)
    # ═══════════════════════════════════════════════════════════════════════
    o_fast = fast_gate * (attn_all @ v)  # [B, H, N, C, D]
    
    # ═══════════════════════════════════════════════════════════════════════
    # CORE 2: S_SLOW - Triton Kernel (recurrence in SRAM)
    # ═══════════════════════════════════════════════════════════════════════
    o_slow = torch.zeros_like(v)
    S_slow = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    
    # Compute mean decay per chunk
    slow_decay_mean = slow_decay.mean(dim=-1)  # [B, H, N]
    slow_gate_mean = slow_gate.mean(dim=-1)    # [B, H, N]
    
    # Launch Triton kernel
    grid = (B, H)
    titans_slow_core_kernel[grid](
        q, k, v, u, w,
        slow_decay_mean, slow_gate_mean, psi,
        attn_all,
        o_slow,
        S_slow,
        B, H, N, C, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # MERGE: Blend cores at end (S_slow dominates)
    # ═══════════════════════════════════════════════════════════════════════
    alpha = 0.1  # S_fast weight
    o = alpha * o_fast + (1.0 - alpha) * o_slow
    
    return o, S_slow


# Fallback PyTorch implementation (for when Triton isn't available)
def titans_dual_core_pytorch(
    q, k, v, u, w,
    slow_decay, slow_gate, psi,
    fast_gate, attn_all,
):
    """
    Pure PyTorch fallback - same logic as Triton kernel
    """
    B, H, N, C, D = q.shape
    
    # CORE 1: S_FAST
    o_fast = fast_gate * (attn_all @ v)
    
    # CORE 2: S_SLOW
    o_slow = torch.zeros_like(v)
    S_slow = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    
    for i in range(N):
        # Extract chunk
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i_base = u[:, :, i]
        w_i = w[:, :, i]
        attn = attn_all[:, :, i]
        
        # Decay
        decay = slow_decay[:, :, i].mean(-1, keepdim=True).unsqueeze(-1)
        gate = slow_gate[:, :, i]
        psi_i = psi[:, :, i].unsqueeze(-1).unsqueeze(-1)
        decay_mod = decay * (1.0 - 0.05 * psi_i)
        
        S_slow = S_slow * decay_mod
        
        # Normalize
        S_norm = S_slow.norm(dim=(-2, -1), keepdim=True) + 1e-8
        S_slow_read = S_slow / S_norm
        
        # Delta rule
        u_i = u_i_base - w_i @ S_slow_read
        
        # Output
        o_inter = q_i @ S_slow_read
        o_slow[:, :, i] = gate * (o_inter + attn @ u_i)
        
        # Update
        update = k_i.transpose(-1, -2) @ u_i
        S_slow = S_slow + update
    
    # MERGE
    alpha = 0.1
    o = alpha * o_fast + (1.0 - alpha) * o_slow
    
    return o, S_slow



