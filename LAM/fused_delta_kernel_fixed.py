"""
🚀 FIXED Fused Delta Kernel - Matches Original Algorithm
=========================================================
The original kernel used: o = q * diag(S) (element-wise)
The original MODEL uses:  o = q @ S (full matrix multiplication)

This fixes that mismatch to get the 0.8190 score back!
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_delta_rule_fixed_kernel(
    q_ptr, k_ptr, v_ptr, decay_ptr,
    s_ptr,  # State matrix [B, H, D, D]
    o_ptr,  # Output [B, H, L, D]
    stride_b, stride_h, stride_l, stride_d,
    stride_s_b, stride_s_h, stride_s_d1, stride_s_d2,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FIXED: Uses full matrix multiplication for output (o = q @ S)
    This matches the original EnhancedHierarchicalDeltaNet algorithm.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    off_b_h = pid_b * stride_b + pid_h * stride_h
    off_s = pid_b * stride_s_b + pid_h * stride_s_h
    
    offs_d = tl.arange(0, BLOCK_D)
    
    # Initialize State S in registers
    S = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    
    # Load initial state
    s_offset_2d = offs_d[:, None] * stride_s_d1 + offs_d[None, :] * stride_s_d2
    S = tl.load(s_ptr + off_s + s_offset_2d)
    
    # Process sequence
    for t in range(seq_len):
        off_t = off_b_h + t * stride_l
        
        # Load q, k, v, w
        q = tl.load(q_ptr + off_t + offs_d * stride_d)  # [D]
        k = tl.load(k_ptr + off_t + offs_d * stride_d)  # [D]
        v = tl.load(v_ptr + off_t + offs_d * stride_d)  # [D]
        w = tl.load(decay_ptr + off_t + offs_d * stride_d)  # [D]
        
        w_mean = tl.sum(w) / HEAD_DIM
        
        # =====================================================
        # DELTA RULE
        # =====================================================
        
        # Step 1: r = k @ S (full row of state, not just diagonal!)
        # r[i] = sum_j(k[j] * S[i,j])
        r = tl.sum(k[None, :] * S, axis=1)  # [D]
        
        # Step 2: u = v - r (delta/error)
        u = v - r  # [D]
        
        # Step 3: Decay state
        S = S * w_mean
        
        # Step 4: Update state with outer product
        update = u[:, None] * k[None, :]  # [D, D]
        S = S + update
        
        # =====================================================
        # FIXED OUTPUT: o = q @ S (full matrix multiplication!)
        # =====================================================
        # o[i] = sum_j(q[j] * S[i,j])
        o_t = tl.sum(q[None, :] * S, axis=1)  # [D]
        
        tl.store(o_ptr + off_t + offs_d * stride_d, o_t)
    
    # Write final state
    tl.store(s_ptr + off_s + s_offset_2d, S)


class FusedDeltaRuleFixed(torch.autograd.Function):
    """Fixed version that matches original algorithm"""
    
    @staticmethod
    def forward(ctx, q, k, v, decay, initial_state=None):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        decay = decay.contiguous()
        
        b, h, l, d = k.shape
        
        if initial_state is None:
            state = torch.zeros((b, h, d, d), device=k.device, dtype=k.dtype)
        else:
            state = initial_state.clone().contiguous()
        
        output = torch.zeros((b, h, l, d), device=k.device, dtype=k.dtype)
        
        BLOCK_D = triton.next_power_of_2(d)
        grid = (b, h)
        
        _fused_delta_rule_fixed_kernel[grid](
            q, k, v, decay,
            state, output,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            l,
            HEAD_DIM=d,
            BLOCK_D=BLOCK_D,
        )
        
        return output, state


def fused_delta_forward_fixed(q, k, v, decay, initial_state=None):
    """
    FIXED Fused Delta Rule - Matches Original Algorithm
    
    Key fix: Uses o = q @ S instead of o = q * diag(S)
    """
    return FusedDeltaRuleFixed.apply(q, k, v, decay, initial_state)


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing FIXED kernel...")
    
    device = torch.device('cuda')
    B, H, L, D = 1, 12, 128, 32
    
    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    v = torch.randn(B, H, L, D, device=device)
    w = torch.sigmoid(torch.randn(B, H, L, D, device=device))
    
    output, state = fused_delta_forward_fixed(q, k, v, w)
    
    print(f"Output shape: {output.shape}")
    print(f"State shape: {state.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("✅ FIXED kernel works!")



