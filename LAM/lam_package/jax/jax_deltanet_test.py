#!/usr/bin/env python3
"""
JAX Implementation Test for Linear Attention (DeltaNet)
======================================================

This is a test implementation using Google JAX to compare performance
with the Cython-compiled version.

Key differences:
- Functional programming (no mutable state)
- Uses jax.lax.scan to compile loops into fused kernels
- Pure JAX/Flax implementation
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import time
import torch
from typing import Dict, Tuple, Optional

# JAX configuration
jax.config.update("jax_enable_x64", False)  # Use float32 for speed
jax.config.update("jax_platform_name", "gpu" if jax.devices()[0].platform == "gpu" else "cpu")


def hierarchical_delta_rule_jax(
    q: jnp.ndarray,  # [b, h, n, c, d_k]
    k: jnp.ndarray,  # [b, h, n, c, d_k]
    v: jnp.ndarray,  # [b, h, n, c, d_v]
    beta: jnp.ndarray,  # [b, h, n]
    fast_decay: jnp.ndarray,  # [b, h]
    slow_decay: jnp.ndarray,  # [b, h]
    fast_gate: jnp.ndarray,  # [b, h]
    slow_gate: jnp.ndarray,  # [b, h]
    resonance_flux_fn,  # Function to compute resonance flux
) -> jnp.ndarray:
    """
    JAX implementation of hierarchical delta rule using jax.lax.scan.
    
    This compiles the chunk loop into a single fused kernel.
    """
    b, h, n, c, d_k = q.shape
    d_v = v.shape[-1]
    
    # Initialize hierarchical states
    S_fast_init = jnp.zeros((b, h, d_k, d_v))
    S_slow_init = jnp.zeros((b, h, d_k, d_v))
    
    # Prepare inputs for scan: [n] chunks
    # Reshape to [n, b, h, c, d] for scan
    q_scan = jnp.transpose(q, (2, 0, 1, 3, 4))  # [n, b, h, c, d_k]
    k_scan = jnp.transpose(k, (2, 0, 1, 3, 4))  # [n, b, h, c, d_k]
    v_scan = jnp.transpose(v, (2, 0, 1, 3, 4))  # [n, b, h, c, d_v]
    # Beta needs to be expanded to match chunk dimension
    beta_expanded = beta[:, :, :, None]  # [b, h, n, 1]
    beta_scan = jnp.transpose(beta_expanded, (2, 0, 1, 3))  # [n, b, h, 1]
    
    def scan_body(carry, inputs):
        """
        Process one chunk.
        
        Args:
            carry: (S_fast, S_slow) - hierarchical states
            inputs: (q_chunk, k_chunk, v_chunk, beta_chunk) - chunk data
        
        Returns:
            (new_carry, output_chunk)
        """
        S_fast, S_slow = carry
        q_chunk, k_chunk, v_chunk, beta_chunk = inputs
        
        # q_chunk: [b, h, c, d_k]
        # k_chunk: [b, h, c, d_k]
        # v_chunk: [b, h, c, d_v]
        # beta_chunk: [b, h, 1]
        beta_chunk = beta_chunk.squeeze(-1)  # [b, h]
        
        # Compute attention (causal mask)
        # q_chunk: [b, h, c, d_k], k_chunk: [b, h, c, d_k]
        attn_scores = jnp.einsum('bhcd,bhkd->bhck', q_chunk, k_chunk)  # [b, h, c, c]
        # Apply causal mask (upper triangular) - mask out future tokens
        mask = jnp.triu(jnp.ones((c, c)), k=1)  # [c, c]
        attn_scores = attn_scores - 1e9 * mask[None, None, :, :]  # [b, h, c, c]
        attn = jax.nn.softmax(attn_scores, axis=-1)  # [b, h, c, c]
        
        # Compute u and w
        u_chunk = jnp.einsum('bhck,bhkd->bhcd', attn, v_chunk)  # [b, h, c, d_v]
        k_beta = k_chunk * beta_chunk[:, :, None, None]  # [b, h, c, d_k]
        w_chunk = jnp.einsum('bhck,bhkd->bhcd', attn, k_beta)  # [b, h, c, d_k]
        
        # Compute resonance flux
        psi = resonance_flux_fn(k_chunk, u_chunk)  # [b, h]
        
        # Update hierarchical states
        # S_fast = fast_decay * S_fast + fast_gate * (k @ u.T)
        # S_slow = slow_decay * S_slow + slow_gate * (k @ u.T)
        # k_chunk: [b, h, c, d_k], u_chunk: [b, h, c, d_v]
        # We need to compute sum over c: sum_c(k_c @ u_c.T) -> [b, h, d_k, d_v]
        k_u_interaction = jnp.einsum('bhck,bhcl->bhkl', k_chunk, u_chunk)  # [b, h, d_k, d_v]
        fast_update = fast_gate[:, :, None, None] * k_u_interaction
        slow_update = slow_gate[:, :, None, None] * k_u_interaction
        
        S_fast_new = fast_decay[:, :, None, None] * S_fast + fast_update
        S_slow_new = slow_decay[:, :, None, None] * S_slow + slow_update
        
        # Flux-modulated readout
        # output = (1 - psi) * (S_fast @ q) + psi * (S_slow @ q)
        # S_fast_new: [b, h, d_k, d_v], q_chunk: [b, h, c, d_k]
        # We compute for each token in chunk: q @ S -> [b, h, c, d_v]
        fast_readout = jnp.einsum('bhcd,bhkl->bhcl', q_chunk, S_fast_new)  # [b, h, c, d_v]
        slow_readout = jnp.einsum('bhcd,bhkl->bhcl', q_chunk, S_slow_new)  # [b, h, c, d_v]
        
        psi_expanded = psi[:, :, None, None]  # [b, h, 1, 1]
        output_chunk = (1 - psi_expanded) * fast_readout + psi_expanded * slow_readout  # [b, h, c, d_v]
        
        new_carry = (S_fast_new, S_slow_new)
        return new_carry, output_chunk
    
    # Initial carry state
    init_carry = (S_fast_init, S_slow_init)
    
    # Scan over chunks
    final_carry, outputs = lax.scan(
        scan_body,
        init_carry,
        (q_scan, k_scan, v_scan, beta_scan)
    )
    
    # Reshape outputs back to [b, h, n, c, d_v]
    outputs = jnp.transpose(outputs, (1, 2, 0, 3, 4))  # [b, h, n, c, d_v]
    
    return outputs


def simple_resonance_flux_jax(k_chunk, u_chunk):
    """
    Simple resonance flux computation.
    """
    # k_chunk: [b, h, c, d_k]
    # u_chunk: [b, h, c, d_v]
    
    # Bilinear interaction
    interaction = jnp.einsum('bhcd,bhcd->bhc', k_chunk, u_chunk)  # [b, h, c]
    avg_attn = jnp.mean(interaction, axis=-1)  # [b, h]
    
    # Simple flux (clamped)
    psi = jax.nn.sigmoid(avg_attn)
    return jnp.clip(psi, 0.01, 0.99)


@jax.jit
def deltanet_forward_jax(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    fast_decay: jnp.ndarray,
    slow_decay: jnp.ndarray,
    fast_gate: jnp.ndarray,
    slow_gate: jnp.ndarray,
):
    """
    JIT-compiled forward pass.
    """
    return hierarchical_delta_rule_jax(
        q, k, v, beta,
        fast_decay, slow_decay,
        fast_gate, slow_gate,
        simple_resonance_flux_jax
    )


def create_test_data(batch_size=2, num_heads=12, num_chunks=4, chunk_size=64, d_model=384):
    """Create test data for benchmarking."""
    d_k = d_v = d_model // num_heads
    
    # PyTorch tensors (original)
    q_torch = torch.randn(batch_size, num_heads, num_chunks, chunk_size, d_k, device='cuda' if torch.cuda.is_available() else 'cpu')
    k_torch = torch.randn(batch_size, num_heads, num_chunks, chunk_size, d_k, device='cuda' if torch.cuda.is_available() else 'cpu')
    v_torch = torch.randn(batch_size, num_heads, num_chunks, chunk_size, d_v, device='cuda' if torch.cuda.is_available() else 'cpu')
    beta_torch = torch.rand(batch_size, num_heads, num_chunks, device=q_torch.device)
    fast_decay_torch = torch.full((batch_size, num_heads), 0.3, device=q_torch.device)
    slow_decay_torch = torch.full((batch_size, num_heads), 0.85, device=q_torch.device)
    fast_gate_torch = torch.full((batch_size, num_heads), 0.5, device=q_torch.device)
    slow_gate_torch = torch.full((batch_size, num_heads), 0.5, device=q_torch.device)
    
    # Convert to JAX
    q_jax = jnp.array(q_torch.cpu().numpy())
    k_jax = jnp.array(k_torch.cpu().numpy())
    v_jax = jnp.array(v_torch.cpu().numpy())
    beta_jax = jnp.array(beta_torch.cpu().numpy())
    fast_decay_jax = jnp.array(fast_decay_torch.cpu().numpy())
    slow_decay_jax = jnp.array(slow_decay_torch.cpu().numpy())
    fast_gate_jax = jnp.array(fast_gate_torch.cpu().numpy())
    slow_gate_jax = jnp.array(slow_gate_torch.cpu().numpy())
    
    return {
        'torch': {
            'q': q_torch, 'k': k_torch, 'v': v_torch,
            'beta': beta_torch,
            'fast_decay': fast_decay_torch, 'slow_decay': slow_decay_torch,
            'fast_gate': fast_gate_torch, 'slow_gate': slow_gate_torch,
        },
        'jax': {
            'q': q_jax, 'k': k_jax, 'v': v_jax,
            'beta': beta_jax,
            'fast_decay': fast_decay_jax, 'slow_decay': slow_decay_jax,
            'fast_gate': fast_gate_jax, 'slow_gate': slow_gate_jax,
        }
    }


def benchmark_jax_vs_cython():
    """Benchmark JAX implementation vs Cython."""
    print("="*70)
    print("🚀 JAX vs Cython Performance Comparison")
    print("="*70)
    
    # Test configurations
    configs = [
        {"batch": 1, "heads": 12, "chunks": 4, "chunk_size": 64, "d_model": 384, "name": "Small"},
        {"batch": 2, "heads": 12, "chunks": 8, "chunk_size": 128, "d_model": 384, "name": "Medium"},
        {"batch": 4, "heads": 12, "chunks": 16, "chunk_size": 256, "d_model": 384, "name": "Large"},
    ]
    
    for config in configs:
        print(f"\n📊 Testing {config['name']} config:")
        print(f"   Batch: {config['batch']}, Heads: {config['heads']}, Chunks: {config['chunks']}, Chunk Size: {config['chunk_size']}")
        
        # Create test data
        data = create_test_data(
            batch_size=config['batch'],
            num_heads=config['heads'],
            num_chunks=config['chunks'],
            chunk_size=config['chunk_size'],
            d_model=config['d_model']
        )
        
        # Warmup JAX (compilation happens here)
        print("   ⚡ Warming up JAX (compiling)...")
        _ = deltanet_forward_jax(**data['jax']).block_until_ready()
        
        # Benchmark JAX
        print("   🧪 Benchmarking JAX...")
        num_runs = 10
        jax_times = []
        for _ in range(num_runs):
            start = time.time()
            result_jax = deltanet_forward_jax(**data['jax']).block_until_ready()
            jax_times.append(time.time() - start)
        
        avg_jax = np.mean(jax_times[1:])  # Skip first run (compilation)
        std_jax = np.std(jax_times[1:])
        
        print(f"   ✅ JAX: {avg_jax*1000:.2f}ms ± {std_jax*1000:.2f}ms")
        print(f"   📦 Output shape: {result_jax.shape}")
        
        # Note: Cython benchmark would require importing the actual compiled module
        print(f"   📝 Note: Cython comparison requires compiled _core.so module")
        print(f"   💡 JAX uses jax.lax.scan to fuse the chunk loop into a single kernel")


if __name__ == "__main__":
    print("🔬 JAX Implementation Test for Linear Attention")
    print("="*70)
    print("\nThis test compares JAX implementation using jax.lax.scan")
    print("with the Cython-compiled version.\n")
    
    try:
        benchmark_jax_vs_cython()
        print("\n" + "="*70)
        print("✅ JAX test completed!")
        print("="*70)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

