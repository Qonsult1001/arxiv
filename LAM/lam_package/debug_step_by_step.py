#!/usr/bin/env python3
"""
Step-by-step comparison of JAX and Cython to find exact divergence point.
"""

import torch
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'build'))

from lam._jax_core import hierarchical_delta_rule_jax
from lam import LAM
from einops import rearrange

def compare_intermediate(name, cython_val, jax_val, threshold=1e-5):
    """Compare intermediate values."""
    if isinstance(cython_val, torch.Tensor):
        cython_val = cython_val.detach().cpu().numpy()
    if isinstance(jax_val, jnp.ndarray):
        jax_val = np.array(jax_val)
    
    diff = np.abs(cython_val - jax_val)
    max_diff = np.max(diff) if diff.size > 0 else 0
    mean_diff = np.mean(diff) if diff.size > 0 else 0
    
    cython_flat = cython_val.flatten()
    jax_flat = jax_val.flatten()
    cos_sim = np.dot(cython_flat, jax_flat) / (np.linalg.norm(cython_flat) * np.linalg.norm(jax_flat) + 1e-9)
    
    match = max_diff < threshold
    status = "✅" if match else "❌"
    
    print(f"{status} {name}:")
    print(f"    Shape: {cython_val.shape} vs {jax_val.shape}")
    print(f"    Max diff: {max_diff:.10e}")
    print(f"    Mean diff: {mean_diff:.10e}")
    print(f"    Cosine sim: {cos_sim:.10f}")
    if not match and cython_val.size <= 10:
        print(f"    Cython: {cython_flat}")
        print(f"    JAX:    {jax_flat}")
    elif not match:
        print(f"    First 3 - Cython: {cython_flat[:3]}")
        print(f"    First 3 - JAX:    {jax_flat[:3]}")
    print()
    
    return match, max_diff, cos_sim

print("="*80)
print("STEP-BY-STEP INTERMEDIATE VALUE COMPARISON")
print("="*80)

model_path = "../LAM-base-v1"
model = LAM(model_path, backend='cython')
test_sentence = "Hello"
input_ids = model.tokenizer.encode(test_sentence)
input_ids_tensor = torch.tensor([input_ids.ids], dtype=torch.long, device=model.device)

model._model.eval()
with torch.no_grad():
    # Forward to first DeltaNet layer
    x = model._model.embeddings['word_embeddings'](input_ids_tensor)
    token_type_ids = torch.zeros_like(input_ids_tensor)
    x = x + model._model.embeddings['token_type_embeddings'](token_type_ids)
    position_ids = torch.arange(input_ids_tensor.shape[1], dtype=torch.long, device=input_ids_tensor.device).unsqueeze(0)
    x = x + model._model.embeddings['position_embeddings'](position_ids)
    x = model._model.embeddings['LayerNorm'](x)
    
    # Get q, k, v from first layer
    layer = model._model.deltanet_layers[0]
    q = layer.q_proj(x)
    k = layer.k_proj(x)
    v = layer.v_proj(x)
    
    # Apply conv and activation
    q, _ = layer.q_conv1d(q)
    k, _ = layer.k_conv1d(k)
    v, _ = layer.v_conv1d(v)
    q = torch.nn.functional.silu(q)
    k = torch.nn.functional.silu(k)
    v = torch.nn.functional.silu(v)
    
    # Reshape for multi-head
    q = rearrange(q, "b l (h d) -> b h l d", h=layer.num_heads)
    k = rearrange(k, "b l (h d) -> b h l d", h=layer.num_heads)
    v = rearrange(v, "b l (h d) -> b h l d", h=layer.num_heads)
    
    # L2 norm
    from lam._core import l2norm
    q_cython = l2norm(q)
    k_cython = l2norm(k)
    
    # Beta scaling
    beta = layer.b_proj(x).sigmoid()
    beta = rearrange(beta, "b l h -> b h l")
    beta_expanded = beta.unsqueeze(-1)
    k_beta_cython = k_cython * beta_expanded
    v_scaled_cython = v * beta_expanded
    
    # Hierarchical decay and gates
    fast_decay = torch.sigmoid(layer.fast_decay_proj(x) + layer.fast_decay_bias)
    slow_decay = torch.sigmoid(layer.slow_decay_proj(x) + layer.slow_decay_bias)
    fast_decay = rearrange(fast_decay, "b l h -> b h l")
    slow_decay = rearrange(slow_decay, "b l h -> b h l")
    
    fast_gate = torch.sigmoid(layer.fast_gate_proj(x)).unsqueeze(-1)
    slow_gate = torch.sigmoid(layer.slow_gate_proj(x)).unsqueeze(-1)
    fast_gate = rearrange(fast_gate, "b l h d -> b h l d")
    slow_gate = rearrange(slow_gate, "b l h d -> b h l d")
    
    # Chunking
    chunk_size = 64
    l = q_cython.shape[2]
    
    if l < chunk_size:
        actual_chunk_size = l
        n_chunks = 1
        pad_len = 0
    else:
        n_chunks = (l + chunk_size - 1) // chunk_size
        pad_len = (n_chunks * chunk_size) - l
        actual_chunk_size = chunk_size
    
    # Pad and chunk
    q_padded = torch.nn.functional.pad(q_cython, (0, 0, 0, pad_len)) if pad_len > 0 else q_cython
    k_padded = torch.nn.functional.pad(k_cython, (0, 0, 0, pad_len)) if pad_len > 0 else k_cython
    v_padded = torch.nn.functional.pad(v_scaled_cython, (0, 0, 0, pad_len)) if pad_len > 0 else v_scaled_cython
    beta_padded = torch.nn.functional.pad(beta, (0, pad_len)) if pad_len > 0 else beta
    fast_decay_padded = torch.nn.functional.pad(fast_decay, (0, pad_len)) if pad_len > 0 else fast_decay
    slow_decay_padded = torch.nn.functional.pad(slow_decay, (0, pad_len)) if pad_len > 0 else slow_decay
    fast_gate_padded = torch.nn.functional.pad(fast_gate, (0, 0, 0, pad_len)) if pad_len > 0 else fast_gate
    slow_gate_padded = torch.nn.functional.pad(slow_gate, (0, 0, 0, pad_len)) if pad_len > 0 else slow_gate
    
    # Reshape to chunks
    q_chunked = q_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, layer.head_k_dim)
    k_chunked = k_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, layer.head_k_dim)
    v_chunked = v_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, layer.head_v_dim)
    beta_chunked = beta_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size)
    fast_decay_chunked = fast_decay_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size)
    slow_decay_chunked = slow_decay_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size)
    fast_gate_chunked = fast_gate_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, 1)
    slow_gate_chunked = slow_gate_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, 1)
    
    # Compute token flux and k_beta before chunking
    token_flux_pre = layer.resonance_flux.compute_token_flux(k_beta_cython, v_scaled_cython)
    token_flux_padded = torch.nn.functional.pad(token_flux_pre, (0, 0, 0, pad_len)) if pad_len > 0 else token_flux_pre
    token_flux_chunked = token_flux_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, 1)
    
    k_beta_padded = torch.nn.functional.pad(k_beta_cython, (0, 0, 0, pad_len)) if pad_len > 0 else k_beta_cython
    k_beta_chunked = k_beta_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, layer.head_k_dim)
    
    # Compute attn_const in Cython style
    actual_chunk_size = q_chunked.shape[3]
    mask_tri_upper_diag = torch.triu(
        torch.ones(actual_chunk_size, actual_chunk_size, dtype=torch.bool, device=q_chunked.device), diagonal=0
    )
    attn_const_cython = -(k_beta_chunked @ k_chunked.transpose(-1, -2))
    attn_const_cython = attn_const_cython.masked_fill(mask_tri_upper_diag, 0)
    
    mask = torch.tril(torch.ones(actual_chunk_size, actual_chunk_size, device=q_chunked.device), diagonal=-1)
    updates_cython = torch.einsum('...ik,...jk->...ij', attn_const_cython, attn_const_cython) * mask.unsqueeze(0).unsqueeze(0)
    eye_cython = torch.eye(actual_chunk_size, dtype=attn_const_cython.dtype, device=q_chunked.device)
    attn_const_cython = attn_const_cython + updates_cython + eye_cython
    
    u_cython = attn_const_cython @ v_chunked
    w_cython = attn_const_cython @ k_beta_chunked
    
    # Compute attn_all in Cython style
    mask_tri_upper = torch.triu(
        torch.ones(actual_chunk_size, actual_chunk_size, dtype=torch.bool, device=q_chunked.device), diagonal=1
    )
    attn_all_cython = q_chunked @ k_chunked.transpose(-1, -2)
    attn_all_cython = attn_all_cython.masked_fill(mask_tri_upper, 0)
    
    # Resonance flux - need to process per chunk
    resonance_flux = layer.resonance_flux
    psi_all_cython_list = []
    for chunk_idx in range(n_chunks):
        k_chunk_i = k_chunked[:, :, chunk_idx]  # [b, h, c, d_k]
        u_chunk_i = u_cython[:, :, chunk_idx]  # [b, h, c, d_v]
        psi_i = resonance_flux(k_chunk_i, u_chunk_i)  # [b, h]
        psi_all_cython_list.append(psi_i)
    psi_all_cython = torch.stack(psi_all_cython_list, dim=2)  # [b, h, n]
    
    print("Comparing pre-computed values:")
    print()
    
    # Convert to JAX for comparison
    q_jax = jnp.array(q_chunked.detach().cpu().numpy())
    k_jax = jnp.array(k_chunked.detach().cpu().numpy())
    v_jax = jnp.array(v_chunked.detach().cpu().numpy())
    k_beta_jax = jnp.array(k_beta_chunked.detach().cpu().numpy())
    
    # Compute attn_const in JAX
    k_T_jax = jnp.transpose(k_jax, (0, 1, 2, 4, 3))
    attn_const_jax = -jnp.matmul(k_beta_jax, k_T_jax)
    mask_tri_upper_diag_jax = jnp.triu(jnp.ones((actual_chunk_size, actual_chunk_size)), k=0)
    attn_const_jax = jnp.where(mask_tri_upper_diag_jax[None, None, None, :, :] > 0, 0.0, attn_const_jax)
    mask_jax = jnp.tril(jnp.ones((actual_chunk_size, actual_chunk_size)), k=-1)
    updates_jax = jnp.einsum('bhnik,bhnjk->bhnij', attn_const_jax, attn_const_jax) * mask_jax[None, None, None, :, :]
    eye_jax = jnp.eye(actual_chunk_size)[None, None, None, :, :]
    attn_const_jax = attn_const_jax + updates_jax + eye_jax
    
    u_jax = jnp.matmul(attn_const_jax, v_jax)
    w_jax = jnp.matmul(attn_const_jax, k_beta_jax)
    
    # Compute attn_all in JAX
    k_T_attn_jax = jnp.transpose(k_jax, (0, 1, 2, 4, 3))
    attn_all_jax = jnp.matmul(q_jax, k_T_attn_jax)
    mask_tri_upper_jax = jnp.triu(jnp.ones((actual_chunk_size, actual_chunk_size)), k=1)
    attn_all_jax = jnp.where(mask_tri_upper_jax[None, None, None, :, :] > 0, 0.0, attn_all_jax)
    
    compare_intermediate("attn_const", attn_const_cython, attn_const_jax, threshold=1e-4)
    compare_intermediate("u", u_cython, u_jax, threshold=1e-4)
    compare_intermediate("w", w_cython, w_jax, threshold=1e-4)
    compare_intermediate("attn_all", attn_all_cython, attn_all_jax, threshold=1e-4)
    
    # Now manually trace through first chunk iteration
    print("="*80)
    print("TRACING FIRST CHUNK ITERATION (i=0)")
    print("="*80)
    print()
    
    i = 0
    S_fast_cython = torch.zeros(1, layer.num_heads, layer.head_k_dim, layer.head_v_dim, device=q_chunked.device)
    S_slow_cython = torch.zeros(1, layer.num_heads, layer.head_k_dim, layer.head_v_dim, device=q_chunked.device)
    
    q_i_cython = q_chunked[:, :, i]
    k_i_cython = k_chunked[:, :, i]
    u_i_cython = u_cython[:, :, i]
    w_i_cython = w_cython[:, :, i]
    attn_i_cython = attn_all_cython[:, :, i]
    fast_decay_i_cython = fast_decay_chunked[:, :, i]
    slow_decay_i_cython = slow_decay_chunked[:, :, i]
    fast_gate_i_cython = fast_gate_chunked[:, :, i]
    slow_gate_i_cython = slow_gate_chunked[:, :, i]
    psi_i_cython = psi_all_cython[:, :, i]
    token_flux_i_cython = token_flux_chunked[:, :, i]
    
    # Flux-modulated decay
    fast_decay_factor_cython = fast_decay_i_cython.mean(-1, keepdim=True).unsqueeze(-1)
    slow_decay_factor_cython = slow_decay_i_cython.mean(-1, keepdim=True).unsqueeze(-1)
    psi_expanded_cython = psi_i_cython.unsqueeze(-1).unsqueeze(-1)
    
    fast_decay_modulated_cython = fast_decay_factor_cython * (1 - 0.1 * psi_expanded_cython)
    slow_decay_modulated_cython = slow_decay_factor_cython * (1 - 0.05 * psi_expanded_cython)
    
    S_fast_cython = S_fast_cython * fast_decay_modulated_cython
    S_slow_cython = S_slow_cython * slow_decay_modulated_cython
    
    S_fast_norm_cython = S_fast_cython.norm(dim=(-2, -1), keepdim=True) + 1e-8
    S_slow_norm_cython = S_slow_cython.norm(dim=(-2, -1), keepdim=True) + 1e-8
    S_fast_read_cython = S_fast_cython / S_fast_norm_cython
    S_slow_read_cython = S_slow_cython / S_slow_norm_cython
    
    # Hierarchical Delta rule
    w_S_fast_cython = w_i_cython @ S_fast_read_cython
    u_i_fast_cython = u_i_cython - w_S_fast_cython
    o_inter_fast_cython = q_i_cython @ S_fast_read_cython
    attn_u_fast_cython = attn_i_cython @ u_i_fast_cython
    o_fast_cython = fast_gate_i_cython * (o_inter_fast_cython + attn_u_fast_cython)
    
    w_S_slow_cython = w_i_cython @ S_slow_read_cython
    u_i_slow_cython = u_i_cython - w_S_slow_cython
    o_inter_slow_cython = q_i_cython @ S_slow_read_cython
    attn_u_slow_cython = attn_i_cython @ u_i_slow_cython
    o_slow_cython = slow_gate_i_cython * (o_inter_slow_cython + attn_u_slow_cython)
    
    # Token-level flux blending
    alpha_cython = 0.5 + 0.3 * token_flux_i_cython
    beta_weight_cython = 1.0 - alpha_cython
    o_chunk_cython = alpha_cython * o_fast_cython + beta_weight_cython * o_slow_cython
    
    print("Cython intermediate values for chunk 0:")
    print(f"  S_fast after decay: shape={S_fast_cython.shape}, norm={S_fast_cython.norm().item():.10f}")
    print(f"  S_fast_read: shape={S_fast_read_cython.shape}, norm={S_fast_read_cython.norm().item():.10f}")
    print(f"  o_fast: shape={o_fast_cython.shape}, first 3 values={o_fast_cython[0,0,0,:3]}")
    print(f"  o_slow: shape={o_slow_cython.shape}, first 3 values={o_slow_cython[0,0,0,:3]}")
    print(f"  o_chunk (before scaling): shape={o_chunk_cython.shape}, first 3 values={o_chunk_cython[0,0,0,:3]}")
    print()
    
    # Now compute same in JAX
    S_fast_jax = jnp.zeros((1, layer.num_heads, layer.head_k_dim, layer.head_v_dim))
    S_slow_jax = jnp.zeros((1, layer.num_heads, layer.head_k_dim, layer.head_v_dim))
    
    q_i_jax = q_jax[:, :, i]  # Should be [b, h, c, d_k]
    k_i_jax = k_jax[:, :, i]
    u_i_jax = u_jax[:, :, i]  # Should be [b, h, c, d_v]
    w_i_jax = w_jax[:, :, i]  # Should be [b, h, c, d_k]
    attn_i_jax = attn_all_jax[:, :, i]  # Should be [b, h, c, c]
    fast_decay_i_jax = jnp.array(fast_decay_chunked[:, :, i].detach().cpu().numpy())  # Should be [b, h, c]
    slow_decay_i_jax = jnp.array(slow_decay_chunked[:, :, i].detach().cpu().numpy())
    fast_gate_i_jax = jnp.array(fast_gate_chunked[:, :, i].detach().cpu().numpy())  # Should be [b, h, c, 1]
    slow_gate_i_jax = jnp.array(slow_gate_chunked[:, :, i].detach().cpu().numpy())
    psi_i_jax = jnp.array(psi_all_cython[:, :, i].detach().cpu().numpy())  # Should be [b, h]
    token_flux_i_jax = jnp.array(token_flux_chunked[:, :, i].detach().cpu().numpy())  # Should be [b, h, c, 1]
    
    print(f"Input shapes for chunk {i}:")
    print(f"  q_i_jax: {q_i_jax.shape}, k_i_jax: {k_i_jax.shape}")
    print(f"  u_i_jax: {u_i_jax.shape}, w_i_jax: {w_i_jax.shape}")
    print(f"  attn_i_jax: {attn_i_jax.shape}")
    print(f"  fast_decay_i_jax: {fast_decay_i_jax.shape}")
    print(f"  fast_gate_i_jax: {fast_gate_i_jax.shape}")
    print(f"  psi_i_jax: {psi_i_jax.shape}")
    print(f"  token_flux_i_jax: {token_flux_i_jax.shape}")
    print()
    
    # Flux-modulated decay
    # fast_decay_i_jax: [b, h, c], mean over c -> [b, h], then expand to [b, h, 1, 1]
    fast_decay_factor_jax = jnp.mean(fast_decay_i_jax, axis=-1)  # [b, h]
    fast_decay_factor_jax = fast_decay_factor_jax[:, :, None, None]  # [b, h, 1, 1]
    slow_decay_factor_jax = jnp.mean(slow_decay_i_jax, axis=-1)  # [b, h]
    slow_decay_factor_jax = slow_decay_factor_jax[:, :, None, None]  # [b, h, 1, 1]
    psi_expanded_jax = psi_i_jax[:, :, None, None]  # [b, h, 1, 1]
    
    print(f"Decay factor shapes:")
    print(f"  fast_decay_factor_jax: {fast_decay_factor_jax.shape}")
    print(f"  psi_expanded_jax: {psi_expanded_jax.shape}")
    print(f"  S_fast_jax before: {S_fast_jax.shape}")
    
    fast_decay_modulated_jax = fast_decay_factor_jax * (1 - 0.1 * psi_expanded_jax)
    slow_decay_modulated_jax = slow_decay_factor_jax * (1 - 0.05 * psi_expanded_jax)
    
    print(f"  fast_decay_modulated_jax: {fast_decay_modulated_jax.shape}")
    
    S_fast_jax = S_fast_jax * fast_decay_modulated_jax
    S_slow_jax = S_slow_jax * slow_decay_modulated_jax
    
    print(f"  S_fast_jax after: {S_fast_jax.shape}")
    print()
    
    S_fast_norm_jax = jnp.linalg.norm(S_fast_jax, axis=(-2, -1), keepdims=True) + 1e-8
    S_slow_norm_jax = jnp.linalg.norm(S_slow_jax, axis=(-2, -1), keepdims=True) + 1e-8
    S_fast_read_jax = S_fast_jax / S_fast_norm_jax
    S_slow_read_jax = S_slow_jax / S_slow_norm_jax
    
    # Hierarchical Delta rule
    # w_i_jax: [b, h, c, d_k], S_fast_read_jax: [b, h, d_k, d_v]
    w_S_fast_jax = jnp.matmul(w_i_jax, S_fast_read_jax)  # [b, h, c, d_v]
    u_i_fast_jax = u_i_jax - w_S_fast_jax  # [b, h, c, d_v]
    o_inter_fast_jax = jnp.matmul(q_i_jax, S_fast_read_jax)  # [b, h, c, d_v]
    attn_u_fast_jax = jnp.matmul(attn_i_jax, u_i_fast_jax)  # [b, h, c, d_v]
    o_fast_jax = fast_gate_i_jax * (o_inter_fast_jax + attn_u_fast_jax)  # [b, h, c, d_v]
    
    w_S_slow_jax = jnp.matmul(w_i_jax, S_slow_read_jax)  # [b, h, c, d_v]
    u_i_slow_jax = u_i_jax - w_S_slow_jax  # [b, h, c, d_v]
    o_inter_slow_jax = jnp.matmul(q_i_jax, S_slow_read_jax)  # [b, h, c, d_v]
    attn_u_slow_jax = jnp.matmul(attn_i_jax, u_i_slow_jax)  # [b, h, c, d_v]
    o_slow_jax = slow_gate_i_jax * (o_inter_slow_jax + attn_u_slow_jax)  # [b, h, c, d_v]
    
    # Token-level flux blending
    alpha_jax = 0.5 + 0.3 * token_flux_i_jax  # [b, h, c, 1]
    beta_weight_jax = 1.0 - alpha_jax
    o_chunk_jax = alpha_jax * o_fast_jax + beta_weight_jax * o_slow_jax  # [b, h, c, d_v]
    
    print("JAX intermediate values for chunk 0:")
    print(f"  S_fast after decay: shape={S_fast_jax.shape}, norm={float(jnp.linalg.norm(S_fast_jax)):.10f}")
    print(f"  S_fast_read: shape={S_fast_read_jax.shape}, norm={float(jnp.linalg.norm(S_fast_read_jax)):.10f}")
    print(f"  o_fast: shape={o_fast_jax.shape}, first 3 values={np.array(o_fast_jax[0,0,0,:3])}")
    print(f"  o_slow: shape={o_slow_jax.shape}, first 3 values={np.array(o_slow_jax[0,0,0,:3])}")
    print(f"  o_chunk (before scaling): shape={o_chunk_jax.shape}, first 3 values={np.array(o_chunk_jax[0,0,0,:3])}")
    print()
    
    # Compare step by step
    print("Step-by-step comparison:")
    print(f"Shapes - S_fast_cython: {S_fast_cython.shape}, S_fast_jax: {S_fast_jax.shape}")
    print(f"Shapes - S_fast_read_cython: {S_fast_read_cython.shape}, S_fast_read_jax: {S_fast_read_jax.shape}")
    print(f"Shapes - o_fast_cython: {o_fast_cython.shape}, o_fast_jax: {o_fast_jax.shape}")
    print(f"Shapes - o_slow_cython: {o_slow_cython.shape}, o_slow_jax: {o_slow_jax.shape}")
    print(f"Shapes - o_chunk_cython: {o_chunk_cython.shape}, o_chunk_jax: {o_chunk_jax.shape}")
    print()
    
    # Only compare if shapes match
    if S_fast_cython.shape == S_fast_jax.shape:
        compare_intermediate("S_fast after decay", S_fast_cython, S_fast_jax, threshold=1e-4)
    else:
        print(f"❌ Shape mismatch: S_fast_cython {S_fast_cython.shape} vs S_fast_jax {S_fast_jax.shape}")
    if S_fast_read_cython.shape == S_fast_read_jax.shape:
        compare_intermediate("S_fast_read", S_fast_read_cython, S_fast_read_jax, threshold=1e-4)
    else:
        print(f"❌ Shape mismatch: S_fast_read_cython {S_fast_read_cython.shape} vs S_fast_read_jax {S_fast_read_jax.shape}")
    if o_fast_cython.shape == o_fast_jax.shape:
        compare_intermediate("o_fast", o_fast_cython, o_fast_jax, threshold=1e-3)
    else:
        print(f"❌ Shape mismatch: o_fast_cython {o_fast_cython.shape} vs o_fast_jax {o_fast_jax.shape}")
    if o_slow_cython.shape == o_slow_jax.shape:
        compare_intermediate("o_slow", o_slow_cython, o_slow_jax, threshold=1e-3)
    else:
        print(f"❌ Shape mismatch: o_slow_cython {o_slow_cython.shape} vs o_slow_jax {o_slow_jax.shape}")
    if o_chunk_cython.shape == o_chunk_jax.shape:
        compare_intermediate("o_chunk (before scaling)", o_chunk_cython, o_chunk_jax, threshold=1e-3)
    else:
        print(f"❌ Shape mismatch: o_chunk_cython {o_chunk_cython.shape} vs o_chunk_jax {o_chunk_jax.shape}")
    compare_intermediate("S_fast_read", S_fast_read_cython, S_fast_read_jax, threshold=1e-4)
    compare_intermediate("o_fast", o_fast_cython, o_fast_jax, threshold=1e-3)
    compare_intermediate("o_slow", o_slow_cython, o_slow_jax, threshold=1e-3)
    compare_intermediate("o_chunk (before scaling)", o_chunk_cython, o_chunk_jax, threshold=1e-3)
    
    # Check if scaling factor fixes it
    scale_factor = 2.856
    o_chunk_jax_scaled = o_chunk_jax / scale_factor
    compare_intermediate(f"o_chunk (after scaling by {scale_factor})", o_chunk_cython, o_chunk_jax_scaled, threshold=1e-4)

