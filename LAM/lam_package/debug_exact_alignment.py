#!/usr/bin/env python3
"""
Debug script to find exact divergence point between JAX and Cython
to achieve 1.0 alignment.
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

def compare_values(name, cython_val, jax_val, threshold=1e-6):
    """Compare two values and report if they match."""
    if isinstance(cython_val, torch.Tensor):
        cython_val = cython_val.detach().cpu().numpy()
    if isinstance(jax_val, jnp.ndarray):
        jax_val = np.array(jax_val)
    
    diff = np.abs(cython_val - jax_val)
    max_diff = np.max(diff) if diff.size > 0 else 0
    mean_diff = np.mean(diff) if diff.size > 0 else 0
    
    # Cosine similarity
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
    if not match and cython_val.size <= 20:
        print(f"    Cython: {cython_flat}")
        print(f"    JAX:    {jax_flat}")
    elif not match:
        print(f"    First 5 - Cython: {cython_flat[:5]}")
        print(f"    First 5 - JAX:    {jax_flat[:5]}")
    print()
    
    return match, max_diff, cos_sim

print("="*80)
print("EXACT ALIGNMENT DEBUG - Finding Root Cause")
print("="*80)

model_path = "../LAM-base-v1"
model = LAM(model_path, backend='cython')
test_sentence = "Hello"
input_ids = model.tokenizer.encode(test_sentence)
input_ids_tensor = torch.tensor([input_ids.ids], dtype=torch.long, device=model.device)
attention_mask = torch.tensor([input_ids.attention_mask], dtype=torch.long, device=model.device)

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
    
    # Convert to JAX for comparison
    q_jax = jnp.array(q_cython.detach().cpu().numpy())
    k_jax = jnp.array(k_cython.detach().cpu().numpy())
    
    # Compare q and k after normalization
    compare_values("q after L2 norm", q_cython, q_jax, threshold=1e-6)
    compare_values("k after L2 norm", k_cython, k_jax, threshold=1e-6)
    
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
    
    # Call Cython hierarchical delta rule
    from lam._core import enhanced_hierarchical_delta_rule
    resonance_flux = layer.resonance_flux
    
    o_cython, _ = enhanced_hierarchical_delta_rule(
        q=q_cython, k=k_cython, v=v_scaled_cython, beta=beta,
        fast_decay=fast_decay, slow_decay=slow_decay,
        fast_gate=fast_gate, slow_gate=slow_gate,
        resonance_flux=resonance_flux,
        training=False
    )
    
    print("Cython output computed")
    print(f"  Shape: {o_cython.shape}")
    print(f"  First 5 values: {o_cython[0,0,0,:5]}")
    print()
    
    # Now prepare for JAX - need to chunk
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
    
    # Compute token flux before chunking
    token_flux_pre = resonance_flux.compute_token_flux(k_beta_cython, v_scaled_cython)
    token_flux_padded = torch.nn.functional.pad(token_flux_pre, (0, 0, 0, pad_len)) if pad_len > 0 else token_flux_pre
    token_flux_chunked = token_flux_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, 1)
    
    # Compute k_beta before chunking
    k_beta_padded = torch.nn.functional.pad(k_beta_cython, (0, 0, 0, pad_len)) if pad_len > 0 else k_beta_cython
    k_beta_chunked = k_beta_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, layer.head_k_dim)
    
    # JAX weights
    resonance_flux_w = layer.resonance_flux
    W_bilinear = jnp.array(resonance_flux_w.W_bilinear.detach().cpu().numpy())
    temp = jnp.array(resonance_flux_w.temp.detach().cpu().numpy())
    flux_net_w0 = jnp.array(resonance_flux_w.flux_net[0].weight.detach().cpu().numpy())
    flux_net_b0 = jnp.array(resonance_flux_w.flux_net[0].bias.detach().cpu().numpy())
    flux_net_w2 = jnp.array(resonance_flux_w.flux_net[2].weight.detach().cpu().numpy())
    flux_net_b2 = jnp.array(resonance_flux_w.flux_net[2].bias.detach().cpu().numpy())
    token_flux_w0 = jnp.array(resonance_flux_w.token_flux_proj[0].weight.detach().cpu().numpy())
    token_flux_b0 = jnp.array(resonance_flux_w.token_flux_proj[0].bias.detach().cpu().numpy())
    token_flux_w2 = jnp.array(resonance_flux_w.token_flux_proj[2].weight.detach().cpu().numpy())
    token_flux_b2 = jnp.array(resonance_flux_w.token_flux_proj[2].bias.detach().cpu().numpy())
    
    # Convert to JAX
    q_jax = jnp.array(q_chunked.detach().cpu().numpy())
    k_jax = jnp.array(k_chunked.detach().cpu().numpy())
    v_jax = jnp.array(v_chunked.detach().cpu().numpy())
    beta_jax = jnp.array(beta_chunked.detach().cpu().numpy())
    fast_decay_jax = jnp.array(fast_decay_chunked.detach().cpu().numpy())
    slow_decay_jax = jnp.array(slow_decay_chunked.detach().cpu().numpy())
    fast_gate_jax = jnp.array(fast_gate_chunked.detach().cpu().numpy())
    slow_gate_jax = jnp.array(slow_gate_chunked.detach().cpu().numpy())
    token_flux_jax = jnp.array(token_flux_chunked.detach().cpu().numpy())
    k_beta_jax = jnp.array(k_beta_chunked.detach().cpu().numpy())
    
    # Compare inputs
    print("Comparing inputs to hierarchical_delta_rule_jax:")
    compare_values("q (chunked)", q_chunked, q_jax, threshold=1e-6)
    compare_values("k (chunked)", k_chunked, k_jax, threshold=1e-6)
    compare_values("v (chunked)", v_chunked, v_jax, threshold=1e-6)
    compare_values("beta (chunked)", beta_chunked, beta_jax, threshold=1e-6)
    compare_values("token_flux (chunked)", token_flux_chunked, token_flux_jax, threshold=1e-6)
    compare_values("k_beta (chunked)", k_beta_chunked, k_beta_jax, threshold=1e-6)
    
    # Call JAX
    o_jax = hierarchical_delta_rule_jax(
        q_jax, k_jax, v_jax,
        beta_jax,
        fast_decay_jax, slow_decay_jax,
        fast_gate_jax, slow_gate_jax,
        resonance_flux_fn=None,
        resonance_flux_W_bilinear=W_bilinear,
        resonance_flux_temp=temp,
        resonance_flux_net_w0=flux_net_w0,
        resonance_flux_net_b0=flux_net_b0,
        resonance_flux_net_w2=flux_net_w2,
        resonance_flux_net_b2=flux_net_b2,
        token_flux_proj_w0=token_flux_w0,
        token_flux_proj_b0=token_flux_b0,
        token_flux_proj_w2=token_flux_w2,
        token_flux_proj_b2=token_flux_b2,
        token_flux_precomputed=token_flux_jax,
        k_beta_precomputed=k_beta_jax,
    )
    
    # Reshape back and remove padding
    o_jax_reshaped = o_jax.reshape(1, layer.num_heads, n_chunks * actual_chunk_size, layer.head_v_dim)
    o_jax_final = o_jax_reshaped[:, :, :l, :]  # Remove padding
    
    # Reshape Cython output to match
    o_cython_reshaped = rearrange(o_cython, "b h l d -> b h l d")
    
    print("Final output comparison:")
    match, max_diff, cos_sim = compare_values(
        "Hierarchical Delta Rule Output", 
        o_cython_reshaped, 
        o_jax_final, 
        threshold=1e-5
    )
    
    print("="*80)
    if match and cos_sim > 0.999999:
        print("✅ PERFECT ALIGNMENT ACHIEVED!")
    else:
        print(f"❌ Not aligned: max_diff={max_diff:.10e}, cos_sim={cos_sim:.10f}")
        print("Need to investigate further to find root cause.")
    print("="*80)




