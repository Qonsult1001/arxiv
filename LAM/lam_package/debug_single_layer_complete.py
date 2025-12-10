#!/usr/bin/env python3
"""
Debug a single complete layer (DeltaNet + FFN + LayerNorms) to find the issue.
"""

import torch
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'build'))

from lam import LAM
from einops import rearrange

def compare(name, cython_val, jax_val):
    """Compare values."""
    if isinstance(cython_val, torch.Tensor):
        cython_np = cython_val.detach().cpu().numpy()
    else:
        cython_np = cython_val
    
    if isinstance(jax_val, jnp.ndarray):
        jax_np = np.array(jax_val)
    else:
        jax_np = jax_val
    
    if cython_np.shape != jax_np.shape:
        print(f"⚠️  {name}: Shape mismatch - Cython {cython_np.shape} vs JAX {jax_np.shape}")
        return
    
    diff = np.abs(cython_np - jax_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    cython_flat = cython_np.flatten()
    jax_flat = jax_np.flatten()
    cos_sim = np.dot(cython_flat, jax_flat) / (np.linalg.norm(cython_flat) * np.linalg.norm(jax_flat) + 1e-9)
    
    status = "✅" if cos_sim > 0.99 and max_diff < 1e-2 else "❌"
    print(f"{status} {name}:")
    print(f"    Max diff: {max_diff:.10e}")
    print(f"    Mean diff: {mean_diff:.10e}")
    print(f"    Cosine sim: {cos_sim:.10f}")
    if cos_sim < 0.99:
        print(f"    First 3 - Cython: {cython_flat[:3]}")
        print(f"    First 3 - JAX:    {jax_flat[:3]}")
    print()

print("="*80)
print("SINGLE COMPLETE LAYER DEBUG")
print("="*80)

model_path = "../LAM-base-v1"
test_sentence = "Hello"

# Create models
model_cython = LAM(model_path, backend='cython')
model_jax = LAM(model_path, backend='jax')

# Get inputs
input_ids = model_cython.tokenizer.encode(test_sentence)
input_ids_tensor = torch.tensor([input_ids.ids], dtype=torch.long, device=model_cython.device)
attention_mask = torch.tensor([input_ids.attention_mask], dtype=torch.long, device=model_cython.device)

# Get embeddings (same for both)
model_cython._model.eval()
with torch.no_grad():
    x_cython = model_cython._model.embeddings['word_embeddings'](input_ids_tensor)
    token_type_ids = torch.zeros_like(input_ids_tensor)
    x_cython = x_cython + model_cython._model.embeddings['token_type_embeddings'](token_type_ids)
    position_ids = torch.arange(input_ids_tensor.shape[1], dtype=torch.long, device=input_ids_tensor.device).unsqueeze(0)
    x_cython = x_cython + model_cython._model.embeddings['position_embeddings'](position_ids)
    x_cython = model_cython._model.embeddings['LayerNorm'](x_cython)
    x_cython = model_cython._model.embeddings['dropout'](x_cython)

# JAX embeddings
from lam._jax_model_optimized import jax_forward_pass_optimized
input_ids_jax = jnp.array(input_ids_tensor.detach().cpu().numpy())
attention_mask_jax = jnp.array(attention_mask.detach().cpu().numpy())
params = model_jax._jax_params
config = model_jax._jax_config_values

word_emb = params['embeddings']['word']
x_jax = word_emb[input_ids_jax]
token_type_emb = params['embeddings']['token_type']
token_type_ids_jax = jnp.zeros_like(input_ids_jax)
x_jax = x_jax + token_type_emb[token_type_ids_jax]
position_ids_jax = jnp.arange(input_ids_jax.shape[1])[None, :]
position_emb = model_jax._jax_position_emb_weight
x_jax = x_jax + position_emb[position_ids_jax]
emb_ln_w = params['embeddings']['ln_weight']
emb_ln_b = params['embeddings']['ln_bias']
mean = jnp.mean(x_jax, axis=-1, keepdims=True)
variance = jnp.var(x_jax, axis=-1, keepdims=True)
x_jax = (x_jax - mean) / jnp.sqrt(variance + config['layer_norm_eps'])
x_jax = x_jax * emb_ln_w[None, None, :] + emb_ln_b[None, None, :]
# Note: JAX doesn't apply dropout in eval mode

compare("Embeddings (after dropout)", x_cython, x_jax)

# Process Layer 0 completely
layer_idx = 0
print(f"\n{'='*80}")
print(f"PROCESSING LAYER {layer_idx} COMPLETELY")
print(f"{'='*80}\n")

# ============================================================================
# CYTHON: Complete Layer 0
# ============================================================================
print("--- CYTHON Layer 0 ---")
with torch.no_grad():
    # DeltaNet
    residual_attn_cython = x_cython
    x_attn_cython, _, _, _ = model_cython._model.deltanet_layers[layer_idx](x_cython, attention_mask)
    x_attn_cython = model_cython._model.deltanet_norms[layer_idx](residual_attn_cython + x_attn_cython)
    
    # FFN
    residual_ffn_cython = x_attn_cython
    x_ffn_cython = model_cython._model.deltanet_ffns[layer_idx]['dense'](x_attn_cython)
    x_ffn_cython = model_cython._model.deltanet_ffns[layer_idx].intermediate_act_fn(x_ffn_cython)  # GELU
    x_ffn_cython = torch.nn.functional.gelu(x_ffn_cython)  # Another GELU
    x_ffn_cython = model_cython._model.output_denses[layer_idx]['dense'](x_ffn_cython)
    x_ffn_cython = model_cython._model.output_denses[layer_idx]['dropout'](x_ffn_cython)  # No-op in eval
    x_cython_layer0 = model_cython._model.output_denses[layer_idx]['LayerNorm'](residual_ffn_cython + x_ffn_cython)

print(f"  After DeltaNet: shape={x_attn_cython.shape}, first 3={x_attn_cython[0,0,:3]}")
print(f"  After FFN: shape={x_cython_layer0.shape}, first 3={x_cython_layer0[0,0,:3]}")

# ============================================================================
# JAX: Complete Layer 0
# ============================================================================
print("\n--- JAX Layer 0 ---")
from lam._jax_model_optimized import jax_deltanet_layer_optimized

# DeltaNet
residual_attn_jax = x_jax
x_attn_jax = jax_deltanet_layer_optimized(
    params['deltanet']['q_proj'][layer_idx],
    params['deltanet']['k_proj'][layer_idx],
    params['deltanet']['v_proj'][layer_idx],
    params['deltanet']['b_proj'][layer_idx],
    params['deltanet']['fast_decay_proj'][layer_idx],
    params['deltanet']['fast_decay_bias'][layer_idx],
    params['deltanet']['slow_decay_proj'][layer_idx],
    params['deltanet']['slow_decay_bias'][layer_idx],
    params['deltanet']['fast_gate_proj'][layer_idx],
    params['deltanet']['slow_gate_proj'][layer_idx],
    params['deltanet']['g_proj'][layer_idx],
    params['deltanet']['o_proj'][layer_idx],
    params['deltanet']['o_norm_weight'][layer_idx],
    params['deltanet']['o_norm_gate'][layer_idx],
    params['deltanet']['q_conv_weight'][layer_idx],
    params['deltanet']['q_conv_bias'][layer_idx],
    params['deltanet']['k_conv_weight'][layer_idx],
    params['deltanet']['k_conv_bias'][layer_idx],
    params['deltanet']['v_conv_weight'][layer_idx],
    params['deltanet']['v_conv_bias'][layer_idx],
    params['deltanet']['resonance_flux_W_bilinear'][layer_idx],
    params['deltanet']['resonance_flux_temp'][layer_idx],
    params['deltanet']['resonance_flux_net_w0'][layer_idx],
    params['deltanet']['resonance_flux_net_b0'][layer_idx],
    params['deltanet']['resonance_flux_net_w2'][layer_idx],
    params['deltanet']['resonance_flux_net_b2'][layer_idx],
    params['deltanet']['token_flux_proj_w0'][layer_idx],
    params['deltanet']['token_flux_proj_b0'][layer_idx],
    params['deltanet']['token_flux_proj_w2'][layer_idx],
    params['deltanet']['token_flux_proj_b2'][layer_idx],
    x_jax,
    config['num_heads'],
    config['head_k_dim'],
    config['head_v_dim'],
    chunk_size=64
)

# LayerNorm after DeltaNet
mean = jnp.mean(x_attn_jax, axis=-1, keepdims=True)
variance = jnp.var(x_attn_jax, axis=-1, keepdims=True, ddof=0)
x_attn_jax = (x_attn_jax - mean) / jnp.sqrt(variance + config['layer_norm_eps'])
x_attn_jax = x_attn_jax * params['norms']['attn']['weight'][layer_idx][None, None, :] + params['norms']['attn']['bias'][layer_idx][None, None, :]
x_attn_jax = residual_attn_jax + x_attn_jax

# FFN
residual_ffn_jax = x_attn_jax
x_ffn_jax = jnp.dot(x_attn_jax, params['ffn']['intermediate']['weight'][layer_idx].T) + params['ffn']['intermediate']['bias'][layer_idx][None, :]
x_ffn_jax = jax.nn.gelu(x_ffn_jax)  # First GELU (intermediate_act_fn)
x_ffn_jax = jax.nn.gelu(x_ffn_jax)  # Second GELU (redundant but matches Cython)
x_ffn_jax = jnp.dot(x_ffn_jax, params['ffn']['output']['weight'][layer_idx].T) + params['ffn']['output']['bias'][layer_idx][None, :]
# Skip dropout (eval mode)

# LayerNorm after FFN
x_ffn_sum_jax = residual_ffn_jax + x_ffn_jax
mean = jnp.mean(x_ffn_sum_jax, axis=-1, keepdims=True)
variance = jnp.var(x_ffn_sum_jax, axis=-1, keepdims=True, ddof=0)
x_ffn_sum_jax = (x_ffn_sum_jax - mean) / jnp.sqrt(variance + config['layer_norm_eps'])
x_jax_layer0 = x_ffn_sum_jax * params['norms']['ffn']['weight'][layer_idx][None, None, :] + params['norms']['ffn']['bias'][layer_idx][None, None, :]

print(f"  After DeltaNet: shape={x_attn_jax.shape}, first 3={np.array(x_attn_jax[0,0,:3])}")
print(f"  After FFN: shape={x_jax_layer0.shape}, first 3={np.array(x_jax_layer0[0,0,:3])}")

# Compare
print("\n--- COMPARISON ---")
compare("After DeltaNet + LayerNorm", x_attn_cython, x_attn_jax)
compare("After Complete Layer 0 (DeltaNet + FFN)", x_cython_layer0, x_jax_layer0)

print("="*80)
print("Debug complete!")
print("="*80)

