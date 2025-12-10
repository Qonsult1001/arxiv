#!/usr/bin/env python3
"""
Trace through each layer to find where divergence occurs.
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

def compare(name, cython_val, jax_val, threshold=1e-2):
    """Compare values and return statistics."""
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
        return None, None, None
    
    diff = np.abs(cython_np - jax_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    cython_flat = cython_np.flatten()
    jax_flat = jax_np.flatten()
    cos_sim = np.dot(cython_flat, jax_flat) / (np.linalg.norm(cython_flat) * np.linalg.norm(jax_flat) + 1e-9)
    
    status = "✅" if cos_sim > 0.99 and max_diff < threshold else "❌"
    print(f"{status} {name}:")
    print(f"    Max diff: {max_diff:.10e}")
    print(f"    Mean diff: {mean_diff:.10e}")
    print(f"    Cosine sim: {cos_sim:.10f}")
    if cos_sim < 0.99:
        print(f"    First 3 - Cython: {cython_flat[:3]}")
        print(f"    First 3 - JAX:    {jax_flat[:3]}")
    print()
    
    return cos_sim, max_diff, mean_diff

print("="*80)
print("LAYER-BY-LAYER TRACE TO FIND DIVERGENCE")
print("="*80)

model_path = "../LAM-base-v1"
test_sentence = "Hello"

# Create models
print("Loading models...")
model_cython = LAM(model_path, backend='cython')
model_jax = LAM(model_path, backend='jax')
print("✅ Models loaded\n")

# Get inputs
input_ids = model_cython.tokenizer.encode(test_sentence)
input_ids_tensor = torch.tensor([input_ids.ids], dtype=torch.long, device=model_cython.device)
attention_mask = torch.tensor([input_ids.attention_mask], dtype=torch.long, device=model_cython.device)

print(f"Input: '{test_sentence}'")
print(f"Input IDs: {input_ids.ids}")
print(f"Sequence length: {len(input_ids.ids)}\n")

# ============================================================================
# EMBEDDINGS
# ============================================================================
print("="*80)
print("1. EMBEDDINGS")
print("="*80)

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

emb_cos, emb_max, emb_mean = compare("Embeddings", x_cython, x_jax)

# ============================================================================
# LAYER-BY-LAYER PROCESSING
# ============================================================================
print("="*80)
print("2. LAYER-BY-LAYER PROCESSING")
print("="*80)

from lam._jax_model_optimized import jax_deltanet_layer_optimized

x_cython_layer = x_cython
x_jax_layer = x_jax

layer_cos_sims = []
layer_max_diffs = []

for layer_idx in range(config['num_layers']):
    print(f"\n{'='*80}")
    print(f"LAYER {layer_idx}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # CYTHON: Process layer
    # ========================================================================
    with torch.no_grad():
        # DeltaNet
        residual_attn_cython = x_cython_layer
        x_attn_cython, _, _, _ = model_cython._model.deltanet_layers[layer_idx](x_cython_layer, attention_mask)
        x_attn_cython = model_cython._model.deltanet_norms[layer_idx](residual_attn_cython + x_attn_cython)
        
        # FFN
        residual_ffn_cython = x_attn_cython
        x_ffn_cython = model_cython._model.deltanet_ffns[layer_idx]['dense'](x_attn_cython)
        x_ffn_cython = model_cython._model.deltanet_ffns[layer_idx].intermediate_act_fn(x_ffn_cython)
        x_ffn_cython = torch.nn.functional.gelu(x_ffn_cython)
        x_ffn_cython = model_cython._model.output_denses[layer_idx]['dense'](x_ffn_cython)
        x_ffn_cython = model_cython._model.output_denses[layer_idx]['dropout'](x_ffn_cython)
        x_cython_layer = model_cython._model.output_denses[layer_idx]['LayerNorm'](residual_ffn_cython + x_ffn_cython)
    
    # ========================================================================
    # JAX: Process layer
    # ========================================================================
    # DeltaNet
    residual_attn_jax = x_jax_layer
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
        x_jax_layer,
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
    x_ffn_jax = jax.nn.gelu(x_ffn_jax)
    x_ffn_jax = jax.nn.gelu(x_ffn_jax)
    x_ffn_jax = jnp.dot(x_ffn_jax, params['ffn']['output']['weight'][layer_idx].T) + params['ffn']['output']['bias'][layer_idx][None, :]
    
    # LayerNorm after FFN
    x_ffn_sum_jax = residual_ffn_jax + x_ffn_jax
    mean = jnp.mean(x_ffn_sum_jax, axis=-1, keepdims=True)
    variance = jnp.var(x_ffn_sum_jax, axis=-1, keepdims=True, ddof=0)
    x_ffn_sum_jax = (x_ffn_sum_jax - mean) / jnp.sqrt(variance + config['layer_norm_eps'])
    x_jax_layer = x_ffn_sum_jax * params['norms']['ffn']['weight'][layer_idx][None, None, :] + params['norms']['ffn']['bias'][layer_idx][None, None, :]
    
    # ========================================================================
    # COMPARE
    # ========================================================================
    print("--- After DeltaNet + LayerNorm ---")
    attn_cos, attn_max, attn_mean = compare(f"Layer {layer_idx} After DeltaNet", x_attn_cython, x_attn_jax)
    
    print("--- After Complete Layer (DeltaNet + FFN) ---")
    layer_cos, layer_max, layer_mean = compare(f"Layer {layer_idx} Complete", x_cython_layer, x_jax_layer)
    
    layer_cos_sims.append(layer_cos)
    layer_max_diffs.append(layer_max)
    
    # Check if divergence occurred
    if layer_cos is not None and layer_cos < 0.95:
        print(f"⚠️  WARNING: Significant divergence detected at layer {layer_idx}!")
        print(f"    Cosine similarity dropped to {layer_cos:.6f}")
        if layer_idx > 0:
            prev_cos = layer_cos_sims[layer_idx - 1]
            if prev_cos is not None:
                print(f"    Previous layer cosine sim: {prev_cos:.6f}")
                print(f"    Drop: {prev_cos - layer_cos:.6f}")
        break

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Embeddings cosine sim: {emb_cos:.10f}")
print(f"\nLayer-by-layer cosine similarities:")
for i, cos_sim in enumerate(layer_cos_sims):
    if cos_sim is not None:
        status = "✅" if cos_sim > 0.99 else "⚠️" if cos_sim > 0.95 else "❌"
        print(f"  {status} Layer {i}: {cos_sim:.10f} (max diff: {layer_max_diffs[i]:.10e})")
    else:
        print(f"  ❌ Layer {i}: Failed comparison")

print("\n" + "="*80)
print("Trace complete!")
print("="*80)


