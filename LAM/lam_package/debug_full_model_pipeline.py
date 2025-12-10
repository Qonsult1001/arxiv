#!/usr/bin/env python3
"""
Debug full model pipeline to find where accuracy drops.
Compares Cython and JAX layer-by-layer.
"""

import torch
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'build'))

from lam import LAM
from einops import rearrange

def compare_outputs(name, cython_val, jax_val, threshold=1e-3):
    """Compare outputs and return statistics."""
    if isinstance(cython_val, torch.Tensor):
        cython_np = cython_val.detach().cpu().numpy()
    else:
        cython_np = cython_val
    
    if isinstance(jax_val, jnp.ndarray):
        jax_np = np.array(jax_val)
    else:
        jax_np = jax_val
    
    # Handle shape mismatches
    if cython_np.shape != jax_np.shape:
        print(f"⚠️  {name}: Shape mismatch - Cython {cython_np.shape} vs JAX {jax_np.shape}")
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(cython_np.shape, jax_np.shape))
        cython_np = cython_np[tuple(slice(0, s) for s in min_shape)]
        jax_np = jax_np[tuple(slice(0, s) for s in min_shape)]
    
    diff = np.abs(cython_np - jax_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    cython_flat = cython_np.flatten()
    jax_flat = jax_np.flatten()
    cos_sim = np.dot(cython_flat, jax_flat) / (np.linalg.norm(cython_flat) * np.linalg.norm(jax_flat) + 1e-9)
    
    status = "✅" if cos_sim > 0.99 and max_diff < threshold else "❌"
    
    print(f"{status} {name}:")
    print(f"    Shape: {cython_np.shape}")
    print(f"    Max diff: {max_diff:.10e}")
    print(f"    Mean diff: {mean_diff:.10e}")
    print(f"    Cosine sim: {cos_sim:.10f}")
    if cos_sim < 0.99:
        print(f"    First 3 - Cython: {cython_flat[:3]}")
        print(f"    First 3 - JAX:    {jax_flat[:3]}")
    print()
    
    return cos_sim, max_diff

print("="*80)
print("FULL MODEL PIPELINE DEBUG")
print("="*80)

model_path = "../LAM-base-v1"
test_sentence = "Hello"

# Create models
print("\nCreating models...")
model_cython = LAM(model_path, backend='cython')
model_jax = LAM(model_path, backend='jax')
print("✅ Models created\n")

# Get inputs
input_ids = model_cython.tokenizer.encode(test_sentence)
input_ids_tensor = torch.tensor([input_ids.ids], dtype=torch.long, device=model_cython.device)
attention_mask = torch.tensor([input_ids.attention_mask], dtype=torch.long, device=model_cython.device)

print(f"Input: '{test_sentence}'")
print(f"Input IDs: {input_ids.ids}")
print(f"Sequence length: {len(input_ids.ids)}\n")

# ============================================================================
# 1. EMBEDDINGS
# ============================================================================
print("="*80)
print("1. EMBEDDINGS")
print("="*80)

model_cython._model.eval()
with torch.no_grad():
    # Cython embeddings
    x_cython = model_cython._model.embeddings['word_embeddings'](input_ids_tensor)
    token_type_ids = torch.zeros_like(input_ids_tensor)
    x_cython = x_cython + model_cython._model.embeddings['token_type_embeddings'](token_type_ids)
    position_ids = torch.arange(input_ids_tensor.shape[1], dtype=torch.long, device=input_ids_tensor.device).unsqueeze(0)
    x_cython = x_cython + model_cython._model.embeddings['position_embeddings'](position_ids)
    x_cython = model_cython._model.embeddings['LayerNorm'](x_cython)

# JAX embeddings
from lam._jax_model_optimized import jax_forward_pass_optimized
input_ids_jax = jnp.array(input_ids_tensor.detach().cpu().numpy())
attention_mask_jax = jnp.array(attention_mask.detach().cpu().numpy())

# Get JAX embeddings manually
params = model_jax._jax_params
config = model_jax._jax_config_values

# Word embeddings
word_emb = params['embeddings']['word']
x_jax = word_emb[input_ids_jax]  # [b, l, d_model]

# Token type embeddings
token_type_emb = params['embeddings']['token_type']
token_type_ids_jax = jnp.zeros_like(input_ids_jax)
x_jax = x_jax + token_type_emb[token_type_ids_jax]

# Position embeddings
position_ids_jax = jnp.arange(input_ids_jax.shape[1])[None, :]
position_emb = model_jax._jax_position_emb_weight
x_jax = x_jax + position_emb[position_ids_jax]

# LayerNorm
emb_ln_w = params['embeddings']['ln_weight']
emb_ln_b = params['embeddings']['ln_bias']
mean = jnp.mean(x_jax, axis=-1, keepdims=True)
variance = jnp.var(x_jax, axis=-1, keepdims=True)
x_jax = (x_jax - mean) / jnp.sqrt(variance + config['layer_norm_eps'])
x_jax = x_jax * emb_ln_w[None, None, :] + emb_ln_b[None, None, :]

compare_outputs("Embeddings", x_cython, x_jax)

# ============================================================================
# 2. DELTANET LAYERS
# ============================================================================
print("="*80)
print("2. DELTANET LAYERS")
print("="*80)

x_cython_layer = x_cython
x_jax_layer = x_jax

for layer_idx in range(config['num_layers']):
    print(f"\n--- DeltaNet Layer {layer_idx} ---")
    
    # Cython layer
    layer = model_cython._model.deltanet_layers[layer_idx]
    with torch.no_grad():
        x_cython_layer_result = layer(x_cython_layer)
        # Handle tuple return if needed
        if isinstance(x_cython_layer_result, tuple):
            x_cython_layer = x_cython_layer_result[0]
        else:
            x_cython_layer = x_cython_layer_result
    
    # JAX layer
    from lam._jax_model_optimized import jax_deltanet_layer_optimized
    layer_params = params['deltanet']
    
    x_jax_layer_result = jax_deltanet_layer_optimized(
        layer_params['q_proj'][layer_idx],  # [key_dim, d_model]
        layer_params['k_proj'][layer_idx],
        layer_params['v_proj'][layer_idx],
        layer_params['b_proj'][layer_idx],  # [num_heads, d_model]
        layer_params['fast_decay_proj'][layer_idx],
        layer_params['fast_decay_bias'][layer_idx],
        layer_params['slow_decay_proj'][layer_idx],
        layer_params['slow_decay_bias'][layer_idx],
        layer_params['fast_gate_proj'][layer_idx],
        layer_params['slow_gate_proj'][layer_idx],
        layer_params['g_proj'][layer_idx],
        layer_params['o_proj'][layer_idx],
        layer_params['o_norm_weight'][layer_idx],  # [head_v_dim]
        layer_params['o_norm_gate'][layer_idx],  # [head_v_dim]
        layer_params['q_conv_weight'][layer_idx],  # [d_model, 1, 4]
        layer_params['q_conv_bias'][layer_idx],  # [d_model]
        layer_params['k_conv_weight'][layer_idx],
        layer_params['k_conv_bias'][layer_idx],
        layer_params['v_conv_weight'][layer_idx],
        layer_params['v_conv_bias'][layer_idx],
        layer_params['resonance_flux_W_bilinear'][layer_idx],  # [num_heads, head_k_dim, head_v_dim]
        layer_params['resonance_flux_temp'][layer_idx],  # [num_heads]
        layer_params['resonance_flux_net_w0'][layer_idx],  # [16, head_k_dim + head_v_dim + 1]
        layer_params['resonance_flux_net_b0'][layer_idx],  # [16]
        layer_params['resonance_flux_net_w2'][layer_idx],  # [1, 16]
        layer_params['resonance_flux_net_b2'][layer_idx],  # [1]
        layer_params['token_flux_proj_w0'][layer_idx],  # [head_k_dim // 2, head_k_dim + head_v_dim]
        layer_params['token_flux_proj_b0'][layer_idx],  # [head_k_dim // 2]
        layer_params['token_flux_proj_w2'][layer_idx],  # [1, head_k_dim // 2]
        layer_params['token_flux_proj_b2'][layer_idx],  # [1]
        x_jax_layer,
        config['num_heads'],
        config['head_k_dim'],
        config['head_v_dim'],
        chunk_size=64
    )
    
    # Handle return value
    if isinstance(x_jax_layer_result, tuple):
        x_jax_layer = x_jax_layer_result[0]
        print(f"  Note: Function returned tuple, using first element")
    else:
        x_jax_layer = x_jax_layer_result
    
    print(f"  JAX output type: {type(x_jax_layer)}, shape: {getattr(x_jax_layer, 'shape', 'N/A')}")
    print(f"  Cython output type: {type(x_cython_layer)}, shape: {x_cython_layer.shape}")
    
    cos_sim, max_diff = compare_outputs(f"DeltaNet Layer {layer_idx} Output", x_cython_layer, x_jax_layer)
    
    if cos_sim < 0.99:
        print(f"⚠️  Accuracy dropped at layer {layer_idx}!")
        break

# ============================================================================
# 3. FFN LAYERS (if any)
# ============================================================================
print("="*80)
print("3. FFN LAYERS")
print("="*80)

# Check if there are FFN layers
if 'ffn' in params and len(params['ffn']['intermediate']) > 0:
    for layer_idx in range(len(params['ffn']['intermediate'])):
        print(f"\n--- FFN Layer {layer_idx} ---")
        
        # Cython FFN
        ffn_layer = model_cython._model.ffn_layers[layer_idx]
        with torch.no_grad():
            residual = x_cython_layer
            x_cython_layer = ffn_layer.intermediate(x_cython_layer)
            x_cython_layer = ffn_layer.output(x_cython_layer)
            x_cython_layer = ffn_layer.norm(residual + x_cython_layer)
        
        # JAX FFN
        # TODO: Implement JAX FFN if needed
        print("⚠️  FFN layers not yet implemented in JAX")
else:
    print("No FFN layers in model")

# ============================================================================
# 4. FINAL OUTPUT
# ============================================================================
print("="*80)
print("4. FINAL OUTPUT COMPARISON")
print("="*80)

# Cython final output
with torch.no_grad():
    output_cython = model_cython._model(input_ids_tensor, attention_mask=attention_mask)

# JAX final output
output_jax_dict = model_jax.encode([test_sentence])
if isinstance(output_jax_dict, list):
    output_jax = np.array(output_jax_dict[0])
else:
    output_jax = np.array(output_jax_dict)

compare_outputs("Final Model Output", output_cython['last_hidden_state'], output_jax)

print("="*80)
print("Debug complete!")
print("="*80)

