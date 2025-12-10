#!/usr/bin/env python3
"""Compare intermediate outputs at each layer to find where errors accumulate."""

import torch
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'build'))

from lam import LAM

def compare_arrays(a1, a2, name):
    a1_np = a1.detach().cpu().numpy() if isinstance(a1, torch.Tensor) else a1
    a2_np = np.array(a2) if isinstance(a2, jnp.ndarray) else a2
    diff = np.abs(a1_np - a2_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    a1_flat = a1_np.flatten()
    a2_flat = a2_np.flatten()
    cos_sim = np.dot(a1_flat, a2_flat) / (np.linalg.norm(a1_flat) * np.linalg.norm(a2_flat) + 1e-9)
    print(f"{name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, cos_sim={cos_sim:.9f}")
    return max_diff, cos_sim

model_path = "../LAM-base-v1"
model_cython = LAM(model_path, backend='cython')
model_jax = LAM(model_path, backend='jax')

input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model_cython.device)
attention_mask = torch.ones_like(input_ids)

# Cython forward pass - layer by layer
with torch.no_grad():
    # Embeddings
    inputs_embeds = model_cython._model.embeddings['word_embeddings'](input_ids)
    token_type_emb = model_cython._model.embeddings['token_type_embeddings'](torch.zeros_like(input_ids))
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    position_emb = model_cython._model.embeddings['position_embeddings'](position_ids)
    embeddings = inputs_embeds + token_type_emb + position_emb
    embeddings = model_cython._model.embeddings['LayerNorm'](embeddings)
    x_cython = model_cython._model.embeddings['dropout'](embeddings)
    
    print("After embeddings:")
    print(f"  Cython shape: {x_cython.shape}, first 5: {x_cython[0,0,:5]}")
    
    # Process first layer
    i = 0
    residual = x_cython
    x_attn, _, _, _ = model_cython._model.deltanet_layers[i](x_cython, attention_mask)
    x_attn = model_cython._model.deltanet_norms[i](residual + x_attn)
    
    print(f"\nAfter DeltaNet layer {i}:")
    print(f"  Cython shape: {x_attn.shape}, first 5: {x_attn[0,0,:5]}")
    
    # FFN
    residual = x_attn
    x_ffn = model_cython._model.deltanet_ffns[i]['dense'](x_attn)
    x_ffn = model_cython._model.deltanet_ffns[i].intermediate_act_fn(x_ffn)
    x_ffn = torch.nn.functional.gelu(x_ffn)
    x_ffn = model_cython._model.output_denses[i]['dense'](x_ffn)
    x_ffn = model_cython._model.output_denses[i]['dropout'](x_ffn)
    x_cython = model_cython._model.output_denses[i]['LayerNorm'](residual + x_ffn)
    
    print(f"\nAfter FFN layer {i}:")
    print(f"  Cython shape: {x_cython.shape}, first 5: {x_cython[0,0,:5]}")

# JAX forward pass
input_ids_jax = jnp.array(input_ids.cpu().numpy())
attention_mask_jax = jnp.array(attention_mask.cpu().numpy())

from lam._jax_model_optimized import jax_forward_pass_optimized
position_emb_weight = jnp.array(model_cython._model.embeddings['position_embeddings'].weight.detach().cpu().numpy())

outputs_jax = jax_forward_pass_optimized(
    model_jax._jax_params, input_ids_jax, attention_mask_jax, None, position_emb_weight,
    model_jax._jax_config_values['d_model'],
    model_jax._jax_config_values['num_layers'],
    model_jax._jax_config_values['num_heads'],
    model_jax._jax_config_values['head_k_dim'],
    model_jax._jax_config_values['head_v_dim'],
    model_jax._jax_config_values['layer_norm_eps'],
    model_jax._jax_config_values.get('original_max_pos', 512),
    model_jax._jax_config_values.get('license_limit', 0x2000)
)

x_jax = outputs_jax['last_hidden_state']
print(f"\nJAX final output shape: {x_jax.shape}, first 5: {x_jax[0,0,:5]}")

compare_arrays(x_cython, x_jax, "Final output")





