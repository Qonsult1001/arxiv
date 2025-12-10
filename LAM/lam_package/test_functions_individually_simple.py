#!/usr/bin/env python3
"""
Simplified systematic testing - test compute_token_flux_jax with real data.
"""

import torch
import jax.numpy as jnp
import jax
import numpy as np
from lam._jax_core import compute_token_flux_jax
from lam import LAM

model_path = "../LAM-base-v1"
model = LAM(model_path, backend='cython')

# Get REAL data from forward pass
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
    x = model._model.embeddings['dropout'](x)
    
    # Get q, k, v
    layer = model._model.deltanet_layers[0]
    q = layer.q_proj(x)
    k = layer.k_proj(x)
    v = layer.v_proj(x)
    
    # Conv + SiLU
    q = layer.q_conv1d(q.transpose(1, 2)).transpose(1, 2)
    k = layer.k_conv1d(k.transpose(1, 2)).transpose(1, 2)
    v = layer.v_conv1d(v.transpose(1, 2)).transpose(1, 2)
    q = torch.nn.functional.silu(q)
    k = torch.nn.functional.silu(k)
    v = torch.nn.functional.silu(v)
    
    # Reshape for heads
    from einops import rearrange
    q = rearrange(q, "b l (h d) -> b h l d", h=layer.num_heads)
    k = rearrange(k, "b l (h d) -> b h l d", h=layer.num_heads)
    v = rearrange(v, "b l (h d) -> b h l d", h=layer.num_heads)
    
    # L2 norm
    from lam._core import l2norm
    q = l2norm(q)
    k = l2norm(k)
    
    # Beta scaling
    beta = layer.b_proj(x).sigmoid()
    beta = rearrange(beta, "b l h -> b h l")
    k_beta = k * beta.unsqueeze(-1)
    v_scaled = v * beta.unsqueeze(-1)
    
    # Cython token flux
    resonance_flux = layer.resonance_flux
    token_flux_cython = resonance_flux.compute_token_flux(k_beta, v_scaled)
    print(f"Cython: {token_flux_cython[0,0,0,0].item():.6f}")
    
    # JAX token flux
    w0 = jnp.array(resonance_flux.token_flux_proj[0].weight.detach().cpu().numpy())
    b0 = jnp.array(resonance_flux.token_flux_proj[0].bias.detach().cpu().numpy())
    w2 = jnp.array(resonance_flux.token_flux_proj[2].weight.detach().cpu().numpy())
    b2 = jnp.array(resonance_flux.token_flux_proj[2].bias.detach().cpu().numpy())
    
    k_beta_jax = jnp.array(k_beta.detach().cpu().numpy())
    v_scaled_jax = jnp.array(v_scaled.detach().cpu().numpy())
    
    token_flux_jax = compute_token_flux_jax(k_beta_jax, v_scaled_jax, w0, b0, w2, b2)
    print(f"JAX:    {token_flux_jax[0,0,0,0]:.6f}")
    
    diff = np.abs(token_flux_cython[0,0,0,0].item() - token_flux_jax[0,0,0,0])
    print(f"Diff:   {diff:.6e}")
    print(f"Match:  {'✅' if diff < 1e-5 else '❌'}")

