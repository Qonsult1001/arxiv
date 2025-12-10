#!/usr/bin/env python3
"""
Test intermediate values to find exact divergence point.
"""

import torch
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'build'))

# We need to modify the JAX function to return intermediate values
# For now, let's check if the issue is in attn_const computation

from lam import LAM
from einops import rearrange

print("="*80)
print("TESTING INTERMEDIATE VALUES")
print("="*80)

model_path = "../LAM-base-v1"
model = LAM(model_path, backend='cython')
test_sentence = "Hello"
input_ids = model.tokenizer.encode(test_sentence)
input_ids_tensor = torch.tensor([input_ids.ids], dtype=torch.long, device=model.device)

model._model.eval()
with torch.no_grad():
    x = model._model.embeddings['word_embeddings'](input_ids_tensor)
    token_type_ids = torch.zeros_like(input_ids_tensor)
    x = x + model._model.embeddings['token_type_embeddings'](token_type_ids)
    position_ids = torch.arange(input_ids_tensor.shape[1], dtype=torch.long, device=input_ids_tensor.device).unsqueeze(0)
    x = x + model._model.embeddings['position_embeddings'](position_ids)
    x = model._model.embeddings['LayerNorm'](x)
    
    layer = model._model.deltanet_layers[0]
    q = layer.q_proj(x)
    k = layer.k_proj(x)
    v = layer.v_proj(x)
    q, _ = layer.q_conv1d(q)
    k, _ = layer.k_conv1d(k)
    v, _ = layer.v_conv1d(v)
    q = torch.nn.functional.silu(q)
    k = torch.nn.functional.silu(k)
    v = torch.nn.functional.silu(v)
    q = rearrange(q, "b l (h d) -> b h l d", h=layer.num_heads)
    k = rearrange(k, "b l (h d) -> b h l d", h=layer.num_heads)
    v = rearrange(v, "b l (h d) -> b h l d", h=layer.num_heads)
    from lam._core import l2norm
    q = l2norm(q)
    k = l2norm(k)
    beta = layer.b_proj(x).sigmoid()
    beta = rearrange(beta, "b l h -> b h l")
    beta_expanded = beta.unsqueeze(-1)
    k_beta = k * beta_expanded
    v_scaled = v * beta_expanded
    
    # Test attn_const computation
    chunk_size = 64
    l = q.shape[2]
    if l < chunk_size:
        actual_chunk_size = l
        n_chunks = 1
        pad_len = 0
    else:
        n_chunks = (l + chunk_size - 1) // chunk_size
        pad_len = (n_chunks * chunk_size) - l
        actual_chunk_size = chunk_size
    
    q_padded = torch.nn.functional.pad(q, (0, 0, 0, pad_len)) if pad_len > 0 else q
    k_padded = torch.nn.functional.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k
    k_beta_padded = torch.nn.functional.pad(k_beta, (0, 0, 0, pad_len)) if pad_len > 0 else k_beta
    
    q_chunked = q_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, layer.head_k_dim)
    k_chunked = k_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, layer.head_k_dim)
    k_beta_chunked = k_beta_padded.reshape(1, layer.num_heads, n_chunks, actual_chunk_size, layer.head_k_dim)
    
    # Cython: attn_const = -(k_beta @ k.transpose(-1, -2))
    k_T_cython = k_chunked.transpose(-1, -2)  # [b, h, n, d_k, c]
    attn_const_cython = -(k_beta_chunked @ k_T_cython)  # [b, h, n, c, c]
    
    # JAX: using matmul
    k_T_jax = jnp.transpose(jnp.array(k_chunked.detach().cpu().numpy()), (0, 1, 2, 4, 3))
    k_beta_jax = jnp.array(k_beta_chunked.detach().cpu().numpy())
    attn_const_jax = -jnp.matmul(k_beta_jax, k_T_jax)
    
    cython_np = attn_const_cython.detach().cpu().numpy()
    jax_np = np.array(attn_const_jax)
    
    diff = np.abs(cython_np - jax_np)
    print(f'attn_const comparison:')
    print(f'  Max diff: {np.max(diff):.10e}')
    print(f'  Mean diff: {np.mean(diff):.10e}')
    cos_sim = np.dot(cython_np.flatten(), jax_np.flatten()) / (np.linalg.norm(cython_np.flatten()) * np.linalg.norm(jax_np.flatten()) + 1e-9)
    print(f'  Cosine sim: {cos_sim:.10f}')
    if np.max(diff) > 1e-6:
        print(f'  ❌ attn_const does not match!')
        print(f'  First 5 - Cython: {cython_np.flatten()[:5]}')
        print(f'  First 5 - JAX:    {jax_np.flatten()[:5]}')
    else:
        print(f'  ✅ attn_const matches!')


