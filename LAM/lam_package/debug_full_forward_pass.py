#!/usr/bin/env python3
"""
Debug full forward pass to find discrepancies between JAX and Cython.
Compare each component systematically.
"""

import torch
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'build'))

from lam import LAM

def compare_arrays(a1, a2, name, threshold=1e-4):
    """Compare two arrays and report differences."""
    if isinstance(a1, torch.Tensor):
        a1 = a1.detach().cpu().numpy()
    if isinstance(a2, jnp.ndarray):
        a2 = np.array(a2)
    
    diff = np.abs(a1 - a2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Cosine similarity
    a1_flat = a1.flatten()
    a2_flat = a2.flatten()
    cos_sim = np.dot(a1_flat, a2_flat) / (np.linalg.norm(a1_flat) * np.linalg.norm(a2_flat) + 1e-9)
    
    match = max_diff < threshold
    status = "✅" if match else "❌"
    
    print(f"{status} {name}:")
    print(f"    Shape: {a1.shape} vs {a2.shape}")
    print(f"    Max diff: {max_diff:.6e}")
    print(f"    Mean diff: {mean_diff:.6e}")
    print(f"    Cosine sim: {cos_sim:.9f}")
    if not match and a1.size <= 20:
        print(f"    Cython: {a1_flat}")
        print(f"    JAX:    {a2_flat}")
    elif not match:
        print(f"    First 5 - Cython: {a1_flat[:5]}")
        print(f"    First 5 - JAX:    {a2_flat[:5]}")
    print()
    
    return match, max_diff, cos_sim

def test_embeddings():
    """Test embeddings layer."""
    print("=" * 80)
    print("TEST 1: Embeddings Layer")
    print("=" * 80)
    
    model_path = "../LAM-base-v1"
    model_cython = LAM(model_path, backend='cython')
    model_jax = LAM(model_path, backend='jax')
    
    # Simple input
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model_cython.device)
    attention_mask = torch.ones_like(input_ids)
    
    # Get embeddings from Cython
    with torch.no_grad():
        # Access the model's embedding layer
        word_emb = model_cython._model.embeddings['word_embeddings'](input_ids)
        token_type_emb = model_cython._model.embeddings['token_type_embeddings'](torch.zeros_like(input_ids))
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_emb = model_cython._model.embeddings['position_embeddings'](position_ids)
        
        embeddings_cython = word_emb + token_type_emb + position_emb
        
        # LayerNorm
        ln_weight = model_cython._model.embeddings['LayerNorm'].weight
        ln_bias = model_cython._model.embeddings['LayerNorm'].bias
        ln_eps = model_cython._model.embeddings['LayerNorm'].eps
        
        mean = embeddings_cython.mean(dim=-1, keepdim=True)
        variance = embeddings_cython.var(dim=-1, keepdim=True, unbiased=False)
        embeddings_cython = (embeddings_cython - mean) / torch.sqrt(variance + ln_eps)
        embeddings_cython = embeddings_cython * ln_weight[None, None, :] + ln_bias[None, None, :]
    
    # Get embeddings from JAX
    input_ids_jax = jnp.array(input_ids.cpu().numpy())
    attention_mask_jax = jnp.array(attention_mask.cpu().numpy())
    
    # Access JAX embeddings
    from lam._jax_model_optimized import jax_forward_pass_optimized
    from lam._jax_model_optimized import create_jax_lam_model_optimized
    
    jax_params, jax_config = create_jax_lam_model_optimized(model_path)
    
    # Get just the embeddings part
    position_emb_weight = jnp.array(model_cython._model.embeddings['position_embeddings'].weight.detach().cpu().numpy())
    
    # Run forward pass to get embeddings
    outputs = jax_forward_pass_optimized(
        jax_params, input_ids_jax, attention_mask_jax, None, position_emb_weight,
        jax_config['d_model'], jax_config['num_layers'], jax_config['num_heads'],
        jax_config['head_k_dim'], jax_config['head_v_dim'], jax_config['layer_norm_eps']
    )
    
    # For now, just compare final output
    embeddings_jax = outputs['last_hidden_state']
    
    # Compare
    compare_arrays(embeddings_cython, embeddings_jax, "Embeddings", threshold=1e-3)

def test_single_layer():
    """Test a single DeltaNet layer."""
    print("=" * 80)
    print("TEST 2: Single DeltaNet Layer")
    print("=" * 80)
    
    model_path = "../LAM-base-v1"
    model_cython = LAM(model_path, backend='cython')
    
    # Get input from first layer
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model_cython.device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        # Get embeddings
        word_emb = model_cython._model.embeddings['word_embeddings'](input_ids)
        token_type_emb = model_cython._model.embeddings['token_type_embeddings'](torch.zeros_like(input_ids))
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_emb = model_cython._model.embeddings['position_embeddings'](position_ids)
        
        embeddings = word_emb + token_type_emb + position_emb
        
        # LayerNorm
        ln_weight = model_cython._model.embeddings['LayerNorm'].weight
        ln_bias = model_cython._model.embeddings['LayerNorm'].bias
        ln_eps = model_cython._model.embeddings['LayerNorm'].eps
        
        mean = embeddings.mean(dim=-1, keepdim=True)
        variance = embeddings.var(dim=-1, keepdim=True, unbiased=False)
        embeddings = (embeddings - mean) / torch.sqrt(variance + ln_eps)
        embeddings = embeddings * ln_weight[None, None, :] + ln_bias[None, None, :]
        
        # First DeltaNet layer
        x = embeddings
        layer = model_cython._model.deltanet_layers[0]
        norm = model_cython._model.deltanet_norms[0]
        
        residual = x
        x_attn, _, _, _ = layer(x, attention_mask)
        
        # LayerNorm
        mean = x_attn.mean(dim=-1, keepdim=True)
        variance = x_attn.var(dim=-1, keepdim=True, unbiased=False)
        x_attn = (x_attn - mean) / torch.sqrt(variance + norm.eps)
        x_attn = x_attn * norm.weight[None, None, :] + norm.bias[None, None, :]
        x = residual + x_attn
        
        # FFN
        residual = x
        x_ffn = model_cython._model.deltanet_ffns[0]['dense'](x)
        x_ffn = model_cython._model.deltanet_ffns[0].intermediate_act_fn(x_ffn)
        x_ffn = torch.nn.functional.gelu(x_ffn)
        x_ffn = model_cython._model.output_denses[0]['dense'](x_ffn)
        x_ffn = model_cython._model.output_denses[0]['dropout'](x_ffn)
        x = model_cython._model.output_denses[0]['LayerNorm'](residual + x_ffn)
        
        output_cython = x
    
    print(f"Cython output shape: {output_cython.shape}")
    print(f"Cython output[0,0,:5]: {output_cython[0,0,:5]}")
    
    # TODO: Compare with JAX
    print("JAX comparison not yet implemented for single layer")
    print()

if __name__ == "__main__":
    test_embeddings()
    test_single_layer()





