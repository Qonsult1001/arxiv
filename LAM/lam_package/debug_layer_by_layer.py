#!/usr/bin/env python3
"""
Debug forward pass layer by layer to find exact discrepancies.
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
    """Compare two arrays."""
    if isinstance(a1, torch.Tensor):
        a1 = a1.detach().cpu().numpy()
    if isinstance(a2, jnp.ndarray):
        a2 = np.array(a2)
    
    diff = np.abs(a1 - a2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    a1_flat = a1.flatten()
    a2_flat = a2.flatten()
    cos_sim = np.dot(a1_flat, a2_flat) / (np.linalg.norm(a1_flat) * np.linalg.norm(a2_flat) + 1e-9)
    
    status = "✅" if max_diff < threshold else "❌"
    print(f"{status} {name}:")
    print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, Cosine: {cos_sim:.9f}")
    if max_diff >= threshold and a1.size <= 10:
        print(f"    Cython: {a1_flat}")
        print(f"    JAX:    {a2_flat}")
    print()
    return max_diff, cos_sim

def test_layer_by_layer():
    """Test each layer systematically."""
    print("=" * 80)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 80)
    
    model_path = "../LAM-base-v1"
    model_cython = LAM(model_path, backend='cython')
    
    # Simple input
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model_cython.device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = model_cython._model.embeddings['word_embeddings'](input_ids)
        token_type_embeddings = model_cython._model.embeddings['token_type_embeddings'](torch.zeros_like(input_ids))
        
        # Position embeddings
        position_emb_weight = model_cython._model.embeddings['position_embeddings'].weight
        seq_length = input_ids.shape[1]
        original_max_pos = 512
        
        try:
            from lam import _secrets
            position_emb = _secrets.interpolate_positions(
                position_emb_weight,
                seq_length,
                original_max_pos,
                device=str(input_ids.device),
                license_limit=0x2000
            )
            position_embeddings = position_emb.unsqueeze(0).expand(1, -1, -1)
        except:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            position_embeddings = position_emb_weight[position_ids]
        
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = model_cython._model.embeddings['LayerNorm'](embeddings)
        x = model_cython._model.embeddings['dropout'](embeddings)
        
        print(f"After embeddings:")
        print(f"  Shape: {x.shape}")
        print(f"  First 5: {x[0,0,:5]}")
        print()
        
        # Process first layer only
        i = 0
        residual = x
        x_attn, _, _, _ = model_cython._model.deltanet_layers[i](x, attention_mask)
        x_attn = model_cython._model.deltanet_norms[i](residual + x_attn)
        
        print(f"After DeltaNet layer {i}:")
        print(f"  Shape: {x_attn.shape}")
        print(f"  First 5: {x_attn[0,0,:5]}")
        print()
        
        # FFN
        residual = x_attn
        x_ffn = model_cython._model.deltanet_ffns[i]['dense'](x_attn)
        x_ffn = model_cython._model.deltanet_ffns[i].intermediate_act_fn(x_ffn)
        x_ffn = torch.nn.functional.gelu(x_ffn)
        x_ffn = model_cython._model.output_denses[i]['dense'](x_ffn)
        x_ffn = model_cython._model.output_denses[i]['dropout'](x_ffn)
        x = model_cython._model.output_denses[i]['LayerNorm'](residual + x_ffn)
        
        print(f"After FFN layer {i}:")
        print(f"  Shape: {x.shape}")
        print(f"  First 5: {x[0,0,:5]}")
        print()
        
        output_cython = x
    
    # Now get JAX output for same input
    model_jax = LAM(model_path, backend='jax')
    input_ids_jax = jnp.array(input_ids.cpu().numpy())
    attention_mask_jax = jnp.array(attention_mask.cpu().numpy())
    
    # Get JAX embeddings
    embeddings_jax = model_jax._jax_get_embeddings(
        model_jax._jax_params, input_ids_jax, attention_mask_jax,
        **model_jax._jax_config_values
    )
    embeddings_jax_torch = torch.from_numpy(np.array(embeddings_jax)).to(model_cython.device)
    
    # Compare final embeddings
    final_emb_cython = model_cython.encode(["test"], batch_size=1, show_progress_bar=False)
    final_emb_jax = model_jax.encode(["test"], batch_size=1, show_progress_bar=False)
    
    final_emb_cython_np = final_emb_cython if isinstance(final_emb_cython, np.ndarray) else final_emb_cython.cpu().numpy()
    final_emb_jax_np = final_emb_jax if isinstance(final_emb_jax, np.ndarray) else final_emb_jax.cpu().numpy()
    
    print("=" * 80)
    print("FINAL EMBEDDINGS COMPARISON")
    print("=" * 80)
    compare_arrays(final_emb_cython_np, final_emb_jax_np, "Final Embeddings", threshold=1e-3)

if __name__ == "__main__":
    test_layer_by_layer()
