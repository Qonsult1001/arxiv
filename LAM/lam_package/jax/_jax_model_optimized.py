"""
Optimized JAX Model Implementation for LAM
==========================================

Pure JAX arrays - fully JIT-compilable forward pass.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from ._jax_core import hierarchical_delta_rule_jax, simple_resonance_flux_jax, enhanced_resonance_flux_jax

# JAX configuration
jax.config.update("jax_enable_x64", False)


def pytorch_to_jax(tensor: torch.Tensor) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(tensor.detach().cpu().numpy())


def create_jax_lam_model_optimized(model_path, license_limit=0x2000, tier="free"):
    """
    Create optimized JAX LAM model - pure arrays, no dicts/dataclasses.
    Returns a tuple of arrays that can be JIT-compiled.
    """
    weights_path = Path(model_path) / 'pytorch_model.bin'
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    loaded_data = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    
    is_raw_state_dict = False
    if isinstance(loaded_data, dict):
        has_checkpoint_keys = any(k in loaded_data for k in ['model_state_dict', 'deltanet_layers', 'config', 'step', 'lam_layers'])
        has_model_keys = any('deltanet_layers.' in str(k) or 'teacher_model.' in str(k) for k in loaded_data.keys())
        is_raw_state_dict = not has_checkpoint_keys and has_model_keys
    
    if is_raw_state_dict:
        checkpoint = {'model_state_dict': loaded_data, 'config': {}, 'step': 0}
    else:
        checkpoint = loaded_data
    
    config_path = Path(model_path) / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            lam_config = json.load(f)
        config = checkpoint.get('config', lam_config)
        if 'lam_config' in lam_config:
            config.update(lam_config.get('lam_config', {}))
    else:
        config = checkpoint.get('config', {})
    
    teacher_config = {
        "vocab_size": config.get('vocab_size', 30522),
        "hidden_size": config.get('hidden_size', 384),
        "max_position_embeddings": config.get('max_position_embeddings', 512),
        "type_vocab_size": config.get('type_vocab_size', 2),
        "layer_norm_eps": config.get('layer_norm_eps', 1e-12),
        "hidden_dropout_prob": config.get('hidden_dropout_prob', 0.1),
        "intermediate_size": config.get('intermediate_size', 1536),
        "num_hidden_layers": config.get('num_layers', 6),
        "num_attention_heads": config.get('num_heads', 12),
    }
    
    d_model = config.get('hidden_size', teacher_config.get('hidden_size', 384))
    num_layers = config.get('num_layers', teacher_config.get('num_hidden_layers', 6))
    num_heads = config.get('num_heads', teacher_config.get('num_attention_heads', 12))
    vocab_size = teacher_config.get('vocab_size', 30522)
    max_pos = teacher_config.get('max_position_embeddings', 512)
    type_vocab_size = teacher_config.get('type_vocab_size', 2)
    layer_norm_eps = teacher_config.get('layer_norm_eps', 1e-12)
    intermediate_size = teacher_config.get('intermediate_size', 1536)
    head_k_dim = d_model // num_heads
    head_v_dim = d_model // num_heads
    
    # Extract weights
    teacher_state_dict = {}
    for key, value in loaded_data.items():
        if key.startswith('teacher_model.'):
            new_key = key.replace('teacher_model.', '')
            teacher_state_dict[new_key] = value
    
    # Embeddings - pure arrays
    word_emb = pytorch_to_jax(teacher_state_dict.get('embeddings.word_embeddings.weight', torch.randn(vocab_size, d_model)))
    pos_emb = pytorch_to_jax(teacher_state_dict.get('embeddings.position_embeddings.weight', torch.randn(max_pos, d_model)))
    token_type_emb = pytorch_to_jax(teacher_state_dict.get('embeddings.token_type_embeddings.weight', torch.randn(type_vocab_size, d_model)))
    emb_ln_weight = pytorch_to_jax(teacher_state_dict.get('embeddings.LayerNorm.weight', torch.ones(d_model)))
    emb_ln_bias = pytorch_to_jax(teacher_state_dict.get('embeddings.LayerNorm.bias', torch.zeros(d_model)))
    
    # Extract deltanet layer weights
    model_state_dict = checkpoint.get('model_state_dict', {})
    deltanet_layers_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('deltanet_layers.'):
            new_key = key.replace('deltanet_layers.', '')
            deltanet_layers_dict[new_key] = value
    
    # Build layer weights as arrays (not dicts)
    # Shape: [num_layers, ...] for each weight type
    q_proj_weights = []
    k_proj_weights = []
    v_proj_weights = []
    b_proj_weights = []
    fast_decay_proj_weights = []
    fast_decay_biases = []
    slow_decay_proj_weights = []
    slow_decay_biases = []
    fast_gate_proj_weights = []
    slow_gate_proj_weights = []
    g_proj_weights = []
    o_proj_weights = []
    o_norm_weights = []
    o_norm_gates = []
    # Convolution weights
    q_conv_weights = []
    q_conv_biases = []
    k_conv_weights = []
    k_conv_biases = []
    v_conv_weights = []
    v_conv_biases = []
    # Resonance flux weights
    resonance_flux_W_bilinear = []
    resonance_flux_temp = []
    resonance_flux_net_weights = []
    token_flux_weights = []
    resonance_flux_net_biases = []
    attn_norm_weights = []
    attn_norm_biases = []
    ffn_intermediate_weights = []
    ffn_intermediate_biases = []
    ffn_output_weights = []
    ffn_output_biases = []
    ffn_norm_weights = []
    ffn_norm_biases = []
    
    for i in range(num_layers):
        # DeltaNet layer weights
        layer_state = {}
        for k, v in deltanet_layers_dict.items():
            if k.startswith(f'{i}.'):
                new_key = k[len(f'{i}.'):]
                layer_state[new_key] = pytorch_to_jax(v)
        
        # Load weights with correct shapes
        # q_proj: [key_dim, d_model] where key_dim = d_model (384, 384)
        q_proj_w = layer_state.get('q_proj.weight', pytorch_to_jax(torch.randn(d_model, d_model)))
        k_proj_w = layer_state.get('k_proj.weight', pytorch_to_jax(torch.randn(d_model, d_model)))
        v_proj_w = layer_state.get('v_proj.weight', pytorch_to_jax(torch.randn(d_model, d_model)))
        b_proj_w = layer_state.get('b_proj.weight', pytorch_to_jax(torch.randn(num_heads, d_model)))
        fast_decay_proj_w = layer_state.get('fast_decay_proj.weight', pytorch_to_jax(torch.randn(num_heads, d_model)))
        fast_decay_bias = layer_state.get('fast_decay_bias', pytorch_to_jax(torch.full((num_heads,), np.log(0.3))))
        slow_decay_proj_w = layer_state.get('slow_decay_proj.weight', pytorch_to_jax(torch.randn(num_heads, d_model)))
        slow_decay_bias = layer_state.get('slow_decay_bias', pytorch_to_jax(torch.full((num_heads,), np.log(0.9))))
        fast_gate_proj_w = layer_state.get('fast_gate_proj.weight', pytorch_to_jax(torch.randn(num_heads, d_model)))
        slow_gate_proj_w = layer_state.get('slow_gate_proj.weight', pytorch_to_jax(torch.randn(num_heads, d_model)))
        g_proj_w = layer_state.get('g_proj.weight', pytorch_to_jax(torch.randn(d_model, d_model)))
        o_proj_w = layer_state.get('o_proj.weight', pytorch_to_jax(torch.randn(d_model, d_model)))
        o_norm_w = layer_state.get('o_norm.weight', pytorch_to_jax(torch.ones(head_v_dim)))
        o_norm_gate = layer_state.get('o_norm.gate', pytorch_to_jax(torch.ones(head_v_dim)))
        
        # Convolution weights: [hidden_size, 1, kernel_size] = [384, 1, 4]
        q_conv_w = layer_state.get('q_conv1d.conv.weight', pytorch_to_jax(torch.randn(d_model, 1, 4)))
        q_conv_b = layer_state.get('q_conv1d.conv.bias', pytorch_to_jax(torch.zeros(d_model)))
        k_conv_w = layer_state.get('k_conv1d.conv.weight', pytorch_to_jax(torch.randn(d_model, 1, 4)))
        k_conv_b = layer_state.get('k_conv1d.conv.bias', pytorch_to_jax(torch.zeros(d_model)))
        v_conv_w = layer_state.get('v_conv1d.conv.weight', pytorch_to_jax(torch.randn(d_model, 1, 4)))
        v_conv_b = layer_state.get('v_conv1d.conv.bias', pytorch_to_jax(torch.zeros(d_model)))
        
        # Resonance flux weights
        resonance_flux_W_bilinear_w = layer_state.get('resonance_flux.W_bilinear', pytorch_to_jax(torch.randn(num_heads, head_k_dim, head_v_dim)))
        resonance_flux_temp_w = layer_state.get('resonance_flux.temp', pytorch_to_jax(torch.ones(num_heads)))
        # Flux net: 2-layer MLP with 16 hidden units
        flux_net_w0 = layer_state.get('resonance_flux.flux_net.0.weight', pytorch_to_jax(torch.randn(16, head_k_dim + head_v_dim + 1)))
        flux_net_b0 = layer_state.get('resonance_flux.flux_net.0.bias', pytorch_to_jax(torch.zeros(16)))
        flux_net_w2 = layer_state.get('resonance_flux.flux_net.2.weight', pytorch_to_jax(torch.randn(1, 16)))
        flux_net_b2 = layer_state.get('resonance_flux.flux_net.2.bias', pytorch_to_jax(torch.zeros(1)))
        # Token flux projection: 2-layer MLP (d_k + d_v) -> (d_k // 2) -> 1
        token_flux_w0 = layer_state.get('resonance_flux.token_flux_proj.0.weight', pytorch_to_jax(torch.randn(head_k_dim // 2, head_k_dim + head_v_dim)))
        token_flux_b0 = layer_state.get('resonance_flux.token_flux_proj.0.bias', pytorch_to_jax(torch.zeros(head_k_dim // 2)))
        token_flux_w2 = layer_state.get('resonance_flux.token_flux_proj.2.weight', pytorch_to_jax(torch.randn(1, head_k_dim // 2)))
        token_flux_b2 = layer_state.get('resonance_flux.token_flux_proj.2.bias', pytorch_to_jax(torch.zeros(1)))
        
        q_proj_weights.append(q_proj_w)
        k_proj_weights.append(k_proj_w)
        v_proj_weights.append(v_proj_w)
        b_proj_weights.append(b_proj_w)
        fast_decay_proj_weights.append(fast_decay_proj_w)
        fast_decay_biases.append(fast_decay_bias)
        slow_decay_proj_weights.append(slow_decay_proj_w)
        slow_decay_biases.append(slow_decay_bias)
        fast_gate_proj_weights.append(fast_gate_proj_w)
        slow_gate_proj_weights.append(slow_gate_proj_w)
        g_proj_weights.append(g_proj_w)
        o_proj_weights.append(o_proj_w)
        o_norm_weights.append(o_norm_w)
        o_norm_gates.append(o_norm_gate)
        q_conv_weights.append(q_conv_w)
        q_conv_biases.append(q_conv_b)
        k_conv_weights.append(k_conv_w)
        k_conv_biases.append(k_conv_b)
        v_conv_weights.append(v_conv_w)
        v_conv_biases.append(v_conv_b)
        resonance_flux_W_bilinear.append(resonance_flux_W_bilinear_w)
        resonance_flux_temp.append(resonance_flux_temp_w)
        # Flatten flux net weights for JIT compilation
        resonance_flux_net_weights.append((flux_net_w0, flux_net_b0, flux_net_w2, flux_net_b2))
        # Token flux weights
        token_flux_weights.append((token_flux_w0, token_flux_b0, token_flux_w2, token_flux_b2))
        
        # Norms
        attn_norm_w = pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.attention.output.LayerNorm.weight', torch.ones(d_model)))
        attn_norm_b = pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.attention.output.LayerNorm.bias', torch.zeros(d_model)))
        attn_norm_weights.append(attn_norm_w)
        attn_norm_biases.append(attn_norm_b)
        
        # FFN
        ffn_int_w = pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.intermediate.dense.weight', torch.randn(intermediate_size, d_model)))
        ffn_int_b = pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.intermediate.dense.bias', torch.zeros(intermediate_size)))
        ffn_intermediate_weights.append(ffn_int_w)
        ffn_intermediate_biases.append(ffn_int_b)
        
        ffn_out_w = pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.output.dense.weight', torch.randn(d_model, intermediate_size)))
        ffn_out_b = pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.output.dense.bias', torch.zeros(d_model)))
        ffn_output_weights.append(ffn_out_w)
        ffn_output_biases.append(ffn_out_b)
        
        ffn_norm_w = pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.output.LayerNorm.weight', torch.ones(d_model)))
        ffn_norm_b = pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.output.LayerNorm.bias', torch.zeros(d_model)))
        ffn_norm_weights.append(ffn_norm_w)
        ffn_norm_biases.append(ffn_norm_b)
    
    # Stack into arrays
    q_proj_weights = jnp.stack(q_proj_weights)  # [num_layers, key_dim, d_model]
    k_proj_weights = jnp.stack(k_proj_weights)
    v_proj_weights = jnp.stack(v_proj_weights)
    b_proj_weights = jnp.stack(b_proj_weights)  # [num_layers, num_heads, d_model]
    fast_decay_proj_weights = jnp.stack(fast_decay_proj_weights)
    fast_decay_biases = jnp.stack(fast_decay_biases)  # [num_layers, num_heads]
    slow_decay_proj_weights = jnp.stack(slow_decay_proj_weights)
    slow_decay_biases = jnp.stack(slow_decay_biases)
    fast_gate_proj_weights = jnp.stack(fast_gate_proj_weights)
    slow_gate_proj_weights = jnp.stack(slow_gate_proj_weights)
    g_proj_weights = jnp.stack(g_proj_weights)  # [num_layers, d_model, d_model]
    o_proj_weights = jnp.stack(o_proj_weights)  # [num_layers, d_model, value_dim]
    o_norm_weights = jnp.stack(o_norm_weights)  # [num_layers, head_v_dim]
    o_norm_gates = jnp.stack(o_norm_gates)  # [num_layers, head_v_dim]
    q_conv_weights = jnp.stack(q_conv_weights)  # [num_layers, d_model, 1, 4]
    q_conv_biases = jnp.stack(q_conv_biases)  # [num_layers, d_model]
    k_conv_weights = jnp.stack(k_conv_weights)
    k_conv_biases = jnp.stack(k_conv_biases)
    v_conv_weights = jnp.stack(v_conv_weights)
    v_conv_biases = jnp.stack(v_conv_biases)
    resonance_flux_W_bilinear = jnp.stack(resonance_flux_W_bilinear)  # [num_layers, num_heads, head_k_dim, head_v_dim]
    resonance_flux_temp = jnp.stack(resonance_flux_temp)  # [num_layers, num_heads]
    # Stack flux net weights: [num_layers, 4] where each element is a tuple, but we'll store as separate arrays
    flux_net_w0_list = [w[0] for w in resonance_flux_net_weights]
    flux_net_b0_list = [w[1] for w in resonance_flux_net_weights]
    flux_net_w2_list = [w[2] for w in resonance_flux_net_weights]
    flux_net_b2_list = [w[3] for w in resonance_flux_net_weights]
    resonance_flux_net_w0 = jnp.stack(flux_net_w0_list)  # [num_layers, 16, head_k_dim + head_v_dim + 1]
    resonance_flux_net_b0 = jnp.stack(flux_net_b0_list)  # [num_layers, 16]
    resonance_flux_net_w2 = jnp.stack(flux_net_w2_list)  # [num_layers, 1, 16]
    resonance_flux_net_b2 = jnp.stack(flux_net_b2_list)  # [num_layers, 1]
    # Stack token flux weights
    token_flux_w0_list = [w[0] for w in token_flux_weights]
    token_flux_b0_list = [w[1] for w in token_flux_weights]
    token_flux_w2_list = [w[2] for w in token_flux_weights]
    token_flux_b2_list = [w[3] for w in token_flux_weights]
    token_flux_proj_w0 = jnp.stack(token_flux_w0_list)  # [num_layers, head_k_dim // 2, head_k_dim + head_v_dim]
    token_flux_proj_b0 = jnp.stack(token_flux_b0_list)  # [num_layers, head_k_dim // 2]
    token_flux_proj_w2 = jnp.stack(token_flux_w2_list)  # [num_layers, 1, head_k_dim // 2]
    token_flux_proj_b2 = jnp.stack(token_flux_b2_list)  # [num_layers, 1]
    attn_norm_weights = jnp.stack(attn_norm_weights)  # [num_layers, d_model]
    attn_norm_biases = jnp.stack(attn_norm_biases)
    ffn_intermediate_weights = jnp.stack(ffn_intermediate_weights)  # [num_layers, intermediate_size, d_model]
    ffn_intermediate_biases = jnp.stack(ffn_intermediate_biases)  # [num_layers, intermediate_size]
    ffn_output_weights = jnp.stack(ffn_output_weights)  # [num_layers, d_model, intermediate_size]
    ffn_output_biases = jnp.stack(ffn_output_biases)  # [num_layers, d_model]
    ffn_norm_weights = jnp.stack(ffn_norm_weights)  # [num_layers, d_model]
    ffn_norm_biases = jnp.stack(ffn_norm_biases)
    
    return {
        'embeddings': {
            'word': word_emb,
            'position': pos_emb,
            'token_type': token_type_emb,
            'ln_weight': emb_ln_weight,
            'ln_bias': emb_ln_bias,
        },
        'deltanet': {
            'q_proj': q_proj_weights,
            'k_proj': k_proj_weights,
            'v_proj': v_proj_weights,
            'b_proj': b_proj_weights,
            'fast_decay_proj': fast_decay_proj_weights,
            'fast_decay_bias': fast_decay_biases,
            'slow_decay_proj': slow_decay_proj_weights,
            'slow_decay_bias': slow_decay_biases,
            'fast_gate_proj': fast_gate_proj_weights,
            'slow_gate_proj': slow_gate_proj_weights,
            'g_proj': g_proj_weights,
            'o_proj': o_proj_weights,
            'o_norm_weight': o_norm_weights,
            'o_norm_gate': o_norm_gates,
            'q_conv_weight': q_conv_weights,
            'q_conv_bias': q_conv_biases,
            'k_conv_weight': k_conv_weights,
            'k_conv_bias': k_conv_biases,
            'v_conv_weight': v_conv_weights,
            'v_conv_bias': v_conv_biases,
            'resonance_flux_W_bilinear': resonance_flux_W_bilinear,
            'resonance_flux_temp': resonance_flux_temp,
            'resonance_flux_net_w0': resonance_flux_net_w0,
            'resonance_flux_net_b0': resonance_flux_net_b0,
            'resonance_flux_net_w2': resonance_flux_net_w2,
            'resonance_flux_net_b2': resonance_flux_net_b2,
            'token_flux_proj_w0': token_flux_proj_w0,
            'token_flux_proj_b0': token_flux_proj_b0,
            'token_flux_proj_w2': token_flux_proj_w2,
            'token_flux_proj_b2': token_flux_proj_b2,
        },
        'norms': {
            'attn': {'weight': attn_norm_weights, 'bias': attn_norm_biases},
            'ffn': {'weight': ffn_norm_weights, 'bias': ffn_norm_biases},
        },
        'ffn': {
            'intermediate': {'weight': ffn_intermediate_weights, 'bias': ffn_intermediate_biases},
            'output': {'weight': ffn_output_weights, 'bias': ffn_output_biases},
        },
        'config': {
            'd_model': d_model,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'head_k_dim': head_k_dim,
            'head_v_dim': head_v_dim,
            'vocab_size': vocab_size,
            'max_pos': max_pos,
            'type_vocab_size': type_vocab_size,
            'layer_norm_eps': layer_norm_eps,
            'intermediate_size': intermediate_size,
            'license_limit': license_limit,
            'tier': tier
        }
    }


def jax_conv1d(x, weight, bias, kernel_size=4):
    """Depthwise conv1d: [b, l, d] -> [b, l, d] (JIT-friendly vectorized)"""
    # x: [b, l, d], weight: [d, 1, kernel_size], bias: [d]
    b, l, d = x.shape
    padding = kernel_size // 2
    # Pad: [b, l, d] -> [b, l + 2*padding, d]
    x_padded = jnp.pad(x, ((0, 0), (padding, padding), (0, 0)), mode='constant')
    
    # Create sliding windows using gather: [b, l, d, kernel_size]
    # For each position i, gather x_padded[:, i:i+kernel_size, :]
    indices = jnp.arange(l)[:, None] + jnp.arange(kernel_size)[None, :]  # [l, kernel_size]
    windows = x_padded[:, indices, :]  # [b, l, kernel_size, d]
    windows = jnp.transpose(windows, (0, 1, 3, 2))  # [b, l, d, kernel_size]
    
    # Apply conv: [b, l, d, kernel_size] * [d, kernel_size] -> [b, l, d]
    weight_flat = weight[:, 0, :]  # [d, kernel_size]
    output = jnp.sum(windows * weight_flat[None, None, :, :], axis=-1)  # [b, l, d]
    output = output + bias[None, None, :]
    
    return output


def jax_deltanet_layer_optimized(
    q_proj_w, k_proj_w, v_proj_w, b_proj_w,
    fast_decay_proj_w, fast_decay_bias,
    slow_decay_proj_w, slow_decay_bias,
    fast_gate_proj_w, slow_gate_proj_w,
    g_proj_w, o_proj_w, o_norm_w, o_norm_gate,
    q_conv_w, q_conv_b, k_conv_w, k_conv_b, v_conv_w, v_conv_b,
    resonance_flux_W_bilinear, resonance_flux_temp,
    resonance_flux_net_w0, resonance_flux_net_b0, resonance_flux_net_w2, resonance_flux_net_b2,
    token_flux_proj_w0, token_flux_proj_b0, token_flux_proj_w2, token_flux_proj_b2,
    hidden_states,
    num_heads, head_k_dim, head_v_dim, chunk_size
):
    """Fully JIT-compiled DeltaNet layer with all components."""
    batch_size, seq_len, d_model = hidden_states.shape
    
    # Q/K/V projections
    q = jnp.dot(hidden_states, q_proj_w.T)  # [b, l, key_dim]
    k = jnp.dot(hidden_states, k_proj_w.T)
    v = jnp.dot(hidden_states, v_proj_w.T)
    
    # Convolutions (depthwise conv1d)
    q = jax_conv1d(q, q_conv_w, q_conv_b, kernel_size=4)
    k = jax_conv1d(k, k_conv_w, k_conv_b, kernel_size=4)
    v = jax_conv1d(v, v_conv_w, v_conv_b, kernel_size=4)
    
    # SiLU activation
    q = jax.nn.silu(q)
    k = jax.nn.silu(k)
    v = jax.nn.silu(v)
    
    # Reshape for multi-head
    # Compute head dimensions from static projection weights, not traced arrays
    # q_proj_w has shape [key_dim, d_model], so key_dim = q_proj_w.shape[0]
    # Since key_dim = num_heads * head_k_dim, and num_heads is static:
    key_dim_static = q_proj_w.shape[0]  # This is static (from weight shape)
    value_dim_static = v_proj_w.shape[0]  # This is static
    head_k_dim_computed = key_dim_static // num_heads  # Static computation
    head_v_dim_computed = value_dim_static // num_heads  # Static computation
    
    # Now reshape using computed static dimensions
    q = q.reshape(batch_size, seq_len, num_heads, head_k_dim_computed)  # [b, l, h, d_k]
    k = k.reshape(batch_size, seq_len, num_heads, head_k_dim_computed)
    v = v.reshape(batch_size, seq_len, num_heads, head_v_dim_computed)
    
    # L2 normalization
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)
    
    # Beta projection
    beta = jax.nn.sigmoid(jnp.dot(hidden_states, b_proj_w.T))  # [b, l, h]
    
    # Hierarchical decay
    fast_decay = jax.nn.sigmoid(jnp.dot(hidden_states, fast_decay_proj_w.T) + fast_decay_bias[None, None, :])
    slow_decay = jax.nn.sigmoid(jnp.dot(hidden_states, slow_decay_proj_w.T) + slow_decay_bias[None, None, :])
    
    # Gates
    fast_gate = jax.nn.sigmoid(jnp.dot(hidden_states, fast_gate_proj_w.T))
    slow_gate = jax.nn.sigmoid(jnp.dot(hidden_states, slow_gate_proj_w.T))
    
    # Reshape to [b, h, l, d] for chunking
    q = jnp.transpose(q, (0, 2, 1, 3))  # [b, h, l, d_k]
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    beta = jnp.transpose(beta, (0, 2, 1))  # [b, h, l]
    fast_decay = jnp.transpose(fast_decay, (0, 2, 1))
    slow_decay = jnp.transpose(slow_decay, (0, 2, 1))
    fast_gate = jnp.transpose(fast_gate, (0, 2, 1))
    slow_gate = jnp.transpose(slow_gate, (0, 2, 1))
    
    # Compute token-level flux BEFORE chunking (Cython does this)
    # beta_expanded = beta.unsqueeze(-1)  # [b, h, l, 1]
    # v = v * beta_expanded  # Scale v by beta
    # k_beta = k * beta_expanded
    beta_expanded = beta[:, :, :, None]  # [b, h, l, 1]
    v_scaled = v * beta_expanded  # [b, h, l, d_v]
    k_beta_pre = k * beta_expanded  # [b, h, l, d_k]
    
    # Compute token flux BEFORE chunking
    from ._jax_core import compute_token_flux_jax
    token_flux_pre = compute_token_flux_jax(
        k_beta_pre, v_scaled,
        token_flux_proj_w0, token_flux_proj_b0,
        token_flux_proj_w2, token_flux_proj_b2
    )  # [b, h, l, 1]
    
    # Use scaled v for chunking (v is scaled by beta)
    v = v_scaled
    
    # Get actual dimensions from shapes
    actual_num_heads = q.shape[1]
    actual_head_k_dim = q.shape[-1]
    actual_head_v_dim = v.shape[-1]
    
    # Chunking: [b, h, l, d] -> [b, h, n, c, d]
    n_chunks = (seq_len + chunk_size - 1) // chunk_size
    pad_len = (n_chunks * chunk_size) - seq_len
    
    # Pad k_beta and token_flux BEFORE chunking (they're computed before padding)
    k_beta_padded = jnp.pad(k_beta_pre, ((0, 0), (0, 0), (0, pad_len), (0, 0))) if pad_len > 0 else k_beta_pre
    token_flux_padded = jnp.pad(token_flux_pre, ((0, 0), (0, 0), (0, pad_len), (0, 0))) if pad_len > 0 else token_flux_pre
    
    # Use jnp.where for conditional padding (JIT-friendly)
    q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0))) if pad_len > 0 else q
    k_padded = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0))) if pad_len > 0 else k
    v_padded = jnp.pad(v_scaled, ((0, 0), (0, 0), (0, pad_len), (0, 0))) if pad_len > 0 else v_scaled
    beta_padded = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_len))) if pad_len > 0 else beta
    
    # actual_chunk_size should be the padded sequence length divided by n_chunks
    # After padding, the sequence length is n_chunks * chunk_size
    # So each chunk has size chunk_size (or the padded length / n_chunks)
    actual_chunk_size = (seq_len + pad_len) // n_chunks if n_chunks > 0 else seq_len
    
    q_chunked = q_padded.reshape(batch_size, actual_num_heads, n_chunks, actual_chunk_size, actual_head_k_dim)
    k_chunked = k_padded.reshape(batch_size, actual_num_heads, n_chunks, actual_chunk_size, actual_head_k_dim)
    v_chunked = v_padded.reshape(batch_size, actual_num_heads, n_chunks, actual_chunk_size, actual_head_v_dim)
    beta_chunked = beta_padded.reshape(batch_size, actual_num_heads, n_chunks, actual_chunk_size)
    k_beta_chunked = k_beta_padded.reshape(batch_size, actual_num_heads, n_chunks, actual_chunk_size, actual_head_k_dim)
    
    # Chunk decay, gate, and token flux to match Cython
    fast_decay_padded = jnp.pad(fast_decay, ((0, 0), (0, 0), (0, pad_len))) if pad_len > 0 else fast_decay
    slow_decay_padded = jnp.pad(slow_decay, ((0, 0), (0, 0), (0, pad_len))) if pad_len > 0 else slow_decay
    fast_gate_padded = jnp.pad(fast_gate, ((0, 0), (0, 0), (0, pad_len))) if pad_len > 0 else fast_gate
    slow_gate_padded = jnp.pad(slow_gate, ((0, 0), (0, 0), (0, pad_len))) if pad_len > 0 else slow_gate
    
    fast_decay_chunked = fast_decay_padded.reshape(batch_size, actual_num_heads, n_chunks, actual_chunk_size)
    slow_decay_chunked = slow_decay_padded.reshape(batch_size, actual_num_heads, n_chunks, actual_chunk_size)
    fast_gate_chunked = fast_gate_padded.reshape(batch_size, actual_num_heads, n_chunks, actual_chunk_size, 1)
    slow_gate_chunked = slow_gate_padded.reshape(batch_size, actual_num_heads, n_chunks, actual_chunk_size, 1)
    token_flux_chunked = token_flux_padded.reshape(batch_size, actual_num_heads, n_chunks, actual_chunk_size, 1)
    
    # Call optimized hierarchical delta rule with enhanced resonance flux
    # Pass pre-computed token_flux_chunked and k_beta_chunked
    o = hierarchical_delta_rule_jax(
        q_chunked, k_chunked, v_chunked,
        beta_chunked,  # [b, h, n, c] - per-chunk, per-token
        fast_decay_chunked, slow_decay_chunked,  # [b, h, n, c]
        fast_gate_chunked, slow_gate_chunked,  # [b, h, n, c, 1]
        resonance_flux_fn=None,  # Use enhanced version instead
        resonance_flux_W_bilinear=resonance_flux_W_bilinear,
        resonance_flux_temp=resonance_flux_temp,
        resonance_flux_net_w0=resonance_flux_net_w0,
        resonance_flux_net_b0=resonance_flux_net_b0,
        resonance_flux_net_w2=resonance_flux_net_w2,
        resonance_flux_net_b2=resonance_flux_net_b2,
        token_flux_proj_w0=token_flux_proj_w0,
        token_flux_proj_b0=token_flux_proj_b0,
        token_flux_proj_w2=token_flux_proj_w2,
        token_flux_proj_b2=token_flux_proj_b2,
        token_flux_precomputed=token_flux_chunked,  # Pass pre-computed token flux
        k_beta_precomputed=k_beta_chunked,  # Pass pre-computed k_beta
    )  # [b, h, n, c, d_v]
    
    # Extract token flux weights for this layer (they're passed as layer-specific)
    # token_flux_proj_w0, etc. are already per-layer weights
    
    # Reshape back
    actual_head_v_dim = o.shape[-1]
    actual_num_heads = o.shape[1]
    o = o.reshape(batch_size, actual_num_heads, n_chunks * actual_chunk_size, actual_head_v_dim)
    o = o[:, :, :seq_len, :]  # Remove padding
    o = jnp.transpose(o, (0, 2, 1, 3))  # [b, l, h, d_v]
    # o is currently [b, l, h, d_v] from transpose above
    # Keep it in this shape for normalization
    
    # G projection from hidden_states
    g = jnp.dot(hidden_states, g_proj_w.T)  # [b, l, value_dim]
    # Reshape g to [b, l, h, d_v] to match o
    g = g.reshape(batch_size, seq_len, actual_num_heads, actual_head_v_dim)  # [b, l, h, d_v]
    
    # FusedRMSNormGated: o is [b, l, h, d_v], g is [b, l, h, d_v]
    # Cython: x * rsqrt(mean(x^2) + eps) * weight * sigmoid(g) where eps=1e-6
    # RMS norm: rsqrt = 1 / sqrt(mean(x^2) + eps)
    norm = jax.lax.rsqrt(jnp.mean(o ** 2, axis=-1, keepdims=True) + 1e-6)  # [b, l, h, 1] - use rsqrt with eps=1e-6 to match Cython
    # Apply: x * norm * weight * sigmoid(g) - matches Cython exactly
    o = o * norm * o_norm_w[None, None, None, :] * jax.nn.sigmoid(g)  # [b, l, h, d_v]
    
    # Reshape back to [b, l, value_dim]
    o = o.reshape(batch_size, seq_len, actual_num_heads * actual_head_v_dim)  # [b, l, value_dim]
    
    # Output projection
    output = jnp.dot(o, o_proj_w.T)  # [b, l, d_model]
    
    return output


def jax_interpolate_positions(position_emb_weight, seq_length, original_max_pos=512, license_limit=0x2000):
    """
    JAX implementation of position embedding interpolation (matches Cython _secrets.interpolate_positions).
    For sequences longer than original_max_pos, linearly interpolate between position embeddings.
    """
    if seq_length > license_limit:
        raise ValueError("CONTEXT_LIMIT_EXCEEDED")
    
    if seq_length <= original_max_pos:
        # No interpolation needed
        position_ids = jnp.arange(seq_length)
        return position_emb_weight[position_ids]  # [seq_length, d_model]
    
    # Interpolate for sequences longer than original_max_pos
    scale_factor = (original_max_pos - 1) / (seq_length - 1)
    
    def interpolate_pos(i):
        original_pos = i * scale_factor
        lower_pos = jnp.int32(original_pos)
        upper_pos = jnp.minimum(lower_pos + 1, original_max_pos - 1)
        weight = original_pos - lower_pos
        
        lower_emb = position_emb_weight[lower_pos]  # [d_model]
        upper_emb = position_emb_weight[upper_pos]  # [d_model]
        interp_emb = (1 - weight) * lower_emb + weight * upper_emb
        return interp_emb
    
    # Vectorized interpolation
    positions = jnp.arange(seq_length)
    position_embeddings = jax.vmap(interpolate_pos)(positions)  # [seq_length, d_model]
    
    return position_embeddings


@jax.jit(static_argnames=['d_model', 'num_layers', 'num_heads', 'head_k_dim', 'head_v_dim', 'layer_norm_eps', 'original_max_pos', 'license_limit'])
def jax_forward_pass_optimized(params, input_ids, attention_mask, token_type_ids, position_emb_weight,
                               d_model, num_layers, num_heads, head_k_dim, head_v_dim, layer_norm_eps,
                               original_max_pos=512, license_limit=0x2000):
    """Fully JIT-compiled forward pass."""
    
    batch_size, seq_length = input_ids.shape
    
    # Handle token_type_ids - if None or wrong shape, use zeros
    if token_type_ids is None:
        token_type_ids = jnp.zeros_like(input_ids)
    elif token_type_ids.shape[1] != seq_length:
        # Pad or truncate to match sequence length
        if token_type_ids.shape[1] < seq_length:
            token_type_ids = jnp.pad(token_type_ids, ((0, 0), (0, seq_length - token_type_ids.shape[1])), mode='constant')
        else:
            token_type_ids = token_type_ids[:, :seq_length]
    
    # Embeddings
    inputs_embeds = params['embeddings']['word'][input_ids]  # [b, l, d_model]
    token_type_embeddings = params['embeddings']['token_type'][token_type_ids]  # [b, l, d_model]
    
    # Position embeddings with interpolation (matches Cython _secrets.interpolate_positions)
    if position_emb_weight is not None:
        # Use interpolation for sequences > original_max_pos
        position_emb = jax_interpolate_positions(position_emb_weight, seq_length, original_max_pos, license_limit)
        position_embeddings = position_emb[None, :, :]  # [1, l, d_model]
        position_embeddings = jnp.repeat(position_embeddings, batch_size, axis=0)  # [b, l, d_model]
    else:
        position_ids = jnp.arange(seq_length)[None, :]  # [1, l]
        position_embeddings = params['embeddings']['position'][position_ids]  # [1, l, d_model]
        position_embeddings = jnp.repeat(position_embeddings, batch_size, axis=0)  # [b, l, d_model]
    
    embeddings = inputs_embeds + token_type_embeddings + position_embeddings
    
    # LayerNorm (Cython: self.model.embeddings['LayerNorm'](embeddings))
    # LayerNorm (Cython uses unbiased=False for variance, which is ddof=0 in JAX)
    mean = jnp.mean(embeddings, axis=-1, keepdims=True)
    variance = jnp.var(embeddings, axis=-1, keepdims=True, ddof=0)  # ddof=0 matches PyTorch unbiased=False
    embeddings = (embeddings - mean) / jnp.sqrt(variance + layer_norm_eps)
    embeddings = embeddings * params['embeddings']['ln_weight'][None, None, :] + params['embeddings']['ln_bias'][None, None, :]
    
    # Dropout (Cython: self.model.embeddings['dropout'](embeddings))
    # In eval mode, dropout is disabled, but we need to match Cython exactly
    # Cython applies dropout even in eval mode if training flag is set
    # For now, skip dropout to match eval mode behavior
    x = embeddings
    
    # Process through layers using lax.fori_loop (JIT-friendly)
    def layer_body(i, x):
        # DeltaNet attention
        residual = x
        x_attn = jax_deltanet_layer_optimized(
            params['deltanet']['q_proj'][i],
            params['deltanet']['k_proj'][i],
            params['deltanet']['v_proj'][i],
            params['deltanet']['b_proj'][i],
            params['deltanet']['fast_decay_proj'][i],
            params['deltanet']['fast_decay_bias'][i],
            params['deltanet']['slow_decay_proj'][i],
            params['deltanet']['slow_decay_bias'][i],
            params['deltanet']['fast_gate_proj'][i],
            params['deltanet']['slow_gate_proj'][i],
            params['deltanet']['g_proj'][i],
            params['deltanet']['o_proj'][i],
            params['deltanet']['o_norm_weight'][i],
            params['deltanet']['o_norm_gate'][i],
            params['deltanet']['q_conv_weight'][i],
            params['deltanet']['q_conv_bias'][i],
            params['deltanet']['k_conv_weight'][i],
            params['deltanet']['k_conv_bias'][i],
            params['deltanet']['v_conv_weight'][i],
            params['deltanet']['v_conv_bias'][i],
            params['deltanet']['resonance_flux_W_bilinear'][i],
            params['deltanet']['resonance_flux_temp'][i],
            params['deltanet']['resonance_flux_net_w0'][i],
            params['deltanet']['resonance_flux_net_b0'][i],
            params['deltanet']['resonance_flux_net_w2'][i],
            params['deltanet']['resonance_flux_net_b2'][i],
            params['deltanet']['token_flux_proj_w0'][i],
            params['deltanet']['token_flux_proj_b0'][i],
            params['deltanet']['token_flux_proj_w2'][i],
            params['deltanet']['token_flux_proj_b2'][i],
            x,
            num_heads, head_k_dim, head_v_dim, 64
        )
        
        # LayerNorm (Cython: self.model.deltanet_norms[i](residual + x_attn))
        # CRITICAL FIX: Cython adds residual FIRST, then applies LayerNorm
        # Order: residual + x_attn -> LayerNorm(result)
        x_attn_sum = residual + x_attn
        mean = jnp.mean(x_attn_sum, axis=-1, keepdims=True)
        variance = jnp.var(x_attn_sum, axis=-1, keepdims=True, ddof=0)  # ddof=0 matches PyTorch unbiased=False
        x_attn_sum = (x_attn_sum - mean) / jnp.sqrt(variance + layer_norm_eps)
        x = x_attn_sum * params['norms']['attn']['weight'][i][None, None, :] + params['norms']['attn']['bias'][i][None, None, :]
        
        # FFN (Cython: intermediate_act_fn + F.gelu)
        residual = x
        x_ffn = jnp.dot(x, params['ffn']['intermediate']['weight'][i].T) + params['ffn']['intermediate']['bias'][i][None, :]
        # Cython: intermediate_act_fn is F.gelu, then applies F.gelu again
        x_ffn = jax.nn.gelu(x_ffn)  # First GELU (intermediate_act_fn)
        x_ffn = jax.nn.gelu(x_ffn)  # Second GELU (redundant but matches Cython)
        x_ffn = jnp.dot(x_ffn, params['ffn']['output']['weight'][i].T) + params['ffn']['output']['bias'][i][None, :]
        # Cython: dropout is applied here, but in eval mode it's disabled
        # x_ffn = dropout(x_ffn)  # Skip in eval mode
        
        # LayerNorm (Cython: self.model.output_denses[i]['LayerNorm'](residual + x_ffn))
        # CRITICAL FIX: Cython applies LayerNorm to (residual + x_ffn), not just x_ffn
        x_ffn_sum = residual + x_ffn
        mean = jnp.mean(x_ffn_sum, axis=-1, keepdims=True)
        variance = jnp.var(x_ffn_sum, axis=-1, keepdims=True, ddof=0)  # ddof=0 matches PyTorch unbiased=False
        x_ffn_sum = (x_ffn_sum - mean) / jnp.sqrt(variance + layer_norm_eps)
        x = x_ffn_sum * params['norms']['ffn']['weight'][i][None, None, :] + params['norms']['ffn']['bias'][i][None, None, :]
        
        return x  # CRITICAL: Must return x for lax.fori_loop
    
    # CRITICAL FIX: Actually call lax.fori_loop to process all layers
    x = lax.fori_loop(0, num_layers, layer_body, x)
    return {'last_hidden_state': x}


@jax.jit(static_argnames=['d_model', 'num_layers', 'num_heads', 'head_k_dim', 'head_v_dim', 'layer_norm_eps', 'original_max_pos', 'license_limit'])
def jax_get_sentence_embeddings_optimized(params, input_ids, attention_mask,
                                           d_model, num_layers, num_heads, head_k_dim, head_v_dim, layer_norm_eps,
                                           position_emb_weight=None, original_max_pos=512, license_limit=0x2000):
    """Fully JIT-compiled sentence embeddings."""
    if attention_mask is None:
        attention_mask = jnp.ones_like(input_ids)
    
    outputs = jax_forward_pass_optimized(params, input_ids, attention_mask, None, position_emb_weight,
                                         d_model, num_layers, num_heads, head_k_dim, head_v_dim, layer_norm_eps,
                                         original_max_pos, license_limit)
    last_hidden_state = outputs['last_hidden_state']
    
    # Mean pooling
    input_mask_expanded = attention_mask[:, :, None]  # [b, l, 1]
    embeddings = jnp.sum(last_hidden_state * input_mask_expanded, axis=1) / jnp.clip(
        jnp.sum(input_mask_expanded, axis=1), a_min=1e-9
    )
    
    # L2 normalize
    norm = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    embeddings = embeddings / jnp.clip(norm, a_min=1e-9)
    
    return embeddings

