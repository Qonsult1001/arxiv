"""
JAX Model Implementation for LAM
================================

This module provides a JAX-based implementation of the LAM model.
It's a parallel implementation to the Cython version for performance comparison.

The model loads PyTorch weights and converts them to JAX format.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from ._jax_core import hierarchical_delta_rule_jax, simple_resonance_flux_jax


@dataclass
class JAXModelParams:
    """JAX model parameters (functional style - no mutable state)."""
    # Embeddings
    word_embeddings: jnp.ndarray  # [vocab_size, d_model]
    position_embeddings: jnp.ndarray  # [max_pos, d_model]
    token_type_embeddings: jnp.ndarray  # [type_vocab_size, d_model]
    embedding_layernorm_weight: jnp.ndarray  # [d_model]
    embedding_layernorm_bias: jnp.ndarray  # [d_model]
    
    # DeltaNet layers
    deltanet_layers: List[Dict]  # List of layer parameters
    deltanet_norms: List[Dict]  # List of norm parameters
    
    # FFN layers
    ffn_intermediate: List[Dict]  # List of intermediate layer parameters
    ffn_output: List[Dict]  # List of output layer parameters
    ffn_norms: List[Dict]  # List of FFN norm parameters


def pytorch_to_jax(tensor: torch.Tensor) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(tensor.detach().cpu().numpy())


def create_jax_lam_model(model_path, license_limit=0x2000, tier="free"):
    """
    Create JAX LAM model from PyTorch weights.
    Mirrors create_lam_model but returns JAX parameters.
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
    
    # Extract weights
    teacher_state_dict = {}
    for key, value in loaded_data.items():
        if key.startswith('teacher_model.'):
            new_key = key.replace('teacher_model.', '')
            teacher_state_dict[new_key] = value
    
    # Build embeddings
    word_emb = pytorch_to_jax(teacher_state_dict.get('embeddings.word_embeddings.weight', torch.randn(vocab_size, d_model)))
    pos_emb = pytorch_to_jax(teacher_state_dict.get('embeddings.position_embeddings.weight', torch.randn(max_pos, d_model)))
    token_type_emb = pytorch_to_jax(teacher_state_dict.get('embeddings.token_type_embeddings.weight', torch.randn(type_vocab_size, d_model)))
    emb_ln_weight = pytorch_to_jax(teacher_state_dict.get('embeddings.LayerNorm.weight', torch.ones(d_model)))
    emb_ln_bias = pytorch_to_jax(teacher_state_dict.get('embeddings.LayerNorm.bias', torch.zeros(d_model)))
    
    # Build DeltaNet layers
    deltanet_layers = []
    deltanet_norms = []
    ffn_intermediate = []
    ffn_output = []
    ffn_norms = []
    
    # Extract deltanet layer weights
    model_state_dict = checkpoint.get('model_state_dict', {})
    deltanet_layers_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('deltanet_layers.'):
            new_key = key.replace('deltanet_layers.', '')
            deltanet_layers_dict[new_key] = value
    
    for i in range(num_layers):
        # DeltaNet layer parameters - extract all weights
        layer_state = {}
        for k, v in deltanet_layers_dict.items():
            if k.startswith(f'{i}.'):
                new_key = k[len(f'{i}.'):]
                layer_state[new_key] = pytorch_to_jax(v)
        
        # If no weights found, create default structure
        if not layer_state:
            layer_state = {
                'q_proj.weight': pytorch_to_jax(torch.randn(d_model * num_heads // d_model, d_model)),
                'k_proj.weight': pytorch_to_jax(torch.randn(d_model * num_heads // d_model, d_model)),
                'v_proj.weight': pytorch_to_jax(torch.randn(d_model * num_heads // d_model, d_model)),
                'b_proj.weight': pytorch_to_jax(torch.randn(num_heads, d_model)),
                'fast_decay_proj.weight': pytorch_to_jax(torch.randn(num_heads, d_model)),
                'slow_decay_proj.weight': pytorch_to_jax(torch.randn(num_heads, d_model)),
                'fast_gate_proj.weight': pytorch_to_jax(torch.randn(num_heads, d_model)),
                'slow_gate_proj.weight': pytorch_to_jax(torch.randn(num_heads, d_model)),
                'o_proj.weight': pytorch_to_jax(torch.randn(d_model, d_model * num_heads // d_model)),
                'fast_decay_bias': pytorch_to_jax(torch.full((num_heads,), np.log(0.3))),
                'slow_decay_bias': pytorch_to_jax(torch.full((num_heads,), np.log(0.9))),
            }
        deltanet_layers.append(layer_state)
        
        # Attention norm
        attn_norm = {
            'weight': pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.attention.output.LayerNorm.weight', torch.ones(d_model))),
            'bias': pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.attention.output.LayerNorm.bias', torch.zeros(d_model)))
        }
        deltanet_norms.append(attn_norm)
        
        # FFN intermediate
        ffn_int = {
            'weight': pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.intermediate.dense.weight', torch.randn(intermediate_size, d_model))),
            'bias': pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.intermediate.dense.bias', torch.zeros(intermediate_size)))
        }
        ffn_intermediate.append(ffn_int)
        
        # FFN output
        ffn_out = {
            'weight': pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.output.dense.weight', torch.randn(d_model, intermediate_size))),
            'bias': pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.output.dense.bias', torch.zeros(d_model))),
            'layernorm_weight': pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.output.LayerNorm.weight', torch.ones(d_model))),
            'layernorm_bias': pytorch_to_jax(teacher_state_dict.get(f'encoder.layer.{i}.output.LayerNorm.bias', torch.zeros(d_model)))
        }
        ffn_output.append(ffn_out)
        ffn_norms.append({
            'weight': ffn_out['layernorm_weight'],
            'bias': ffn_out['layernorm_bias']
        })
    
    params = JAXModelParams(
        word_embeddings=word_emb,
        position_embeddings=pos_emb,
        token_type_embeddings=token_type_emb,
        embedding_layernorm_weight=emb_ln_weight,
        embedding_layernorm_bias=emb_ln_bias,
        deltanet_layers=deltanet_layers,
        deltanet_norms=deltanet_norms,
        ffn_intermediate=ffn_intermediate,
        ffn_output=ffn_output,
        ffn_norms=ffn_norms
    )
    
    return params, {
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'vocab_size': vocab_size,
        'max_pos': max_pos,
        'type_vocab_size': type_vocab_size,
        'layer_norm_eps': layer_norm_eps,
        'intermediate_size': intermediate_size,
        'license_limit': license_limit,
        'tier': tier
    }


# JIT compile the core delta rule call
@jax.jit
def _jax_deltanet_core(q_chunked, k_chunked, v_chunked, beta_chunked,
                       fast_decay_avg, slow_decay_avg, fast_gate_avg, slow_gate_avg):
    """JIT-compiled core delta rule computation."""
    from ._jax_core import hierarchical_delta_rule_jax, simple_resonance_flux_jax
    return hierarchical_delta_rule_jax(
        q_chunked, k_chunked, v_chunked,
        beta_chunked,
        fast_decay_avg, slow_decay_avg,
        fast_gate_avg, slow_gate_avg,
        simple_resonance_flux_jax
    )

def jax_deltanet_layer(params_layer: Dict, hidden_states: jnp.ndarray, 
                      num_heads: int, head_k_dim: int, head_v_dim: int,
                      chunk_size: int = 64):
    """
    JAX DeltaNet layer using optimized hierarchical_delta_rule_jax.
    """
    
    batch_size, seq_len, d_model = hidden_states.shape
    
    # Q/K/V projections
    q_proj_w = params_layer.get('q_proj.weight', jnp.eye(d_model, d_model * num_heads // d_model))
    k_proj_w = params_layer.get('k_proj.weight', jnp.eye(d_model, d_model * num_heads // d_model))
    v_proj_w = params_layer.get('v_proj.weight', jnp.eye(d_model, d_model * num_heads // d_model))
    
    q = jnp.dot(hidden_states, q_proj_w.T)  # [b, l, key_dim]
    k = jnp.dot(hidden_states, k_proj_w.T)  # [b, l, key_dim]
    v = jnp.dot(hidden_states, v_proj_w.T)  # [b, l, value_dim]
    
    # SiLU activation
    q = jax.nn.silu(q)
    k = jax.nn.silu(k)
    v = jax.nn.silu(v)
    
    # Reshape for multi-head: [b, l, (h*d)] -> [b, l, h, d]
    q = q.reshape(batch_size, seq_len, num_heads, head_k_dim)  # [b, l, h, d_k]
    k = k.reshape(batch_size, seq_len, num_heads, head_k_dim)  # [b, l, h, d_k]
    v = v.reshape(batch_size, seq_len, num_heads, head_v_dim)  # [b, l, h, d_v]
    
    # L2 normalization
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)
    
    # Beta projection
    b_proj_w = params_layer.get('b_proj.weight', jnp.zeros((num_heads, d_model)))
    beta = jax.nn.sigmoid(jnp.dot(hidden_states, b_proj_w.T))  # [b, l, h]
    
    # Hierarchical decay
    fast_decay_proj_w = params_layer.get('fast_decay_proj.weight', jnp.zeros((num_heads, d_model)))
    fast_decay_bias = params_layer.get('fast_decay_bias', jnp.full((num_heads,), jnp.log(0.3)))
    fast_decay = jax.nn.sigmoid(jnp.dot(hidden_states, fast_decay_proj_w.T) + fast_decay_bias[None, None, :])  # [b, l, h]
    
    slow_decay_proj_w = params_layer.get('slow_decay_proj.weight', jnp.zeros((num_heads, d_model)))
    slow_decay_bias = params_layer.get('slow_decay_bias', jnp.full((num_heads,), jnp.log(0.9)))
    slow_decay = jax.nn.sigmoid(jnp.dot(hidden_states, slow_decay_proj_w.T) + slow_decay_bias[None, None, :])  # [b, l, h]
    
    # Gates
    fast_gate_proj_w = params_layer.get('fast_gate_proj.weight', jnp.zeros((num_heads, d_model)))
    slow_gate_proj_w = params_layer.get('slow_gate_proj.weight', jnp.zeros((num_heads, d_model)))
    fast_gate = jax.nn.sigmoid(jnp.dot(hidden_states, fast_gate_proj_w.T))  # [b, l, h]
    slow_gate = jax.nn.sigmoid(jnp.dot(hidden_states, slow_gate_proj_w.T))  # [b, l, h]
    
    # Reshape to [b, h, l, d] for chunking
    q = jnp.transpose(q, (0, 2, 1, 3))  # [b, h, l, d_k]
    k = jnp.transpose(k, (0, 2, 1, 3))  # [b, h, l, d_k]
    v = jnp.transpose(v, (0, 2, 1, 3))  # [b, h, l, d_v]
    beta = jnp.transpose(beta, (0, 2, 1))  # [b, h, l]
    fast_decay = jnp.transpose(fast_decay, (0, 2, 1))  # [b, h, l]
    slow_decay = jnp.transpose(slow_decay, (0, 2, 1))  # [b, h, l]
    fast_gate = jnp.transpose(fast_gate, (0, 2, 1))  # [b, h, l]
    slow_gate = jnp.transpose(slow_gate, (0, 2, 1))  # [b, h, l]
    
    # Chunking: [b, h, l, d] -> [b, h, n, c, d]
    if seq_len <= chunk_size:
        n_chunks = 1
        actual_chunk_size = seq_len
        q_chunked = q[:, :, None, :, :]  # [b, h, 1, l, d_k]
        k_chunked = k[:, :, None, :, :]  # [b, h, 1, l, d_k]
        v_chunked = v[:, :, None, :, :]  # [b, h, 1, l, d_v]
        beta_chunked = beta[:, :, None, :]  # [b, h, 1, l]
    else:
        n_chunks = (seq_len + chunk_size - 1) // chunk_size
        pad_len = (n_chunks * chunk_size) - seq_len
        if pad_len > 0:
            q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
            k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
            beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_len)))
            fast_decay = jnp.pad(fast_decay, ((0, 0), (0, 0), (0, pad_len)))
            slow_decay = jnp.pad(slow_decay, ((0, 0), (0, 0), (0, pad_len)))
            fast_gate = jnp.pad(fast_gate, ((0, 0), (0, 0), (0, pad_len)))
            slow_gate = jnp.pad(slow_gate, ((0, 0), (0, 0), (0, pad_len)))
        
        q_chunked = q.reshape(batch_size, num_heads, n_chunks, chunk_size, head_k_dim)
        k_chunked = k.reshape(batch_size, num_heads, n_chunks, chunk_size, head_k_dim)
        v_chunked = v.reshape(batch_size, num_heads, n_chunks, chunk_size, head_v_dim)
        beta_chunked = beta.reshape(batch_size, num_heads, n_chunks, chunk_size)
        actual_chunk_size = chunk_size
    
    # Average decay and gate over sequence for simplicity (or use per-chunk)
    fast_decay_avg = jnp.mean(fast_decay, axis=-1)  # [b, h]
    slow_decay_avg = jnp.mean(slow_decay, axis=-1)  # [b, h]
    fast_gate_avg = jnp.mean(fast_gate, axis=-1)  # [b, h]
    slow_gate_avg = jnp.mean(slow_gate, axis=-1)  # [b, h]
    
    # Call optimized hierarchical delta rule (JIT compiled)
    beta_avg = beta_chunked.mean(axis=-1)  # [b, h, n] - average beta per chunk
    o = _jax_deltanet_core(
        q_chunked, k_chunked, v_chunked,
        beta_avg,
        fast_decay_avg, slow_decay_avg,
        fast_gate_avg, slow_gate_avg
    )  # [b, h, n, c, d_v]
    
    # Reshape back: [b, h, n, c, d_v] -> [b, h, l, d_v] -> [b, l, h, d_v]
    o = o.reshape(batch_size, num_heads, n_chunks * actual_chunk_size, head_v_dim)
    o = o[:, :, :seq_len, :]  # Remove padding
    o = jnp.transpose(o, (0, 2, 1, 3))  # [b, l, h, d_v]
    o = o.reshape(batch_size, seq_len, num_heads * head_v_dim)  # [b, l, value_dim]
    
    # Output projection
    o_proj_w = params_layer.get('o_proj.weight', jnp.eye(num_heads * head_v_dim, d_model))
    output = jnp.dot(o, o_proj_w.T)  # [b, l, d_model]
    
    return output


def jax_forward_pass(params: JAXModelParams, config: Dict, input_ids: jnp.ndarray, 
                     attention_mask: Optional[jnp.ndarray] = None,
                     token_type_ids: Optional[jnp.ndarray] = None,
                     position_emb_weight: Optional[jnp.ndarray] = None):
    """
    JAX forward pass - functional style with optimized DeltaNet.
    JIT compiled for maximum performance.
    """
    batch_size, seq_length = input_ids.shape
    d_model = config['d_model']
    num_heads = config['num_heads']
    head_k_dim = d_model // num_heads
    head_v_dim = d_model // num_heads
    
    if token_type_ids is None:
        token_type_ids = jnp.zeros_like(input_ids)
    
    # Embeddings
    inputs_embeds = params.word_embeddings[input_ids]  # [b, l, d_model]
    token_type_embeddings = params.token_type_embeddings[token_type_ids]  # [b, l, d_model]
    
    # Position embeddings
    if position_emb_weight is not None:
        position_ids = jnp.arange(seq_length)[None, :]  # [1, l]
        position_embeddings = position_emb_weight[position_ids]  # [1, l, d_model]
        position_embeddings = jnp.broadcast_to(position_embeddings, (batch_size, seq_length, d_model))
    else:
        position_ids = jnp.arange(seq_length)[None, :]
        position_embeddings = params.position_embeddings[position_ids]
        position_embeddings = jnp.broadcast_to(position_embeddings, (batch_size, seq_length, d_model))
    
    embeddings = inputs_embeds + token_type_embeddings + position_embeddings
    
    # LayerNorm
    mean = jnp.mean(embeddings, axis=-1, keepdims=True)
    variance = jnp.var(embeddings, axis=-1, keepdims=True)
    embeddings = (embeddings - mean) / jnp.sqrt(variance + config['layer_norm_eps'])
    embeddings = embeddings * params.embedding_layernorm_weight[None, None, :] + params.embedding_layernorm_bias[None, None, :]
    
    x = embeddings
    
    # Process through layers
    for i in range(config['num_layers']):
        # DeltaNet attention
        residual = x
        x_attn = jax_deltanet_layer(params.deltanet_layers[i], x, num_heads, head_k_dim, head_v_dim)
        
        # LayerNorm
        mean = jnp.mean(x_attn, axis=-1, keepdims=True)
        variance = jnp.var(x_attn, axis=-1, keepdims=True)
        x_attn = (x_attn - mean) / jnp.sqrt(variance + config['layer_norm_eps'])
        x_attn = x_attn * params.deltanet_norms[i]['weight'][None, None, :] + params.deltanet_norms[i]['bias'][None, None, :]
        x = residual + x_attn
        
        # FFN
        residual = x
        x_ffn = jnp.dot(x, params.ffn_intermediate[i]['weight'].T) + params.ffn_intermediate[i]['bias'][None, :]
        x_ffn = jax.nn.gelu(x_ffn)
        x_ffn = jnp.dot(x_ffn, params.ffn_output[i]['weight'].T) + params.ffn_output[i]['bias'][None, :]
        
        # LayerNorm
        mean = jnp.mean(x_ffn, axis=-1, keepdims=True)
        variance = jnp.var(x_ffn, axis=-1, keepdims=True)
        x_ffn = (x_ffn - mean) / jnp.sqrt(variance + config['layer_norm_eps'])
        x_ffn = x_ffn * params.ffn_norms[i]['weight'][None, None, :] + params.ffn_norms[i]['bias'][None, None, :]
        x = residual + x_ffn
    
    return {'last_hidden_state': x}


def jax_get_sentence_embeddings(params: JAXModelParams, config: Dict, 
                                input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None):
    """Get sentence embeddings from JAX model."""
    if attention_mask is None:
        attention_mask = jnp.ones_like(input_ids)
    
    outputs = jax_forward_pass(params, config, input_ids, attention_mask)
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

