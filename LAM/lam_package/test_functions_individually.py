#!/usr/bin/env python3
"""
Systematic function-by-function comparison with Cython.
Test each function in isolation until it matches exactly.
"""

import torch
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'build'))

from lam._jax_core import (
    compute_token_flux_jax,
    enhanced_resonance_flux_jax,
    hierarchical_delta_rule_jax
)
from lam._jax_model_optimized import jax_conv1d
from lam import LAM

def compare_arrays(a1, a2, name, threshold=1e-5):
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
    if not match:
        print(f"    First 5 values - Cython: {a1_flat[:5]}")
        print(f"    First 5 values - JAX:    {a2_flat[:5]}")
    print()
    
    return match, max_diff, cos_sim

def test_1_token_flux():
    """Test 1: Token flux computation"""
    print("="*80)
    print("TEST 1: Token Flux Computation")
    print("="*80)
    
    model_path = "../LAM-base-v1"
    model = LAM(model_path, backend='cython')
    
    # Get weights for layer 0
    resonance_flux = model._model.deltanet_layers[0].resonance_flux
    
    # Get actual dimensions from model
    head_k_dim = model._model.deltanet_layers[0].head_k_dim
    head_v_dim = model._model.deltanet_layers[0].head_v_dim
    
    # Create test data - use fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create test data - Cython expects [b, h, l, d_k] and [b, h, l, d_v]
    b, h, l = 1, 4, 3
    d_k = head_k_dim
    d_v = head_v_dim
    k = torch.randn(b, h, l, d_k, device=model.device)
    v = torch.randn(b, h, l, d_v, device=model.device)
    
    # Convert to numpy for JAX (before JAX computation to ensure same values)
    k_np = k.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    
    # Cython - expects [b, h, l, d] format
    token_flux_cython = resonance_flux.compute_token_flux(k, v)  # [b, h, l, 1]
    
    # JAX
    token_flux_proj_w0 = jnp.array(resonance_flux.token_flux_proj[0].weight.detach().cpu().numpy())
    token_flux_proj_b0 = jnp.array(resonance_flux.token_flux_proj[0].bias.detach().cpu().numpy())
    token_flux_proj_w2 = jnp.array(resonance_flux.token_flux_proj[2].weight.detach().cpu().numpy())
    token_flux_proj_b2 = jnp.array(resonance_flux.token_flux_proj[2].bias.detach().cpu().numpy())
    
    k_jax = jnp.array(k_np)
    v_jax = jnp.array(v_np)
    
    # JAX expects [b, h, l, d] - same as input
    token_flux_jax = compute_token_flux_jax(
        k_jax, v_jax,
        token_flux_proj_w0, token_flux_proj_b0,
        token_flux_proj_w2, token_flux_proj_b2
    )  # [b, h, l, 1]
    
    match, max_diff, cos_sim = compare_arrays(
        token_flux_cython, token_flux_jax, "Token Flux", threshold=1e-4
    )
    
    return match

def test_2_enhanced_resonance_flux():
    """Test 2: Enhanced resonance flux (4D case)"""
    print("="*80)
    print("TEST 2: Enhanced Resonance Flux (4D)")
    print("="*80)
    
    model_path = "../LAM-base-v1"
    model = LAM(model_path, backend='cython')
    
    # Get weights for layer 0
    resonance_flux = model._model.deltanet_layers[0].resonance_flux
    
    # Create test data (4D: [b, h, c, d])
    # Get actual dimensions from model
    head_k_dim = model._model.deltanet_layers[0].head_k_dim
    head_v_dim = model._model.deltanet_layers[0].head_v_dim
    num_heads = model._model.deltanet_layers[0].num_heads
    b, c = 1, 3
    h = num_heads
    d_k = head_k_dim
    d_v = head_v_dim
    k = torch.randn(b, h, c, d_k, device=model.device)
    u = torch.randn(b, h, c, d_v, device=model.device)
    
    # Cython
    psi_cython = resonance_flux(k, u)  # [b, h]
    
    # JAX
    W_bilinear = jnp.array(resonance_flux.W_bilinear.detach().cpu().numpy())
    temp = jnp.array(resonance_flux.temp.detach().cpu().numpy())
    flux_net_w0 = jnp.array(resonance_flux.flux_net[0].weight.detach().cpu().numpy())
    flux_net_b0 = jnp.array(resonance_flux.flux_net[0].bias.detach().cpu().numpy())
    flux_net_w2 = jnp.array(resonance_flux.flux_net[2].weight.detach().cpu().numpy())
    flux_net_b2 = jnp.array(resonance_flux.flux_net[2].bias.detach().cpu().numpy())
    
    k_jax = jnp.array(k.detach().cpu().numpy())
    u_jax = jnp.array(u.detach().cpu().numpy())
    
    psi_jax = enhanced_resonance_flux_jax(
        k_jax, u_jax,
        W_bilinear, temp,
        flux_net_w0, flux_net_b0,
        flux_net_w2, flux_net_b2
    )
    
    match, max_diff, cos_sim = compare_arrays(
        psi_cython, psi_jax, "Enhanced Resonance Flux (4D)", threshold=2e-5
    )
    
    return match

def test_3_enhanced_resonance_flux_5d():
    """Test 3: Enhanced resonance flux (5D case)"""
    print("="*80)
    print("TEST 3: Enhanced Resonance Flux (5D)")
    print("="*80)
    
    model_path = "../LAM-base-v1"
    model = LAM(model_path, backend='cython')
    
    # Get weights for layer 0
    resonance_flux = model._model.deltanet_layers[0].resonance_flux
    
    # Create test data (5D: [b, h, n, c, d])
    # Get actual dimensions from model
    head_k_dim = model._model.deltanet_layers[0].head_k_dim
    head_v_dim = model._model.deltanet_layers[0].head_v_dim
    num_heads = model._model.deltanet_layers[0].num_heads
    b, n, c = 1, 2, 3
    h = num_heads
    d_k = head_k_dim
    d_v = head_v_dim
    k = torch.randn(b, h, n, c, d_k, device=model.device)
    u = torch.randn(b, h, n, c, d_v, device=model.device)
    
    # Cython
    psi_cython = resonance_flux(k, u)  # [b, h, n]
    
    # JAX
    W_bilinear = jnp.array(resonance_flux.W_bilinear.detach().cpu().numpy())
    temp = jnp.array(resonance_flux.temp.detach().cpu().numpy())
    flux_net_w0 = jnp.array(resonance_flux.flux_net[0].weight.detach().cpu().numpy())
    flux_net_b0 = jnp.array(resonance_flux.flux_net[0].bias.detach().cpu().numpy())
    flux_net_w2 = jnp.array(resonance_flux.flux_net[2].weight.detach().cpu().numpy())
    flux_net_b2 = jnp.array(resonance_flux.flux_net[2].bias.detach().cpu().numpy())
    
    k_jax = jnp.array(k.detach().cpu().numpy())
    u_jax = jnp.array(u.detach().cpu().numpy())
    
    psi_jax = enhanced_resonance_flux_jax(
        k_jax, u_jax,
        W_bilinear, temp,
        flux_net_w0, flux_net_b0,
        flux_net_w2, flux_net_b2
    )
    
    match, max_diff, cos_sim = compare_arrays(
        psi_cython, psi_jax, "Enhanced Resonance Flux (5D)", threshold=2e-5
    )
    
    return match

def test_4_short_convolution():
    """Test 4: Short convolution"""
    print("="*80)
    print("TEST 4: Short Convolution")
    print("="*80)
    
    model_path = "../LAM-base-v1"
    model = LAM(model_path, backend='cython')
    
    # Get conv weights for layer 0
    q_conv1d = model._model.deltanet_layers[0].q_conv1d
    
    # Create test data - use actual model dimensions
    # First get q from a forward pass to match the actual usage
    test_sentence = "Hello"
    input_ids = model.tokenizer.encode(test_sentence)
    input_ids_tensor = torch.tensor([input_ids.ids], dtype=torch.long, device=model.device)
    
    model._model.eval()
    with torch.no_grad():
        # Get to first layer
        x = model._model.embeddings['word_embeddings'](input_ids_tensor)
        token_type_ids = torch.zeros_like(input_ids_tensor)
        x = x + model._model.embeddings['token_type_embeddings'](token_type_ids)
        position_ids = torch.arange(input_ids_tensor.shape[1], dtype=torch.long, device=input_ids_tensor.device).unsqueeze(0)
        x = x + model._model.embeddings['position_embeddings'](position_ids)
        x = model._model.embeddings['LayerNorm'](x)
        x = model._model.embeddings['dropout'](x)
        
        # Get q projection
        layer = model._model.deltanet_layers[0]
        q = layer.q_proj(x)  # [b, l, key_dim]
        
        # Cython conv - ShortConvolution.forward does transpose internally
        # It expects [b, l, c] and does x.transpose(1, 2) internally
        x_cython, _ = q_conv1d(q)  # Input: [b, l, key_dim], output: [b, l, key_dim]
    
    # JAX
    conv_weight = jnp.array(q_conv1d.conv.weight.detach().cpu().numpy())  # [d, 1, k]
    conv_bias = jnp.array(q_conv1d.conv.bias.detach().cpu().numpy())  # [d]
    
    q_jax = jnp.array(q.detach().cpu().numpy())
    
    x_jax_out = jax_conv1d(q_jax, conv_weight, conv_bias, kernel_size=4)
    
    # Apply SiLU activation (ShortConvolution does this)
    import jax
    x_jax_out = jax.nn.silu(x_jax_out)
    
    match, max_diff, cos_sim = compare_arrays(
        x_cython, x_jax_out, "Short Convolution", threshold=1e-4
    )
    
    return match

def main():
    """Run all tests in order, stopping at first failure."""
    tests = [
        ("Token Flux", test_1_token_flux),
        ("Enhanced Resonance Flux (4D)", test_2_enhanced_resonance_flux),
        ("Enhanced Resonance Flux (5D)", test_3_enhanced_resonance_flux_5d),
        ("Short Convolution", test_4_short_convolution),
    ]
    
    print("\n" + "="*80)
    print("SYSTEMATIC FUNCTION TESTING")
    print("="*80)
    print("Testing each function individually until it matches Cython exactly.\n")
    
    for test_name, test_func in tests:
        try:
            match = test_func()
            if not match:
                print(f"\n❌ {test_name} FAILED - Fix this before continuing!")
                return False
            print(f"✅ {test_name} PASSED - Moving to next test\n")
        except Exception as e:
            print(f"\n❌ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("="*80)
    print("✅ ALL BASIC FUNCTIONS PASSED!")
    print("="*80)
    print("\nNext: Test hierarchical_delta_rule_jax with real data from Cython")
    
    return True

if __name__ == "__main__":
    main()

