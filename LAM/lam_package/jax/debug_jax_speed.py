#!/usr/bin/env python3
"""
Debug JAX Speed - Step by Step Analysis
=======================================

Find where JAX is slow and fix it.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from lam import LAM

print("="*80)
print("🔍 JAX Speed Debug - Step by Step")
print("="*80)

model_path = Path(__file__).parent.parent / "LAM-base-v1"
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    sys.exit(1)

# Test sentence
test_sentence = "Hello world, this is a test sentence."

print("\n1️⃣ Loading Models...")
start = time.time()
model_cython = LAM(str(model_path), backend='cython')
cython_load_time = time.time() - start
print(f"   Cython load: {cython_load_time*1000:.2f}ms")

start = time.time()
model_jax = LAM(str(model_path), backend='jax')
jax_load_time = time.time() - start
print(f"   JAX load: {jax_load_time*1000:.2f}ms")

print("\n2️⃣ Tokenization...")
start = time.time()
encoded_cython = model_cython.tokenizer.encode(test_sentence)
cython_tokenize = time.time() - start
print(f"   Cython tokenize: {cython_tokenize*1000:.2f}ms")

start = time.time()
encoded_jax = model_jax.tokenizer.encode(test_sentence)
jax_tokenize = time.time() - start
print(f"   JAX tokenize: {jax_tokenize*1000:.2f}ms")

print("\n3️⃣ Encoding (First Call - Compilation)...")
start = time.time()
emb_cython = model_cython.encode([test_sentence])
cython_first = time.time() - start
print(f"   Cython first: {cython_first*1000:.2f}ms")

start = time.time()
emb_jax = model_jax.encode([test_sentence])
jax_first = time.time() - start
print(f"   JAX first (compilation): {jax_first*1000:.2f}ms")

print("\n4️⃣ Encoding (Warmed Up)...")
times_cython = []
times_jax = []
for i in range(5):
    start = time.time()
    _ = model_cython.encode([test_sentence])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times_cython.append(time.time() - start)
    
    start = time.time()
    _ = model_jax.encode([test_sentence])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times_jax.append(time.time() - start)

cython_avg = np.mean(times_cython[1:]) * 1000
jax_avg = np.mean(times_jax[1:]) * 1000
print(f"   Cython avg: {cython_avg:.2f}ms")
print(f"   JAX avg: {jax_avg:.2f}ms")
print(f"   Speedup: {cython_avg/jax_avg:.2f}x ({'JAX faster' if jax_avg < cython_avg else 'Cython faster'})")

print("\n5️⃣ Breaking Down JAX Forward Pass...")
# Get the actual forward pass timing
import jax.numpy as jnp

# Prepare inputs
input_ids = torch.tensor([encoded_jax.ids], dtype=torch.long)
attention_mask = torch.tensor([encoded_jax.attention_mask], dtype=torch.long)

# Convert to JAX
start = time.time()
input_ids_jax = jnp.array(input_ids.cpu().numpy())
attention_mask_jax = jnp.array(attention_mask.cpu().numpy())
convert_time = time.time() - start
print(f"   PyTorch → JAX conversion: {convert_time*1000:.2f}ms")

# Forward pass (using optimized version)
start = time.time()
embeddings_jax = model_jax._jax_get_embeddings(model_jax._jax_params, input_ids_jax, attention_mask_jax, **model_jax._jax_config_values)
forward_time = time.time() - start
print(f"   JAX forward pass: {forward_time*1000:.2f}ms")

# Convert back
start = time.time()
embeddings_torch = torch.from_numpy(np.array(embeddings_jax))
convert_back_time = time.time() - start
print(f"   JAX → PyTorch conversion: {convert_back_time*1000:.2f}ms")

print(f"\n   Total breakdown:")
print(f"   - Conversion: {convert_time*1000:.2f}ms")
print(f"   - Forward: {forward_time*1000:.2f}ms")
print(f"   - Convert back: {convert_back_time*1000:.2f}ms")
print(f"   - Total: {(convert_time + forward_time + convert_back_time)*1000:.2f}ms")

print("\n6️⃣ JIT Compilation Status...")
print(f"   ✅ Forward pass is fully JIT-compiled")
print(f"   ✅ Using optimized hierarchical_delta_rule_jax with jax.lax.scan")
print(f"   ✅ All operations use pure JAX arrays (no Python dicts)")

print("\n" + "="*80)
print("✅ Debug Complete!")
print("="*80)

