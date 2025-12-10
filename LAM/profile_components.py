#!/usr/bin/env python3
"""Profile individual components of DeltaNet layer"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import time
from final_solution_formula_final import EnhancedHierarchicalDeltaNet
from einops import rearrange

device = 'cuda'

print('='*60)
print('DETAILED PROFILING: DeltaNet Layer Components')
print('='*60)

teacher = AutoModel.from_pretrained('/workspace/LAM/all-MiniLM-L6-v2').to(device)
tokenizer = AutoTokenizer.from_pretrained('/workspace/LAM/all-MiniLM-L6-v2')

layer = EnhancedHierarchicalDeltaNet(d_model=384, num_heads=12,
                                      use_hierarchical_decay=True, use_enhanced_flux=True).to(device)

state_dict = torch.load('/workspace/LAM/best/pytorch_model.bin', map_location=device, weights_only=False)
layer_dict = {k.replace('deltanet_layers.0.', ''): v for k, v in state_dict.items() if 'deltanet_layers.0.' in k}
if layer_dict:
    layer.load_state_dict(layer_dict, strict=False)

layer.eval()

text = 'The quick brown fox ' * 30
tokens = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt').to(device)
x = teacher.embeddings(tokens['input_ids'])
mask = tokens['attention_mask']

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = layer(x, mask)
torch.cuda.synchronize()

batch_size, seq_len, _ = x.shape

# Component 1: Projections
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        q = layer.q_proj(x)
        k = layer.k_proj(x)
        v = layer.v_proj(x)
    torch.cuda.synchronize()
    proj_time = (time.time() - start) / 100 * 1000

# Component 2: Convolutions
with torch.no_grad():
    q = layer.q_proj(x)
    k = layer.k_proj(x)
    v = layer.v_proj(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        q_conv, _ = layer.q_conv1d(q)
        k_conv, _ = layer.k_conv1d(k)
        v_conv, _ = layer.v_conv1d(v)
    torch.cuda.synchronize()
    conv_time = (time.time() - start) / 100 * 1000

# Component 3: Reshape operations
with torch.no_grad():
    q, _ = layer.q_conv1d(layer.q_proj(x))
    k, _ = layer.k_conv1d(layer.k_proj(x))
    v, _ = layer.v_conv1d(layer.v_proj(x))
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        q_r = rearrange(q, 'b l (h d) -> b h l d', h=layer.num_heads)
        k_r = rearrange(k, 'b l (h d) -> b h l d', h=layer.num_heads)
        v_r = rearrange(v, 'b l (h d) -> b h l d', h=layer.num_heads)
    torch.cuda.synchronize()
    reshape_time = (time.time() - start) / 100 * 1000

# Component 4: Beta/Decay/Gate projections
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        beta = layer.b_proj(x).sigmoid()
        fast_decay = torch.sigmoid(layer.fast_decay_proj(x) + layer.fast_decay_bias)
        slow_decay = torch.sigmoid(layer.slow_decay_proj(x) + layer.slow_decay_bias)
        fast_gate = torch.sigmoid(layer.fast_gate_proj(x))
        slow_gate = torch.sigmoid(layer.slow_gate_proj(x))
    torch.cuda.synchronize()
    control_time = (time.time() - start) / 100 * 1000

# Component 5: Full forward (to get delta rule time by subtraction)
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = layer(x, mask)
    torch.cuda.synchronize()
    total_time = (time.time() - start) / 100 * 1000

delta_rule_time = total_time - proj_time - conv_time - reshape_time - control_time

print(f'\nComponent Breakdown (single layer, 128 tokens):')
print(f'{"Component":<25} {"Time (ms)":<12} {"% of Total"}')
print('-'*50)
print(f'{"1. Projections (QKV)":<25} {proj_time:.3f}       {proj_time/total_time*100:.1f}%')
print(f'{"2. Convolutions":<25} {conv_time:.3f}       {conv_time/total_time*100:.1f}%')
print(f'{"3. Reshapes":<25} {reshape_time:.3f}       {reshape_time/total_time*100:.1f}%')
print(f'{"4. Control (β/decay/gate)":<25} {control_time:.3f}       {control_time/total_time*100:.1f}%')
print(f'{"5. Delta Rule + Mixing":<25} {delta_rule_time:.3f}       {delta_rule_time/total_time*100:.1f}%')
print('-'*50)
print(f'{"TOTAL":<25} {total_time:.3f}       100.0%')

print(f'\n💡 OPTIMIZATION PRIORITY:')
components = [
    ('Delta Rule + Mixing', delta_rule_time, 5),
    ('Control projections', control_time, 4),
    ('Projections (QKV)', proj_time, 3),
    ('Convolutions', conv_time, 2),
    ('Reshapes', reshape_time, 1),
]
components.sort(key=lambda x: x[1], reverse=True)

for i, (name, t, _) in enumerate(components, 1):
    print(f'   {i}. {name}: {t:.3f}ms ({t/total_time*100:.0f}%)')

print('='*60)
