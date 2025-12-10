#!/usr/bin/env python3
"""
Detailed analysis of model structure to understand frozen vs trained parameters.
"""
import torch
from pathlib import Path
from collections import defaultdict

checkpoint_path = "/workspace/LAM/best/pytorch_model.bin"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("="*80)
print("DETAILED MODEL STRUCTURE ANALYSIS")
print("="*80)

# Categorize all parameters
categories = {
    'teacher_embeddings': [],
    'teacher_attention': [],
    'teacher_ffn': [],
    'teacher_norms': [],
    'deltanet': [],
    'other': []
}

for name, param in checkpoint.items():
    if not isinstance(param, torch.Tensor):
        continue
    
    num_params = param.numel()
    name_lower = name.lower()
    
    if 'teacher_model' in name:
        if 'embedding' in name_lower:
            categories['teacher_embeddings'].append((name, num_params))
        elif 'attention' in name_lower:
            categories['teacher_attention'].append((name, num_params))
        elif 'intermediate' in name_lower or 'output.dense' in name_lower:
            categories['teacher_ffn'].append((name, num_params))
        elif 'norm' in name_lower:
            categories['teacher_norms'].append((name, num_params))
        else:
            categories['other'].append((name, num_params))
    elif 'deltanet' in name_lower:
        categories['deltanet'].append((name, num_params))
    else:
        categories['other'].append((name, num_params))

# Calculate totals
totals = {}
for cat, items in categories.items():
    totals[cat] = sum(count for _, count in items)

total_all = sum(totals.values())

print(f"\n📊 PARAMETER BREAKDOWN:")
print(f"{'Category':<25} {'Parameters':>15} {'Million':>12} {'Percent':>10}")
print("-" * 65)

for cat in ['teacher_embeddings', 'teacher_attention', 'teacher_ffn', 'teacher_norms', 'deltanet', 'other']:
    count = totals[cat]
    millions = count / 1e6
    percent = (count / total_all * 100) if total_all > 0 else 0
    print(f"{cat:<25} {count:>15,} {millions:>12.2f} {percent:>9.1f}%")

print("-" * 65)
print(f"{'TOTAL':<25} {total_all:>15,} {total_all/1e6:>12.2f} {100:>9.1f}%")

# Calculate frozen vs trained
teacher_total = totals['teacher_embeddings'] + totals['teacher_attention'] + totals['teacher_ffn'] + totals['teacher_norms']
deltanet_total = totals['deltanet']

print(f"\n🔒 FROZEN vs TRAINED:")
print(f"   Teacher Model (Frozen): {teacher_total:>12,} ({teacher_total/1e6:>6.2f}M)")
print(f"   DeltaNet Layers (Trained): {deltanet_total:>12,} ({deltanet_total/1e6:>6.2f}M)")
print(f"   Total:                    {total_all:>12,} ({total_all/1e6:>6.2f}M)")

# Check if teacher_model matches all-mini size (should be ~22M)
print(f"\n📋 VERIFICATION:")
print(f"   Expected all-mini base: ~22M")
print(f"   Actual teacher_model: {teacher_total/1e6:.2f}M")
if abs(teacher_total/1e6 - 22) < 1:
    print(f"   ✅ Matches expected ~22M (frozen all-mini base)")
else:
    print(f"   ⚠️  Differs from expected 22M by {abs(teacher_total/1e6 - 22):.2f}M")

print(f"\n   Expected DeltaNet: ~5M")
print(f"   Actual DeltaNet: {deltanet_total/1e6:.2f}M")
if abs(deltanet_total/1e6 - 5) < 1:
    print(f"   ✅ Matches expected ~5M (trained DeltaNet)")
else:
    print(f"   ⚠️  Differs from expected 5M by {abs(deltanet_total/1e6 - 5):.2f}M")

# Show some example keys
print(f"\n📝 SAMPLE KEYS:")
print(f"\n   Teacher Embeddings (first 3):")
for name, count in categories['teacher_embeddings'][:3]:
    print(f"      {name}: {count:,}")

print(f"\n   Teacher Attention (first 3):")
for name, count in categories['teacher_attention'][:3]:
    print(f"      {name}: {count:,}")

print(f"\n   DeltaNet (first 5):")
for name, count in categories['deltanet'][:5]:
    print(f"      {name}: {count:,}")

