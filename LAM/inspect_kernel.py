#!/usr/bin/env python3
"""
Inspect pretrained semantic kernel metrics
"""
import torch
from pathlib import Path

# Load the pretrained kernel
kernel_path = Path("/workspace/LAM/data/pretrained_semantic_kernel_FOCUSED.pt")

if not kernel_path.exists():
    print(f"❌ Kernel file not found: {kernel_path}")
    exit(1)

kernel_state = torch.load(kernel_path, map_location='cpu', weights_only=False)

print("="*80)
print("📊 PRETRAINED SEMANTIC KERNEL METRICS")
print("="*80)
print()

# Extract all metrics
kernel_norm = kernel_state.get('final_norm', kernel_state.get('kernel_norm', 0.0))
kernel_volume = kernel_state.get('final_volume', kernel_state.get('kernel_volume', 0.0))
volume_growth = kernel_state.get('volume_growth_pct', 0.0)
avg_novelty = kernel_state.get('avg_novelty', 0.0)
num_memories = kernel_state.get('num_memories', 0)
active_clusters = kernel_state.get('active_clusters', 0)
kernel_count = kernel_state.get('kernel_count', 0)
d_model = kernel_state.get('d_model', 384)

# Compute kernel quality indicators
print("📈 KERNEL STATISTICS:")
print(f"   Kernel norm: {kernel_norm:.4f} (target: 40-50)")
print(f"   Kernel volume: {kernel_volume:.4f}")
print(f"   Volume growth: {volume_growth:+.2f}%")
print(f"   Average novelty: {avg_novelty:.3f} (target: 0.5-0.8)")
print(f"   Memories stored: {num_memories:,}")
print(f"   Active clusters: {active_clusters} (target: >50)")
print(f"   Kernel updates: {kernel_count:,}")
print(f"   Embedding dimension: {d_model}")
print()

# Quality assessment
print("🎯 QUALITY ASSESSMENT:")
quality_score = 0.0
issues = []
warnings = []

# 1. Kernel norm check
if 40 <= kernel_norm <= 50:
    print("   ✅ Kernel norm: EXCELLENT (near capacity)")
    quality_score += 0.3
elif 30 <= kernel_norm < 40:
    print("   ⚠️  Kernel norm: GOOD (but not at capacity)")
    quality_score += 0.2
    warnings.append("Kernel norm below capacity (40-50 ideal)")
else:
    print(f"   ❌ Kernel norm: TOO LOW or TOO HIGH ({kernel_norm:.2f})")
    issues.append(f"Kernel norm {kernel_norm:.2f} not in ideal range (40-50)")

# 2. Volume growth check
if volume_growth > 100:
    print(f"   ✅ Volume growth: EXCELLENT ({volume_growth:+.1f}%)")
    quality_score += 0.3
elif volume_growth > 50:
    print(f"   ⚠️  Volume growth: GOOD ({volume_growth:+.1f}%)")
    quality_score += 0.2
elif volume_growth > 0:
    print(f"   ⚠️  Volume growth: POSITIVE but LOW ({volume_growth:+.1f}%)")
    quality_score += 0.1
    warnings.append("Volume growth is low")
else:
    print(f"   ❌ Volume growth: NEGATIVE or ZERO ({volume_growth:+.1f}%)")
    issues.append("Volume did not grow during training")

# 3. Novelty check
if 0.5 <= avg_novelty <= 0.8:
    print(f"   ✅ Average novelty: EXCELLENT ({avg_novelty:.3f})")
    quality_score += 0.2
elif 0.3 <= avg_novelty < 0.5:
    print(f"   ⚠️  Average novelty: MODERATE ({avg_novelty:.3f})")
    quality_score += 0.1
    warnings.append("Novelty is on the lower side")
else:
    print(f"   ❌ Average novelty: TOO LOW or TOO HIGH ({avg_novelty:.3f})")
    issues.append(f"Novelty {avg_novelty:.3f} not in ideal range (0.5-0.8)")

# 4. Active clusters check
if active_clusters >= 50:
    print(f"   ✅ Active clusters: EXCELLENT ({active_clusters})")
    quality_score += 0.2
elif active_clusters >= 20:
    print(f"   ⚠️  Active clusters: MODERATE ({active_clusters})")
    quality_score += 0.1
    warnings.append("Fewer clusters than ideal")
else:
    print(f"   ❌ Active clusters: TOO FEW ({active_clusters})")
    issues.append(f"Only {active_clusters} active clusters (need >50)")

print()
print("="*80)
print(f"📊 OVERALL QUALITY SCORE: {quality_score:.2f}/1.0")
print("="*80)

if quality_score >= 0.8:
    print("   ✅ EXCELLENT: Kernel is well-trained! Safe to use weight 0.7-1.0")
elif quality_score >= 0.6:
    print("   ⚠️  GOOD: Kernel is decent. Use weight 0.5-0.7")
elif quality_score >= 0.4:
    print("   ⚠️  MODERATE: Kernel needs improvement. Use weight 0.3-0.5")
else:
    print("   ❌ POOR: Kernel may not be well-trained. Use weight 0.1-0.3")

if issues:
    print()
    print("❌ ISSUES FOUND:")
    for issue in issues:
        print(f"   - {issue}")

if warnings:
    print()
    print("⚠️  WARNINGS:")
    for warning in warnings:
        print(f"   - {warning}")

print()
print("="*80)
print("💡 RECOMMENDED KERNEL GUIDANCE WEIGHT:")
if quality_score >= 0.8:
    print("   Use 0.7-1.0 (kernel is excellent)")
elif quality_score >= 0.6:
    print("   Use 0.5-0.7 (kernel is good)")
elif quality_score >= 0.4:
    print("   Use 0.3-0.5 (kernel is moderate)")
else:
    print("   Use 0.1-0.3 (kernel needs improvement)")
print("="*80)

