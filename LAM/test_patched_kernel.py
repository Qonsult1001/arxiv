#!/usr/bin/env python3
"""
Test script to validate patched semantic memory kernel

Expected results after patches:
✅ Kernel norm grows from ~1.5 → 30+ (not saturate at 1.0)
✅ Novel phase shows higher novelty (>0.6) than familiar phase (<0.3)
✅ Kernel change (Δ) higher for novel than familiar

Run: python test_patched_kernel.py
"""
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from semantic_memory_kernel import SemanticMemoryLatentSpace

print("="*80)
print("🧪 TESTING PATCHED SEMANTIC MEMORY KERNEL")
print("="*80)

# Initialize - use d_model=64 to match test_space_thinking.py exactly
memory = SemanticMemoryLatentSpace(
    d_model=64,  # ⭐ Match test_space_thinking.py (uses 64 for tests)
    use_teacher=False,  # Use projection for fast testing
    num_clusters=10,
    kernel_momentum=0.99,
)

print("\n📊 PHASE 1: Diverse Novel Memories")
print("-" * 80)

# Phase 1: Diverse topics (should be novel) - INCREASED to 3333 for ~10k total
# Aligned with test_space_thinking.py which uses 1000 diverse memories
texts_phase1 = []
for i in range(3333):
    texts_phase1.append(f"quantum computing research topic number {i}")
    texts_phase1.append(f"cooking pasta recipe variation {i}")
    texts_phase1.append(f"mountain climbing expedition log {i}")

novelties_1 = []
print(f"Processing {len(texts_phase1)} diverse memories...")
for i, text in enumerate(texts_phase1):
    refined, stats = memory.process([text], update_memory=True, return_stats=True)
    novelties_1.append(stats['novelty_scores'][0])
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(texts_phase1)} memories processed...")

state1 = memory.get_memory_state()
print(f"Processed {len(texts_phase1)} diverse memories")
print(f"  Kernel norm: {state1['kernel_norm']:.2f}")
print(f"  Avg novelty: {np.mean(novelties_1):.3f}")
print(f"  Active clusters: {state1['active_clusters']}")

print("\n📊 PHASE 2: Familiar Memories (Repeat Similar)")
print("-" * 80)

# Phase 2: Similar to phase 1 (should be familiar) - INCREASED to 1667
texts_phase2 = []
for i in range(1667):
    texts_phase2.append(f"quantum mechanics lecture notes {i}")  # Similar to quantum computing
    texts_phase2.append(f"pasta cooking instructions {i}")  # Similar to cooking
    texts_phase2.append(f"hiking trail guide {i}")  # Similar to mountain climbing

novelties_2 = []
kernel_changes_2 = []
print(f"Processing {len(texts_phase2)} familiar memories...")
for i, text in enumerate(texts_phase2):
    norm_before = memory.kernel.kernel.norm().item()
    refined, stats = memory.process([text], update_memory=True, return_stats=True)
    norm_after = memory.kernel.kernel.norm().item()
    
    novelties_2.append(stats['novelty_scores'][0])
    kernel_changes_2.append(norm_after - norm_before)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(texts_phase2)} memories processed...")

state2 = memory.get_memory_state()
print(f"Processed {len(texts_phase2)} familiar memories")
print(f"  Kernel norm: {state2['kernel_norm']:.2f} (Δ = {state2['kernel_norm'] - state1['kernel_norm']:.2f})")
print(f"  Avg novelty: {np.mean(novelties_2):.3f}")
print(f"  Avg kernel change: {np.mean(kernel_changes_2):.4f}")

print("\n📊 PHASE 3: New Novel Memories")
print("-" * 80)

# Phase 3: Completely new domain (should be novel again) - INCREASED to 1667
texts_phase3 = []
for i in range(1667):
    texts_phase3.append(f"Renaissance art history discussion {i}")
    texts_phase3.append(f"medieval architecture analysis {i}")
    texts_phase3.append(f"baroque music composition theory {i}")

novelties_3 = []
kernel_changes_3 = []
print(f"Processing {len(texts_phase3)} new novel memories...")
for i, text in enumerate(texts_phase3):
    norm_before = memory.kernel.kernel.norm().item()
    refined, stats = memory.process([text], update_memory=True, return_stats=True)
    norm_after = memory.kernel.kernel.norm().item()
    
    novelties_3.append(stats['novelty_scores'][0])
    kernel_changes_3.append(norm_after - norm_before)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(texts_phase3)} memories processed...")

state3 = memory.get_memory_state()
print(f"Processed {len(texts_phase3)} new novel memories")
print(f"  Kernel norm: {state3['kernel_norm']:.2f} (Δ = {state3['kernel_norm'] - state2['kernel_norm']:.2f})")
print(f"  Avg novelty: {np.mean(novelties_3):.3f}")
print(f"  Avg kernel change: {np.mean(kernel_changes_3):.4f}")

print("\n" + "="*80)
print("📈 VALIDATION RESULTS")
print("="*80)

# Test 1: Kernel growth (measure VOLUME like test_space_thinking.py, not norm!)
# Volume can grow even if norm is clamped at 50.0
vol1 = state1.get('kernel_volume', 0.0)
vol2 = state2.get('kernel_volume', 0.0)
vol3 = state3.get('kernel_volume', 0.0)

growth_1_to_2 = ((vol2 - vol1) / abs(vol1) * 100) if vol1 > 0 else 0.0
growth_2_to_3 = ((vol3 - vol2) / abs(vol2) * 100) if vol2 > 0 else 0.0
total_growth = ((vol3 - vol1) / abs(vol1) * 100) if vol1 > 0 else 0.0

print(f"\n✅ Test 1: Kernel Growth (Volume)")
print(f"  Phase 1→2: {growth_1_to_2:+.2f}%")
print(f"  Phase 2→3: {growth_2_to_3:+.2f}%")
print(f"  Total: {vol1:.2f} → {vol3:.2f} ({total_growth:+.2f}%)")

test1_pass = total_growth > 50.0  # 50% volume growth (like test_space_thinking.py)
print(f"  Status: {'✅ PASS' if test1_pass else '❌ FAIL'} (need >50% volume growth)")

# Test 2: Novelty discrimination
avg_novel = (np.mean(novelties_1) + np.mean(novelties_3)) / 2
avg_familiar = np.mean(novelties_2)

print(f"\n✅ Test 2: Novelty Discrimination")
print(f"  Novel phases: {avg_novel:.3f}")
print(f"  Familiar phase: {avg_familiar:.3f}")
print(f"  Ratio: {avg_novel / (avg_familiar + 1e-8):.2f}x")

test2_pass = avg_novel > avg_familiar * 1.5  # Novel should be 50% higher
print(f"  Status: {'✅ PASS' if test2_pass else '❌ FAIL'} (novel > 1.5x familiar)")

# Test 3: Kernel evolution tracks novelty
avg_change_novel = np.mean(kernel_changes_3)
avg_change_familiar = np.mean(kernel_changes_2)

print(f"\n✅ Test 3: Kernel Evolution")
print(f"  Δ(novel): {avg_change_novel:.4f}")
print(f"  Δ(familiar): {avg_change_familiar:.4f}")
print(f"  Ratio: {avg_change_novel / (avg_change_familiar + 1e-8):.2f}x")

test3_pass = avg_change_novel > avg_change_familiar * 1.2  # Novel causes 20% more change
print(f"  Status: {'✅ PASS' if test3_pass else '❌ FAIL'} (novel > 1.2x familiar)")

# Final verdict
print("\n" + "="*80)
all_pass = test1_pass and test2_pass and test3_pass

if all_pass:
    print("🎉 ALL TESTS PASSED! 🎉")
    print("\n✅ Your patched kernel is working correctly!")
    print("✅ Ready for integration with Enhanced Hierarchical DeltaNet")
else:
    print("⚠️  SOME TESTS FAILED")
    print("\nDebug:")
    if not test1_pass:
        print("  - Kernel not growing enough → Check PATCH 1 & 2")
    if not test2_pass:
        print("  - Novelty not discriminating → Check PATCH 4")
    if not test3_pass:
        print("  - Kernel evolution not tracking novelty → Check PATCH 2")

print("="*80)
exit(0 if all_pass else 1)

