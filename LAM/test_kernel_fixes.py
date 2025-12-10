"""
Quick diagnostic to verify kernel learning fixes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

# Test the distance metric
print("="*70)
print("🔬 TESTING DISTANCE METRIC")
print("="*70)

# Load teacher model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# Test sentences
test_sentences = [
    "The cat sat on the mat.",           # Reference
    "The cat is sitting on the mat.",    # Very similar
    "Dogs are playing in the park.",     # Different topic
    "Quantum physics explains reality.", # Very different
]

embeddings = teacher.encode(test_sentences, convert_to_tensor=True, normalize_embeddings=True).to(device)

print("\nTest sentences:")
for i, sent in enumerate(test_sentences):
    print(f"  {i}: {sent}")

# Compute distances using the fixed metric
ref = embeddings[0]
print(f"\nDistances from reference (sentence 0):")
print(f"{'Sentence':<50} {'Cosine Sim':>12} {'Novelty':>12}")
print("-"*75)

for i, emb in enumerate(embeddings):
    sim = F.cosine_similarity(ref, emb, dim=0).item()
    novelty = (1.0 - sim) ** 0.5
    print(f"{test_sentences[i]:<50} {sim:>12.4f} {novelty:>12.4f}")

print("\n✅ Expected pattern:")
print("   Similar sentences → sim≈0.9, novelty≈0.3")
print("   Different sentences → sim≈0.3, novelty≈0.8")

# Test kernel learning
print("\n" + "="*70)
print("🔬 TESTING KERNEL LEARNING")
print("="*70)

# Import from pretrain_semantic_focus.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# The classes are defined inside pretrain_semantic_focus.py
# We need to import the module and access the classes
import pretrain_semantic_focus
AdaptiveMemoryKernel = pretrain_semantic_focus.AdaptiveMemoryKernel

kernel = AdaptiveMemoryKernel(d_model=384).to(device)

print(f"\nInitial kernel norm: {torch.norm(kernel.kernel, p='fro').item():.4f}")

# Simulate 10 updates with high novelty
norms = [torch.norm(kernel.kernel, p='fro').item()]
lrs = []

for i in range(20):
    # Create random embedding
    emb = torch.randn(384, device=device)
    emb = F.normalize(emb, p=2, dim=-1)
    
    # Update with high importance
    stats = kernel.update_kernel(emb, importance=0.9)
    norms.append(stats['kernel_norm'])
    lrs.append(stats['learning_rate'])

print("\nKernel growth over 20 updates (novelty=0.9):")
print(f"{'Update':<10} {'Norm':>12} {'LR':>12}")
print("-"*35)
for i in range(0, 21, 5):
    if i < len(norms):
        lr = lrs[i-1] if i > 0 else 0
        print(f"{i:<10} {norms[i]:>12.4f} {lr:>12.6f}")

print("\n✅ Expected pattern:")
print("   Norm should grow: 0.x → 5 → 15 → 25 → 35 (not jump to 50)")
print("   LR should decay: 0.005 → 0.001 → 0.0001 (as kernel fills)")

# Test clustering
print("\n" + "="*70)
print("🔬 TESTING CLUSTERING")
print("="*70)

SemanticNoveltyTracker = pretrain_semantic_focus.SemanticNoveltyTracker

tracker = SemanticNoveltyTracker(d_model=384, num_clusters=10).to(device)

# Add diverse embeddings
test_topics = [
    "cats and animals",
    "dogs and pets", 
    "quantum physics",
    "machine learning",
    "cooking recipes",
]

print(f"\nAdding 5 diverse topics:")
for i, topic in enumerate(test_topics):
    emb = teacher.encode(topic, convert_to_tensor=True, normalize_embeddings=True).to(device)
    novelty, cluster = tracker.compute_novelty(emb)
    tracker.update_clusters(emb, cluster, novelty)
    print(f"  Topic '{topic}': novelty={novelty:.3f}, cluster={cluster}, active={tracker.active_clusters}")

print(f"\n✅ Expected: 3-5 clusters created (diverse topics)")
print(f"   Actual: {tracker.active_clusters} clusters")

# Test with similar sentences (should NOT create new clusters)
print(f"\nAdding 5 similar sentences (to 'cats'):")
similar_to_cats = [
    "feline creatures",
    "kitty cats",
    "domestic cats",
    "tabby cats",
    "persian cats",
]

for i, sent in enumerate(similar_to_cats):
    emb = teacher.encode(sent, convert_to_tensor=True, normalize_embeddings=True).to(device)
    novelty, cluster = tracker.compute_novelty(emb)
    tracker.update_clusters(emb, cluster, novelty)
    print(f"  '{sent}': novelty={novelty:.3f}, cluster={cluster}, active={tracker.active_clusters}")

print(f"\n✅ Expected: Still 3-5 clusters (similar to existing)")
print(f"   Actual: {tracker.active_clusters} clusters (should not grow much)")

print("\n" + "="*70)
print("🎯 DIAGNOSTIC COMPLETE")
print("="*70)
print("\nIf all tests show expected patterns, the fixes are working!")
print("Run the full pre-training script to verify at scale.")