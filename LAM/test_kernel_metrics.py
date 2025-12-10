#!/usr/bin/env python3
"""
Test kernel metrics to verify they're working correctly
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from semantic_memory_kernel import SemanticMemoryLatentSpace
from sentence_transformers import SentenceTransformer

print("="*80)
print("🧪 TESTING KERNEL METRICS")
print("="*80)
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {device}\n")

# Initialize kernel
print("1️⃣  Initializing semantic kernel...")
semantic_kernel = SemanticMemoryLatentSpace(
    d_model=384,
    num_clusters=50,
)
semantic_kernel.to(device)

# Get initial state
initial_state = semantic_kernel.get_memory_state()
print(f"   Initial kernel norm: {initial_state['kernel_norm']:.4f}")
print(f"   Initial kernel count: {initial_state['kernel_count']}")
print(f"   Initial active clusters: {initial_state['active_clusters']}")
print()

# Load teacher model
print("2️⃣  Loading teacher model...")
teacher_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
teacher_model.eval()
print()

# Test with a few sentences
test_sentences = [
    "The cat sat on the mat",
    "A feline rested on a rug",
    "Dogs are loyal pets",
    "Canines make faithful companions",
    "Python is a programming language",
    "Java is another programming language",
    "The weather is sunny today",
    "It's raining outside",
    "I love machine learning",
    "Deep learning is fascinating",
]

print("3️⃣  Processing test sentences...")
novelty_scores_all = []

for i, sentence in enumerate(test_sentences):
    # Encode
    with torch.no_grad():
        embedding = teacher_model.encode([sentence], convert_to_tensor=True, normalize_embeddings=True)[0].to(device)
    
    # Process through kernel
    _, stats = semantic_kernel.process_embeddings(
        embedding.unsqueeze(0),
        texts=[sentence],
        update_memory=True
    )
    
    if stats:
        novelty = stats.get('avg_novelty', 0.0)
        novelty_scores_all.append(novelty)
        print(f"   Sentence {i+1}: novelty={novelty:.3f}")
    
    # Check state after each update
    state = semantic_kernel.get_memory_state()
    print(f"      Kernel norm: {state['kernel_norm']:.4f}, count: {state['kernel_count']}, clusters: {state['active_clusters']}")

print()
print("4️⃣  Final state:")
final_state = semantic_kernel.get_memory_state()
print(f"   Kernel norm: {final_state['kernel_norm']:.4f}")
print(f"   Kernel count: {final_state['kernel_count']}")
print(f"   Active clusters: {final_state['active_clusters']}")
print(f"   Memories stored: {final_state['num_memories']}")
print(f"   Average novelty: {np.mean(novelty_scores_all):.3f}" if novelty_scores_all else "   Average novelty: N/A")
print()

print("="*80)
print("✅ TEST COMPLETE")
print("="*80)
print()
print("📊 EXPECTED BEHAVIOR:")
print("   - Kernel norm: Should grow from ~0.1 to ~50.0")
print("   - Kernel count: Should equal number of sentences processed (10)")
print("   - Active clusters: Should grow from 0 to some number (1-10)")
print("   - Novelty scores: Should decrease as similar sentences are added")
print("   - Average novelty: Should be > 0.0 and < 1.0")
print()

