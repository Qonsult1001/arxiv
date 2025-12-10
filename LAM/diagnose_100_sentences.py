"""
🔬 STANDALONE DIAGNOSTIC: No imports from broken modules!

All code is self-contained here to avoid caching issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import gzip
from sentence_transformers import SentenceTransformer

print("="*80)
print("🔬 STANDALONE DIAGNOSTIC: 100 SENTENCES")
print("="*80)
print("\nAll code self-contained - no module caching issues!\n")

# ============================================================================
# KERNEL CLASS - BUILT IN
# ============================================================================
class StandaloneKernel:
    def __init__(self, d_model=384):
        self.d_model = d_model
        # ✅ CORRECT LR: Target 40-48 at 1.5M (not 21!)
        # Diagnostic showed 21.39, need ~2x higher
        self.base_lr = 0.00000092
        self.count = 0
        
        # ✅ CREATE TINY KERNEL CORRECTLY
        print("Creating initial kernel...")
        initial = torch.randn(d_model, d_model, dtype=torch.float32) * 0.00001
        initial = (initial + initial.T) / 2
        initial = initial + torch.eye(d_model, dtype=torch.float32) * 0.00001
        
        # VERIFY IT'S SMALL
        init_norm = torch.norm(initial, p='fro').item()
        print(f"   Created kernel with norm: {init_norm:.6f}")
        
        if init_norm > 0.1:
            print(f"   ⚠️  WARNING: Initial norm {init_norm:.4f} > 0.1!")
            print(f"   Something is wrong with initialization!")
        else:
            print(f"   ✅ Initial norm is correctly small!")
        
        self.kernel = initial
    
    def update(self, embedding, novelty):
        """Update kernel"""
        # Perturbation
        pert = torch.outer(embedding, embedding)
        pert = (pert + pert.T) / 2
        
        # Trace normalize
        trace = torch.trace(pert)
        if trace > 1e-8:
            pert = pert / trace * self.d_model
        
        # Adaptive LR with hard cap
        current_norm = torch.norm(self.kernel, p='fro').item()
        
        if current_norm > 48.0:
            lr = 0.0
        else:
            capacity = current_norm / 50.0
            lr = self.base_lr * (0.01 ** (capacity * 3))
            # No minimum floor - let base_lr control it!
        
        # Apply update with clipping
        update = lr * novelty * pert
        update_norm = torch.norm(update, p='fro').item()
        if update_norm > 0.1:
            update = update * (0.1 / update_norm)
        
        self.kernel = self.kernel + update
        self.kernel = (self.kernel + self.kernel.T) / 2
        
        # Clamp to 50
        kernel_norm = torch.norm(self.kernel, p='fro')
        if kernel_norm > 50.0:
            self.kernel = self.kernel / kernel_norm * 50.0
            kernel_norm = torch.tensor(50.0, device=self.kernel.device)
        
        self.count += 1
        return kernel_norm.item(), lr

# ============================================================================
# NOVELTY TRACKER - BUILT IN
# ============================================================================
class StandaloneNovelty:
    def __init__(self, d_model=384, num_clusters=100):
        self.d_model = d_model
        self.num_clusters = num_clusters
        self.centroids = torch.zeros(num_clusters, d_model)
        self.counts = torch.zeros(num_clusters, dtype=torch.long)
        self.active = 0
        self.threshold = 0.80
    
    def compute_novelty(self, embedding):
        """Compute novelty"""
        # FIRST SENTENCE IS ALWAYS NOVEL
        if self.active == 0:
            print(f"   First sentence - returning novelty = 1.0")
            return 1.0, -1
        
        # Normalize
        emb_norm = F.normalize(embedding.flatten(), p=2, dim=-1)
        
        # Compare to active centroids
        sims = []
        for i in range(self.active):
            cent = F.normalize(self.centroids[i], p=2, dim=-1)
            sim = F.cosine_similarity(emb_norm, cent, dim=0).item()
            sim = max(0.0, min(1.0, sim))  # Clamp
            sims.append(sim)
        
        if len(sims) == 0:
            return 1.0, -1
        
        max_sim = max(sims)
        closest = sims.index(max_sim)
        
        # Convert to novelty
        novelty = (1.0 - max_sim) ** 0.5
        novelty = max(0.0, min(1.0, novelty))
        
        return novelty, closest
    
    def update_clusters(self, embedding, cluster_id, novelty):
        """Update clusters"""
        if novelty > self.threshold and self.active < self.num_clusters:
            # New cluster
            self.centroids[self.active] = embedding.flatten()
            self.counts[self.active] = 1
            self.active += 1
        elif cluster_id >= 0 and cluster_id < self.active:
            # Update existing
            cent = self.centroids[cluster_id]
            count = self.counts[cluster_id].item()
            momentum = min(0.99, count / (count + 1))
            
            new_cent = momentum * cent + (1 - momentum) * embedding.flatten()
            new_cent = F.normalize(new_cent, p=2, dim=-1)
            
            self.centroids[cluster_id] = new_cent
            self.counts[cluster_id] += 1

# ============================================================================
# LOAD DATA
# ============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Load teacher
print("Loading teacher model...")
teacher = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
teacher.eval()

# Load 100 sentences
print("\n📚 Loading 100 sentences...")
script_dir = Path(__file__).parent
data_dir = script_dir / "data"
allnli_path = data_dir / "AllNLI.jsonl.gz"

sentences = []
if allnli_path.exists():
    with gzip.open(allnli_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            try:
                data = json.loads(line)
                if isinstance(data, list) and len(data) >= 2:
                    s1, s2 = str(data[0]).strip(), str(data[1]).strip()
                    if len(s1) > 5 and len(s2) > 5:
                        sentences.append(s1)
                        sentences.append(s2)
            except:
                continue

print(f"✅ Loaded {len(sentences)} sentences\n")

# Encode
print("⚡ Encoding...")
with torch.no_grad():
    embeddings = teacher.encode(
        sentences,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=True
    ).to(device)

print(f"✅ Embeddings: {embeddings.shape}\n")

# ============================================================================
# INITIALIZE & TEST
# ============================================================================
print("🧠 Initializing kernel and novelty tracker...\n")
kernel = StandaloneKernel(d_model=384)
novelty_tracker = StandaloneNovelty(d_model=384, num_clusters=100)

# CRITICAL: Move tensors to device (embeddings are on CUDA, kernel must match)
kernel.kernel = kernel.kernel.to(device)
novelty_tracker.centroids = novelty_tracker.centroids.to(device)
novelty_tracker.counts = novelty_tracker.counts.to(device)

initial_norm = torch.norm(kernel.kernel, p='fro').item()

print(f"\n📊 VERIFIED INITIAL STATE:")
print(f"   Kernel norm: {initial_norm:.6f}")
print(f"   Active clusters: {novelty_tracker.active}")

if initial_norm > 1.0:
    print(f"\n❌ CRITICAL ERROR: Initial norm {initial_norm:.4f} is STILL too large!")
    print(f"   Expected: < 0.1")
    print(f"   This means the initialization code is still broken!")
    print(f"   Stopping diagnostic.")
    import sys
    sys.exit(1)
else:
    print(f"   ✅ Initial norm is good!")

# ============================================================================
# PROCESS STEP BY STEP
# ============================================================================
print("\n" + "="*80)
print("📊 PROCESSING SENTENCE BY SENTENCE")
print("="*80)
print(f"\n{'Step':<6} {'Norm':>10} {'ΔNorm':>10} {'Novel':>8} {'Clust':>7} {'LR':>12}")
print("-"*70)

prev_norm = initial_norm

with torch.no_grad():
    for i in range(min(len(embeddings), 100)):
        emb = embeddings[i]
        
        # Compute novelty
        novelty, cluster_id = novelty_tracker.compute_novelty(emb)
        
        # Verify novelty is valid
        if i == 0 and abs(novelty - 1.0) > 0.01:
            print(f"\n⚠️  WARNING: First sentence novelty = {novelty:.4f}, expected 1.0!")
        
        # Update kernel
        current_norm, lr = kernel.update(emb, novelty)
        
        # Update clusters
        novelty_tracker.update_clusters(emb, cluster_id, novelty)
        
        delta_norm = current_norm - prev_norm
        
        # Print
        if i < 20 or i % 10 == 0 or i == len(embeddings) - 1:
            print(f"{i:<6} {current_norm:>10.6f} {delta_norm:>10.6f} {novelty:>8.4f} "
                  f"{novelty_tracker.active:>3}/100 {lr:>12.8f}")
        
        prev_norm = current_norm
        
        # Check for explosion
        if current_norm > 45.0 and i < 50:
            print(f"\n❌ Kernel exploded at step {i}: norm = {current_norm:.4f}")
            break

# ============================================================================
# FINAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("📊 FINAL ANALYSIS")
print("="*80)

final_norm = torch.norm(kernel.kernel, p='fro').item()
total_growth = final_norm - initial_norm
avg_growth = total_growth / len(embeddings)
projected = initial_norm + avg_growth * 1_500_000

print(f"\nAfter {len(embeddings)} sentences:")
print(f"   Initial norm: {initial_norm:.6f}")
print(f"   Final norm: {final_norm:.6f}")
print(f"   Total growth: {total_growth:.6f}")
print(f"   Avg per sentence: {avg_growth:.6f}")
print(f"   Projected at 1.5M: {projected:.2f} / 50.0")

print(f"\n🎯 VERDICT:")
if projected < 20:
    print(f"   ⚠️  TOO SLOW: Would only reach {projected:.1f}")
    print(f"   Need to increase LR by: {45 / projected:.1f}x")
elif projected > 48:
    print(f"   ⚠️  TOO FAST: Would reach {projected:.1f}")
    print(f"   Need to decrease LR by: {projected / 45:.1f}x")
else:
    print(f"   ✅ PERFECT: Would reach {projected:.1f} (target 40-48)")
    print(f"   Parameters are CORRECT!")
    print(f"   Ready for full training!")

print("\n" + "="*80)
print("🎯 STANDALONE DIAGNOSTIC COMPLETE")
print("="*80)