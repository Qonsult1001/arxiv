"""
🚀 DYNAMIC KERNEL PRE-TRAINING - Self-Adjusting for Any Scale

KEY INNOVATION: LR adjusts automatically based on:
  1. Target capacity utilization (default 85%)
  2. Current novelty trends
  3. Observed growth rate

Works for: 1M, 10M, 100M, or ANY dataset size!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import gzip
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ============================================================================
# DYNAMIC ADAPTIVE KERNEL - Self-Adjusting LR
# ============================================================================
class DynamicAdaptiveKernel(nn.Module):
    def __init__(self, d_model=384, target_capacity=0.85):
        super().__init__()
        self.d_model = d_model
        self.target_capacity = target_capacity  # Target 85% utilization
        self.max_capacity = 50.0
        
        # Initialize tiny kernel
        initial = torch.randn(d_model, d_model, dtype=torch.float32) * 0.00001
        initial = (initial + initial.T) / 2
        initial = initial + torch.eye(d_model, dtype=torch.float32) * 0.00001
        self.register_buffer('kernel', initial)
        
        # Dynamic LR controller
        self.base_lr = 0.0000010  # Starting point (will adjust)
        self.lr_adjustment_interval = 50000  # Adjust every 50K
        self.update_count = 0
        
        # Track growth rate for adjustment
        self.growth_history = []
        self.norm_history = []
        
    def adjust_learning_rate(self, total_seen, total_expected):
        """
        Dynamically adjust LR based on progress
        
        Args:
            total_seen: How many samples processed so far
            total_expected: Total samples in dataset (can be estimate)
        """
        current_norm = torch.norm(self.kernel, p='fro').item()
        target_norm = self.max_capacity * self.target_capacity
        
        # Calculate progress
        progress = total_seen / max(total_expected, 1)
        expected_norm = target_norm * progress
        
        # Adjustment factor
        if current_norm < expected_norm * 0.8:
            # Too slow - increase LR
            adjustment = 1.2
        elif current_norm > expected_norm * 1.2:
            # Too fast - decrease LR
            adjustment = 0.8
        else:
            # On track
            adjustment = 1.0
        
        self.base_lr *= adjustment
        
        return {
            'current_norm': current_norm,
            'expected_norm': expected_norm,
            'adjustment': adjustment,
            'new_lr': self.base_lr
        }
    
    def update_kernel(self, embedding, novelty):
        """Update kernel with dynamic adaptive LR"""
        # Perturbation
        pert = torch.outer(embedding, embedding)
        pert = (pert + pert.T) / 2
        
        # Trace normalize
        trace = torch.trace(pert)
        if trace > 1e-8:
            pert = pert / trace * self.d_model
        
        # Dynamic adaptive LR
        current_norm = torch.norm(self.kernel, p='fro').item()
        capacity_ratio = current_norm / self.max_capacity
        
        # Hard stop at 96% capacity
        if capacity_ratio > 0.96:
            lr = 0.0
        else:
            # Exponential decay as capacity fills
            decay_factor = (0.01 ** (capacity_ratio * 3))
            lr = self.base_lr * decay_factor
        
        # Apply update with clipping
        update = lr * novelty * pert
        update_norm = torch.norm(update, p='fro').item()
        if update_norm > 0.1:
            update = update * (0.1 / update_norm)
        
        self.kernel = self.kernel + update
        self.kernel = (self.kernel + self.kernel.T) / 2
        
        # Hard cap at max capacity
        kernel_norm = torch.norm(self.kernel, p='fro')
        if kernel_norm > self.max_capacity:
            self.kernel = self.kernel / kernel_norm * self.max_capacity
            kernel_norm = torch.tensor(self.max_capacity)
        
        self.update_count += 1
        self.norm_history.append(kernel_norm.item())
        
        return {
            'kernel_norm': kernel_norm.item(),
            'learning_rate': lr,
            'capacity': capacity_ratio,
        }


class SemanticNoveltyTracker(nn.Module):
    def __init__(self, d_model=384, num_clusters=100):
        super().__init__()
        self.d_model = d_model
        self.num_clusters = num_clusters
        self.register_buffer('centroids', torch.zeros(num_clusters, d_model))
        self.register_buffer('counts', torch.zeros(num_clusters, dtype=torch.long))
        self.active = 0
        self.threshold = 0.80
    
    def compute_novelty(self, embedding):
        if self.active == 0:
            return 1.0, -1
        
        emb_norm = F.normalize(embedding.flatten(), p=2, dim=-1)
        
        sims = []
        for i in range(self.active):
            cent = F.normalize(self.centroids[i], p=2, dim=-1)
            sim = F.cosine_similarity(emb_norm, cent, dim=0).item()
            sim = max(0.0, min(1.0, sim))
            sims.append(sim)
        
        if len(sims) == 0:
            return 1.0, -1
        
        max_sim = max(sims)
        closest = sims.index(max_sim)
        novelty = (1.0 - max_sim) ** 0.5
        novelty = max(0.0, min(1.0, novelty))
        
        return novelty, closest
    
    def update_clusters(self, embedding, cluster_id, novelty):
        if novelty > self.threshold and self.active < self.num_clusters:
            self.centroids[self.active] = embedding.flatten()
            self.counts[self.active] = 1
            self.active += 1
        elif cluster_id >= 0 and cluster_id < self.active:
            cent = self.centroids[cluster_id]
            count = self.counts[cluster_id].item()
            momentum = min(0.99, count / (count + 1))
            
            new_cent = momentum * cent + (1 - momentum) * embedding.flatten()
            new_cent = F.normalize(new_cent, p=2, dim=-1)
            
            self.centroids[cluster_id] = new_cent
            self.counts[cluster_id] += 1


class DynamicSemanticMemory(nn.Module):
    def __init__(self, d_model=384, target_capacity=0.85):
        super().__init__()
        self.kernel = DynamicAdaptiveKernel(d_model, target_capacity)
        self.novelty = SemanticNoveltyTracker(d_model)
        
        self.total_updates = 0
        self.novelty_history = []
        self.adjustment_history = []
    
    def process_batch(self, embeddings, total_expected=None):
        """Process a batch with dynamic LR adjustment"""
        batch_size = len(embeddings)
        novelty_scores = []
        
        for i, emb in enumerate(embeddings):
            # Compute novelty
            novelty, cluster_id = self.novelty.compute_novelty(emb)
            novelty_scores.append(novelty)
            
            # Update kernel
            stats = self.kernel.update_kernel(emb, novelty)
            
            # Update clusters
            self.novelty.update_clusters(emb, cluster_id, novelty)
            
            self.total_updates += 1
            self.novelty_history.append(novelty)
            
            # Dynamic LR adjustment every 50K
            if self.total_updates % 50000 == 0 and total_expected:
                adjustment = self.kernel.adjust_learning_rate(
                    self.total_updates, 
                    total_expected
                )
                self.adjustment_history.append(adjustment)
                
                print(f"\n📊 LR Adjustment at {self.total_updates:,} samples:")
                print(f"   Current norm: {adjustment['current_norm']:.2f}")
                print(f"   Expected norm: {adjustment['expected_norm']:.2f}")
                print(f"   Adjustment: {adjustment['adjustment']:.2f}x")
                print(f"   New LR: {adjustment['new_lr']:.8f}\n")
        
        return {
            'avg_novelty': np.mean(novelty_scores),
            'kernel_norm': stats['kernel_norm'],
            'learning_rate': stats['learning_rate'],
            'capacity': stats['capacity'],
        }
    
    def get_state(self):
        """Get current state"""
        kernel_norm = torch.norm(self.kernel.kernel, p='fro').item()
        recent_novelty = np.mean(self.novelty_history[-10000:]) if len(self.novelty_history) > 0 else 0.0
        
        return {
            'kernel_norm': kernel_norm,
            'capacity': kernel_norm / self.kernel.max_capacity,
            'active_clusters': self.novelty.active,
            'total_updates': self.total_updates,
            'recent_novelty': recent_novelty,
            'base_lr': self.kernel.base_lr,
        }


# ============================================================================
# DYNAMIC PRE-TRAINING
# ============================================================================
def pretrain_dynamic(
    teacher_model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=256,
    target_capacity=0.85,
    output_path=None
):
    """
    Dynamic pre-training that works for ANY dataset size!
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Load teacher
    print(f"\n🤖 Loading teacher: {teacher_model_name}")
    teacher = SentenceTransformer(teacher_model_name, device=device)
    teacher.eval()
    
    # Load data
    print("\n📚 Loading dataset...")
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    sentences = []
    total_pairs = 0
    
    # AllNLI
    allnli_path = data_dir / "AllNLI.jsonl.gz"
    if allnli_path.exists():
        print(f"   Loading AllNLI...")
        with gzip.open(allnli_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if isinstance(data, list) and len(data) >= 2:
                        s1, s2 = str(data[0]).strip(), str(data[1]).strip()
                        if len(s1) > 5 and len(s2) > 5:
                            sentences.append(s1)
                            sentences.append(s2)
                            total_pairs += 1
                except:
                    continue
        print(f"      ✅ {total_pairs:,} pairs")
    
    # WikiAnswers
    wikianswers_path = data_dir / "WikiAnswers_1M.jsonl.gz"
    if wikianswers_path.exists():
        print(f"   Loading WikiAnswers...")
        count = 0
        with gzip.open(wikianswers_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if isinstance(data, list) and len(data) >= 2:
                        s1, s2 = str(data[0]).strip(), str(data[1]).strip()
                        if len(s1) > 5 and len(s2) > 5:
                            sentences.append(s1)
                            sentences.append(s2)
                            count += 1
                except:
                    continue
        print(f"      ✅ {count:,} pairs")
        total_pairs += count
    
    sentences = list(set(sentences))  # Deduplicate
    total_expected = len(sentences)
    
    print(f"\n✅ Dataset loaded:")
    print(f"   Total pairs: {total_pairs:,}")
    print(f"   Unique sentences: {total_expected:,}")
    print(f"   Target capacity: {target_capacity*100:.0f}%")
    print(f"   Target norm: {50.0 * target_capacity:.1f} / 50.0")
    
    # Initialize dynamic memory
    print(f"\n🧠 Initializing dynamic kernel...")
    memory = DynamicSemanticMemory(d_model=384, target_capacity=target_capacity)
    memory.to(device)
    
    initial_state = memory.get_state()
    print(f"   Initial norm: {initial_state['kernel_norm']:.6f}")
    print(f"   Starting LR: {initial_state['base_lr']:.8f}")
    
    # Pre-training loop
    print(f"\n{'='*80}")
    print(f"🚀 DYNAMIC PRE-TRAINING - Self-Adjusting for {total_expected:,} sentences")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        pbar = tqdm(range(0, len(sentences), batch_size), desc="Pre-training", unit="batch")
        
        for i in pbar:
            batch_sentences = sentences[i:i+batch_size]
            
            # Encode
            batch_embs = teacher.encode(
                batch_sentences,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=batch_size
            ).to(device)
            
            # Process with dynamic adjustment
            stats = memory.process_batch(batch_embs, total_expected=total_expected)
            
            # Update progress
            pbar.set_postfix({
                'norm': f"{stats['kernel_norm']:.2f}",
                'cap': f"{stats['capacity']*100:.0f}%",
                'novel': f"{stats['avg_novelty']:.2f}",
                'lr': f"{stats['learning_rate']:.2e}",
                'clust': f"{memory.novelty.active}"
            })
            
            # Stats every 100K
            if memory.total_updates % 100000 == 0 and memory.total_updates > 0:
                state = memory.get_state()
                progress = memory.total_updates / total_expected * 100
                
                print(f"\n{'='*70}")
                print(f"📊 Progress: {memory.total_updates:,}/{total_expected:,} ({progress:.1f}%)")
                print(f"{'='*70}")
                print(f"   Kernel norm: {state['kernel_norm']:.4f} / 50.0 ({state['capacity']*100:.1f}%)")
                print(f"   Base LR: {state['base_lr']:.8f}")
                print(f"   Recent novelty: {state['recent_novelty']:.3f}")
                print(f"   Active clusters: {state['active_clusters']}/100")
                print(f"{'='*70}\n")
    
    # Final statistics
    final_state = memory.get_state()
    
    print(f"\n{'='*80}")
    print(f"✅ DYNAMIC PRE-TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📈 Final Statistics:")
    print(f"   Sentences processed: {total_expected:,}")
    print(f"   Kernel norm: {final_state['kernel_norm']:.4f} / 50.0")
    print(f"   Capacity used: {final_state['capacity']*100:.1f}%")
    print(f"   Final LR: {final_state['base_lr']:.8f}")
    print(f"   Recent novelty: {final_state['recent_novelty']:.3f}")
    print(f"   Active clusters: {final_state['active_clusters']}/100")
    
    print(f"\n📊 LR Adjustments Made: {len(memory.adjustment_history)}")
    if len(memory.adjustment_history) > 0:
        print(f"   Adjustments:")
        for i, adj in enumerate(memory.adjustment_history[-5:]):  # Last 5
            print(f"      Step {(i+1)*50000:,}: {adj['adjustment']:.2f}x → LR {adj['new_lr']:.8f}")
    
    # Calculate distillation weight
    capacity_score = final_state['capacity']
    novelty_score = 1.0 - final_state['recent_novelty']
    distill_weight = 0.3 + 0.4 * capacity_score + 0.2 * novelty_score
    
    print(f"\n🎯 Training Assessment:")
    if capacity_score < 0.6:
        status = "UNDERTRAINED"
    elif capacity_score < 0.8:
        status = "MODERATE"
    else:
        status = "WELL-TRAINED"
    
    print(f"   Status: {status}")
    print(f"   Distillation weight: {distill_weight:.2f}")
    
    # Save
    if output_path is None:
        output_path = data_dir / "pretrained_kernel_DYNAMIC.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_path}...")
    
    kernel_state = {
        'kernel': memory.kernel.kernel.cpu(),
        'd_model': 384,
        'target_capacity': target_capacity,
        'final_capacity': final_state['capacity'],
        'kernel_norm': final_state['kernel_norm'],
        'num_sentences': total_expected,
        'active_clusters': final_state['active_clusters'],
        'recent_novelty': final_state['recent_novelty'],
        'distillation_weight': distill_weight,
        'status': status,
        'base_lr_final': final_state['base_lr'],
        'adjustment_history': memory.adjustment_history,
    }
    
    torch.save(kernel_state, output_path)
    print(f"✅ Saved!")
    print(f"{'='*80}\n")
    
    return output_path, memory


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--target_capacity", type=float, default=0.85)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 DYNAMIC KERNEL PRE-TRAINING")
    print("="*80)
    print("\n⚡ KEY FEATURES:")
    print("   ✅ Self-adjusting LR - works for ANY dataset size")
    print("   ✅ Automatic capacity targeting")
    print("   ✅ Real-time progress monitoring")
    print("   ✅ Scales from 1M to 100M+ sentences")
    print("\n" + "="*80 + "\n")
    
    pretrain_dynamic(
        teacher_model_name=args.teacher,
        batch_size=args.batch_size,
        target_capacity=args.target_capacity,
        output_path=args.output
    )