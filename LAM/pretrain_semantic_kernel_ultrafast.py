"""
🚀 PAIR-AWARE SEMANTIC KERNEL PRE-TRAINING

CRITICAL FIX: Trains kernel on sentence PAIRS to learn semantic refinement
- Similar pairs → Preserve/enhance similarity
- Different pairs → Maintain difference
- Result: Kernel that IMPROVES semantic relationships (not destroys them!)

Usage:
    python pretrain_semantic_kernel_PAIRS.py
    python pretrain_semantic_kernel_PAIRS.py --max_pairs 500000
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
from scipy.stats import spearmanr
import random

# ============================================================================
# PAIR-AWARE KERNEL LEARNING
# ============================================================================

class PairAwareSemanticKernel(nn.Module):
    """
    Learns kernel from sentence PAIRS to refine semantic relationships
    
    Key insight: Train on (s1, s2, label) where label indicates similarity
    - label=1: Similar pairs → kernel should preserve/enhance similarity
    - label=0: Different pairs → kernel should maintain difference
    """
    
    def __init__(self, d_model=384, target_norm=50.0):
        super().__init__()
        self.d_model = d_model
        self.target_norm = target_norm
        
        # Initialize small symmetric kernel
        initial_kernel = torch.randn(d_model, d_model) * 0.0001
        initial_kernel = (initial_kernel + initial_kernel.T) / 2
        initial_kernel = initial_kernel + torch.eye(d_model) * 0.001
        self.register_buffer('kernel', initial_kernel)
        
        # Learning parameters
        self.base_lr = 0.000005  # Conservative for stability
        self.updates_count = 0
        
        # Track metrics
        self.metrics = {
            'similar_pair_sims': [],  # Similarity for similar pairs
            'different_pair_sims': [], # Similarity for different pairs
            'kernel_norms': [],
        }
    
    def compute_pair_objective(self, emb1, emb2, label):
        """
        Compute update direction based on pair relationship
        
        Args:
            emb1, emb2: [d_model] embeddings
            label: 1 (similar) or 0 (different)
        
        Returns:
            perturbation: [d_model, d_model] kernel update
            weight: scalar multiplier for update strength
        """
        # Current similarity
        current_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        
        if label == 1:  # SIMILAR PAIR
            # Goal: High similarity (target ~0.85-0.95)
            target_sim = 0.90
            similarity_gap = target_sim - current_sim
            
            if similarity_gap > 0.05:  # Need significant improvement
                # Strong update: Pull embeddings together
                # Use their average (shared semantic content)
                combined = F.normalize(emb1 + emb2, p=2, dim=0)
                perturbation = torch.outer(combined, combined)
                weight = min(similarity_gap * 3.0, 1.0)  # Cap at 1.0
                
            elif similarity_gap > 0:  # Minor improvement needed
                # Gentle update
                combined = F.normalize(emb1 + emb2, p=2, dim=0)
                perturbation = torch.outer(combined, combined)
                weight = similarity_gap
                
            else:  # Already good (>0.90)
                # Minimal reinforcement
                perturbation = torch.outer(emb1, emb1)
                weight = 0.1
        
        else:  # DIFFERENT PAIR (label == 0)
            # Goal: Low similarity (target ~0.0-0.20)
            target_sim = 0.15
            
            if current_sim > 0.35:  # Too similar, need separation
                # Strong update: Push embeddings apart
                # Emphasize their difference
                difference = F.normalize(emb1 - emb2, p=2, dim=0)
                perturbation = torch.outer(difference, difference)
                weight = min((current_sim - 0.35) * 3.0, 1.0)
                
            elif current_sim > 0.20:  # Moderate separation needed
                # Medium update
                difference = F.normalize(emb1 - emb2, p=2, dim=0)
                perturbation = torch.outer(difference, difference)
                weight = (current_sim - 0.20) * 2.0
                
            else:  # Already different enough (<0.20)
                # Minimal reinforcement
                perturbation = torch.outer(emb1, emb1)
                weight = 0.05
        
        return perturbation, weight, current_sim
    
    def update_from_pair(self, emb1, emb2, label):
        """
        Update kernel from a single pair
        
        Args:
            emb1, emb2: [d_model] normalized embeddings
            label: 1 (similar) or 0 (different)
        """
        device = self.kernel.device
        emb1 = emb1.to(device)
        emb2 = emb2.to(device)
        
        # Compute update
        perturbation, weight, current_sim = self.compute_pair_objective(emb1, emb2, label)
        
        # Make symmetric
        perturbation = (perturbation + perturbation.T) / 2
        
        # Trace normalization (preserve scale)
        trace_pert = torch.trace(perturbation)
        if trace_pert > 1e-8:
            perturbation = perturbation / trace_pert * self.d_model
        
        # Adaptive learning rate based on kernel capacity
        kernel_norm = torch.norm(self.kernel, p='fro').item()
        
        if kernel_norm >= self.target_norm:
            # At capacity, stop learning
            adaptive_lr = 0.0
        else:
            # Decay LR as capacity fills
            capacity_ratio = kernel_norm / self.target_norm
            decay_factor = (1.0 - capacity_ratio) ** 2  # Quadratic decay
            adaptive_lr = self.base_lr * decay_factor
        
        # Apply update
        update = adaptive_lr * weight * perturbation
        
        # Clip update norm for stability
        update_norm = torch.norm(update, p='fro')
        max_update = 0.05
        if update_norm > max_update:
            update = update * (max_update / update_norm)
        
        # Update kernel
        self.kernel = self.kernel + update
        self.kernel = (self.kernel + self.kernel.T) / 2  # Keep symmetric
        
        # Hard norm cap
        final_norm = torch.norm(self.kernel, p='fro')
        if final_norm > self.target_norm:
            self.kernel = self.kernel / final_norm * self.target_norm
        
        self.updates_count += 1
        
        # Track metrics
        if label == 1:
            self.metrics['similar_pair_sims'].append(current_sim)
        else:
            self.metrics['different_pair_sims'].append(current_sim)
        self.metrics['kernel_norms'].append(final_norm.item())
        
        return final_norm.item()
    
    def get_metrics_summary(self, window=10000):
        """Get recent metrics summary"""
        recent_similar = self.metrics['similar_pair_sims'][-window:]
        recent_different = self.metrics['different_pair_sims'][-window:]
        recent_norms = self.metrics['kernel_norms'][-window:]
        
        return {
            'avg_similar_sim': np.mean(recent_similar) if recent_similar else 0.0,
            'avg_different_sim': np.mean(recent_different) if recent_different else 0.0,
            'separation': (np.mean(recent_similar) - np.mean(recent_different)) if (recent_similar and recent_different) else 0.0,
            'kernel_norm': np.mean(recent_norms) if recent_norms else 0.0,
        }


def load_paired_dataset(max_pairs=None):
    """
    Load sentence pairs with labels (KEEP PAIR STRUCTURE!)
    
    Returns:
        pairs: List of (sentence1, sentence2, label)
               label=1 for similar, label=0 for different
    """
    print("\n📚 Loading PAIRED dataset...")
    print("   ⚠️  Keeping pair structure (not breaking into individual sentences!)")
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    pairs = []
    
    # AllNLI - Contains (anchor, positive, negative) triplets
    allnli_path = data_dir / "AllNLI.jsonl.gz"
    if allnli_path.exists():
        print(f"\n   Loading AllNLI.jsonl.gz...")
        count = 0
        
        with gzip.open(allnli_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if max_pairs and len(pairs) >= max_pairs:
                    break
                
                try:
                    data = json.loads(line)
                    if isinstance(data, list) and len(data) == 3:
                        anchor, positive, negative = data
                        s_anchor = str(anchor).strip()
                        s_positive = str(positive).strip()
                        s_negative = str(negative).strip()
                        
                        if len(s_anchor) > 10 and len(s_positive) > 10 and len(s_negative) > 10:
                            # Create similar pair (anchor, positive, label=1)
                            pairs.append((s_anchor, s_positive, 1))
                            
                            # Create different pair (anchor, negative, label=0)
                            pairs.append((s_anchor, s_negative, 0))
                            
                            count += 1
                            
                            if max_pairs and len(pairs) >= max_pairs:
                                break
                except:
                    continue
        
        print(f"      ✅ Loaded {count:,} triplets → {len(pairs):,} pairs")
    
    # Shuffle pairs for better training dynamics
    random.shuffle(pairs)
    
    # Print distribution
    similar_count = sum(1 for p in pairs if p[2] == 1)
    different_count = sum(1 for p in pairs if p[2] == 0)
    
    print(f"\n✅ Dataset loaded:")
    print(f"   Total pairs: {len(pairs):,}")
    print(f"   Similar pairs (label=1): {similar_count:,} ({similar_count/len(pairs)*100:.1f}%)")
    print(f"   Different pairs (label=0): {different_count:,} ({different_count/len(pairs)*100:.1f}%)")
    
    return pairs


def evaluate_kernel_quality(kernel, teacher_model, validation_pairs, device):
    """
    Evaluate kernel quality on validation set
    
    Measures:
    1. Similar pairs: avg similarity BEFORE and AFTER kernel
    2. Different pairs: avg similarity BEFORE and AFTER kernel
    3. Separation: (similar_sim - different_sim) - should INCREASE
    """
    print(f"\n📊 Evaluating kernel on {len(validation_pairs)} validation pairs...")
    
    teacher_model.eval()
    
    sims_before_similar = []
    sims_after_similar = []
    sims_before_different = []
    sims_after_different = []
    
    with torch.no_grad():
        for s1, s2, label in tqdm(validation_pairs, desc="Evaluating"):
            # Encode
            emb1 = teacher_model.encode(s1, convert_to_tensor=True, normalize_embeddings=True).to(device)
            emb2 = teacher_model.encode(s2, convert_to_tensor=True, normalize_embeddings=True).to(device)
            
            # Before kernel
            sim_before = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            
            # After kernel
            warped1 = torch.matmul(emb1, kernel)
            warped2 = torch.matmul(emb2, kernel)
            warped1 = F.normalize(warped1, p=2, dim=0)
            warped2 = F.normalize(warped2, p=2, dim=0)
            sim_after = F.cosine_similarity(warped1.unsqueeze(0), warped2.unsqueeze(0)).item()
            
            # Track by label
            if label == 1:  # Similar
                sims_before_similar.append(sim_before)
                sims_after_similar.append(sim_after)
            else:  # Different
                sims_before_different.append(sim_before)
                sims_after_different.append(sim_after)
    
    # Compute metrics
    avg_before_similar = np.mean(sims_before_similar)
    avg_after_similar = np.mean(sims_after_similar)
    avg_before_different = np.mean(sims_before_different)
    avg_after_different = np.mean(sims_after_different)
    
    separation_before = avg_before_similar - avg_before_different
    separation_after = avg_after_similar - avg_after_different
    
    print(f"\n   Results:")
    print(f"   {'='*60}")
    print(f"   Similar pairs:")
    print(f"      Before kernel: {avg_before_similar:.4f}")
    print(f"      After kernel:  {avg_after_similar:.4f} ({avg_after_similar-avg_before_similar:+.4f})")
    print(f"\n   Different pairs:")
    print(f"      Before kernel: {avg_before_different:.4f}")
    print(f"      After kernel:  {avg_after_different:.4f} ({avg_after_different-avg_before_different:+.4f})")
    print(f"\n   Separation (similar - different):")
    print(f"      Before kernel: {separation_before:.4f}")
    print(f"      After kernel:  {separation_after:.4f} ({separation_after-separation_before:+.4f})")
    print(f"   {'='*60}")
    
    quality_score = separation_after - separation_before
    
    if quality_score > 0.05:
        status = "✅ EXCELLENT - Kernel significantly improves separation"
    elif quality_score > 0.01:
        status = "✅ GOOD - Kernel improves separation"
    elif quality_score > -0.01:
        status = "⚠️  NEUTRAL - Kernel has minimal effect"
    else:
        status = "❌ BAD - Kernel hurts separation"
    
    print(f"\n   {status}")
    print(f"   Quality score: {quality_score:+.4f}\n")
    
    return {
        'similar_before': avg_before_similar,
        'similar_after': avg_after_similar,
        'different_before': avg_before_different,
        'different_after': avg_after_different,
        'separation_before': separation_before,
        'separation_after': separation_after,
        'quality_score': quality_score,
    }


def pretrain_kernel_pairs(
    teacher_model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_pairs=None,
    encoding_batch_size=128,
    d_model=384,
    validation_size=1000,
    stats_interval=50000,
    output_path=None
):
    """
    Train kernel on sentence pairs to learn semantic refinement
    
    Args:
        teacher_model_name: Teacher model for encoding
        max_pairs: Maximum pairs to use (None = all)
        encoding_batch_size: Batch size for encoding
        d_model: Embedding dimension
        validation_size: Number of pairs for validation
        stats_interval: Print stats every N pairs
        output_path: Where to save kernel
    
    Returns:
        output_path: Path to saved kernel
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load teacher model
    print(f"\n🤖 Loading teacher model: {teacher_model_name}")
    teacher_model = SentenceTransformer(teacher_model_name, device=device)
    teacher_model.eval()
    
    # Load paired dataset
    pairs = load_paired_dataset(max_pairs=max_pairs)
    
    if len(pairs) == 0:
        print("\n❌ No pairs loaded!")
        return None
    
    # Split train/validation
    validation_pairs = pairs[:validation_size]
    training_pairs = pairs[validation_size:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Training pairs: {len(training_pairs):,}")
    print(f"   Validation pairs: {len(validation_pairs):,}")
    
    # ==========================================
    # PHASE 1: ENCODE ALL TRAINING PAIRS
    # ==========================================
    print(f"\n{'='*70}")
    print(f"⚡ PHASE 1: ENCODING {len(training_pairs):,} TRAINING PAIRS")
    print(f"{'='*70}")
    
    embeddings_s1 = []
    embeddings_s2 = []
    labels = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(training_pairs), encoding_batch_size), desc="Encoding", unit="batch"):
            batch_pairs = training_pairs[i:i+encoding_batch_size]
            
            # Separate components
            batch_s1 = [p[0] for p in batch_pairs]
            batch_s2 = [p[1] for p in batch_pairs]
            batch_labels = [p[2] for p in batch_pairs]
            
            # Encode both sentences
            emb1 = teacher_model.encode(
                batch_s1,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=encoding_batch_size
            )
            
            emb2 = teacher_model.encode(
                batch_s2,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=encoding_batch_size
            )
            
            embeddings_s1.append(emb1.to(device))
            embeddings_s2.append(emb2.to(device))
            labels.extend(batch_labels)
    
    # Concatenate
    embeddings_s1 = torch.cat(embeddings_s1, dim=0)
    embeddings_s2 = torch.cat(embeddings_s2, dim=0)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    
    print(f"\n✅ Phase 1 complete!")
    print(f"   Encoded: {embeddings_s1.shape[0]:,} pairs")
    print(f"   Memory: {(embeddings_s1.element_size() * embeddings_s1.nelement() * 2) / 1024**2:.1f} MB")
    
    # ==========================================
    # PHASE 2: LEARN KERNEL FROM PAIRS
    # ==========================================
    print(f"\n{'='*70}")
    print(f"⚡ PHASE 2: LEARNING SEMANTIC KERNEL FROM PAIRS")
    print(f"{'='*70}")
    print(f"   Objective: Preserve similarity for similar pairs")
    print(f"              Maintain difference for different pairs\n")
    
    # Initialize kernel
    semantic_kernel = PairAwareSemanticKernel(d_model=d_model, target_norm=50.0)
    semantic_kernel.to(device)
    
    initial_norm = torch.norm(semantic_kernel.kernel, p='fro').item()
    print(f"   Initial kernel norm: {initial_norm:.4f}")
    
    # Training loop
    print(f"\n   Training on {len(embeddings_s1):,} pairs...")
    
    pbar = tqdm(total=len(embeddings_s1), desc="Learning kernel", unit="pair")
    
    for i in range(len(embeddings_s1)):
        emb1 = embeddings_s1[i]
        emb2 = embeddings_s2[i]
        label = labels[i].item()
        
        # Update kernel
        kernel_norm = semantic_kernel.update_from_pair(emb1, emb2, label)
        
        pbar.update(1)
        
        # Print stats at intervals
        if (i + 1) % stats_interval == 0 or (i + 1) == len(embeddings_s1):
            metrics = semantic_kernel.get_metrics_summary()
            
            pbar.write(f"\n{'='*70}")
            pbar.write(f"📊 Progress: {i+1:,}/{len(embeddings_s1):,} ({(i+1)/len(embeddings_s1)*100:.1f}%)")
            pbar.write(f"{'='*70}")
            pbar.write(f"   Kernel norm: {metrics['kernel_norm']:.4f} / 50.0")
            pbar.write(f"   Avg similarity (similar pairs): {metrics['avg_similar_sim']:.4f}")
            pbar.write(f"   Avg similarity (different pairs): {metrics['avg_different_sim']:.4f}")
            pbar.write(f"   Separation: {metrics['separation']:.4f}")
            pbar.write(f"   Total updates: {semantic_kernel.updates_count:,}")
            pbar.write(f"{'='*70}\n")
    
    pbar.close()
    
    final_norm = torch.norm(semantic_kernel.kernel, p='fro').item()
    
    print(f"\n✅ Training complete!")
    print(f"   Final kernel norm: {final_norm:.4f}")
    print(f"   Total updates: {semantic_kernel.updates_count:,}")
    
    # ==========================================
    # PHASE 3: EVALUATE ON VALIDATION SET
    # ==========================================
    print(f"\n{'='*70}")
    print(f"⚡ PHASE 3: VALIDATION")
    print(f"{'='*70}")
    
    evaluation_results = evaluate_kernel_quality(
        semantic_kernel.kernel,
        teacher_model,
        validation_pairs,
        device
    )
    
    # ==========================================
    # SAVE KERNEL
    # ==========================================
    if output_path is None:
        output_path = Path(__file__).parent / "data" / "pretrained_semantic_kernel_PAIRS.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving kernel to {output_path}...")
    
    kernel_state = {
        'kernel': semantic_kernel.kernel.cpu(),
        'd_model': d_model,
        'training_type': 'PAIR_AWARE',
        'num_training_pairs': len(training_pairs),
        'num_validation_pairs': len(validation_pairs),
        'final_norm': final_norm,
        'total_updates': semantic_kernel.updates_count,
        'teacher_model': teacher_model_name,
        
        # Validation metrics
        'validation_metrics': evaluation_results,
        
        # Training metrics
        'final_similar_sim': semantic_kernel.get_metrics_summary()['avg_similar_sim'],
        'final_different_sim': semantic_kernel.get_metrics_summary()['avg_different_sim'],
        'final_separation': semantic_kernel.get_metrics_summary()['separation'],
    }
    
    torch.save(kernel_state, output_path)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved ({size_mb:.1f} MB)")
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print(f"\n{'='*70}")
    print(f"🎉 PAIR-AWARE KERNEL TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📈 Training Summary:")
    print(f"   Pairs processed: {len(training_pairs):,}")
    print(f"   Kernel norm: {final_norm:.4f} / 50.0")
    print(f"   Training separation: {semantic_kernel.get_metrics_summary()['separation']:.4f}")
    
    print(f"\n📊 Validation Results:")
    print(f"   Quality score: {evaluation_results['quality_score']:+.4f}")
    print(f"   Separation improvement: {evaluation_results['separation_after'] - evaluation_results['separation_before']:+.4f}")
    
    if evaluation_results['quality_score'] > 0.01:
        print(f"\n   ✅ Kernel successfully learned semantic refinement!")
        print(f"   Ready to use for distillation.")
    else:
        print(f"\n   ⚠️  Kernel quality is low. Consider:")
        print(f"      - Training on more pairs")
        print(f"      - Adjusting learning rate")
        print(f"      - Using better quality data")
    
    print(f"\n{'='*70}\n")
    
    return output_path, semantic_kernel


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pair-aware semantic kernel pre-training")
    parser.add_argument("--teacher_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_pairs", type=int, default=None, help="Max pairs to use (None=all)")
    parser.add_argument("--encoding_batch_size", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--validation_size", type=int, default=1000)
    parser.add_argument("--stats_interval", type=int, default=50000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 PAIR-AWARE SEMANTIC KERNEL PRE-TRAINING")
    print("="*80)
    print("\n💡 KEY DIFFERENCE:")
    print("   ❌ OLD: Trained on individual sentences (diversity objective)")
    print("   ✅ NEW: Trained on sentence PAIRS (semantic refinement objective)")
    print("\n🎯 OBJECTIVE:")
    print("   - Similar pairs → Preserve/enhance similarity")
    print("   - Different pairs → Maintain difference")
    print("   - Result: Kernel that IMPROVES semantic relationships!")
    print("\n" + "="*80)
    
    pretrain_kernel_pairs(
        teacher_model_name=args.teacher_model,
        max_pairs=args.max_pairs,
        encoding_batch_size=args.encoding_batch_size,
        d_model=args.d_model,
        validation_size=args.validation_size,
        stats_interval=args.stats_interval,
        output_path=args.output
    )