"""
🚀 SIMPLE 2-STAGE KERNEL BOOST TEST

Your kernel improves teacher embeddings:
- Similar pairs: 0.6947 → 0.8467 (+0.1520)
- Different pairs: 0.4341 → 0.5441 (+0.1100)
- Separation: +0.0420

Test Plan:
1. Student learns from Teacher → Baseline Score
2. Student learns from Kernel(Teacher) → Boosted Score
3. Compare: Boosted should be > Baseline!

Usage:
    python test_kernel_boost_SIMPLE.py --steps 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import sys
import json
import gzip
import random

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet


def load_pairs(n=5000):
    """Load training pairs"""
    pairs, labels = [], []
    path = Path(__file__).parent / "data" / "AllNLI.jsonl.gz"
    
    if path.exists():
        with gzip.open(path, 'rt') as f:
            for line in f:
                if len(pairs) >= n:
                    break
                try:
                    data = json.loads(line)
                    if len(data) == 3:
                        a, p, neg = data  # ⭐ FIX: Rename 'n' to 'neg' to avoid shadowing parameter 'n'
                        pairs.append((str(a).strip(), str(p).strip()))
                        labels.append(1)
                        pairs.append((str(a).strip(), str(neg).strip()))
                        labels.append(0)
                except:
                    continue
    
    return pairs, labels


def load_stsb():
    """Load STS-B test set"""
    try:
        from datasets import load_dataset
        stsb = load_dataset('glue', 'stsb', split='validation')
        return stsb['sentence1'], stsb['sentence2'], [s/5.0 for s in stsb['label']]
    except:
        return ["test"]*100, ["test"]*100, [0.5]*100


class StudentModel(nn.Module):
    """Simple student model"""
    def __init__(self, d=384, layers=4):
        super().__init__()
        self.embed = nn.Embedding(30522, d)
        self.layers = nn.ModuleList([
            EnhancedHierarchicalDeltaNet(d_model=d, num_heads=6) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.proj = nn.Linear(d, d, bias=False)
    
    def forward(self, ids, mask):
        # ⭐ FIX: Clamp token IDs to valid range [0, vocab_size-1]
        vocab_size = self.embed.num_embeddings
        ids = torch.clamp(ids, 0, vocab_size - 1)
        x = self.embed(ids)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x), mask)[0]
        
        # Mean pooling
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (x * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        return F.normalize(self.proj(pooled), p=2, dim=-1)


def train_student(model, pairs, labels, targets_s1, targets_s2, tokenizer, device, steps=50):
    """
    Train student to match target embeddings
    
    Args:
        targets_s1, targets_s2: Pre-computed target embeddings for all pairs
    """
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    model.train()
    
    losses = []
    
    for step in tqdm(range(steps), desc="Training"):
        # Sample batch
        idx = random.sample(range(len(pairs)), 16)
        
        s1 = [pairs[i][0] for i in idx]
        s2 = [pairs[i][1] for i in idx]
        
        # Tokenize
        tok1 = tokenizer(s1, padding=True, truncation=True, max_length=64, return_tensors='pt')
        tok2 = tokenizer(s2, padding=True, truncation=True, max_length=64, return_tensors='pt')
        
        tok1 = {k: v.to(device) for k, v in tok1.items()}
        tok2 = {k: v.to(device) for k, v in tok2.items()}
        
        # Get targets for this batch
        batch_t1 = torch.stack([targets_s1[i] for i in idx])
        batch_t2 = torch.stack([targets_s2[i] for i in idx])
        
        # Forward
        opt.zero_grad()
        e1 = model(tok1['input_ids'], tok1['attention_mask'])
        e2 = model(tok2['input_ids'], tok2['attention_mask'])
        
        # Loss: match targets
        loss = (
            (1 - F.cosine_similarity(e1, batch_t1).mean()) +
            (1 - F.cosine_similarity(e2, batch_t2).mean())
        ) / 2
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)


def evaluate(model, s1, s2, scores, tokenizer, device):
    """Evaluate on STS-B"""
    model.eval()
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(s1), 32):
            batch_s1 = s1[i:i+32]
            batch_s2 = s2[i:i+32]
            
            tok1 = tokenizer(batch_s1, padding=True, truncation=True, max_length=128, return_tensors='pt')
            tok2 = tokenizer(batch_s2, padding=True, truncation=True, max_length=128, return_tensors='pt')
            
            tok1 = {k: v.to(device) for k, v in tok1.items()}
            tok2 = {k: v.to(device) for k, v in tok2.items()}
            
            e1 = model(tok1['input_ids'], tok1['attention_mask'])
            e2 = model(tok2['input_ids'], tok2['attention_mask'])
            
            sims = F.cosine_similarity(e1, e2, dim=-1)
            preds.extend(sims.cpu().numpy())
    
    pearson = pearsonr(scores[:len(preds)], preds)[0]
    spearman = spearmanr(scores[:len(preds)], preds)[0]
    
    return pearson, spearman


def main(steps=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load teacher
    print("Loading teacher model...")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    
    teacher = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    teacher.eval()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Teacher loaded\n")
    
    # Load kernel
    kernel_path = Path(__file__).parent / "data" / "pretrained_semantic_kernel_PAIRS.pt"
    if not kernel_path.exists():
        print(f"❌ Kernel not found: {kernel_path}")
        return
    
    print("Loading kernel...")
    kernel_state = torch.load(kernel_path, map_location=device, weights_only=False)
    kernel = kernel_state['kernel'].to(device)
    
    if 'validation_metrics' in kernel_state:
        m = kernel_state['validation_metrics']
        print(f"Kernel validation:")
        print(f"  Similar: {m['similar_before']:.3f} → {m['similar_after']:.3f} (+{m['similar_after']-m['similar_before']:.3f})")
        print(f"  Different: {m['different_before']:.3f} → {m['different_after']:.3f} (+{m['different_after']-m['different_before']:.3f})")
        print(f"  Separation: +{m['quality_score']:.4f}")
    print("✅ Kernel loaded\n")
    
    # Load data
    print("Loading data...")
    pairs, labels = load_pairs(n=5000)
    test_s1, test_s2, test_scores = load_stsb()
    print(f"Training: {len(pairs)} pairs")
    print(f"Test: {len(test_s1)} pairs\n")
    
    # ========================================================================
    # PRE-COMPUTE EMBEDDINGS
    # ========================================================================
    print("="*80)
    print("PRE-COMPUTING TARGET EMBEDDINGS")
    print("="*80)
    
    print("\n1. Encoding with teacher...")
    teacher_s1, teacher_s2 = [], []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), 128), desc="Encoding"):
            batch = pairs[i:i+128]
            s1 = [p[0] for p in batch]
            s2 = [p[1] for p in batch]
            
            e1 = teacher.encode(s1, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            e2 = teacher.encode(s2, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            
            teacher_s1.append(e1.to(device))
            teacher_s2.append(e2.to(device))
    
    teacher_s1 = torch.cat(teacher_s1)
    teacher_s2 = torch.cat(teacher_s2)
    print(f"✅ Encoded {len(teacher_s1)} pairs\n")
    
    print("2. Applying kernel...")
    kernel_s1 = F.normalize(torch.matmul(teacher_s1, kernel), p=2, dim=-1)
    kernel_s2 = F.normalize(torch.matmul(teacher_s2, kernel), p=2, dim=-1)
    print(f"✅ Kernel applied\n")
    
    # Verify kernel improves separation on training data
    with torch.no_grad():
        sim_idx = [i for i, l in enumerate(labels) if l == 1][:100]
        diff_idx = [i for i, l in enumerate(labels) if l == 0][:100]
        
        sim_before = F.cosine_similarity(teacher_s1[sim_idx], teacher_s2[sim_idx]).mean().item()
        sim_after = F.cosine_similarity(kernel_s1[sim_idx], kernel_s2[sim_idx]).mean().item()
        
        diff_before = F.cosine_similarity(teacher_s1[diff_idx], teacher_s2[diff_idx]).mean().item()
        diff_after = F.cosine_similarity(kernel_s1[diff_idx], kernel_s2[diff_idx]).mean().item()
        
        sep_before = sim_before - diff_before
        sep_after = sim_after - diff_after
        
        print(f"Training data check (100 samples each):")
        print(f"  Similar: {sim_before:.3f} → {sim_after:.3f} ({sim_after-sim_before:+.3f})")
        print(f"  Different: {diff_before:.3f} → {diff_after:.3f} ({diff_after-diff_before:+.3f})")
        print(f"  Separation: {sep_before:.3f} → {sep_after:.3f} ({sep_after-sep_before:+.3f})")
        
        if sep_after > sep_before:
            print(f"  ✅ Kernel improves separation!\n")
        else:
            print(f"  ⚠️  WARNING: Kernel doesn't improve on training data!\n")
    
    # ========================================================================
    # STAGE 1: BASELINE (Student → Teacher)
    # ========================================================================
    print("="*80)
    print("STAGE 1: BASELINE DISTILLATION")
    print("="*80)
    print(f"Target: Teacher embeddings (plain)\n")
    
    model1 = StudentModel(d=384, layers=4).to(device)
    
    loss1 = train_student(
        model1, pairs, labels, 
        teacher_s1, teacher_s2,  # ← Plain teacher targets
        tokenizer, device, steps
    )
    
    print(f"Training loss: {loss1:.4f}\n")
    print("Evaluating...")
    pearson1, spearman1 = evaluate(model1, test_s1, test_s2, test_scores, tokenizer, device)
    
    print(f"\n✅ Stage 1 Results:")
    print(f"   Pearson:  {pearson1:.4f}")
    print(f"   Spearman: {spearman1:.4f} ⭐\n")
    
    # ========================================================================
    # STAGE 2: KERNEL BOOST (Student → Kernel(Teacher))
    # ========================================================================
    print("="*80)
    print("STAGE 2: KERNEL-BOOSTED DISTILLATION")
    print("="*80)
    print(f"Target: Kernel-enhanced teacher embeddings\n")
    
    model2 = StudentModel(d=384, layers=4).to(device)
    
    loss2 = train_student(
        model2, pairs, labels,
        kernel_s1, kernel_s2,  # ← Kernel-boosted targets
        tokenizer, device, steps
    )
    
    print(f"Training loss: {loss2:.4f}\n")
    print("Evaluating...")
    pearson2, spearman2 = evaluate(model2, test_s1, test_s2, test_scores, tokenizer, device)
    
    print(f"\n✅ Stage 2 Results:")
    print(f"   Pearson:  {pearson2:.4f}")
    print(f"   Spearman: {spearman2:.4f} ⭐\n")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    improvement = spearman2 - spearman1
    improvement_pct = (improvement / spearman1 * 100) if spearman1 > 0 else 0
    
    print(f"\n📊 Stage 1 (Baseline): {spearman1:.4f}")
    print(f"📊 Stage 2 (Kernel):   {spearman2:.4f}")
    print(f"📊 Improvement:        {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    if 'validation_metrics' in kernel_state:
        expected = kernel_state['validation_metrics']['quality_score']
        print(f"\n💡 Expected kernel boost: +{expected:.4f}")
        print(f"💡 Achieved boost:        {improvement:+.4f}")
        
        if improvement >= expected * 0.3:  # At least 30% of expected
            print(f"\n✅ SUCCESS! Kernel boost transferred to student!")
        elif improvement > 0:
            print(f"\n⚠️  Partial success - kernel helps but not as much as expected")
        else:
            print(f"\n❌ Kernel didn't help - need more training steps or debugging")
    
    print("\n" + "="*80)
    
    return {
        'baseline': spearman1,
        'kernel': spearman2,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Training steps per stage")
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 SIMPLE KERNEL BOOST TEST")
    print("="*80)
    print(f"\nStage 1: Student learns from Teacher")
    print(f"Stage 2: Student learns from Kernel(Teacher)")
    print(f"Goal: Stage 2 > Stage 1 by ~0.04 Spearman")
    print(f"\nTraining: {args.steps} steps per stage")
    print(f"Expected time: ~5-10 minutes\n")
    print("="*80 + "\n")
    
    results = main(steps=args.steps)
    
    if results:
        print(f"\n🎉 Test complete!")
        print(f"Kernel added: {results['improvement']:+.4f} Spearman")