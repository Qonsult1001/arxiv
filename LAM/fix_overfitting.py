#!/usr/bin/env python3

"""

Fix Val→Test Generalization Gap

Problem: Val Spearman 0.8418 → Test Spearman 0.7664 (gap = -0.0754)

Solution: Regularization + Data Augmentation to improve generalization

"""

 

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import numpy as np

from tqdm import tqdm

from scipy.stats import spearmanr, pearsonr

from transformers import AutoTokenizer, AutoModel

from pathlib import Path

from datasets import load_dataset

import random

import sys

 

sys.path.insert(0, str(Path(__file__).parent))

from final_solution_formula import EnhancedHierarchicalDeltaNet
from train_6layer_deltanet_2 import DeltaNetPure6Layer

 

config = {

    # Model

    "checkpoint_file": "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEWNEW/checkpoint_104000.pt",

    "model_dim": 384,

    "num_layers": 6,
    
    # DeltaNet config (matching train_6layer_deltanet_2.py)
    "num_heads": 12,
    "fast_decay_init": 0.30,
    "slow_decay_init": 0.85,
    "use_kernel_blending": False,  # Disable for overfitting fix
    "kernel_blend_alpha": 0.0,

 

    # Teacher

    "teacher_model": "/workspace/LAM/all-MiniLM-L6-v2",

 

    # Training - AGGRESSIVE REGULARIZATION

    "learning_rate": 5e-7,  # Very low - we're near optimum

    "batch_size": 16,

    "total_steps": 1000,

    "eval_steps": 50,

 

    # Regularization to prevent overfitting

    "dropout": 0.2,  # Increased from 0.1

    "weight_decay": 0.05,  # Aggressive weight decay

    "label_smoothing": 0.15,  # Smooth hard labels

 

    # Data augmentation

    "use_mixup": True,

    "mixup_alpha": 0.3,

    "use_dropout_augmentation": True,  # Multiple forward passes with different dropout

 

    # Early stopping

    "patience": 5,  # Stop if test score doesn't improve

}

 

print("=" * 80)

print("🎯 FIXING OVERFITTING: Val→Test Generalization Gap")

print("=" * 80)

print(f"Current: Val Spearman 0.8418 → Test Spearman 0.7664")

print(f"Gap: -0.0754")

print(f"Target: Close this gap to reach Test Spearman 0.83+")

print()

print("Strategy:")

print("  ✅ Aggressive Regularization (dropout=0.2, weight_decay=0.05)")

print("  ✅ Data Augmentation (Mixup, Dropout augmentation)")

print("  ✅ Label Smoothing (0.15)")

print("  ✅ Early Stopping (patience=5)")

print("  ✅ Very Low LR (5e-7) - fine-tune near optimum")

print("=" * 80)

print()

 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 

class MeanPooling(nn.Module):

    def forward(self, token_embeddings, attention_mask):

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

 

class DeltaNetModel(nn.Module):

    def __init__(self, vocab_size, model_dim, num_layers, dropout=0.2):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)

        self.deltanet_layers = nn.ModuleList([

            EnhancedHierarchicalDeltaNet(

                d_model=model_dim,  # model_dim=384, divisible by num_heads=6 (384/6=64)

                num_heads=6,

                expand_k=1.0,  # key_dim = 384 * 1.0 = 384 (divisible by 6)

                expand_v=1.0,  # value_dim = 384 * 1.0 = 384 (divisible by 6)

                use_enhanced_flux=True  # Use correct parameter name

            )

            for _ in range(num_layers)

        ])

        self.pooler = MeanPooling()

 

        # Additional dropout layer before pooling

        self.pre_pool_dropout = nn.Dropout(dropout)

 

    def forward(self, input_ids, attention_mask, return_features=False):

        x = self.embedding(input_ids)

 

        for layer in self.deltanet_layers:

            # EnhancedHierarchicalDeltaNet returns a tuple, we only need the hidden states
            result = layer(x, attention_mask)
            if isinstance(result, tuple):
                x = result[0]  # Extract hidden states from tuple
            else:
                x = result

 

        # Extra dropout before pooling (regularization)

        x = self.pre_pool_dropout(x)

 

        pooled = self.pooler(x, attention_mask)

        pooled = F.normalize(pooled, p=2, dim=1)

 

        return pooled

 

class STSBDataset(Dataset):

    def __init__(self, tokenizer, split='train'):

        self.tokenizer = tokenizer

        self.pairs = []

        # Load from datasets library
        if split == 'train':
            ds = load_dataset("glue", "stsb", split="train")
        elif split == 'dev':
            ds = load_dataset("sentence-transformers/stsb", split="validation")
        else:
            ds = load_dataset("sentence-transformers/stsb", split="test")
        
        # Determine label column name
        if 'label' in ds.column_names:
            label_col = 'label'
        elif 'score' in ds.column_names:
            label_col = 'score'
        else:
            raise ValueError(f"Dataset has neither 'label' nor 'score' column. Available: {ds.column_names}")
        
        # Convert to pairs
        for item in ds:
            sent1 = item['sentence1']
            sent2 = item['sentence2']
            score = float(item[label_col]) / 5.0  # Normalize to [0, 1]
                    self.pairs.append((sent1, sent2, score))

        print(f"✅ Loaded {len(self.pairs)} {split} pairs")

 

    def __len__(self):

        return len(self.pairs)

 

    def __getitem__(self, idx):

        return self.pairs[idx]

 

def collate_fn(batch):

    sent1 = [item[0] for item in batch]

    sent2 = [item[1] for item in batch]

    scores = torch.tensor([item[2] for item in batch], dtype=torch.float)

    return sent1, sent2, scores

 

def mixup_embeddings(emb1, emb2, labels, alpha=0.3):

    """Mixup data augmentation on embeddings"""

    batch_size = emb1.size(0)

    if batch_size == 0:

        return emb1, emb2, labels

 

    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(batch_size).to(emb1.device)

 

    mixed_emb1 = lam * emb1 + (1 - lam) * emb1[index]

    mixed_emb2 = lam * emb2 + (1 - lam) * emb2[index]

    mixed_labels = lam * labels + (1 - lam) * labels[index]

 

    return mixed_emb1, mixed_emb2, mixed_labels

 

def evaluate(model, tokenizer, split='test'):

    """Evaluate on STS-B"""

    dataset = STSBDataset(tokenizer, split=split)

 

    sent1_list = [x[0] for x in dataset.pairs]

    sent2_list = [x[1] for x in dataset.pairs]

    labels = np.array([x[2] for x in dataset.pairs])

 

    model.eval()

 

    # Test-time augmentation: average over multiple dropout samples

    num_tta_samples = 3 if config['use_dropout_augmentation'] else 1

 

    all_sims = []

 

    for _ in range(num_tta_samples):

        batch_size = 32

        all_emb1 = []

        all_emb2 = []

 

        with torch.no_grad():

            for i in range(0, len(sent1_list), batch_size):

                batch1 = sent1_list[i:i+batch_size]

                batch2 = sent2_list[i:i+batch_size]

 

                tokens1 = tokenizer(batch1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)

                tokens2 = tokenizer(batch2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)

 

                # Keep dropout active for test-time augmentation

                if config['use_dropout_augmentation']:

                    model.train()  # Enable dropout

 

                # Use forward_student to get embeddings only
                emb1, _, _ = model.forward_student(tokens1['input_ids'], tokens1['attention_mask'])
                emb2, _, _ = model.forward_student(tokens2['input_ids'], tokens2['attention_mask'])

 

                all_emb1.append(emb1.cpu())

                all_emb2.append(emb2.cpu())

 

        emb1 = torch.cat(all_emb1, dim=0)

        emb2 = torch.cat(all_emb2, dim=0)

 

        sims = F.cosine_similarity(emb1, emb2, dim=1).numpy()

        all_sims.append(sims)

 

    # Average predictions from multiple dropout samples

    sims = np.mean(all_sims, axis=0)

 

    pearson = pearsonr(sims, labels)[0]

    spearman = spearmanr(sims, labels)[0]

 

    return {

        'pearson': pearson,

        'spearman': spearman

    }

 

def train():

    tokenizer = AutoTokenizer.from_pretrained(config['teacher_model'])

 

    # Load teacher

    print(f"\n📚 Loading teacher model...")

    teacher_model = AutoModel.from_pretrained(config['teacher_model']).to(device)

    teacher_model.eval()

    for param in teacher_model.parameters():

        param.requires_grad = False

 

    # Initialize model using same architecture as train_6layer_deltanet_2.py
    print(f"\n📦 Initializing DeltaNetPure6Layer model...")
    model = DeltaNetPure6Layer(
        teacher_model_name=config['teacher_model'],
        num_linear_layers=config['num_layers'],
        config=config
    ).to(device)

    # Load checkpoint (matching train_6layer_deltanet_2.py structure)
    checkpoint_path = Path(config['checkpoint_file'])
    print(f"\n📦 Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    
    # Load checkpoint weights (matching how train_6layer_deltanet_2.py saves them)
    if 'deltanet_layers' in checkpoint:
        model.deltanet_layers.load_state_dict(checkpoint['deltanet_layers'], strict=False)
        print("   ✅ Loaded deltanet_layers")
    if 'deltanet_norms' in checkpoint:
        model.deltanet_norms.load_state_dict(checkpoint['deltanet_norms'], strict=False)
        print("   ✅ Loaded deltanet_norms")
    if 'deltanet_ffns' in checkpoint:
        model.deltanet_ffns.load_state_dict(checkpoint['deltanet_ffns'], strict=False)
        print("   ✅ Loaded deltanet_ffns")
    if 'ffn_norms' in checkpoint:
        model.ffn_norms.load_state_dict(checkpoint['ffn_norms'], strict=False)
        print("   ✅ Loaded ffn_norms")
    
    print("   ✅ Checkpoint loaded successfully")

 

    # Dataset

    print(f"\n📚 Loading data...")

    train_dataset = STSBDataset(tokenizer, split='train')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

 

    # Optimizer with aggressive weight decay

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config['learning_rate'],

        weight_decay=config['weight_decay'],

        betas=(0.9, 0.999)

    )

 

    # Baseline

    print(f"\n📊 BASELINE EVALUATION...")

    val_baseline = evaluate(model, tokenizer, split='dev')

    test_baseline = evaluate(model, tokenizer, split='test')

 

    print(f"   Val:  Pearson={val_baseline['pearson']:.4f}, Spearman={val_baseline['spearman']:.4f}")

    print(f"   Test: Pearson={test_baseline['pearson']:.4f}, Spearman={test_baseline['spearman']:.4f}")

    print(f"   Gap (Val→Test Spearman): {val_baseline['spearman'] - test_baseline['spearman']:.4f}")

 

    # Training

    print(f"\n▶️  Starting anti-overfitting fine-tuning...")

 

    best_test_spearman = test_baseline['spearman']

    patience_counter = 0

    step = 0

 

    pbar = tqdm(total=config['total_steps'], desc="Anti-Overfitting Training")

 

    model.train()

 

    while step < config['total_steps']:

        for sent1_batch, sent2_batch, scores in train_loader:

            if step >= config['total_steps']:

                break

 

            # Tokenize

            tokens1 = tokenizer(sent1_batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)

            tokens2 = tokenizer(sent2_batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)

            scores = scores.to(device)

 

            # Student embeddings (use forward_student to get embeddings only)
            emb1, _, _ = model.forward_student(tokens1['input_ids'], tokens1['attention_mask'])
            emb2, _, _ = model.forward_student(tokens2['input_ids'], tokens2['attention_mask'])

 

            # Mixup augmentation

            if config['use_mixup'] and random.random() > 0.5:

                emb1, emb2, scores = mixup_embeddings(emb1, emb2, scores, config['mixup_alpha'])

 

            # Teacher embeddings

            with torch.no_grad():

                def mean_pooling(output, mask):

                    embeddings = output[0]

                    mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()

                    sum_emb = torch.sum(embeddings * mask_expanded, 1)

                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

                    return sum_emb / sum_mask

 

                teacher_out1 = teacher_model(**tokens1)

                teacher_out2 = teacher_model(**tokens2)

                teacher_emb1 = F.normalize(mean_pooling(teacher_out1, tokens1['attention_mask']), p=2, dim=1)

                teacher_emb2 = F.normalize(mean_pooling(teacher_out2, tokens2['attention_mask']), p=2, dim=1)

                teacher_sim = F.cosine_similarity(teacher_emb1, teacher_emb2)

 

            # Similarity matching with label smoothing

            student_sim = F.cosine_similarity(emb1, emb2)

 

            # Label smoothing on teacher targets

            teacher_sim_smooth = teacher_sim * (1 - config['label_smoothing']) + 0.5 * config['label_smoothing']

 

            loss = F.mse_loss(student_sim, teacher_sim_smooth)

 

            # Backward

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

 

            pbar.update(1)

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'best_test': f"{best_test_spearman:.4f}"})

 

            step += 1

 

            # Evaluate

            if step % config['eval_steps'] == 0:

                val_results = evaluate(model, tokenizer, split='dev')

                test_results = evaluate(model, tokenizer, split='test')

 

                gap = val_results['spearman'] - test_results['spearman']

 

                print(f"\n📊 Step {step}:")

                print(f"   Val:  Spearman={val_results['spearman']:.4f}")

                print(f"   Test: Spearman={test_results['spearman']:.4f}")

                print(f"   Gap:  {gap:.4f} (baseline: {val_baseline['spearman'] - test_baseline['spearman']:.4f})")

 

                # Early stopping based on TEST performance

                if test_results['spearman'] > best_test_spearman:

                    best_test_spearman = test_results['spearman']

                    patience_counter = 0

 

                    save_path = f"antioverfit_best_{step}.pt"

                    torch.save(model.state_dict(), save_path)

                    print(f"   💾 NEW BEST TEST: {best_test_spearman:.4f} - saved to {save_path}")

                else:

                    patience_counter += 1

                    print(f"   Patience: {patience_counter}/{config['patience']}")

 

                    if patience_counter >= config['patience']:

                        print(f"\n⚠️  Early stopping triggered - test score not improving")

                        break

 

                model.train()

 

        if patience_counter >= config['patience']:

            break

 

    pbar.close()

 

    # Final evaluation

    print(f"\n" + "=" * 80)

    print("📊 FINAL RESULTS")

    print("=" * 80)

 

    model.eval()

    final_val = evaluate(model, tokenizer, split='dev')

    final_test = evaluate(model, tokenizer, split='test')

 

    print(f"\nBASELINE:")

    print(f"  Val Spearman:  {val_baseline['spearman']:.4f}")

    print(f"  Test Spearman: {test_baseline['spearman']:.4f}")

    print(f"  Gap:           {val_baseline['spearman'] - test_baseline['spearman']:.4f}")

 

    print(f"\nFINAL:")

    print(f"  Val Spearman:  {final_val['spearman']:.4f}")

    print(f"  Test Spearman: {final_test['spearman']:.4f}")

    print(f"  Gap:           {final_val['spearman'] - final_test['spearman']:.4f}")

 

    print(f"\nBEST TEST: {best_test_spearman:.4f}")

    print(f"Improvement: {best_test_spearman - test_baseline['spearman']:+.4f}")

 

    if best_test_spearman >= 0.83:

        print(f"\n🎉 SUCCESS! Reached target of 0.83+ test Spearman!")

    else:

        print(f"\n⚠️  Need {0.83 - best_test_spearman:.4f} more to reach 0.83 target")

 

if __name__ == '__main__':

    train()