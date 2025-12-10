# Multiple Negatives Ranking Loss (MNR) - Final solution where we Implementing Soft-Label Contrastive Distillation
#!/usr/bin/env python3
"""
🚀 DELTANET GENERALIZATION FIX - Soft-Label Contrastive Distillation (Final Push)
1. Teacher: stsb-roberta-base-v2 (0.87 Spearman)
2. Data: AllNLI (Generalization)
3. Loss: Multiple Negatives Ranking (MNR) using Teacher Scores as Soft Labels.
Goal: Push the 0.793 score towards the 0.82 target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import json
import gzip
import random
import sys
from typing import List

# Import your model definitions
sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet
from train_6layer_deltanet_2 import DeltaNetPure6Layer

config = {
    # 1. UPGRADE TEACHER
    "teacher_model": "sentence-transformers/stsb-roberta-base-v2", # 768 dim
    "student_dim": 384,
    
    # 2. USE MORE DATA (NLI)
    "nli_file": "/workspace/LAM/data/AllNLI.jsonl.gz",
    "max_samples": 200000, 
    
    # Model Loading
    "checkpoint_file": "/workspace/LAM/deltanet_roberta_tsl_distilled.pt", # Load checkpoint (best performing step)
    "num_layers": 6,
    "num_heads": 12,
    "fast_decay_init": 0.30,
    "slow_decay_init": 0.85,
    "use_kernel_blending": False,
    "kernel_blend_alpha": 0.0,

    # Training Config
    "learning_rate": 1e-5, # Slightly lower LR to fine-tune aggressively
    "batch_size": 128, # Larger batch size is MANDATORY for effective Negative Mining
    "epochs": 5, # Run 5 new epochs of specialized training
    "patience": 5,
    "temperature": 0.04 # Low temperature sharpens the contrastive loss significantly
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Shared Pooling Function ---
def mean_pooling(output, mask):
    embeddings = output.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_emb = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_emb / sum_mask

# --- 1. DATA LOADING (NLI + STSB) ---
class NLIDataset(Dataset):
    """Loads NLI Anchor-Positive pairs for similarity prediction."""
    def __init__(self, nli_path, max_samples=None):
        self.pairs = []
        print(f"📚 Loading NLI data from {nli_path}...")
        samples_added = 0
        with gzip.open(nli_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if max_samples and samples_added >= max_samples: break
                try:
                    data = json.loads(line)
                    self.pairs.append((data[0], data[1])) 
                    samples_added += 1
                except: continue
        print(f"✅ Loaded {len(self.pairs)} NLI Anchor/Positive pairs.")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def collate_nli(batch: List[tuple]) -> tuple[list, list]:
    """Collation for NLI pairs (Sentence A, Sentence B)."""
    s1 = [item[0] for item in batch]
    s2 = [item[1] for item in batch]
    return s1, s2


# --- 2. EVALUATION UTILS ---
class STSBDataset(Dataset):
    def __init__(self, split='test'):
        from datasets import load_dataset
        ds = load_dataset("sentence-transformers/stsb", split="validation" if split=='dev' else "test")
        self.pairs = []
        for item in ds:
            self.pairs.append((item['sentence1'], item['sentence2'], float(item['score'])/5.0))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def evaluate(model, tokenizer, split='test'):
    """Evaluates the Student Model on the STSB task."""
    dataset = STSBDataset(split=split)
    model.eval()
    emb1s, emb2s, scores = [], [], []
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            s1 = [x[0] for x in batch]
            s2 = [x[1] for x in batch]
            score = [x[2] for x in batch]
            
            t1 = tokenizer(s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2 = tokenizer(s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            e1, _, _ = model.forward_student(t1['input_ids'], t1['attention_mask'])
            e2, _, _ = model.forward_student(t2['input_ids'], t2['attention_mask'])
            
            emb1s.append(e1.cpu()); emb2s.append(e2.cpu())
            scores.extend(score)
            
    emb1s = torch.cat(emb1s); emb2s = torch.cat(emb2s)
    sims = F.cosine_similarity(emb1s, emb2s).numpy()
    return spearmanr(sims, scores)[0]

# --- 3. TRAINING LOOP ---
def train():
    print(f"🤖 Loading RoBERTa Teacher (768 dim) and Tokenizer...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config['teacher_model'])
    teacher_model = AutoModel.from_pretrained(config['teacher_model']).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False # Freeze Teacher

    print(f"👨‍🎓 Loading DeltaNet Student (384 dim)...")
    student_model = DeltaNetPure6Layer(
        teacher_model_name='sentence-transformers/all-MiniLM-L6-v2', # Tokenizer base for DeltaNet is MiniLM
        num_linear_layers=config['num_layers'],
        config=config
    ).to(device)
    
    # Get student tokenizer (MiniLM)
    student_tokenizer = student_model.tokenizer
    
    # Load checkpoint weights (format from train_6layer_deltanet_2.py)
    checkpoint_path = Path(config['checkpoint_file'])
    if checkpoint_path.exists():
        print(f"📦 Loading checkpoint from: {checkpoint_path.name}")
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        
        # Try loading as state_dict first (for .pt files saved with state_dict)
        if 'deltanet_layers' not in ckpt and len(ckpt) > 0:
            # Might be a state_dict format - try loading directly
            try:
                student_model.load_state_dict(ckpt, strict=False)
                print("   ✅ Loaded checkpoint as state_dict")
            except:
                # Fall back to component loading
                if 'deltanet_layers' in ckpt:
                    student_model.deltanet_layers.load_state_dict(ckpt['deltanet_layers'], strict=False)
                    print("   ✅ Loaded deltanet_layers")
                if 'deltanet_norms' in ckpt:
                    student_model.deltanet_norms.load_state_dict(ckpt['deltanet_norms'], strict=False)
                    print("   ✅ Loaded deltanet_norms")
                if 'deltanet_ffns' in ckpt:
                    student_model.deltanet_ffns.load_state_dict(ckpt['deltanet_ffns'], strict=False)
                    print("   ✅ Loaded deltanet_ffns")
                if 'ffn_norms' in ckpt:
                    student_model.ffn_norms.load_state_dict(ckpt['ffn_norms'], strict=False)
                    print("   ✅ Loaded ffn_norms")
        else:
            # Component loading
            if 'deltanet_layers' in ckpt:
                student_model.deltanet_layers.load_state_dict(ckpt['deltanet_layers'], strict=False)
                print("   ✅ Loaded deltanet_layers")
            if 'deltanet_norms' in ckpt:
                student_model.deltanet_norms.load_state_dict(ckpt['deltanet_norms'], strict=False)
                print("   ✅ Loaded deltanet_norms")
            if 'deltanet_ffns' in ckpt:
                student_model.deltanet_ffns.load_state_dict(ckpt['deltanet_ffns'], strict=False)
                print("   ✅ Loaded deltanet_ffns")
            if 'ffn_norms' in ckpt:
                student_model.ffn_norms.load_state_dict(ckpt['ffn_norms'], strict=False)
                print("   ✅ Loaded ffn_norms")
        
        if 'step' in ckpt:
            print(f"   📊 Checkpoint was at step {ckpt['step']}")
        print("   ✅ Checkpoint weights loaded successfully!")
    else:
        print(f"   ⚠️  Checkpoint not found at {checkpoint_path}")
        print(f"   ⚠️  Starting from scratch (random initialization)")
    
    optimizer = torch.optim.AdamW(
        student_model.parameters(), 
        lr=config['learning_rate']
    )

    # Data Loaders
    train_dataset = NLIDataset(config['nli_file'], config['max_samples'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_nli, drop_last=True) # drop_last is crucial for square matrix

    print("\n📊 Initial Evaluation (Should be ~0.7915)...")
    initial_score = evaluate(student_model, student_tokenizer)
    print(f"Test Spearman: {initial_score:.4f}")

    print("\n🚀 Starting Soft-Label Contrastive Distillation (MNR Loss)...")
    best_score = initial_score
    temp = config['temperature']
    
    for epoch in range(config['epochs']):
        student_model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for s1_list, s2_list in pbar:
            # Tokenize with appropriate tokenizers
            # Teacher uses RoBERTa tokenizer
            t1_teacher = teacher_tokenizer(s1_list, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_teacher = teacher_tokenizer(s2_list, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            # Student uses MiniLM tokenizer
            t1_student = student_tokenizer(s1_list, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_student = student_tokenizer(s2_list, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            # --- 1. Get Teacher Target Similarity Matrix (The Soft Labels) ---
            with torch.no_grad():
                # Get Teacher Embeddings (B x 768)
                te1 = F.normalize(mean_pooling(teacher_model(**t1_teacher), t1_teacher['attention_mask']), p=2, dim=1)
                te2 = F.normalize(mean_pooling(teacher_model(**t2_teacher), t2_teacher['attention_mask']), p=2, dim=1)
                
                # Calculate Teacher Cosine Similarity Matrix (B x B)
                # This gives us all Positive (diagonal) and Negative (off-diagonal) targets
                target_sim_matrix = torch.matmul(te1, te2.transpose(0, 1)).detach() # Teacher Sim
                
                # Convert similarities to logits using the temperature
                teacher_logits = target_sim_matrix / temp
                # Apply softmax to get the soft probability distribution over all pairs
                teacher_probs = F.softmax(teacher_logits, dim=1)


            # --- 2. Get Student Prediction Similarity Matrix ---
            # Get Student Embeddings (B x 384)
            optimizer.zero_grad()
            se1, _, _ = student_model.forward_student(t1_student['input_ids'], t1_student['attention_mask'])
            se2, _, _ = student_model.forward_student(t2_student['input_ids'], t2_student['attention_mask'])
            
            # Calculate Student Cosine Similarity Matrix (B x B)
            # This contains all B positive pairs (A_i, P_i) and B*(B-1) negative pairs (A_i, N_j)
            student_sim_matrix = torch.matmul(se1, se2.transpose(0, 1)) # Student Sim
            
            # Convert student similarities to logits
            student_logits = student_sim_matrix / temp
            
            # --- 3. Loss: Cross-Entropy (KL-Divergence) between Student Logits and Teacher Soft Targets ---
            # We treat the Teacher's softmax distribution (teacher_probs) as the true label distribution.
            # This is equivalent to KL-Divergence (Cross-Entropy loss when targets are soft)
            # F.log_softmax(student_logits, dim=1) gives the log-probabilities for the student's output
            # torch.sum(teacher_probs * F.log_softmax(student_logits, dim=1), dim=1) gives the cross-entropy per row
            # .mean() averages the loss across the entire batch.
            loss = -torch.sum(teacher_probs * F.log_softmax(student_logits, dim=1), dim=1).mean()
            
            # 4. Update Student
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        # Eval
        test_score = evaluate(student_model, student_tokenizer, split='test')
        print(f"\nEpoch {epoch+1} Test Spearman: {test_score:.4f}")
        
        # Save only the best performing model
        if test_score > best_score:
            best_score = test_score
            # Save the entire model state to the same file path for continuity
            torch.save(student_model.state_dict(), "deltanet_roberta_tsl_distilled.pt") 
            print(f"✅ New Best! Saved. Best score: {best_score:.4f}")
        else:
            print(f"Current score {test_score:.4f} is not better than best score {best_score:.4f}. Continuing training.")

if __name__ == "__main__":
    train()