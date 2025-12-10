#!/usr/bin/env python3
"""
🚀 DELTANET MULTITASK DISTILLATION (Fixed)
Correction: Uses Teacher Scores for BOTH NLI and STS-B to prevent objective conflict.
Reduced STS-B weight to prevent noisy gradient interference.
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
from datasets import load_dataset

# Import your model definitions
sys.path.insert(0, str(Path(__file__).parent))
from train_6layer_deltanet_2 import DeltaNetPure6Layer

config = {
    # 1. TEACHER & STUDENT
    "teacher_model": "sentence-transformers/stsb-roberta-base-v2", 
    "student_dim": 384,
    
    # 2. DATA
    "nli_file": "/workspace/LAM/data/AllNLI.jsonl.gz",
    "max_nli_samples": 200000, 
    
    # 3. MODEL LOADING
    # Load your BEST NLI checkpoint (0.7979)
    "checkpoint_file": "deltanet_roberta_tsl_distilled.pt",
    "num_layers": 6,
    "num_heads": 12,
    "fast_decay_init": 0.30,
    "slow_decay_init": 0.85,
    "use_kernel_blending": False,
    "kernel_blend_alpha": 0.0,

    # 4. TRAINING CONFIG
    "learning_rate": 5e-6, 
    "batch_size_nli": 64,
    "batch_size_stsb": 32,
    "epochs": 5, 
    "temperature": 0.04,
    # ⬇️ CHANGED: Reduced weight to prevent destabilization
    "stsb_loss_weight": 1.0, 
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- SHARED UTILS ---
def mean_pooling(output, mask):
    embeddings = output.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_emb = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_emb / sum_mask

# --- DATASETS ---
class NLIDataset(Dataset):
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
        print(f"✅ Loaded {len(self.pairs)} NLI pairs.")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

class STSBDataset(Dataset):
    def __init__(self, split='train'):
        ds = load_dataset("sentence-transformers/stsb", split=split)
        self.pairs = []
        for item in ds:
            # Return (s1, s2) - we don't need human score, we use Teacher score!
            self.pairs.append((item['sentence1'], item['sentence2']))
        print(f"✅ Loaded {len(self.pairs)} STS-B {split} pairs.")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

# --- COLLATORS ---
def collate_nli(batch):
    s1 = [item[0] for item in batch]
    s2 = [item[1] for item in batch]
    return s1, s2

def collate_stsb(batch):
    s1 = [item[0] for item in batch]
    s2 = [item[1] for item in batch]
    return s1, s2

# --- EVALUATION ---
def evaluate(model, tokenizer, split='test'):
    ds = load_dataset("sentence-transformers/stsb", split="validation" if split=='dev' else "test")
    pairs = []
    for item in ds:
        pairs.append((item['sentence1'], item['sentence2'], float(item['score'])/5.0))

    model.eval()
    emb1s, emb2s, scores = [], [], []
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
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

# --- MAIN TRAINING ---
def train():
    print(f"🤖 Loading Teacher: {config['teacher_model']}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config['teacher_model'])
    teacher_model = AutoModel.from_pretrained(config['teacher_model']).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters(): p.requires_grad = False

    print(f"👨‍🎓 Loading Student (DeltaNet)...")
    student_model = DeltaNetPure6Layer(
        teacher_model_name='sentence-transformers/all-MiniLM-L6-v2',
        num_linear_layers=config['num_layers'],
        config=config
    ).to(device)
    
    # Get student tokenizer (MiniLM)
    student_tokenizer = student_model.tokenizer
    
    # Load BEST NLI Checkpoint
    print(f"📦 Loading checkpoint: {config['checkpoint_file']}")
    ckpt = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
    student_model.load_state_dict(ckpt, strict=False)
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config['learning_rate'])
    
    nli_loader = DataLoader(NLIDataset(config['nli_file'], config['max_nli_samples']), 
                           batch_size=config['batch_size_nli'], shuffle=True, collate_fn=collate_nli, drop_last=True)
    
    stsb_loader = DataLoader(STSBDataset(split='train'), 
                             batch_size=config['batch_size_stsb'], shuffle=True, collate_fn=collate_stsb)
    
    print("\n📊 Initial Baseline Score...")
    best_score = evaluate(student_model, student_tokenizer)
    print(f"Starting Test Spearman: {best_score:.4f}")
    
    print("\n🚀 Starting Multitask Distillation (Consistent Teacher Target)...")
    temp = config['temperature']
    mse_loss_fn = nn.MSELoss()

    for epoch in range(config['epochs']):
        student_model.train()
        stsb_iter = iter(stsb_loader)
        
        pbar = tqdm(nli_loader, desc=f"Epoch {epoch+1}")
        total_loss_nli = 0
        total_loss_stsb = 0
        
        for nli_s1, nli_s2 in pbar:
            optimizer.zero_grad()
            
            # --- TASK 1: NLI (Contrastive Distillation) ---
            # Teacher uses RoBERTa tokenizer
            t1_teacher = teacher_tokenizer(nli_s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_teacher = teacher_tokenizer(nli_s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            # Student uses MiniLM tokenizer
            t1_student = student_tokenizer(nli_s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_student = student_tokenizer(nli_s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            with torch.no_grad():
                te1 = F.normalize(mean_pooling(teacher_model(**t1_teacher), t1_teacher['attention_mask']), p=2, dim=1)
                te2 = F.normalize(mean_pooling(teacher_model(**t2_teacher), t2_teacher['attention_mask']), p=2, dim=1)
                teacher_logits = torch.matmul(te1, te2.transpose(0, 1)) / temp
                teacher_probs = F.softmax(teacher_logits, dim=1)
                
            se1, _, _ = student_model.forward_student(t1_student['input_ids'], t1_student['attention_mask'])
            se2, _, _ = student_model.forward_student(t2_student['input_ids'], t2_student['attention_mask'])
            
            student_sim_matrix = torch.matmul(se1, se2.transpose(0, 1))
            student_logits = student_sim_matrix / temp
            loss_nli = -torch.sum(teacher_probs * F.log_softmax(student_logits, dim=1), dim=1).mean()

            # --- TASK 2: STS-B (Teacher Score Distillation) ---
            try:
                stsb_s1, stsb_s2 = next(stsb_iter)
            except StopIteration:
                stsb_iter = iter(stsb_loader)
                stsb_s1, stsb_s2 = next(stsb_iter)
            
            # Teacher uses RoBERTa tokenizer
            t1_stsb_teacher = teacher_tokenizer(stsb_s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_stsb_teacher = teacher_tokenizer(stsb_s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            # Student uses MiniLM tokenizer
            t1_stsb_student = student_tokenizer(stsb_s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_stsb_student = student_tokenizer(stsb_s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            # ⬇️ CHANGED: Use TEACHER to generate target score, not Gold Label
            with torch.no_grad():
                te1_stsb = F.normalize(mean_pooling(teacher_model(**t1_stsb_teacher), t1_stsb_teacher['attention_mask']), p=2, dim=1)
                te2_stsb = F.normalize(mean_pooling(teacher_model(**t2_stsb_teacher), t2_stsb_teacher['attention_mask']), p=2, dim=1)
                teacher_stsb_scores = F.cosine_similarity(te1_stsb, te2_stsb)
            
            se1_stsb, _, _ = student_model.forward_student(t1_stsb_student['input_ids'], t1_stsb_student['attention_mask'])
            se2_stsb, _, _ = student_model.forward_student(t2_stsb_student['input_ids'], t2_stsb_student['attention_mask'])
            
            pred_scores = F.cosine_similarity(se1_stsb, se2_stsb)
            loss_stsb = mse_loss_fn(pred_scores, teacher_stsb_scores)
            
            # Combined Loss
            loss = loss_nli + (config['stsb_loss_weight'] * loss_stsb)
            
            loss.backward()
            optimizer.step()
            
            total_loss_nli += loss_nli.item()
            total_loss_stsb += loss_stsb.item()
            pbar.set_postfix({'NLI': f"{loss_nli.item():.4f}", 'STSB': f"{loss_stsb.item():.4f}"})
            
        # Eval
        test_score = evaluate(student_model, student_tokenizer)
        print(f"\nEpoch {epoch+1} Test Spearman: {test_score:.4f}")
        
        if test_score > best_score:
            best_score = test_score
            torch.save(student_model.state_dict(), "deltanet_multitask_final_corrected.pt")
            print(f"✅ New Best! Saved. ({test_score:.4f})")
            
    print(f"\n🎉 Final Multitask Score: {best_score:.4f}")

if __name__ == "__main__":
    train()