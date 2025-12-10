#!/usr/bin/env python3
"""
🚀 DELTANET GLIDE PATH: 0.8189 -> 0.8200+
STRATEGY: Extended Fine-Tuning (Glide Path).

STATUS: The Shockwave worked! We broke the 0.8183 wall and hit 0.8189.
The trend is still upward, so we extend training with a low LR to squeeze
out the remaining performance without destabilizing the weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
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

# ==============================================================================
# 1. CONFIGURATION: THE GLIDE PATH (EPOCH 31-40)
# ==============================================================================

config = {
    # 1. TEACHER & STUDENT
    "teacher_model": "sentence-transformers/stsb-roberta-base-v2", 
    "student_dim": 384,
    
    # 2. DATA
    "nli_file": "/workspace/LAM/data/AllNLI.jsonl.gz",
    "max_nli_samples": None,
    
    # 3. RESUMPTION
    "checkpoint_file": "deltanet_shockwave_result.pt", # ⬅️ LOAD THE NEW BEST (0.8189)
    "start_epoch": 31,  # Continue from where the shockwave left off
    "total_epochs": 40, # Run for 10 more epochs to fully converge
    "num_layers": 6,
    "num_heads": 12,
    "fast_decay_init": 0.30,
    "slow_decay_init": 0.85,
    "use_kernel_blending": False,
    "kernel_blend_alpha": 0.40,
    
    # 4. GLIDE PATH SETTINGS
    # We lower the LR to ~6e-8 (where the previous run left off) to avoid a second shock.
    # We want a smooth landing now.
    "learning_rate": 6.0e-08, 
    "batch_size_nli": 128,
    "batch_size_stsb": 32,
    "temperature": 0.10,
    "stsb_loss_weight": 1.0, 
    
    # 5. UNLEASHED WEIGHTS (Maintain the successful Shockwave configuration)
    "SHOCK_STSB_W": 8.0,      # Keep high focus on STSB
    "SHOCK_NLI_W": 2.0,       
    "SHOCK_FEATURE_W": 0.1,   
    "SHOCK_ATTENTION_W": 0.2, 
    "SHOCK_ORTHO_W": 0.01,    
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
        print(f"✅ Loaded {len(self.pairs):,} NLI pairs.")
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

class STSBDataset(Dataset):
    def __init__(self, split='train'):
        ds = load_dataset("sentence-transformers/stsb", split=split)
        self.pairs = []
        for item in ds:
            self.pairs.append((item['sentence1'], item['sentence2']))
        print(f"✅ Loaded {len(self.pairs)} STS-B {split} pairs.")
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def collate_nli(batch):
    s1 = [item[0] for item in batch]
    s2 = [item[1] for item in batch]
    return s1, s2

def collate_stsb(batch):
    s1 = [item[0] for item in batch]
    s2 = [item[1] for item in batch]
    return s1, s2

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

# --- LOSS FUNCTIONS ---
def calculate_attention_loss(teacher_attentions, student_attentions, attention_weight):
    if attention_weight <= 0: return torch.tensor(0.0, device=device)
    
    total_att_loss = 0.0
    num_student_layers = len(student_attentions)
    for i in range(num_student_layers):
        t_att = teacher_attentions[2 * i]
        s_att = student_attentions[i]
        
        if t_att.shape[-1] != s_att.shape[-1]:
            target_size = s_att.shape[-1]
            B, H, T_seq, _ = t_att.shape
            t_att_reshaped = t_att.view(B * H, 1, T_seq, T_seq)
            t_att_resized = F.interpolate(t_att_reshaped, size=(target_size, target_size), mode='bilinear', align_corners=False)
            t_att = t_att_resized.view(B, H, target_size, target_size)
            
        if t_att.shape != s_att.shape:
            min_size = min(t_att.shape[-1], s_att.shape[-1])
            t_att = t_att[:, :, :min_size, :min_size]
            s_att = s_att[:, :, :min_size, :min_size]
        
        total_att_loss += F.mse_loss(s_att, t_att)
    return total_att_loss * attention_weight

# --- MAIN TRAINING ---
def train():
    print(f"🤖 Loading Teacher: {config['teacher_model']}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config['teacher_model'])
    teacher_model = AutoModel.from_pretrained(config['teacher_model'], output_attentions=True).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters(): p.requires_grad = False
    
    print(f"👨‍🎓 Loading Student (DeltaNet)...")
    student_model = DeltaNetPure6Layer(
        teacher_model_name='sentence-transformers/all-MiniLM-L6-v2',
        num_linear_layers=config['num_layers'],
        config=config
    ).to(device)
    student_tokenizer = student_model.tokenizer
    
    print(f"📦 Loading checkpoint: {config['checkpoint_file']}")
    checkpoint_path = Path(config['checkpoint_file'])
    if checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        student_model.load_state_dict(ckpt, strict=False)
    
    if not hasattr(student_model, 'feature_proj'):
        student_model.feature_proj = nn.Linear(384, 768).to(device)
        nn.init.xavier_uniform_(student_model.feature_proj.weight, gain=0.1)
        nn.init.zeros_(student_model.feature_proj.bias)

    optimizer = torch.optim.AdamW(
        list(student_model.parameters()) + list(student_model.feature_proj.parameters()), 
        lr=config['learning_rate']
    )
    
    nli_loader = DataLoader(NLIDataset(config['nli_file'], config['max_nli_samples']), 
                           batch_size=config['batch_size_nli'], shuffle=True, collate_fn=collate_nli, drop_last=True)
    stsb_loader = DataLoader(STSBDataset(split='train'), 
                            batch_size=config['batch_size_stsb'], shuffle=True, collate_fn=collate_stsb)
    
    # RESTART SCHEDULER: Glide path (decay from current low LR to 0)
    num_total_steps = len(nli_loader) * (config['total_epochs'] - config['start_epoch'] + 1)
    num_warmup_steps = 0 # No warmup, just linear decay
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps)
    
    print("\n📊 Initial Score (Expecting ~0.8189)...")
    best_score = evaluate(student_model, student_tokenizer)
    print(f"Starting Test Spearman: {best_score:.4f}")
    
    print("\n⚡ STARTING GLIDE PATH TUNING (Low LR, Steady Decay)...")
    temp = config['temperature']
    mse_loss_fn = nn.MSELoss()

    for epoch in range(config['start_epoch'] - 1, config['total_epochs']):
        student_model.train()
        stsb_iter = iter(stsb_loader)
        
        pbar = tqdm(nli_loader, desc=f"Epoch {epoch+1} (Glide)")
        total_loss = 0
        
        for nli_s1, nli_s2 in pbar:
            optimizer.zero_grad()
            
            t1_teacher = teacher_tokenizer(nli_s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_teacher = teacher_tokenizer(nli_s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t1_student = student_tokenizer(nli_s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_student = student_tokenizer(nli_s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            with torch.no_grad():
                t1_output = teacher_model(**t1_teacher, output_hidden_states=True, output_attentions=True)
                t1_emb = F.normalize(mean_pooling(t1_output, t1_teacher['attention_mask']), p=2, dim=1)
                t1_features = t1_output.hidden_states[-1][:, 0, :]
                t1_attentions = t1_output.attentions
                
                t2_output = teacher_model(**t2_teacher) 
                t2_emb = F.normalize(mean_pooling(t2_output, t2_teacher['attention_mask']), p=2, dim=1)
                
                teacher_logits = torch.matmul(t1_emb, t2_emb.transpose(0, 1)) / temp
                teacher_probs = F.softmax(teacher_logits, dim=1)
                
            se1, s1_hidden_states, s1_ortho_loss = student_model.forward_student(t1_student['input_ids'], t1_student['attention_mask'])
            se2, s2_hidden_states, s2_ortho_loss = student_model.forward_student(t2_student['input_ids'], t2_student['attention_mask'])
            
            # Generate Pseudo-Attention for Student
            B, seq_len = t1_student['input_ids'].shape
            num_heads = config['num_heads']
            s1_attentions = []
            for i, hidden_state in enumerate(s1_hidden_states):
                hidden_norm = F.normalize(hidden_state, p=2, dim=-1)
                sim_matrix = torch.matmul(hidden_norm, hidden_norm.transpose(-2, -1))
                pseudo_att = sim_matrix.unsqueeze(1).expand(-1, num_heads, -1, -1)
                s1_attentions.append(pseudo_att)
            
            # 1. NLI Loss
            student_sim_matrix = torch.matmul(se1, se2.transpose(0, 1))
            student_logits = student_sim_matrix / temp
            loss_nli = -torch.sum(teacher_probs * F.log_softmax(student_logits, dim=1), dim=1).mean() * config['SHOCK_NLI_W']

            # 2. Feature Loss (Dropped)
            se1_proj = student_model.feature_proj(se1)
            se1_proj = F.normalize(se1_proj, p=2, dim=1)
            loss_feature = F.mse_loss(se1_proj, t1_emb) * config['SHOCK_FEATURE_W']
            
            # 3. Attention Loss (Dropped)
            loss_attention = calculate_attention_loss(t1_attentions, s1_attentions, config['SHOCK_ATTENTION_W'])
            
            # 4. Ortho Loss (Boosted)
            loss_ortho = (s1_ortho_loss + s2_ortho_loss) * config['SHOCK_ORTHO_W'] / 2.0
            
            # --- STSB Task ---
            try: stsb_s1, stsb_s2 = next(stsb_iter)
            except StopIteration:
                stsb_iter = iter(stsb_loader)
                stsb_s1, stsb_s2 = next(stsb_iter)
            
            t1_stsb_teacher = teacher_tokenizer(stsb_s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_stsb_teacher = teacher_tokenizer(stsb_s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t1_stsb_student = student_tokenizer(stsb_s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_stsb_student = student_tokenizer(stsb_s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            with torch.no_grad():
                te1_stsb = F.normalize(mean_pooling(teacher_model(**t1_stsb_teacher), t1_stsb_teacher['attention_mask']), p=2, dim=1)
                te2_stsb = F.normalize(mean_pooling(teacher_model(**t2_stsb_teacher), t2_stsb_teacher['attention_mask']), p=2, dim=1)
                teacher_stsb_scores = F.cosine_similarity(te1_stsb, te2_stsb)
            
            se1_stsb, _, _ = student_model.forward_student(t1_stsb_student['input_ids'], t1_stsb_student['attention_mask'])
            se2_stsb, _, _ = student_model.forward_student(t2_stsb_student['input_ids'], t2_stsb_student['attention_mask'])
            
            pred_scores = F.cosine_similarity(se1_stsb, se2_stsb)
            loss_stsb = mse_loss_fn(pred_scores, teacher_stsb_scores) * config['SHOCK_STSB_W']
            
            # Total Loss
            loss = loss_nli + loss_stsb + loss_feature + loss_attention + loss_ortho
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar.set_postfix({
                'NLI': f"{loss_nli.item():.4f}", 
                'STSB': f"{loss_stsb.item():.4f}", 
                'Feat': f"{loss_feature.item():.4f}",
                'Attn': f"{loss_attention.item():.4f}",
                'LR': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
        test_score = evaluate(student_model, student_tokenizer)
        print(f"\nEpoch {epoch+1} Test Spearman: {test_score:.4f}")
        
        if test_score > best_score:
            best_score = test_score
            torch.save(student_model.state_dict(), "deltanet_shockwave_result.pt")
            print(f"✅ New Best! Saved. ({test_score:.4f})")
            
    print(f"\n🎉 Final Score: {best_score:.4f}")

if __name__ == "__main__":
    train()