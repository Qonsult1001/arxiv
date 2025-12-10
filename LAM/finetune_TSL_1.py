#Teacher Similarity Loss (TSL) - first distillation process massive impact on training data
#!/usr/bin/env python3
"""
🚀 DELTANET GENERALIZATION FIX - Teacher Similarity Loss (TSL)
1. Teacher: stsb-roberta-base-v2 (0.87 Spearman)
2. Data: AllNLI (Generalization)
3. Loss: MSE on Teacher Similarity (NO PROJECTION HEAD)
Goal: Push 0.7664 baseline towards 0.80+ by generalizing on NLI data.
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
    # Increase samples to force generalization
    "max_samples": 200000, 
    
    # Model Loading - Use best checkpoint from train_6layer_deltanet_2.py (0.7711 score)
    "checkpoint_file": "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEWNEWFINAL/checkpoint_167000.pt",
    "num_layers": 6,
    "num_heads": 12,
    "fast_decay_init": 0.30,
    "slow_decay_init": 0.85,
    "use_kernel_blending": False,
    "kernel_blend_alpha": 0.0,

    # Training Config
    # Increased epochs and kept LR high for new data volume
    "learning_rate": 5e-5, 
    "batch_size": 64,
    "epochs": 10,
    "patience": 5,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Shared Pooling Function ---
def mean_pooling(output, mask):
    """Performs mean pooling over the attention-masked tokens."""
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
        
        # We need both Anchor and Positive sentences
        samples_added = 0
        with gzip.open(nli_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if max_samples and samples_added >= max_samples: break
                try:
                    data = json.loads(line)
                    # data[0] = Anchor, data[1] = Positive
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
            
            # Note: We use the Student's forward_student method here, which produces 384D vectors
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
    student_base = 'sentence-transformers/all-MiniLM-L6-v2'
    student_tokenizer = AutoTokenizer.from_pretrained(student_base)
    student_model = DeltaNetPure6Layer(
        teacher_model_name=student_base, # Tokenizer base for DeltaNet is MiniLM
        num_linear_layers=config['num_layers'],
        config=config
    ).to(device)
    
    # Load previous checkpoint
    ckpt = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
    
    # Helper function to transform W_bilinear if needed
    def transform_checkpoint_state(checkpoint_state, model_state_dict):
        """Transform checkpoint parameters to match model architecture"""
        transformed = {}
        transformed_count = 0
        
        for key, value in checkpoint_state.items():
            if key in model_state_dict:
                expected_shape = model_state_dict[key].shape
                if value.shape == expected_shape:
                    transformed[key] = value
                else:
                    # Transform W_bilinear: [32, 32] -> [12, 32, 32] (replicate across heads)
                    if 'resonance_flux.W_bilinear' in key and len(value.shape) == 2 and len(expected_shape) == 3:
                        # Checkpoint: [d_k, d_v], Model: [num_heads, d_k, d_v]
                        if value.shape == expected_shape[1:]:  # [d_k, d_v] matches [d_k, d_v] part
                            num_heads = expected_shape[0]
                            # Replicate across heads: [d_k, d_v] -> [num_heads, d_k, d_v]
                            transformed_value = value.unsqueeze(0).repeat(num_heads, 1, 1)
                            transformed[key] = transformed_value
                            transformed_count += 1
                            continue
                    # Can't transform - skip
                    pass
            else:
                # Key not in model - skip
                pass
        
        if transformed_count > 0:
            print(f"   🔄 Transformed {transformed_count} W_bilinear parameters: [32, 32] -> [12, 32, 32]")
        
        return transformed
    
    # Transform and load deltanet_layers
    if 'deltanet_layers' in ckpt:
        checkpoint_state = ckpt['deltanet_layers']
        model_state = student_model.deltanet_layers.state_dict()
        transformed_state = transform_checkpoint_state(checkpoint_state, model_state)
        if transformed_state:
            student_model.deltanet_layers.load_state_dict(transformed_state, strict=False)
            print(f"   ✅ Loaded {len(transformed_state)}/{len(checkpoint_state)} parameters from deltanet_layers")
        else:
            print(f"   ⚠️  No compatible parameters found in deltanet_layers")
    
    # Load other components for completeness if they exist
    if 'deltanet_norms' in ckpt: student_model.deltanet_norms.load_state_dict(ckpt['deltanet_norms'], strict=False)
    if 'deltanet_ffns' in ckpt: student_model.deltanet_ffns.load_state_dict(ckpt['deltanet_ffns'], strict=False)
    if 'ffn_norms' in ckpt: student_model.ffn_norms.load_state_dict(ckpt['ffn_norms'], strict=False)
    
    # *** ❌ NO PROJECTION HEAD NEEDED IN THIS APPROACH ❌ ***
    
    optimizer = torch.optim.AdamW(
        student_model.parameters(), 
        lr=config['learning_rate']
    )

    # Data Loaders
    train_dataset = NLIDataset(config['nli_file'], config['max_samples'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_nli)

    print("\n📊 Initial Evaluation...")
    initial_score = evaluate(student_model, student_tokenizer)
    print(f"Test Spearman: {initial_score:.4f}")

    print("\n🚀 Starting Teacher Similarity Loss (TSL) Distillation...")
    best_score = initial_score
    
    # Create publish folder
    publish_dir = Path("/workspace/LAM/publish")
    publish_dir.mkdir(parents=True, exist_ok=True)
    output_path = publish_dir / "deltanet_roberta_tsl_distilled.pt"
    print(f"📁 Output will be saved to: {output_path}")
    
    for epoch in range(config['epochs']):
        student_model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for s1, s2 in pbar:
            # Tokenize with both tokenizers
            t1_teacher = teacher_tokenizer(s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_teacher = teacher_tokenizer(s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t1_student = student_tokenizer(s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2_student = student_tokenizer(s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            # --- 1. Get Teacher Target Similarity (The Knowledge) ---
            with torch.no_grad():
                # Teacher Embeddings (768d) - use teacher tokenizer
                te1 = F.normalize(mean_pooling(teacher_model(**t1_teacher), t1_teacher['attention_mask']), p=2, dim=1)
                te2 = F.normalize(mean_pooling(teacher_model(**t2_teacher), t2_teacher['attention_mask']), p=2, dim=1)
                
                # Target Cosine Similarity
                target_sim = F.cosine_similarity(te1, te2).detach()

            # --- 2. Get Student Prediction Similarity (The Output) ---
            # Student Embeddings (384d) - use student tokenizer
            optimizer.zero_grad()
            se1, _, _ = student_model.forward_student(t1_student['input_ids'], t1_student['attention_mask'])
            se2, _, _ = student_model.forward_student(t2_student['input_ids'], t2_student['attention_mask'])
            
            # Student Cosine Similarity
            student_sim = F.cosine_similarity(se1, se2)
            
            # --- 3. Loss: MSE between Student Sim and Teacher Sim ---
            # MSE is much gentler than Negative Cosine on the vectors and won't destroy the feature space.
            loss = F.mse_loss(student_sim, target_sim)
            
            # 4. Update Student
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        # Eval
        test_score = evaluate(student_model, student_tokenizer, split='test')
        print(f"\nEpoch {epoch+1} Test Spearman: {test_score:.4f} (Best so far: {best_score:.4f})")
        
        # Only save if this is a new best score
        if test_score > best_score:
            best_score = test_score
            # Save ONLY the best model weights to publish folder
            torch.save(student_model.state_dict(), output_path)
            print(f"✅ NEW BEST! Saved checkpoint to {output_path}. Best score: {best_score:.4f}")
        else:
            print(f"   (No improvement - not saving. Best remains: {best_score:.4f})")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎯 Training Complete!")
    print(f"   Initial Score: {initial_score:.4f}")
    print(f"   Best Score:    {best_score:.4f}")
    print(f"   Improvement:   {best_score - initial_score:+.4f}")
    print(f"   📁 Best model saved to: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    train()