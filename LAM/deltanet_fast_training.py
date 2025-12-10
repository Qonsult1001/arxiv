"""
FAST TRAINING SCRIPT - OPTIMIZED FOR LINEAR ASSOCIATED MEMORY (LAM)
Higher LR, Bigger Batches, Mixed Precision, Efficient Data Loading
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm
import sys
import os
import json
from pathlib import Path

# Import your model
sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

# ============================================================================
# CONFIG - OPTIMIZED FOR LINEAR ASSOCIATED MEMORY (LAM)
# ============================================================================
config = {
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    
    # --- LAM OPTIMIZATIONS ---
    "learning_rate": 8e-4,      # INCREASED: Linear models need higher LR than Transformers
    "weight_decay": 0.05,       # INCREASED: Helps stabilize recurrent weights
    "batch_size": 256,          # INCREASED: O(L) memory allows massive batches
    "gradient_accumulation": 1, # Removed: We can fit the real batch now
    "max_length": 128,
    "scale": 20.0,
    
    "warmup_steps": 1000,       # More warmup for recurrent stability
    "total_steps": 50000,       # Faster convergence expected with higher LR
    "num_workers": 4,           # CPU Parallelism
    "output_dir": "./deltanet_fast_training",
    "save_steps": 500,         # Save every 5k steps
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# EFFICIENT DATASET (PRE-TOKENIZED)
# ============================================================================
class ContrastiveDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # We return text here and tokenize in collator for dynamic padding
        item = self.data[idx]
        return item['sentence1'], item['sentence2']

def create_collate_fn(tokenizer, max_len):
    def collate_fn(batch):
        s1_list, s2_list = zip(*batch)
        
        # Tokenize batch at once (Much faster than one by one)
        t1 = tokenizer(list(s1_list), padding=True, truncation=True, 
                       max_length=max_len, return_tensors="pt")
        t2 = tokenizer(list(s2_list), padding=True, truncation=True, 
                       max_length=max_len, return_tensors="pt")
        
        return t1, t2
    return collate_fn

# ============================================================================
# MODEL WRAPPER
# ============================================================================
class DeltaNetForSentenceEmbedding(nn.Module):
    def __init__(self, base_model_name, d_model=384, num_heads=12, num_layers=6):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load Embeddings Only
        base_model = AutoModel.from_pretrained(base_model_name)
        self.embeddings = base_model.embeddings
        
        # LAM Layers
        self.layers = nn.ModuleList([
            EnhancedHierarchicalDeltaNet(
                d_model=d_model,
                num_heads=num_heads,
                use_hierarchical_decay=True,
                use_enhanced_flux=True
            ) for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        # Standard FFN
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            ) for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids=input_ids)
        
        for i in range(len(self.layers)):
            # LAM Block
            residual = x
            x_attn, _, _, _ = self.layers[i](x, attention_mask)
            x = self.norms[i](residual + x_attn)
            
            # FFN Block
            residual = x
            x_ffn = self.ffns[i](x)
            x = self.ffn_norms[i](residual + x_ffn)
            
        embeddings = self.mean_pooling(x, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data_simplified():
    """Load training data - simplified version"""
    print("Loading training data...")
    
    datasets = []
    
    # 1. AllNLI (SNLI + MultiNLI) - main dataset
    try:
        print("   Loading SNLI...")
        snli = load_dataset("snli", split="train")
        snli_pairs = [
            {
                'sentence1': ex['premise'],
                'sentence2': ex['hypothesis']
            }
            for ex in snli
            if ex['label'] != -1
        ]
        datasets.extend(snli_pairs)
        print(f"   ✅ SNLI: {len(snli_pairs):,} pairs")
    except Exception as e:
        print(f"   ⚠️  SNLI failed: {e}")
    
    try:
        print("   Loading MultiNLI...")
        mnli = load_dataset("multi_nli", split="train")
        mnli_pairs = [
            {
                'sentence1': ex['premise'],
                'sentence2': ex['hypothesis']
            }
            for ex in mnli
            if ex['label'] != -1
        ]
        datasets.extend(mnli_pairs)
        print(f"   ✅ MultiNLI: {len(mnli_pairs):,} pairs")
    except Exception as e:
        print(f"   ⚠️  MultiNLI failed: {e}")
    
    # 2. STS-B for validation
    try:
        print("   Loading STS-B...")
        stsb = load_dataset("glue", "stsb", split="train")
        stsb_pairs = [
            {
                'sentence1': ex['sentence1'],
                'sentence2': ex['sentence2']
            }
            for ex in stsb
        ]
        # Oversample small dataset
        datasets.extend(stsb_pairs * 20)
        print(f"   ✅ STS-B: {len(stsb_pairs):,} pairs (×20)")
    except Exception as e:
        print(f"   ⚠️  STS-B failed: {e}")
    
    print(f"\n   Total: {len(datasets):,} training pairs")
    
    if len(datasets) == 0:
        raise ValueError("No training data loaded!")
    
    return datasets  # Return list, not HFDataset

# ============================================================================
# TRAINING LOOP (OPTIMIZED)
# ============================================================================
def train():
    # 1. Load Data
    print("="*80)
    print("FAST TRAINING - OPTIMIZED FOR LAM")
    print("="*80)
    print("Loading Raw Data...")
    dataset_list = load_data_simplified()
    
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    
    # 2. Create Efficient DataLoader
    train_ds = ContrastiveDataset(dataset_list, tokenizer, config['max_length'])
    collate = create_collate_fn(tokenizer, config['max_length'])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # 3. Init Model
    print("\nInitializing model...")
    model = DeltaNetForSentenceEmbedding(
        config['base_model'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model initialized: {total_params:,} trainable parameters")
    
    # 4. Optimizer & Schedule
    optimizer = AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config['warmup_steps'], 
        num_training_steps=config['total_steps']
    )
    
    # 5. Mixed Precision Scaler
    scaler = GradScaler()
    
    # 6. Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print(f"\n🚀 Starting High-Throughput Training")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   LR: {config['learning_rate']}")
    print(f"   AMP Enabled: Yes")
    print(f"   Total Steps: {config['total_steps']:,}")
    print(f"   Output Dir: {config['output_dir']}")
    print("="*80)
    
    model.train()
    step = 0
    
    # Infinite loop wrapper if steps > dataset size
    data_iter = iter(train_loader)
    
    pbar = tqdm(total=config['total_steps'], desc="Training")
    
    while step < config['total_steps']:
        try:
            inputs_a, inputs_b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs_a, inputs_b = next(data_iter)
            
        # Move to GPU
        ids_a = inputs_a['input_ids'].to(device, non_blocking=True)
        mask_a = inputs_a['attention_mask'].to(device, non_blocking=True)
        ids_b = inputs_b['input_ids'].to(device, non_blocking=True)
        mask_b = inputs_b['attention_mask'].to(device, non_blocking=True)
        
        # Autocast Context (FP16)
        with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
            emb_a = model(ids_a, mask_a)
            emb_b = model(ids_b, mask_b)
            
            # Contrastive Loss
            scale = config['scale']
            scores = torch.mm(emb_a, emb_b.transpose(0, 1)) * scale
            labels = torch.arange(len(scores), device=device)
            loss = (F.cross_entropy(scores, labels) + F.cross_entropy(scores.t(), labels)) / 2
        
        # Scaled Backward Pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        step += 1
        pbar.update(1)
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
        
        # Save checkpoint
        if step % config['save_steps'] == 0:
            checkpoint_path = os.path.join(config['output_dir'], f"checkpoint_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"\n💾 Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config['output_dir'], "checkpoint_final.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }, final_path)
    print(f"\n✅ Training complete! Final model saved: {final_path}")
    pbar.close()

if __name__ == "__main__":
    train()

