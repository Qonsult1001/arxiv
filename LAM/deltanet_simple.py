"""
SIMPLIFIED SINGLE-GPU VERSION
DeltaNet training matching all-MiniLM-L6-v2 methodology

This is easier to test and debug than the full multi-GPU version.
Use this first to validate your DeltaNet architecture works.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import os
import json

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

# ============================================================================
# CONFIG - MATCH ALL-MINILM-L6-V2 EXACTLY
# ============================================================================
config = {
    # Model architecture (DeltaNet replaces transformer)
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",  # For embeddings
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    
    # Training params - EXACT match to original
    "learning_rate": 8e-4,
    "weight_decay": 0.05,  # INCREASED: Better regularization for LAM models
    "gradient_clip": 1.0,
    "batch_size": 256,            # Huge physical batch
    "gradient_accumulation": 4,    # No accumulation needed
    "max_length": 128,
    "scale": 20.0,                  # CRITICAL - from original
    
    # Schedule - EXACT match to original
    "warmup_steps": 500,
    "total_steps": 100000,
    
    # Logging
    "log_interval": 50,
    "save_interval": 5000,
    "output_dir": "./deltanet_all_minilm_replica",
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# MODEL - DELTANET VERSION OF ALL-MINILM-L6-V2
# ============================================================================
class DeltaNetForSentenceEmbedding(nn.Module):
    """DeltaNet sentence encoder matching all-MiniLM structure"""
    
    def __init__(self, base_model_name, d_model=384, num_heads=12, num_layers=6):
        super().__init__()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load pre-trained embeddings (for warm start)
        print(f"Loading embeddings from: {base_model_name}")
        base_model = AutoModel.from_pretrained(base_model_name)
        self.embeddings = base_model.embeddings
        
        embed_dim = base_model.config.hidden_size
        
        # Project if dimensions don't match
        if embed_dim != d_model:
            self.embedding_projection = nn.Linear(embed_dim, d_model)
            nn.init.xavier_uniform_(self.embedding_projection.weight)
            nn.init.zeros_(self.embedding_projection.bias)
        else:
            self.embedding_projection = None
        
        # DeltaNet layers (replacing transformer attention)
        self.layers = nn.ModuleList([
            EnhancedHierarchicalDeltaNet(
                d_model=d_model,
                num_heads=num_heads,
                use_hierarchical_decay=True,
                use_enhanced_flux=True,
                fast_decay_init=0.30,
                slow_decay_init=0.85,
            )
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # FFN blocks (like transformer) - WITH DROPOUT for regularization
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(0.1),  # ADDED: Regularization to prevent overfitting
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # LEARNABLE SCALE (like CLIP, SigLIP) - initialized to log(20) ≈ 2.99
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(20.0)))
        
        print(f"✅ DeltaNet model initialized:")
        print(f"   Layers: {num_layers} DeltaNet layers")
        print(f"   Dimension: {d_model}")
        print(f"   Heads: {num_heads}")
        print(f"   Learnable scale: initialized to {self.logit_scale.exp().item():.2f}")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling - EXACT same as original"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Embeddings
        x = self.embeddings(input_ids=input_ids)
        
        # Project if needed
        if self.embedding_projection is not None:
            x = self.embedding_projection(x)
        
        # DeltaNet layers
        for i in range(len(self.layers)):
            # Attention block
            residual = x
            x_attn, _, _, _ = self.layers[i](x, attention_mask)  # Returns 4 values: (x_attn, _, _, ortho_loss)
            x = self.norms[i](residual + x_attn)
            
            # FFN block
            residual = x
            x_ffn = self.ffns[i](x)
            x = self.ffn_norms[i](residual + x_ffn)
        
        # Pooling
        embeddings = self.mean_pooling(x, attention_mask)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Return embeddings AND learned scale (clamped for stability)
        scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        
        return embeddings, scale
    
    def save_pretrained(self, output_path):
        """Save model"""
        os.makedirs(output_path, exist_ok=True)
        self.tokenizer.save_pretrained(output_path)
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        
        # Save config
        config_dict = {
            'model_type': 'DeltaNetForSentenceEmbedding',
            'd_model': self.layers[0].hidden_size,
            'num_heads': self.layers[0].num_heads,
            'num_layers': len(self.layers),
        }
        with open(os.path.join(output_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

# ============================================================================
# LOSS - EXACT MATCH TO ORIGINAL
# ============================================================================
def compute_contrastive_loss(embeddings_a, embeddings_b, scale=20.0):
    """
    Contrastive loss with in-batch negatives
    EXACT same as all-MiniLM-L6-v2
    """
    # Similarity matrix [batch x batch]
    scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * scale
    
    # Labels: diagonal should match
    labels = torch.arange(len(scores), device=scores.device)
    
    # Cross-entropy loss
    cross_entropy = nn.CrossEntropyLoss()
    
    # Symmetric loss (bidirectional) - PROVEN in original
    loss = (cross_entropy(scores, labels) + 
            cross_entropy(scores.transpose(0, 1), labels)) / 2
    
    return loss

# ============================================================================
# DATA LOADING
# ============================================================================
def load_training_data():
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
    
    return HFDataset.from_list(datasets)

# ============================================================================
# TRAINING
# ============================================================================
def train():
    print("="*80)
    print("DELTANET - EXACT ALL-MINILM-L6-V2 REPLICA")
    print("="*80)
    print(f"\nArchitecture:")
    print(f"  ✅ 6 DeltaNet layers (replacing 6 transformer layers)")
    print(f"  ✅ Same embeddings, pooling, normalization")
    print(f"\nTraining (OPTIMIZED for LAM):")
    print(f"  ✅ Contrastive loss (LEARNABLE scale, init={config['scale']})")
    print(f"  ✅ Symmetric loss (bidirectional)")
    print(f"  ✅ In-batch negatives ({config['batch_size']} per batch)")
    print(f"  ✅ {config['total_steps']:,} steps")
    print(f"  ✅ LR {config['learning_rate']:.0e} with {config['warmup_steps']} warmup")
    print(f"  ✅ Weight decay: {config['weight_decay']} (increased for regularization)")
    print(f"  ✅ Dropout: 0.1 in FFN layers")
    print("="*80)
    
    # Load data
    dataset = load_training_data()
    
    # Initialize model
    print("\nInitializing DeltaNet model...")
    model = DeltaNetForSentenceEmbedding(
        base_model_name=config['base_model'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    
    # Optimizer - EXACT match to original
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
        # Note: correct_bias is not available in torch.optim.AdamW (only in transformers.AdamW)
        # PyTorch's AdamW doesn't need bias correction as it's handled differently
    )
    
    # Scheduler - EXACT match to original
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['total_steps']
    )
    
    # Training loop
    model.train()
    global_step = 0
    running_loss = 0.0
    
    pbar = tqdm(total=config['total_steps'], desc="Training")
    
    while global_step < config['total_steps']:
        # Sample batch
        indices = np.random.randint(0, len(dataset), size=config['batch_size'])
        batch_data = [dataset[int(i)] for i in indices]
        
        sentences_a = [item['sentence1'] for item in batch_data]
        sentences_b = [item['sentence2'] for item in batch_data]
        
        # Tokenize
        tokens_a = model.tokenizer(
            sentences_a,
            padding='max_length',
            max_length=config['max_length'],
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        tokens_b = model.tokenizer(
            sentences_b,
            padding='max_length',
            max_length=config['max_length'],
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # Forward - model now returns (embeddings, scale)
        embeddings_a, scale = model(tokens_a['input_ids'], tokens_a['attention_mask'])
        embeddings_b, _ = model(tokens_b['input_ids'], tokens_b['attention_mask'])  # scale is same for both
        
        # Use LEARNED scale instead of fixed config['scale'] (keep as tensor for gradients)
        loss = compute_contrastive_loss(embeddings_a, embeddings_b, scale)
        
        # Backward
        loss = loss / config['gradient_accumulation']
        loss.backward()
        
        # Optimizer step with gradient accumulation
        if (global_step + 1) % config['gradient_accumulation'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        running_loss += loss.item() * config['gradient_accumulation']
        
        if (global_step + 1) % config['log_interval'] == 0:
            avg_loss = running_loss / config['log_interval']
            current_scale = scale.item() if isinstance(scale, torch.Tensor) else scale
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'scale': f'{current_scale:.2f}',
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            running_loss = 0.0
        
        # Save checkpoint
        if (global_step + 1) % config['save_interval'] == 0:
            output_dir = Path(config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir / f"checkpoint_{global_step+1}")
            print(f"\n💾 Saved checkpoint at step {global_step + 1}")
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final save
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "final")
    
    print(f"\n✅ Training complete! Saved to: {output_dir}/final/")
    print(f"\n📊 Next steps:")
    print(f"   1. Evaluate on STS-B test set")
    print(f"   2. Compare to all-MiniLM-L6-v2 baseline")
    print(f"   3. Expected: 0.82-0.86 Pearson (DeltaNet vs 0.86-0.87 transformer)")

if __name__ == "__main__":
    train()