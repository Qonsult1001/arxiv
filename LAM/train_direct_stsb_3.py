"""
🚀 PURE CONSTANT HIGH LR - NO WARMUP, NO DECAY
Based on pattern: Higher LR = Better Score

Your data proves it:
- Step 50 @ LR 2.83e-5: 0.8438 ✅
- Step 100 @ LR 2.71e-5: 0.8063 ❌ (LR decayed, score dropped!)

Solution: PURE CONSTANT 3e-5 throughout!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

config = {
    "checkpoint": "/workspace/LAM/pure_constant_lr/checkpoint_6000.pt",
    #"checkpoint": "/workspace/LAM/pure_constant_lr/distill_step0050_val0.8467.pt",
    "base_model": "/workspace/LAM/all-MiniLM-L6-v2",
    "output_dir": "/workspace/LAM/pure_constant_lr2",
    
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    
    # 🚀 PURE CONSTANT - NO WARMUP, NO DECAY!
    "learning_rate": 1e-5,      # CONSTANT throughout
    "weight_decay": 0.01,
    "gradient_clip": 0.01,
    "batch_size": 16,
    "max_length": 256,
    
    # Short training - find the sweet spot
    "max_steps": 200,           # Stop early, we'll see the peak
    "eval_interval": 10,        # Eval VERY frequently
    
    # Save EVERY checkpoint
    "save_every_eval": True,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*80)
print("🚀 PURE CONSTANT HIGH LR - NO WARMUP, NO DECAY")
print("="*80)
print(f"LR: {config['learning_rate']:.2e} (PURE CONSTANT)")
print(f"Max steps: {config['max_steps']}")
print(f"Eval interval: {config['eval_interval']}")
print(f"NO warmup, NO decay, NO scheduler!")
print("="*80)

# ============================================================================
# MODEL
# ============================================================================
class PureConstantModel(nn.Module):
    def __init__(self, base_model_path, checkpoint_path, config):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModel.from_pretrained(base_model_path)
        self.d_model = config['d_model']
        self.num_layers = config['num_layers']
        
        self.embeddings = base_model.embeddings
        for param in self.embeddings.parameters():
            param.requires_grad = False
        
        self.deltanet_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.output_denses = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=self.d_model, num_heads=config['num_heads'],
                    use_hierarchical_decay=True, use_enhanced_flux=True,
                    fast_decay_init=0.3, slow_decay_init=0.85, layer_idx=i,
                )
            )
            self.norms.append(base_model.encoder.layer[i].attention.output.LayerNorm)
            self.ffns.append(base_model.encoder.layer[i].intermediate)
            self.ffn_norms.append(base_model.encoder.layer[i].output.LayerNorm)
            self.output_denses.append(base_model.encoder.layer[i].output.dense)
        
        for param in self.output_denses.parameters():
            param.requires_grad = False
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Check if checkpoint is a dict with 'deltanet_layers' key (standard format)
        if isinstance(checkpoint, dict) and 'deltanet_layers' in checkpoint:
            self.deltanet_layers.load_state_dict(checkpoint['deltanet_layers'], strict=False)
            print(f"✅ Loaded deltanet_layers from checkpoint")
        elif isinstance(checkpoint, dict) and 'lam_layers' in checkpoint:
            self.deltanet_layers.load_state_dict(checkpoint['lam_layers'], strict=False)
            print(f"✅ Loaded lam_layers from checkpoint")
        elif isinstance(checkpoint, dict):
            # Check if it's a direct state dict with deltanet_layers keys (e.g., 'deltanet_layers.0.xxx')
            checkpoint_keys = list(checkpoint.keys())
            has_deltanet_keys = any('deltanet_layers' in k for k in checkpoint_keys)
            
            if has_deltanet_keys:
                # Extract deltanet_layers state dict
                deltanet_state = {}
                for k, v in checkpoint.items():
                    if k.startswith('deltanet_layers.'):
                        new_key = k.replace('deltanet_layers.', '')
                        deltanet_state[new_key] = v
                
                if deltanet_state:
                    self.deltanet_layers.load_state_dict(deltanet_state, strict=False)
                    print(f"✅ Loaded deltanet_layers from state dict format ({len(deltanet_state)} keys)")
                else:
                    print(f"⚠️  Warning: Found 'deltanet_layers' in keys but couldn't extract state dict")
            else:
                print(f"⚠️  Warning: Could not find deltanet_layers in checkpoint")
                print(f"   Available keys (first 10): {checkpoint_keys[:10]}")
                print(f"   ⚠️  Model will use randomly initialized DeltaNet layers!")
        
        # Load norms, FFNs, and ffn_norms
        for key, attr in [('lam_norms', 'norms'), ('deltanet_norms', 'norms'), 
                          ('deltanet_ffns', 'ffns'), ('ffn_norms', 'ffn_norms')]:
            if isinstance(checkpoint, dict) and key in checkpoint:
                try:
                    getattr(self, attr).load_state_dict(checkpoint[key], strict=False)
                    print(f"✅ Loaded {key} from checkpoint")
                except Exception as e:
                    print(f"⚠️  Failed to load {key}: {e}")
            elif isinstance(checkpoint, dict):
                # Try to extract from state dict format
                checkpoint_keys = list(checkpoint.keys())
                if any(key in k for k in checkpoint_keys):
                    state_dict = {}
                    for k, v in checkpoint.items():
                        if k.startswith(f'{key}.'):
                            new_key = k.replace(f'{key}.', '')
                            state_dict[new_key] = v
                    if state_dict:
                        try:
                            getattr(self, attr).load_state_dict(state_dict, strict=False)
                            print(f"✅ Loaded {key} from state dict format ({len(state_dict)} keys)")
                        except Exception as e:
                            print(f"⚠️  Failed to load {key} from state dict: {e}")
        
        # Print checkpoint metadata if available
        if isinstance(checkpoint, dict):
            if 'val_pearson' in checkpoint or 'val_spearman' in checkpoint:
                val_p = checkpoint.get('val_pearson', 'N/A')
                val_s = checkpoint.get('val_spearman', 'N/A')
                print(f"   📊 Checkpoint validation scores: Pearson={val_p}, Spearman={val_s}")
            if 'test_pearson' in checkpoint or 'test_spearman' in checkpoint:
                test_p = checkpoint.get('test_pearson', 'N/A')
                test_s = checkpoint.get('test_spearman', 'N/A')
                print(f"   📊 Checkpoint test scores: Pearson={test_p}, Spearman={test_s}")
            if 'pearson' in checkpoint or 'spearman' in checkpoint:
                p = checkpoint.get('pearson', 'N/A')
                s = checkpoint.get('spearman', 'N/A')
                print(f"   📊 Checkpoint scores: Pearson={p}, Spearman={s}")
            if 'step' in checkpoint:
                print(f"   📍 Checkpoint training step: {checkpoint['step']}")
        
        print(f"✅ Checkpoint loading complete")
        del base_model
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids=input_ids)
        for i in range(self.num_layers):
            residual = x
            x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)
            x = self.norms[i](residual + x_attn)
            residual = x
            x_ffn = self.ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            x_ffn = self.output_denses[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        embeddings = self.mean_pooling(x, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)

def evaluate_stsb(model):
    model.eval()
    ds = load_dataset("glue", "stsb", split="validation")
    all_sim, all_labels = [], []
    
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            batch = ds[i:min(i+32, len(ds))]
            tokens1 = model.tokenizer(batch['sentence1'], padding=True, max_length=256, truncation=True, return_tensors='pt').to(device)
            tokens2 = model.tokenizer(batch['sentence2'], padding=True, max_length=256, truncation=True, return_tensors='pt').to(device)
            emb1 = model(tokens1['input_ids'], tokens1['attention_mask'])
            emb2 = model(tokens2['input_ids'], tokens2['attention_mask'])
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            sim_scaled = (sim + 1) * 2.5
            all_sim.extend(sim_scaled.cpu().numpy().tolist())
            all_labels.extend(list(batch['label']))
    
    pearson = pearsonr(all_sim, all_labels)[0]
    spearman = spearmanr(all_sim, all_labels)[0]
    model.train()
    return pearson, spearman

def train():
    # Data
    stsb_train = load_dataset("glue", "stsb", split="train")
    train_data = [(ex['sentence1'], ex['sentence2'], ex['label']) 
                  for ex in stsb_train if len(ex['sentence1']) > 10 and len(ex['sentence2']) > 10]
    print(f"\n📚 {len(train_data):,} training pairs")
    
    # Model
    model = PureConstantModel(config['base_model'], config['checkpoint'], config).to(device)
    
    # Baseline
    print(f"\n📊 BASELINE...")
    baseline_pearson, _ = evaluate_stsb(model)
    print(f"   Pearson: {baseline_pearson:.4f}")
    
    # Optimizer - NO scheduler!
    trainable = list(model.deltanet_layers.parameters()) + list(model.norms.parameters()) + \
                list(model.ffns.parameters()) + list(model.ffn_norms.parameters())
    
    optimizer = AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # NO SCHEDULER! LR stays constant at 3e-5
    
    print(f"\n▶️  Training with PURE CONSTANT LR {config['learning_rate']:.2e}")
    print(f"   NO warmup, NO decay, NO scheduler!")
    
    # Training
    model.train()
    global_step = 0
    best_pearson = baseline_pearson
    best_step = 0
    
    running_loss = 0.0
    pbar = tqdm(total=config['max_steps'], desc="🚀 Pure Constant")
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Track all scores
    all_scores = []
    
    while global_step < config['max_steps']:
        # Batch
        batch_indices = np.random.choice(len(train_data), size=config['batch_size'], replace=False)
        batch = [train_data[i] for i in batch_indices]
        
        s1, s2 = [item[0] for item in batch], [item[1] for item in batch]
        scores = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
        
        # Tokenize
        t1 = model.tokenizer(s1, padding=True, max_length=256, truncation=True, return_tensors='pt').to(device)
        t2 = model.tokenizer(s2, padding=True, max_length=256, truncation=True, return_tensors='pt').to(device)
        
        # Forward
        e1 = model(t1['input_ids'], t1['attention_mask'])
        e2 = model(t2['input_ids'], t2['attention_mask'])
        
        # Loss
        sim = F.cosine_similarity(e1, e2, dim=1)
        pred = (sim + 1) * 2.5
        loss = F.mse_loss(pred, scores)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip'])
        optimizer.step()
        # NO scheduler.step()!
        
        running_loss += loss.item()
        
        if (global_step + 1) % 5 == 0:
            pbar.set_postfix({'loss': f'{running_loss/5:.4f}', 'lr': f"{config['learning_rate']:.2e}"})
            running_loss = 0.0
        
        if (global_step + 1) % config['eval_interval'] == 0:
            current_pearson, current_spearman = evaluate_stsb(model)
            gain = current_pearson - baseline_pearson
            
            # Track score
            all_scores.append((global_step + 1, current_pearson))
            
            is_best = current_pearson > best_pearson
            if is_best:
                best_pearson = current_pearson
                best_step = global_step + 1
                status = "🎉 NEW BEST!"
            else:
                status = "📊"
            
            print(f"\n📊 Step {global_step+1}:")
            print(f"   Pearson:  {current_pearson:.4f} (Δ {gain:+.4f}) {status}")
            print(f"   Spearman: {current_spearman:.4f}")
            print(f"   LR: {config['learning_rate']:.2e} (CONSTANT)")
            
            # Save checkpoint
            torch.save({
                'deltanet_layers': model.deltanet_layers.state_dict(),
                'deltanet_norms': model.norms.state_dict(),
                'deltanet_ffns': model.ffns.state_dict(),
                'ffn_norms': model.ffn_norms.state_dict(),
                'step': global_step + 1,
                'pearson': current_pearson,
                'spearman': current_spearman
            }, output_dir / f"step{global_step+1:04d}_p{current_pearson:.4f}.pt")
            
            # Continue training even after reaching 0.850 to find the best score
            if current_pearson >= 0.850:
                print(f"\n🎉 TARGET 0.850+ REACHED! Continuing training to find best score...")
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    final_pearson, _ = evaluate_stsb(model)
    print(f"\n📊 RESULTS:")
    print(f"   Baseline: {baseline_pearson:.4f}")
    print(f"   Final:    {final_pearson:.4f} ({final_pearson-baseline_pearson:+.4f})")
    print(f"   Best:     {best_pearson:.4f} at step {best_step} ({best_pearson-baseline_pearson:+.4f})")
    
    # Show score progression
    print(f"\n📈 SCORE PROGRESSION (every {config['eval_interval']} steps):")
    for step, score in all_scores:
        marker = "🎯" if score >= 0.845 else "✅" if score > baseline_pearson else "📊"
        print(f"   Step {step:3d}: {score:.4f} {marker}")
    
    if best_pearson >= 0.850:
        print(f"\n🎉 TARGET ACHIEVED!")
    elif best_pearson >= 0.845:
        print(f"\n✅ SO CLOSE! Only {0.850-best_pearson:.4f} away")
        print(f"\n💡 Try running again with LR {config['learning_rate']*1.2:.2e} (20% higher)")
    
    print(f"\n💾 All checkpoints saved to: {output_dir}/")
    print(f"   Best: step{best_step:04d}_p{best_pearson:.4f}.pt")

if __name__ == "__main__":
    train()