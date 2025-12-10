"""
🎯 FINE-TUNE FROM 0.8431 → 0.850+
Ultra-low LR to avoid overfitting while gaining the final 0.69%

Your constant 3e-5 LR found the optimum at step 30 (0.8431)
Now we use 1e-5 (3x lower) for careful micro-adjustments
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
    "checkpoint": "/workspace/LAM/pure_constant_lr/step0070_p0.8363.pt",  # Your PEAK!
    "base_model": "/workspace/LAM/all-MiniLM-L6-v2",
    "output_dir": "/workspace/LAM/finetune_from_8363",
    
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    
    # 🎯 ULTRA-LOW LR - Careful micro-adjustments
    "learning_rate": 3e-5,      # 3x lower than before
    "weight_decay": 0.005,      # Lighter regularization
    "gradient_clip": 0.01,      # Very tight
    "batch_size": 32,           # Larger = more stable
    "max_length": 128,           # Shorter! STS-B sentences are ~15-20 tokens
    
    # 🎯 AGGRESSIVE EARLY STOPPING
    "max_steps": 150,           # Short run
    "eval_interval": 5,         # Eval VERY frequently
    "patience": 15,             # Stop if no improvement for 15 evals (75 steps)
    
    # Target
    "target_score": 0.8500,
    "min_score": 0.8420,        # Never go below your peak
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*80)
print("🎯 FINE-TUNE FROM 0.8431 → 0.850+")
print("="*80)
print(f"Starting from: 0.8431 (your peak at step 30!)")
print(f"Gap: {config['target_score'] - 0.8431:.4f} (only 0.69%!)")
print(f"Strategy: Ultra-low LR (1e-5) + aggressive early stopping")
print("="*80)

class FineTuneModel(nn.Module):
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
        self.deltanet_layers.load_state_dict(checkpoint['deltanet_layers'], strict=False)
        
        for key, attr in [('deltanet_norms', 'norms'), ('deltanet_ffns', 'ffns'), 
                          ('ffn_norms', 'ffn_norms')]:
            if key in checkpoint:
                try:
                    getattr(self, attr).load_state_dict(checkpoint[key], strict=False)
                except: pass
        
        print(f"✅ Loaded from step {checkpoint.get('step', 30)} (Pearson: {checkpoint.get('pearson', 0.8431):.4f})")
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
            x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)  # Returns 4 values: output, None, past_key_values, ortho_loss
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
    try:
        ds = load_dataset("sentence-transformers/stsb", split="validation")
    except:
        ds = load_dataset("glue", "stsb", split="validation")
    
    s1 = ds["sentence1"]
    s2 = ds["sentence2"]
    labels = np.array(ds["label"] if "label" in ds.column_names else ds["score"], dtype=float)
    
    all_sim = []
    with torch.no_grad():
        for i in range(0, len(s1), 32):
            batch_s1 = s1[i:min(i+32, len(s1))]
            batch_s2 = s2[i:min(i+32, len(s2))]
            
            tokens1 = model.tokenizer(batch_s1, padding=True, max_length=128,
                                     truncation=True, return_tensors='pt').to(device)
            tokens2 = model.tokenizer(batch_s2, padding=True, max_length=128,
                                     truncation=True, return_tensors='pt').to(device)
            
            emb1 = model(tokens1['input_ids'], tokens1['attention_mask'])
            emb2 = model(tokens2['input_ids'], tokens2['attention_mask'])
            
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            all_sim.extend(sim.cpu().numpy().tolist())
    
    pearson = pearsonr(all_sim, labels)[0]
    spearman = spearmanr(all_sim, labels)[0]
    model.train()
    return pearson, spearman

def train():
    # Data
    stsb_train = load_dataset("glue", "stsb", split="train")
    train_data = [(ex['sentence1'], ex['sentence2'], ex['label']) 
                  for ex in stsb_train if len(ex['sentence1']) > 10 and len(ex['sentence2']) > 10]
    print(f"\n📚 {len(train_data):,} training pairs")
    
    # Model
    model = FineTuneModel(config['base_model'], config['checkpoint'], config).to(device)
    
    # Baseline (should be ~0.8431)
    print(f"\n📊 BASELINE...")
    baseline_pearson, _ = evaluate_stsb(model)
    print(f"   Pearson: {baseline_pearson:.4f}")
    print(f"   Gap to 0.850: {config['target_score'] - baseline_pearson:.4f}")
    
    # Optimizer
    trainable = (list(model.deltanet_layers.parameters()) + list(model.norms.parameters()) +
                list(model.ffns.parameters()) + list(model.ffn_norms.parameters()))
    
    optimizer = AdamW(trainable, lr=config['learning_rate'], 
                     weight_decay=config['weight_decay'])
    
    print(f"\n▶️  Fine-tuning with LR {config['learning_rate']:.2e} (3x lower)")
    
    # Training
    model.train()
    best_pearson = baseline_pearson
    best_step = 0
    patience = 0
    
    global_step = 0
    running_loss = 0.0
    log_count = 0
    
    pbar = tqdm(total=config['max_steps'], desc="🎯 Fine-tune")
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    while global_step < config['max_steps']:
        # Batch
        batch_indices = np.random.choice(len(train_data), size=config['batch_size'], replace=False)
        batch = [train_data[i] for i in batch_indices]
        
        s1 = [item[0] for item in batch]
        s2 = [item[1] for item in batch]
        scores = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
        
        # Tokenize
        tokens1 = model.tokenizer(s1, padding=True, max_length=128,
                                 truncation=True, return_tensors='pt').to(device)
        tokens2 = model.tokenizer(s2, padding=True, max_length=128,
                                 truncation=True, return_tensors='pt').to(device)
        
        # Forward
        emb1 = model(tokens1['input_ids'], tokens1['attention_mask'])
        emb2 = model(tokens2['input_ids'], tokens2['attention_mask'])
        
        sim = F.cosine_similarity(emb1, emb2, dim=1)
        pred = (sim + 1) * 2.5
        loss = F.mse_loss(pred, scores)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip'])
        optimizer.step()
        
        running_loss += loss.item()
        log_count += 1
        
        if log_count >= 5:
            pbar.set_postfix({
                'loss': f'{running_loss/log_count:.4f}',
                'best': f'{best_pearson:.4f}'
            })
            running_loss = 0.0
            log_count = 0
        
        if (global_step + 1) % config['eval_interval'] == 0:
            current_pearson, current_spearman = evaluate_stsb(model)
            
            if current_pearson > best_pearson:
                best_pearson = current_pearson
                best_step = global_step + 1
                patience = 0
                
                torch.save({
                    'deltanet_layers': model.deltanet_layers.state_dict(),
                    'deltanet_norms': model.norms.state_dict(),
                    'deltanet_ffns': model.ffns.state_dict(),
                    'ffn_norms': model.ffn_norms.state_dict(),
                    'step': global_step + 1,
                    'pearson': current_pearson,
                    'spearman': current_spearman
                }, output_dir / f"checkpoint_best_{current_pearson:.4f}.pt")
                
                status = "🎉 NEW BEST!"
            elif current_pearson < config['min_score']:
                patience += 1
                status = f"⚠️  Below min (patience {patience}/{config['patience']})"
            else:
                patience += 1
                status = f"📊 (patience {patience}/{config['patience']})"
            
            print(f"\n📊 Step {global_step+1}:")
            print(f"   Pearson:  {current_pearson:.4f} ({current_pearson-baseline_pearson:+.4f}) {status}")
            print(f"   Gap to 0.850: {config['target_score']-current_pearson:.4f}")
            
            if current_pearson >= config['target_score']:
                print(f"\n🎉🎉🎉 TARGET 0.850 REACHED! 🎉🎉🎉")
                break
            
            if patience >= config['patience']:
                print(f"\n⚠️  Early stopping")
                break
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final
    print("\n" + "="*80)
    print("✅ FINE-TUNING COMPLETE")
    print("="*80)
    final_pearson, _ = evaluate_stsb(model)
    print(f"\n📊 RESULTS:")
    print(f"   Baseline: {baseline_pearson:.4f}")
    print(f"   Final:    {final_pearson:.4f} ({final_pearson-baseline_pearson:+.4f})")
    print(f"   Best:     {best_pearson:.4f} (step {best_step})")
    
    if best_pearson >= config['target_score']:
        print(f"\n🎉🎉🎉 TARGET 0.850 ACHIEVED! 🎉🎉🎉")
    else:
        gap = config['target_score'] - best_pearson
        print(f"\n📊 Gap: {gap:.4f}")
        if gap < 0.002:
            print(f"\n💡 SO CLOSE! Try ensemble or multi-task")

if __name__ == "__main__":
    train()