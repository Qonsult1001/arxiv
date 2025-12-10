"""
🎯 MULTI-TASK TRAINING: 0.8358 → 0.850+
Mix STS-B with diverse semantic tasks for better generalization

Research shows: Multi-task learning prevents overfitting and improves test scores
Your conservative model already shows good generalization (test 0.7673)

Strategy:
1. Start from conservative checkpoint (better test score)
2. Mix 4 datasets: STS-B (50%) + AllNLI (25%) + WikiAnswers (15%) + QQP (10%)
3. Use curriculum: Start with high STS-B ratio, gradually add diversity
4. Low LR (2e-6) for stability
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
    # Use conservative (better test score)
    "checkpoint": "/workspace/LAM/final_push_850/step0070_p0.8363.pt",
    "base_model": "/workspace/LAM/all-MiniLM-L6-v2",
    "output_dir": "/workspace/LAM/multitask_850",
    
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    
    # 🎯 MULTI-TASK CONFIGURATION
    "learning_rate": 2e-6,      # Conservative for stability
    "weight_decay": 0.005,
    "gradient_clip": 0.05,
    "batch_size": 32,           # Larger = more stable
    "max_length": 128,
    
    # 📚 DATASET MIX - Curriculum
    # Phase 1 (steps 0-100): Focus on STS-B
    "phase1_mix": {
        "stsb": 0.80,           # 80% STS-B
        "nli": 0.15,            # 15% NLI
        "wiki": 0.05,           # 5% WikiAnswers
        "qqp": 0.00,            # 0% QQP
    },
    # Phase 2 (steps 100-300): Add diversity
    "phase2_mix": {
        "stsb": 0.60,           # 60% STS-B
        "nli": 0.20,            # 20% NLI
        "wiki": 0.10,           # 10% WikiAnswers
        "qqp": 0.10,            # 10% QQP
    },
    # Phase 3 (steps 300+): Balanced
    "phase3_mix": {
        "stsb": 0.50,           # 50% STS-B
        "nli": 0.25,            # 25% NLI
        "wiki": 0.15,           # 15% WikiAnswers
        "qqp": 0.10,            # 10% QQP
    },
    
    # Training
    "max_steps": 500,
    "eval_interval": 10,        # Eval every 10 steps
    "patience": 30,
    
    # Target
    "target_score": 0.8500,
    "min_score": 0.8350,        # Never drop below start
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*80)
print("🎯 MULTI-TASK TRAINING: 0.8358 → 0.850+")
print("="*80)
print(f"Starting from: 0.8358 (conservative checkpoint)")
print(f"Target: 0.8500")
print(f"Gap: 0.0142 (1.7%)")
print()
print("📚 MULTI-TASK STRATEGY:")
print("  Phase 1 (0-100):   80% STS-B, 20% diverse")
print("  Phase 2 (100-300): 60% STS-B, 40% diverse")
print("  Phase 3 (300+):    50% STS-B, 50% diverse")
print(f"  LR: {config['learning_rate']:.2e} (stable)")
print("="*80)

# ============================================================================
# MODEL
# ============================================================================
class MultiTaskModel(nn.Module):
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
        
        print(f"✅ Loaded checkpoint")
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

# ============================================================================
# DATA LOADING
# ============================================================================
def load_multitask_data():
    """Load all datasets for multi-task learning"""
    print("\n📚 Loading datasets...")
    
    # 1. STS-B (regression target)
    stsb_train = load_dataset("glue", "stsb", split="train")
    stsb_data = [(ex['sentence1'], ex['sentence2'], ex['label'], 'stsb') 
                 for ex in stsb_train if len(ex['sentence1']) > 10 and len(ex['sentence2']) > 10]
    print(f"   STS-B: {len(stsb_data):,} pairs")
    
    # 2. AllNLI (contrastive - positive pairs)
    try:
        nli_dataset = load_dataset("sentence-transformers/all-nli", "pair", split="train")
        nli_dataset = nli_dataset.filter(lambda x: x['label'] == 0)  # Entailment only
        nli_dataset = nli_dataset.shuffle(seed=42).select(range(min(50000, len(nli_dataset))))
        nli_data = [(ex['sentence1'], ex['sentence2'], 4.5, 'nli') for ex in nli_dataset]
        print(f"   AllNLI: {len(nli_data):,} pairs")
    except Exception as e:
        print(f"   AllNLI: Failed to load ({e})")
        nli_data = []
    
    # 3. WikiAnswers (paraphrase pairs - high similarity)
    try:
        wiki_dataset = load_dataset("wiki_qa", split="train")
        wiki_dataset = wiki_dataset.filter(lambda x: x['label'] == 1 and len(x['question']) > 10 and len(x['answer']) > 10)
        wiki_dataset = wiki_dataset.shuffle(seed=42).select(range(min(20000, len(wiki_dataset))))
        wiki_data = [(ex['question'], ex['answer'], 4.0, 'wiki') for ex in wiki_dataset]
        print(f"   WikiQA: {len(wiki_data):,} pairs")
    except Exception as e:
        print(f"   WikiQA: Failed to load ({e})")
        wiki_data = []
    
    # 4. QQP (question pairs - moderate similarity)
    try:
        qqp_dataset = load_dataset("glue", "qqp", split="train")
        qqp_dataset = qqp_dataset.filter(lambda x: x['label'] == 1 and len(x['question1']) > 10 and len(x['question2']) > 10)
        qqp_dataset = qqp_dataset.shuffle(seed=42).select(range(min(30000, len(qqp_dataset))))
        qqp_data = [(ex['question1'], ex['question2'], 3.5, 'qqp') for ex in qqp_dataset]
        print(f"   QQP: {len(qqp_data):,} pairs")
    except Exception as e:
        print(f"   QQP: Failed to load ({e})")
        qqp_data = []
    
    return {
        'stsb': stsb_data,
        'nli': nli_data,
        'wiki': wiki_data,
        'qqp': qqp_data,
    }

def get_curriculum_mix(step):
    """Get dataset mix based on training phase"""
    if step < 100:
        return config['phase1_mix']
    elif step < 300:
        return config['phase2_mix']
    else:
        return config['phase3_mix']

def sample_batch(datasets, mix, batch_size):
    """Sample a batch according to current curriculum mix"""
    batch = []
    
    for dataset_name, ratio in mix.items():
        if ratio == 0 or dataset_name not in datasets or len(datasets[dataset_name]) == 0:
            continue
        
        n_samples = int(batch_size * ratio)
        if n_samples == 0:
            continue
        
        dataset = datasets[dataset_name]
        indices = np.random.choice(len(dataset), size=n_samples, replace=True)
        batch.extend([dataset[i] for i in indices])
    
    # Shuffle batch
    np.random.shuffle(batch)
    return batch[:batch_size]  # Ensure exact batch size

def evaluate_stsb(model):
    """Evaluate on STS-B validation"""
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

# ============================================================================
# TRAINING
# ============================================================================
def train_multitask():
    # Load all datasets
    datasets = load_multitask_data()
    
    # Check we have at least STS-B
    if not datasets['stsb']:
        print("\n❌ Failed to load STS-B! Cannot proceed.")
        return
    
    # Model
    model = MultiTaskModel(config['base_model'], config['checkpoint'], config).to(device)
    
    # Baseline
    print(f"\n📊 BASELINE...")
    baseline_pearson, _ = evaluate_stsb(model)
    print(f"   Pearson: {baseline_pearson:.4f}")
    print(f"   Gap to 0.850: {config['target_score'] - baseline_pearson:.4f}")
    
    # Optimizer
    trainable = (list(model.deltanet_layers.parameters()) + list(model.norms.parameters()) +
                list(model.ffns.parameters()) + list(model.ffn_norms.parameters()))
    
    optimizer = AdamW(trainable, lr=config['learning_rate'], 
                     weight_decay=config['weight_decay'])
    
    print(f"\n▶️  Multi-task training with LR {config['learning_rate']:.2e}")
    
    # Training
    model.train()
    best_pearson = baseline_pearson
    best_step = 0
    patience = 0
    
    global_step = 0
    running_loss = 0.0
    log_count = 0
    
    pbar = tqdm(total=config['max_steps'], desc="📚 Multi-task")
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    while global_step < config['max_steps']:
        # Get current curriculum mix
        current_mix = get_curriculum_mix(global_step)
        
        # Sample batch
        batch_data = sample_batch(datasets, current_mix, config['batch_size'])
        
        if len(batch_data) == 0:
            print("\n❌ Empty batch! Skipping...")
            global_step += 1
            continue
        
        s1 = [item[0] for item in batch_data]
        s2 = [item[1] for item in batch_data]
        scores = torch.tensor([item[2] for item in batch_data], 
                             dtype=torch.float32, device=device)
        
        # Tokenize
        tokens1 = model.tokenizer(s1, padding=True, max_length=128,
                                 truncation=True, return_tensors='pt').to(device)
        tokens2 = model.tokenizer(s2, padding=True, max_length=128,
                                 truncation=True, return_tensors='pt').to(device)
        
        # Forward
        emb1 = model(tokens1['input_ids'], tokens1['attention_mask'])
        emb2 = model(tokens2['input_ids'], tokens2['attention_mask'])
        
        # Loss - regression MSE
        sim = F.cosine_similarity(emb1, emb2, dim=1)
        pred = (sim + 1) * 2.5  # Scale to [0, 5]
        loss = F.mse_loss(pred, scores)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip'])
        optimizer.step()
        
        running_loss += loss.item()
        log_count += 1
        
        if log_count >= 5:
            phase = "P1" if global_step < 100 else "P2" if global_step < 300 else "P3"
            stsb_pct = int(current_mix.get('stsb', 0) * 100)
            pbar.set_postfix({
                'loss': f'{running_loss/log_count:.4f}',
                'phase': phase,
                'stsb': f'{stsb_pct}%',
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
                status = f"⚠️  Low (patience {patience}/{config['patience']})"
            else:
                patience += 1
                status = f"📊 (patience {patience}/{config['patience']})"
            
            phase = "Phase 1" if global_step < 100 else "Phase 2" if global_step < 300 else "Phase 3"
            
            print(f"\n📊 Step {global_step+1} ({phase}):")
            print(f"   Pearson:  {current_pearson:.4f} ({current_pearson-baseline_pearson:+.4f}) {status}")
            print(f"   Best:     {best_pearson:.4f} (step {best_step})")
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
    print("✅ MULTI-TASK TRAINING COMPLETE")
    print("="*80)
    final_pearson, _ = evaluate_stsb(model)
    print(f"\n📊 RESULTS:")
    print(f"   Baseline: {baseline_pearson:.4f}")
    print(f"   Final:    {final_pearson:.4f} ({final_pearson-baseline_pearson:+.4f})")
    print(f"   Best:     {best_pearson:.4f} (step {best_step}) ({best_pearson-baseline_pearson:+.4f})")
    
    if best_pearson >= config['target_score']:
        print(f"\n🎉🎉🎉 TARGET 0.850 ACHIEVED! 🎉🎉🎉")
    else:
        gap = config['target_score'] - best_pearson
        print(f"\n📊 Gap: {gap:.4f}")
        if gap < 0.003:
            print(f"\n💡 SO CLOSE! Try ensemble with all your top checkpoints")

if __name__ == "__main__":
    train_multitask()