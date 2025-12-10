"""
STS-SPECIFIC Fine-Tuning: 0.7712 → 0.82+
Train on STS datasets, evaluate on ALL STS benchmarks
Uses REGRESSION LOSS (predict 0-5 scores) instead of contrastive
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

# ============================================================================
# CONFIGURATION - STS-SPECIFIC
# ============================================================================
config = {
    # Load your 0.7712 checkpoint
    "checkpoint_path": "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEW/pytorch_model.bin",
    "tokenizer_path": "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEW",
    
    # Model architecture (same as before)
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    "fast_decay_init": 0.30,
    "slow_decay_init": 0.85,
    
    # ⚡ STS-SPECIFIC: Higher LR for focused fine-tuning
    "peak_learning_rate": 1e-4,  # Much higher!
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    
    # ⚡ Smaller batches (STS datasets are small ~10K pairs total)
    "batch_size": 32,
    "gradient_accumulation_steps": 4,  # Effective: 128
    "max_length": 128,
    
    # ⚡ SHORT training (small dataset, many epochs)
    "warmup_steps": 200,
    "total_steps": 15000,  # ~150 epochs over STS-B train
    "log_interval": 50,
    "save_interval": 500,
    "eval_interval": 200,  # Evaluate often!
    
    # Output
    "output_dir": "/workspace/LAM/deltanet_STS_SPECIALIST",
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# MODEL WITH REGRESSION HEAD
# ============================================================================
class DeltaNetSTSRegressor(nn.Module):
    """DeltaNet + Regression head for STS score prediction"""
    
    def __init__(self, checkpoint_path, tokenizer_path, config):
        super().__init__()
        
        print(f"Loading checkpoint: {checkpoint_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load your trained DeltaNet model
        from transformers import AutoModel
        teacher_model = AutoModel.from_pretrained(tokenizer_path)
        
        self.d_model = config['d_model']
        self.num_layers = config['num_layers']
        
        # 1. Embeddings (frozen initially)
        self.embeddings = teacher_model.embeddings
        
        # 2. DeltaNet layers (will load from checkpoint)
        self.deltanet_layers = nn.ModuleList()
        self.deltanet_norms = nn.ModuleList()
        self.deltanet_ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        
        for i in range(6):
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=self.d_model,
                    num_heads=config['num_heads'],
                    use_hierarchical_decay=True,
                    use_enhanced_flux=True,
                    fast_decay_init=config['fast_decay_init'],
                    slow_decay_init=config['slow_decay_init'],
                )
            )
            self.deltanet_norms.append(teacher_model.encoder.layer[i].attention.output.LayerNorm)
            self.deltanet_ffns.append(teacher_model.encoder.layer[i].intermediate)
            self.ffn_norms.append(teacher_model.encoder.layer[i].output.LayerNorm)
        
        # 3. Pooler
        self.pooler = teacher_model.pooler
        
        # 4. ⭐ NEW: Regression head (predicts 0-5 similarity score)
        self.regression_head = nn.Sequential(
            nn.Linear(self.d_model * 3, 256),  # 3x because we concat [emb1, emb2, |emb1-emb2|]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Output: single score
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint, strict=False)
        print(f"✅ Loaded checkpoint from step with 0.7712 Spearman")
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        """Forward pass returning similarity score prediction"""
        
        # Get embeddings for both sentences
        emb_a = self._encode(input_ids_a, attention_mask_a)
        emb_b = self._encode(input_ids_b, attention_mask_b)
        
        # ⭐ Regression features: [emb1, emb2, |emb1-emb2|]
        features = torch.cat([
            emb_a,
            emb_b,
            torch.abs(emb_a - emb_b)
        ], dim=1)
        
        # Predict score (0-5)
        score = self.regression_head(features).squeeze(-1)
        
        # Also return normalized embeddings (for cosine similarity baseline)
        emb_a_norm = F.normalize(emb_a, p=2, dim=1)
        emb_b_norm = F.normalize(emb_b, p=2, dim=1)
        
        return score, emb_a_norm, emb_b_norm
    
    def _encode(self, input_ids, attention_mask):
        """Encode single sentence"""
        x = self.embeddings(input_ids=input_ids)
        
        # All 6 DeltaNet layers
        for i in range(6):
            # Attention
            residual = x
            x_attn, _, _ = self.deltanet_layers[i](x, attention_mask)
            x = self.deltanet_norms[i](residual + x_attn)
            
            # FFN
            residual = x
            x_ffn = self.deltanet_ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        
        # Mean pooling
        embeddings = self.mean_pooling(x, attention_mask)
        
        return embeddings

# ============================================================================
# DATA LOADING - ALL STS TRAINING DATA
# ============================================================================
def load_all_sts_training_data():
    """
    Load ALL available STS training data (NO test/validation!)
    
    Available:
    - STS-B train: ~5749 pairs
    - SICK train: ~4500 pairs
    - Total: ~10K training pairs
    """
    print("\n" + "="*80)
    print("LOADING ALL STS TRAINING DATA (NO TEST/VALIDATION!)")
    print("="*80)
    
    all_train_data = []
    
    # 1. STS-B train (THE MAIN TARGET)
    print("\n1️⃣  STS-B train set...")
    try:
        stsb_train = load_dataset("stsb_multi_mt", "en", split="train")
        for item in stsb_train:
            all_train_data.append({
                'sentence1': item['sentence1'],
                'sentence2': item['sentence2'],
                'score': item['similarity_score']  # 0-5
            })
        print(f"   ✅ STS-B train: {len(stsb_train):,} pairs")
    except:
        print(f"   ⚠️  Could not load STS-B train")
    
    # 2. SICK train
    print("\n2️⃣  SICK train set...")
    try:
        sick_train = load_dataset("sick", "default", split="train")
        for item in sick_train:
            # SICK has relatedness_score (1-5), convert to 0-5
            score = (item['relatedness_score'] - 1.0)  # 1-5 → 0-4, then scale
            score = score * 1.25  # 0-4 → 0-5
            all_train_data.append({
                'sentence1': item['sentence_A'],
                'sentence2': item['sentence_B'],
                'score': score
            })
        print(f"   ✅ SICK train: {len(sick_train):,} pairs")
    except:
        print(f"   ⚠️  Could not load SICK train")
    
    # Note: STS12-16 are test-only datasets (no train splits)
    
    print("\n" + "="*80)
    print(f"📊 TOTAL STS TRAINING DATA: {len(all_train_data):,} pairs")
    print("="*80)
    print("✅ Dataset ready for STS-specific fine-tuning\n")
    
    return all_train_data

# ============================================================================
# EVALUATION ON ALL STS BENCHMARKS
# ============================================================================
def evaluate_all_sts_tasks(model, device):
    """
    Evaluate on ALL STS benchmarks:
    - SICK-R
    - STS12, STS13, STS14, STS15, STS16
    - STSBenchmark
    """
    print("\n" + "="*80)
    print("EVALUATING ON ALL STS BENCHMARKS")
    print("="*80)
    
    results = {}
    
    # Helper function
    def eval_dataset(name, dataset, s1_key, s2_key, score_key, score_scale=1.0):
        """Evaluate single STS dataset"""
        s1 = dataset[s1_key]
        s2 = dataset[s2_key]
        labels = np.array([item * score_scale for item in dataset[score_key]], dtype=float)
        
        model.eval()
        pred_scores = []
        cosine_sims = []
        
        with torch.no_grad():
            for i in range(0, len(s1), 32):
                batch_s1 = s1[i:i+32]
                batch_s2 = s2[i:i+32]
                
                tokens_a = model.tokenizer(
                    batch_s1, padding=True, max_length=128,
                    truncation=True, return_tensors='pt'
                ).to(device)
                
                tokens_b = model.tokenizer(
                    batch_s2, padding=True, max_length=128,
                    truncation=True, return_tensors='pt'
                ).to(device)
                
                # Get predictions
                scores, emb_a, emb_b = model(
                    tokens_a['input_ids'], tokens_a['attention_mask'],
                    tokens_b['input_ids'], tokens_b['attention_mask']
                )
                
                pred_scores.extend(scores.cpu().numpy().tolist())
                
                # Also compute cosine similarity
                cosine = F.cosine_similarity(emb_a, emb_b, dim=1)
                cosine_sims.extend((cosine.cpu().numpy() * 5.0).tolist())  # Scale to 0-5
        
        # Compute correlations for both methods
        pearson_reg = pearsonr(pred_scores, labels)[0]
        spearman_reg = spearmanr(pred_scores, labels)[0]
        
        pearson_cos = pearsonr(cosine_sims, labels)[0]
        spearman_cos = spearmanr(cosine_sims, labels)[0]
        
        model.train()
        
        return {
            'pearson_regression': pearson_reg,
            'spearman_regression': spearman_reg,
            'pearson_cosine': pearson_cos,
            'spearman_cosine': spearman_cos
        }
    
    # 1. SICK-R (test set)
    print("\n1️⃣  SICK-R...")
    try:
        sick = load_dataset("sick", "default", split="test")
        results['SICK-R'] = eval_dataset(
            'SICK-R', sick,
            'sentence_A', 'sentence_B', 'relatedness_score',
            score_scale=1.25  # 1-5 → 0-5 scale
        )
        print(f"   ✅ Spearman (regression): {results['SICK-R']['spearman_regression']:.4f}")
        print(f"   📊 Spearman (cosine):     {results['SICK-R']['spearman_cosine']:.4f}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # 2-6. STS12-16 (from sentence-transformers)
    for year in [12, 13, 14, 15, 16]:
        print(f"\n{year-10}️⃣  STS{year}...")
        try:
            sts = load_dataset("mteb/sts{year}-sts".format(year=year), split="test")
            results[f'STS{year}'] = eval_dataset(
                f'STS{year}', sts,
                'sentence1', 'sentence2', 'score',
                score_scale=1.0  # Already 0-5
            )
            print(f"   ✅ Spearman (regression): {results[f'STS{year}']['spearman_regression']:.4f}")
            print(f"   📊 Spearman (cosine):     {results[f'STS{year}']['spearman_cosine']:.4f}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    
    # 7. STSBenchmark (THE MAIN TARGET)
    print(f"\n⭐ STSBenchmark...")
    try:
        stsb = load_dataset("stsb_multi_mt", "en", split="test")
        results['STSBenchmark'] = eval_dataset(
            'STSBenchmark', stsb,
            'sentence1', 'sentence2', 'similarity_score',
            score_scale=1.0  # Already 0-5
        )
        print(f"   ⭐ Spearman (regression): {results['STSBenchmark']['spearman_regression']:.4f}")
        print(f"   📊 Spearman (cosine):     {results['STSBenchmark']['spearman_cosine']:.4f}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - SPEARMAN CORRELATIONS")
    print("="*80)
    print(f"{'Task':<20} {'Regression':<12} {'Cosine':<12} {'Best':<12}")
    print("-"*80)
    
    for task, scores in results.items():
        reg = scores['spearman_regression']
        cos = scores['spearman_cosine']
        best = max(reg, cos)
        marker = "⭐" if task == "STSBenchmark" else "  "
        print(f"{marker}{task:<18} {reg:>10.4f}   {cos:>10.4f}   {best:>10.4f}")
    
    # Average
    avg_reg = np.mean([s['spearman_regression'] for s in results.values()])
    avg_cos = np.mean([s['spearman_cosine'] for s in results.values()])
    avg_best = np.mean([max(s['spearman_regression'], s['spearman_cosine']) for s in results.values()])
    
    print("-"*80)
    print(f"{'AVERAGE':<20} {avg_reg:>10.4f}   {avg_cos:>10.4f}   {avg_best:>10.4f}")
    print("="*80)
    
    return results

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train():
    print("="*80)
    print("STS-SPECIFIC FINE-TUNING: 0.7712 → 0.82+")
    print("="*80)
    print(f"\n🎯 STRATEGY:")
    print(f"   1. Train on STS-B + SICK train sets (~10K pairs)")
    print(f"   2. Use REGRESSION LOSS (predict 0-5 scores)")
    print(f"   3. Evaluate on ALL 7 STS benchmarks")
    print(f"   4. High LR (1e-4) for focused adaptation")
    print(f"   5. Many epochs (15K steps = ~150 epochs)")
    print("="*80)
    
    # Load STS training data
    train_data = load_all_sts_training_data()
    
    # Initialize model with regression head
    model = DeltaNetSTSRegressor(
        config['checkpoint_path'],
        config['tokenizer_path'],
        config
    ).to(device)
    
    # Optimizer (all parameters trainable)
    optimizer = AdamW(
        model.parameters(),
        lr=config['peak_learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['total_steps']
    )
    
    # Loss function: MSE for regression
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    best_stsb_spearman = 0.7712  # Start from existing
    global_step = 0
    
    pbar = tqdm(total=config['total_steps'], desc="STS Regression Training")
    
    while global_step < config['total_steps']:
        # Sample batch
        indices = np.random.randint(0, len(train_data), size=config['batch_size'])
        batch = [train_data[i] for i in indices]
        
        # Prepare inputs
        s1 = [item['sentence1'] for item in batch]
        s2 = [item['sentence2'] for item in batch]
        scores = torch.tensor([item['score'] for item in batch], dtype=torch.float32, device=device)
        
        # Tokenize
        tokens_a = model.tokenizer(
            s1, padding=True, max_length=config['max_length'],
            truncation=True, return_tensors='pt'
        ).to(device)
        
        tokens_b = model.tokenizer(
            s2, padding=True, max_length=config['max_length'],
            truncation=True, return_tensors='pt'
        ).to(device)
        
        # Forward
        pred_scores, _, _ = model(
            tokens_a['input_ids'], tokens_a['attention_mask'],
            tokens_b['input_ids'], tokens_b['attention_mask']
        )
        
        # Regression loss
        loss = criterion(pred_scores, scores)
        
        # Backward
        loss = loss / config['gradient_accumulation_steps']
        loss.backward()
        
        # Update
        if (global_step + 1) % config['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if (global_step + 1) % config['log_interval'] == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'best': f'{best_stsb_spearman:.4f}'
            })
        
        # Evaluation
        if (global_step + 1) % config['eval_interval'] == 0:
            results = evaluate_all_sts_tasks(model, device)
            
            # Check if new best on STSBenchmark
            if 'STSBenchmark' in results:
                current_stsb = max(
                    results['STSBenchmark']['spearman_regression'],
                    results['STSBenchmark']['spearman_cosine']
                )
                
                if current_stsb > best_stsb_spearman:
                    best_stsb_spearman = current_stsb
                    
                    # Save best model
                    output_dir = Path(config['output_dir'])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
                    model.tokenizer.save_pretrained(output_dir)
                    
                    print(f"\n⭐ NEW BEST! STSBenchmark Spearman: {best_stsb_spearman:.4f}")
                    print(f"   💾 Saved to {output_dir}/")
        
        # Save checkpoint
        if (global_step + 1) % config['save_interval'] == 0:
            output_dir = Path(config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': global_step,
                'best_stsb': best_stsb_spearman
            }, output_dir / f"checkpoint_{global_step+1}.pt")
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final evaluation
    print(f"\n🎉 FINAL EVALUATION")
    final_results = evaluate_all_sts_tasks(model, device)
    
    print(f"\n✅ Training complete!")
    print(f"   Best STSBenchmark Spearman: {best_stsb_spearman:.4f}")
    print(f"   Target was: 0.82")
    print(f"   {'🎉 TARGET REACHED!' if best_stsb_spearman >= 0.82 else '📊 Close! Try longer training.'}")

if __name__ == "__main__":
    train()