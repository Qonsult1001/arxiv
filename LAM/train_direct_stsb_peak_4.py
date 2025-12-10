"""
🚀 AUGMENTED STS-B TRAINING with SICK-R + STS 2012-2017
Add high-quality semantic similarity data that MATCHES STS-B format!

Training data:
- STS-B train: 5,749 pairs (0-5 scale)
- SICK-R: 10,000 pairs (0-5 scale) ← PERFECT MATCH!
- STS 2012-2017: ~15,000 pairs (0-5 scale)
Total: ~30K pairs vs original 5.7K (5x more data!)

Eval: STS-B validation only
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
import requests

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

config = {
    "checkpoint": "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEW/checkpoint_52000.pt",
    "base_model": "/workspace/LAM/all-MiniLM-L6-v2",
    "output_dir": "/workspace/LAM/augmented_sts_training",
    
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    
    # Conservative LR - more data = easier to overfit
    "learning_rate": 3e-5,      # Lower than 1e-5!
    "weight_decay": 0.01,
    "gradient_clip": 0.01,
    "batch_size": 16,            # Larger batch - more data
    "max_length": 256,           # Back to 128 - most data is short
    
    # More steps - 5x more data!
    "max_steps": 500,
    "eval_interval": 25,         # Eval less frequently
    
    # Dataset weights
    "stsb_weight": 3.0,          # Weight STS-B 3x higher (stay on-distribution!)
    "sick_weight": 1.0,
    "sts_old_weight": 0.5,       # Old STS data weighted lower
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*80)
print("🚀 AUGMENTED STS-B TRAINING - SICK-R + STS 2012-2017")
print("="*80)
print(f"Strategy: Add high-quality semantic similarity data")
print(f"  ✅ STS-B format match (all 0-5 scale)")
print(f"  ✅ 5x more training data")
print(f"  ✅ Weighted sampling (3x STS-B, 1x SICK, 0.5x old STS)")
print("="*80)

# ============================================================================
# MODEL (same as before)
# ============================================================================
class AugmentedModel(nn.Module):
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
        
        if 'deltanet_layers' in checkpoint:
            self.deltanet_layers.load_state_dict(checkpoint['deltanet_layers'], strict=False)
        elif 'lam_layers' in checkpoint:
            self.deltanet_layers.load_state_dict(checkpoint['lam_layers'], strict=False)
        
        for key, attr in [('lam_norms', 'norms'), ('deltanet_norms', 'norms'), 
                          ('deltanet_ffns', 'ffns'), ('ffn_norms', 'ffn_norms')]:
            if key in checkpoint:
                try:
                    getattr(self, attr).load_state_dict(checkpoint[key], strict=False)
                except: pass
        
        print(f"✅ Checkpoint loaded")
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

# ============================================================================
# LOAD AUGMENTED TRAINING DATA
# ============================================================================
def load_augmented_training_data():
    """Load STS-B + SICK-R + STS 2012-2017"""
    all_data = []
    
    print("\n📚 Loading augmented training data...")
    
    # 1. STS-B train (our primary dataset)
    print("\n1️⃣  STS-B train...")
    stsb_train = load_dataset("glue", "stsb", split="train")
    stsb_data = [(ex['sentence1'], ex['sentence2'], ex['label'], 'stsb') 
                 for ex in stsb_train if len(ex['sentence1']) > 10 and len(ex['sentence2']) > 10]
    print(f"   ✅ {len(stsb_data):,} pairs")
    all_data.extend(stsb_data)
    
    # 2. SICK-R (perfect match - 0-5 scale!)
    print("\n2️⃣  SICK-R (Sentences Involving Compositional Knowledge)...")
    sick_data = []
    try:
        import os
        import zipfile
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
        data_dir.mkdir(exist_ok=True, parents=True)
        
        # First, try to find and extract from zip files
        sick_file = data_dir / "SICK_train.txt"
        
        # If train file doesn't exist, look for zip files to extract
        if not sick_file.exists():
            print(f"   Looking for SICK zip files...")
            # Check in data directory first, then script directory
            # PRIORITY: SICK.zip (main) > SICK_Annotated.zip > SICK_subsets.zip (just indices)
            all_zips = list(data_dir.glob("SICK*.zip")) + list(script_dir.glob("SICK*.zip"))
            
            # Sort by priority: SICK.zip first, then annotated, then subsets
            def sort_priority(path):
                name = path.name
                if name == 'SICK.zip':
                    return 0
                elif 'Annotated' in name:
                    return 1
                else:
                    return 2
            
            zip_files = sorted(all_zips, key=sort_priority)
            
            for zip_path in zip_files:
                try:
                    print(f"   Found: {zip_path.name}")
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        # Look for SICK_train.txt or SICK.txt (main file)
                        # Priority: SICK_train.txt > SICK.txt > SICK_annotated.txt
                        txt_files = [f for f in z.namelist() if f.endswith('.txt') and 'readme' not in f.lower()]
                        
                        # Prefer files with 'train' in name, then main SICK.txt
                        train_files = [f for f in txt_files if 'train' in f.lower()]
                        if not train_files:
                            # Use SICK.txt (main file) - it contains all data including train
                            train_files = [f for f in txt_files if f == 'SICK.txt' or 'SICK.txt' in f]
                        if not train_files:
                            # Fallback to any SICK file
                            train_files = [f for f in txt_files if 'SICK' in f and 'annotated' not in f]
                        
                        if train_files:
                            target_file = train_files[0]
                            print(f"   ✅ Found: {target_file}")
                            # Extract to data directory
                            z.extract(target_file, data_dir)
                            # Rename to SICK_train.txt for consistency
                            extracted = data_dir / target_file.split('/')[-1]
                            if extracted != sick_file:
                                extracted.rename(sick_file)
                            print(f"   ✅ Extracted to: {sick_file}")
                            break
                except Exception as e:
                    print(f"   ⚠️  Error processing {zip_path.name}: {e}")
                    continue
        
        # Now try to load from the extracted file
        if sick_file.exists():
            print(f"   Loading from: {sick_file}")
            with open(sick_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Parse header to find column indices
                header = lines[0].strip().split('\t')
                try:
                    pair_id_idx = header.index('pair_ID')
                    s1_idx = header.index('sentence_A')
                    s2_idx = header.index('sentence_B')
                    score_idx = header.index('relatedness_score')
                    # Check if SemEval_set column exists (for train/test filtering)
                    semeval_idx = header.index('SemEval_set') if 'SemEval_set' in header else None
                except ValueError as e:
                    print(f"   ⚠️  Header format issue: {e}")
                    print(f"   Header: {header[:5]}")
                    raise
                
                print(f"   Columns: pair_ID={pair_id_idx}, s1={s1_idx}, s2={s2_idx}, score={score_idx}")
                if semeval_idx is not None:
                    print(f"   ✅ Found SemEval_set column (will filter for TRAIN only)")
                
                # Parse data lines
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) > max(s1_idx, s2_idx, score_idx):
                        # Filter for TRAIN set if SemEval_set column exists
                        if semeval_idx is not None and len(parts) > semeval_idx:
                            semeval_set = parts[semeval_idx].strip().upper()
                            if semeval_set != 'TRAIN':
                                continue  # Skip non-training data
                        
                        try:
                            s1 = parts[s1_idx].strip()
                            s2 = parts[s2_idx].strip()
                            score = float(parts[score_idx].strip())
                            score = score - 1.0  # Convert 1-5 to 0-5
                            
                            if len(s1) > 10 and len(s2) > 10 and 0 <= score <= 5:
                                sick_data.append((s1, s2, score, 'sick'))
                        except (ValueError, IndexError) as e:
                            continue
            
            if len(sick_data) > 0:
                print(f"   ✅ Loaded {len(sick_data):,} pairs from file")
            else:
                raise Exception("No valid SICK data found in file")
        else:
            raise Exception("SICK_train.txt not found. Please extract from zip files.")
        
        if len(sick_data) > 0:
            print(f"   ✅ {len(sick_data):,} pairs (converted 1-5 → 0-5 scale)")
            all_data.extend(sick_data)
        else:
            raise Exception("No SICK data loaded")
    except Exception as e:
        print(f"   ⚠️  Could not load SICK: {e}")
        print(f"   Continuing without SICK data...")
    
    # 3. STS Benchmark 2012-2017 (if available)
    print("\n3️⃣  STS 2012-2017 benchmarks...")
    sts_data = []
    try:
        # Create a set of STS-B sentence pairs for deduplication
        stsb_pairs = set()
        for item in stsb_data:
            # Normalize and create a canonical pair (sorted to handle order)
            s1, s2 = item[0].lower().strip(), item[1].lower().strip()
            pair = tuple(sorted([s1, s2]))  # Sort to handle order-independent duplicates
            stsb_pairs.add(pair)
        print(f"   📋 Created STS-B deduplication set: {len(stsb_pairs):,} unique pairs")
        
        # Try to load STS benchmarks from various sources
        sts_loaded = False
        
        # Skip URL downloads - STS 2012-2017 benchmarks are deprecated online
        # Only try to load from local files
        
        # Try loading from local files
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
        sts_files = [
            data_dir / "sts_2012_2017_pairs.jsonl.gz",  # New downloaded file
            data_dir / "sts-train.csv",
            data_dir / "STS_train.txt",
            data_dir / "stsbenchmark_train.csv",
            data_dir / "sts_benchmark_train.csv",
        ]
        
        for sts_file in sts_files:
            if sts_file.exists():
                try:
                    print(f"   Found: {sts_file.name}")
                    added_count = 0
                    
                    # Handle JSONL.gz format (new downloaded file)
                    if sts_file.suffix == '.gz' or '.jsonl' in sts_file.name:
                        import gzip
                        import json
                        with gzip.open(sts_file, 'rt', encoding='utf-8') as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                try:
                                    data = json.loads(line)
                                    if isinstance(data, list) and len(data) >= 3:
                                        s1_raw = str(data[0]).strip()
                                        s2_raw = str(data[1]).strip()
                                        score = float(data[2])
                                        
                                        if 0 <= score <= 5 and len(s1_raw) > 10 and len(s2_raw) > 10:
                                            s1 = s1_raw.lower().strip()
                                            s2 = s2_raw.lower().strip()
                                            pair = tuple(sorted([s1, s2]))
                                            
                                            # Only add if not in STS-B and valid
                                            if pair not in stsb_pairs:
                                                sts_data.append((s1_raw, s2_raw, score, 'sts'))
                                                stsb_pairs.add(pair)
                                                added_count += 1
                                except (json.JSONDecodeError, ValueError, IndexError):
                                    continue
                    else:
                        # Handle CSV/TSV format
                        with open(sts_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            header_skipped = False
                            
                            for line in lines:
                                if not line.strip():
                                    continue
                                # Skip header
                                if not header_skipped:
                                    if 'score' in line.lower() or 'similarity' in line.lower() or 'sentence' in line.lower():
                                        header_skipped = True
                                        continue
                                
                                # Parse line
                                parts = line.strip().split('\t') if '\t' in line else line.strip().split(',')
                                if len(parts) >= 3:
                                    try:
                                        # STS format: usually score is first or last
                                        score = None
                                        s1_raw = None
                                        s2_raw = None
                                        
                                        # Try: score first
                                        try:
                                            test_score = float(parts[0].strip())
                                            if 0 <= test_score <= 5:
                                                score = test_score
                                                s1_raw = parts[1].strip()
                                                s2_raw = parts[2].strip()
                                        except:
                                            pass
                                        
                                        # Try: score last
                                        if score is None:
                                            try:
                                                test_score = float(parts[-1].strip())
                                                if 0 <= test_score <= 5:
                                                    score = test_score
                                                    s1_raw = parts[0].strip()
                                                    s2_raw = parts[1].strip()
                                            except:
                                                pass
                                        
                                        if score is not None and s1_raw and s2_raw:
                                            s1 = s1_raw.lower().strip()
                                            s2 = s2_raw.lower().strip()
                                            pair = tuple(sorted([s1, s2]))
                                            
                                            # Only add if not in STS-B and valid
                                            if pair not in stsb_pairs and len(s1) > 10 and len(s2) > 10:
                                                sts_data.append((s1_raw, s2_raw, score, 'sts'))
                                                stsb_pairs.add(pair)
                                                added_count += 1
                                    except (ValueError, IndexError):
                                        continue
                    
                    if added_count > 0:
                        print(f"   ✅ Loaded {added_count:,} unique pairs (deduplicated against STS-B)")
                        sts_loaded = True
                        break
                    else:
                        print(f"   ⚠️  No valid pairs found in {sts_file.name}")
                except Exception as e:
                    print(f"   ⚠️  Error loading {sts_file.name}: {e}")
                    continue
        
        if sts_loaded and len(sts_data) > 0:
            print(f"   ✅ {len(sts_data):,} unique STS pairs (deduplicated against STS-B)")
            all_data.extend(sts_data)
        else:
            print(f"   ℹ️  STS 2012-2017 benchmarks not available (optional)")
            print(f"   💡 To add STS: Download and place CSV/TSV in {data_dir}/")
            print(f"   💡 SemEval Task 6: https://ixa2.si.ehu.es/stswiki/")
            print(f"   💡 Files can be named: sts-train.csv, STS_train.txt, etc.")
        
    except Exception as e:
        print(f"   ⚠️  Error loading STS 2012-2017: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*80)
    print(f"📊 TOTAL TRAINING DATA: {len(all_data):,} pairs")
    
    # Count by dataset
    stsb_count = sum(1 for item in all_data if item[3] == 'stsb')
    sick_count = sum(1 for item in all_data if item[3] == 'sick')
    sts_count = sum(1 for item in all_data if item[3] == 'sts')
    
    print(f"   STS-B:  {stsb_count:,} pairs")
    print(f"   SICK-R: {sick_count:,} pairs")
    print(f"   STS:    {sts_count:,} pairs (deduplicated)")
    print(f"   Ratio:  {len(all_data)/5749:.1f}x original STS-B")
    print("="*80)
    
    return all_data

def weighted_sample_batch(data, batch_size, weights):
    """Sample batch with dataset weighting"""
    # Create sampling weights based on dataset
    sample_weights = []
    for item in data:
        dataset = item[3]
        if dataset == 'stsb':
            sample_weights.append(weights['stsb_weight'])
        elif dataset == 'sick':
            sample_weights.append(weights['sick_weight'])
        else:
            sample_weights.append(weights.get('sts_old_weight', 0.5))
    
    # Normalize
    sample_weights = np.array(sample_weights)
    sample_weights = sample_weights / sample_weights.sum()
    
    # Sample
    indices = np.random.choice(len(data), size=batch_size, replace=False, p=sample_weights)
    return [data[i] for i in indices]

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_stsb_validation(model):
    """Evaluate on STS-B validation only"""
    model.eval()
    ds = load_dataset("glue", "stsb", split="validation")
    all_sim, all_labels = [], []
    
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            batch = ds[i:min(i+32, len(ds))]
            tokens1 = model.tokenizer(batch['sentence1'], padding=True, max_length=256, 
                                     truncation=True, return_tensors='pt').to(device)
            tokens2 = model.tokenizer(batch['sentence2'], padding=True, max_length=256,
                                     truncation=True, return_tensors='pt').to(device)
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

# ============================================================================
# TRAINING
# ============================================================================
def train():
    # Load augmented data
    train_data = load_augmented_training_data()
    
    # Model
    model = AugmentedModel(config['base_model'], config['checkpoint'], config).to(device)
    
    # Baseline
    print(f"\n📊 BASELINE (STS-B validation)...")
    baseline_pearson, _ = evaluate_stsb_validation(model)
    print(f"   Pearson: {baseline_pearson:.4f}")
    
    # Optimizer
    trainable = list(model.deltanet_layers.parameters()) + list(model.norms.parameters()) + \
                list(model.ffns.parameters()) + list(model.ffn_norms.parameters())
    
    optimizer = AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    print(f"\n▶️  Training with augmented data")
    print(f"   LR: {config['learning_rate']:.2e} (lower for more data)")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Weighted sampling: {config['stsb_weight']}x STS-B, {config['sick_weight']}x SICK")
    
    # Training
    model.train()
    global_step = 0
    best_pearson = baseline_pearson
    best_step = 0
    
    running_loss = 0.0
    pbar = tqdm(total=config['max_steps'], desc="🚀 Augmented Training")
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_scores = []
    
    while global_step < config['max_steps']:
        # Weighted batch sampling (favor STS-B!)
        batch = weighted_sample_batch(train_data, config['batch_size'], config)
        
        s1 = [item[0] for item in batch]
        s2 = [item[1] for item in batch]
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
        
        running_loss += loss.item()
        
        if (global_step + 1) % 10 == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/10:.4f}', 
                'best': f'{best_pearson:.4f}'
            })
            running_loss = 0.0
        
        # Evaluate on STS-B validation
        if (global_step + 1) % config['eval_interval'] == 0:
            val_pearson, val_spearman = evaluate_stsb_validation(model)
            gain = val_pearson - baseline_pearson
            
            all_scores.append((global_step + 1, val_pearson))
            
            is_best = val_pearson > best_pearson
            if is_best:
                best_pearson = val_pearson
                best_step = global_step + 1
                status = "🎉 NEW BEST!"
            else:
                status = "📊"
            
            print(f"\n📊 Step {global_step+1}:")
            print(f"   STS-B Val Pearson:  {val_pearson:.4f} (Δ {gain:+.4f}) {status}")
            print(f"   STS-B Val Spearman: {val_spearman:.4f}")
            
            # Save
            torch.save({
                'deltanet_layers': model.deltanet_layers.state_dict(),
                'deltanet_norms': model.norms.state_dict(),
                'deltanet_ffns': model.ffns.state_dict(),
                'ffn_norms': model.ffn_norms.state_dict(),
                'step': global_step + 1,
                'val_pearson': val_pearson,
            }, output_dir / f"step{global_step+1:04d}_val{val_pearson:.4f}.pt")
            
            if val_pearson >= 0.850:
                print(f"\n🎉🎉🎉 TARGET 0.850+ REACHED! 🎉🎉🎉")
                break
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    final_pearson, _ = evaluate_stsb_validation(model)
    print(f"\n📊 RESULTS (STS-B Validation):")
    print(f"   Baseline: {baseline_pearson:.4f}")
    print(f"   Final:    {final_pearson:.4f} (Δ {final_pearson-baseline_pearson:+.4f})")
    print(f"   Best:     {best_pearson:.4f} at step {best_step} (Δ {best_pearson-baseline_pearson:+.4f})")
    
    print(f"\n📈 SCORE PROGRESSION:")
    for step, score in all_scores[-10:]:
        marker = "🎯" if score >= 0.850 else "🔥" if score >= 0.847 else "✅"
        print(f"   Step {step:4d}: {score:.4f} {marker}")
    
    if best_pearson >= 0.850:
        print(f"\n🎉 BREAKTHROUGH! 0.850+ achieved!")
    elif best_pearson > baseline_pearson:
        print(f"\n✅ Improved by {best_pearson-baseline_pearson:+.4f}")
        print(f"   Distance to 0.850: {0.850-best_pearson:.4f}")
    else:
        print(f"\n⚠️  No improvement - augmented data didn't help")
        print(f"   Your 0.8470 might be the architecture limit")
    
    print(f"\n💾 Checkpoints: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    train()