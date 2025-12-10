"""
Automated Ablation Suite for Scientific Paper
==============================================

This script runs the exact four ablation scenarios defined for your paper's table:
1. Enhanced LAM (Ours) - Full model
2. w/o Resonance Flux - Disable enhanced flux
3. w/o Hierarchy (Undifferentiated States) - Disable hierarchical decay (fast=slow decay)
4. w/o Orthogonal Reg. - Zero orthogonal regularization weight
5. w/o Delta Rule (Standard Linear) - Disable delta rule subtraction

The script loads your trained model from /workspace/LAM/best/pytorch_model.bin,
applies ablation settings, fine-tunes with REAL multi-domain data (same as train_6layer_deltanet_2.py),
and evaluates on STS-B test set.

NOTE: Uses REAL multi-domain training data (AllNLI, QQP, MS MARCO, WikiAnswers, SNLI)
      and proper contrastive learning + distillation (same as main training script).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import time
import gzip
from scipy.stats import pearsonr, spearmanr
import numpy as np
from pathlib import Path
import sys
from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import os

# Set cache directory
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import model classes
from train_6layer_deltanet_2 import DeltaNetPure6Layer, compute_loss, spearman_correlation_loss, pairwise_ranking_loss
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def load_data():
    """
    Multi-domain training data - SAME as train_6layer_deltanet_2.py
    """
    print("\n" + "="*80)
    print("LOADING MULTI-DOMAIN TRAINING DATA")
    print("="*80)
    all_data = []
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # 1. LOCAL AllNLI triplets
    print("\n1️⃣  Loading LOCAL AllNLI triplets...")
    try:
        allnli_path = data_dir / "AllNLI.jsonl.gz"
        if allnli_path.exists():
            with gzip.open(allnli_path, 'rt', encoding='utf-8') as f:
                count = 0
                for line in f:
                    triplet = json.loads(line)
                    if len(triplet) == 3:
                        anchor, positive, negative = triplet
                        if len(anchor) > 10 and len(positive) > 10:
                            all_data.append({'sentence1': anchor, 'sentence2': positive})
                            count += 1
            print(f"   ✅ Local AllNLI: {count:,} pairs")
        else:
            print(f"   ⚠️  Local file not found: {allnli_path} - skipping AllNLI")
    except Exception as e:
        print(f"   ⚠️  Could not load AllNLI: {e} - skipping")
    
    # 2. QQP
    print("\n2️⃣  Loading QQP...")
    try:
        qqp = load_dataset("glue", "qqp", split="train[:200000]", cache_dir="/workspace/.cache/huggingface/datasets")
        qqp_count = 0
        for item in qqp:
            if item['label'] != -1 and len(item['question1']) > 10 and len(item['question2']) > 10:
                all_data.append({'sentence1': item['question1'], 'sentence2': item['question2']})
                qqp_count += 1
        print(f"   ✅ QQP: {qqp_count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  Could not load QQP: {e}")
    
    # 3. MS MARCO
    print("\n3️⃣  Loading MS MARCO...")
    msmarco_count = 0
    try:
        msmarco = None
        for config in ["v1.1", "v2.1"]:
            try:
                msmarco = load_dataset("ms_marco", config, split="train[:100000]", cache_dir="/workspace/.cache/huggingface/datasets")
                print(f"   📦 Loaded MS MARCO {config}")
                break
            except:
                continue
        
        if msmarco is not None:
            for item in msmarco:
                if 'query' not in item or 'passages' not in item:
                    continue
                query = item['query']
                passages = item['passages']
                if isinstance(passages, str):
                    passages = json.loads(passages)
                if isinstance(passages, dict) and 'is_selected' in passages and 'passage_text' in passages:
                    selected_indices = [i for i, selected in enumerate(passages['is_selected']) if selected == 1]
                    for idx in selected_indices:
                        if idx < len(passages['passage_text']):
                            passage = passages['passage_text'][idx][:200]  # Truncate
                            if len(query) > 10 and len(passage) > 20:
                                all_data.append({'sentence1': query, 'sentence2': passage})
                                msmarco_count += 1
                                break
            print(f"   ✅ MS MARCO: {msmarco_count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  Could not load MS MARCO: {e}")
    
    # 4. WikiAnswers (from pre-generated file if available)
    print("\n4️⃣  Loading WikiAnswers...")
    wiki_count = 0
    try:
        wiki_path = data_dir / "WikiAnswers_1M.jsonl.gz"
        if wiki_path.exists() and wiki_path.stat().st_size > 0:
            with gzip.open(wiki_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    pair = json.loads(line)
                    if len(pair) == 2:
                        q1, q2 = pair
                        if len(q1) > 10 and len(q2) > 10:
                            all_data.append({'sentence1': q1, 'sentence2': q2})
                            wiki_count += 1
                            if wiki_count >= 500000:  # Limit for ablation (faster)
                                break
            print(f"   ✅ WikiAnswers: {wiki_count:,} pairs")
        else:
            print(f"   ⚠️  WikiAnswers file not found - skipping")
    except Exception as e:
        print(f"   ⚠️  Could not load WikiAnswers: {e}")
    
    # 5. SNLI
    print("\n5️⃣  Loading SNLI...")
    try:
        snli = load_dataset("snli", split="train", cache_dir="/workspace/.cache/huggingface/datasets")
        snli_count = 0
        for item in snli:
            if item['label'] in [0, 1]:
                if len(item['premise']) > 10 and len(item['hypothesis']) > 10:
                    all_data.append({'sentence1': item['premise'], 'sentence2': item['hypothesis']})
                    snli_count += 1
                    if snli_count >= 100000:
                        break
        print(f"   ✅ SNLI: {snli_count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  Could not load SNLI: {e}")
    
    print("\n" + "="*80)
    print(f"📊 TOTAL: {len(all_data):,} pairs")
    print("="*80)
    
    final_dataset = Dataset.from_list(all_data)
    final_dataset = final_dataset.shuffle(seed=42)
    
    print("✅ Dataset ready\n")
    return final_dataset


class AblationDeltaNetPure6Layer(DeltaNetPure6Layer):
    """
    DeltaNetPure6Layer with configurable ablation settings.
    """
    def __init__(self, teacher_model_name, num_linear_layers, config, ablation_config=None):
        """
        Args:
            ablation_config: Dict with ablation settings:
                - use_enhanced_flux: bool
                - use_hierarchical_decay: bool
                - use_delta_rule: bool
        """
        # Temporarily override config for layer creation
        original_config = config.copy()
        if ablation_config:
            # Override layer creation config
            config = config.copy()
            config.update(ablation_config)
        
        super().__init__(teacher_model_name, num_linear_layers, config)
        
        # Now replace layers with ablation settings if provided
        if ablation_config is not None:
            d_model = self.d_model
            for i in range(6):
                self.deltanet_layers[i] = EnhancedHierarchicalDeltaNet(
                    d_model=d_model,
                    num_heads=original_config['num_heads'],
                    use_hierarchical_decay=ablation_config.get('use_hierarchical_decay', True),
                    use_enhanced_flux=ablation_config.get('use_enhanced_flux', True),
                    use_delta_rule=ablation_config.get('use_delta_rule', True),
                    fast_decay_init=original_config.get('fast_decay_init', 0.3),
                    slow_decay_init=original_config.get('slow_decay_init', 0.85),
                ).to(self.deltanet_layers[i].q_proj.weight.device)


def evaluate_test(model, device, batch_size=32):
    """Evaluate on STS-B test set"""
    try:
        sts_test = load_dataset("sentence-transformers/stsb", split="test", cache_dir="/workspace/.cache/huggingface/datasets")
    except:
        try:
            sts_test = load_dataset("glue", "stsb", split="test", cache_dir="/workspace/.cache/huggingface/datasets")
        except:
            print("   ⚠️  Could not load STS-B test set")
            return None, None
    
    s1 = sts_test["sentence1"]
    s2 = sts_test["sentence2"]
    if 'label' in sts_test.column_names:
        labels = np.array(sts_test["label"], dtype=float)
    else:
        labels = np.array(sts_test["score"], dtype=float)
    
    model.eval()
    all_sims = []
    
    with torch.no_grad():
        for i in range(0, len(s1), batch_size):
            batch_s1 = s1[i:min(i+batch_size, len(s1))]
            batch_s2 = s2[i:min(i+batch_size, len(s2))]
            
            tokens1 = model.tokenizer(
                batch_s1, padding=True, max_length=128, 
                truncation=True, return_tensors='pt'
            ).to(device)
            tokens2 = model.tokenizer(
                batch_s2, padding=True, max_length=128,
                truncation=True, return_tensors='pt'
            ).to(device)
            
            # Get embeddings using encode (pure student, no teacher)
            emb1 = model.encode(tokens1['input_ids'], tokens1['attention_mask'])
            emb2 = model.encode(tokens2['input_ids'], tokens2['attention_mask'])
            
            # Cosine similarity
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            all_sims.extend(sim.cpu().numpy().tolist())
    
    # Compute correlations
    pearson = pearsonr(all_sims, labels)[0]
    spearman = spearmanr(all_sims, labels)[0]
    
    model.train()
    return pearson, spearman


def run_ablation_suite(
    output_file="ablation_results.json",
    num_steps=5000,  # Fine-tuning steps per experiment
    batch_size=64,
    gradient_accumulation_steps=4,  # Effective batch: 64 * 4 = 256
    checkpoint_path="/workspace/LAM/best/pytorch_model.bin",
    max_length=128
):
    print(f"🧪 STARTING ABLATION SUITE for Scientific Paper")
    print(f"==================================================")
    print(f"Using device: {device}")
    print(f"Training steps per experiment: {num_steps}")
    print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    
    # Load real multi-domain training data
    print("\n📚 Loading training data (same as train_6layer_deltanet_2.py)...")
    dataset = load_data()
    
    if len(dataset) == 0:
        print("❌ No training data loaded! Exiting.")
        return
    
    # Training config (same as train_6layer_deltanet_2.py)
    training_config = {
        'distillation_weight': 1.0,
        'layer_distill_weight': 1.5,
        'identity_reg_weight': 0.01,
        'ortho_state_reg_weight': 0.002,
        'label_smoothing': 0.1,
        'use_spearman_loss': True,
        'spearman_loss_weight': 0.3,
        'ranking_loss_weight': 0.2,
        'max_length': max_length,
    }
    
    # 1. DEFINE THE EXPERIMENTS
    experiments = [
        {
            "name": "Enhanced LAM (Ours)",
            "ablation_config": {
                "use_enhanced_flux": True,
                "use_hierarchical_decay": True,
                "use_delta_rule": True
            },
            "ortho_weight": 0.002  # Same as training
        },
        {
            "name": "w/o Resonance Flux",
            "ablation_config": {
                "use_enhanced_flux": False,
                "use_hierarchical_decay": True,
                "use_delta_rule": True
            },
            "ortho_weight": 0.002
        },
        {
            "name": "w/o Hierarchy (Undifferentiated States)",
            "ablation_config": {
                "use_enhanced_flux": True,
                "use_hierarchical_decay": False,
                "use_delta_rule": True
            },
            "ortho_weight": 0.002
        },
        {
            "name": "w/o Orthogonal Reg.",
            "ablation_config": {
                "use_enhanced_flux": True,
                "use_hierarchical_decay": True,
                "use_delta_rule": True
            },
            "ortho_weight": 0.0  # Zero orthogonal regularization
        },
        {
            "name": "w/o Delta Rule (Standard Linear)",
            "ablation_config": {
                "use_enhanced_flux": True,
                "use_hierarchical_decay": True,
                "use_delta_rule": False
            },
            "ortho_weight": 0.002
        }
    ]

    results = []

    # 2. THE LOOP
    for exp_idx, exp in enumerate(experiments):
        exp_name = exp['name']
        print(f"\n{'='*80}")
        print(f"Running Experiment {exp_idx+1}/{len(experiments)}: {exp_name}...")
        print(f"{'='*80}")
        
        # Initialize Model with ablation settings
        model_config = {
            'teacher_model': "/workspace/LAM/all-MiniLM-L6-v2",
            'num_heads': 12,
            'fast_decay_init': 0.30,
            'slow_decay_init': 0.85,
            **training_config
        }
        
        model = AblationDeltaNetPure6Layer(
            teacher_model_name=model_config['teacher_model'],
            num_linear_layers=6,
            config=model_config,
            ablation_config=exp['ablation_config']
        ).to(device)
        
        # Load checkpoint weights if available
        if Path(checkpoint_path).exists():
            print(f"📦 Loading weights from {checkpoint_path}...")
            try:
                loaded_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
                
                # Handle different checkpoint formats
                if 'deltanet_layers' in loaded_data:
                    model.deltanet_layers.load_state_dict(loaded_data['deltanet_layers'], strict=False)
                    print("   ✅ Loaded model weights")
                elif isinstance(loaded_data, dict) and any('deltanet_layers' in str(k) for k in loaded_data.keys()):
                    # Raw state dict format
                    deltanet_dict = {}
                    for key, value in loaded_data.items():
                        if 'deltanet_layers' in key:
                            new_key = key.replace('deltanet_layers.', '')
                            deltanet_dict[new_key] = value
                    if deltanet_dict:
                        # Load layer by layer
                        for i in range(6):
                            layer_dict = {}
                            for key, value in deltanet_dict.items():
                                if key.startswith(f'{i}.'):
                                    layer_dict[key.replace(f'{i}.', '')] = value
                            if layer_dict:
                                try:
                                    model.deltanet_layers[i].load_state_dict(layer_dict, strict=False)
                                except:
                                    pass
                        print("   ✅ Loaded compatible weights")
            except Exception as e:
                print(f"   ⚠️  Could not load checkpoint: {e}")
                print("   Starting from random initialization")
        
        # Setup training
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=2e-5, weight_decay=0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=500, num_training_steps=num_steps
        )
        
        # Override ortho_state_reg_weight for this experiment
        exp_config = training_config.copy()
        exp_config['ortho_state_reg_weight'] = exp['ortho_weight']
        
        # 3. TRAINING LOOP (same structure as train_6layer_deltanet_2.py)
        model.train()
        loss_history = []
        start_time = time.time()
        
        for step in range(num_steps):
            # Sample batch
            indices = np.random.randint(0, len(dataset), size=batch_size)
            batch_data = [dataset[int(i)] for i in indices]
            
            sentences_a = [item['sentence1'] for item in batch_data]
            sentences_b = [item['sentence2'] for item in batch_data]
            
            # Tokenize
            tokens_a = model.tokenizer(
                sentences_a, padding='max_length', max_length=max_length,
                truncation=True, return_tensors='pt'
            ).to(device)
            
            tokens_b = model.tokenizer(
                sentences_b, padding='max_length', max_length=max_length,
                truncation=True, return_tensors='pt'
            ).to(device)
            
            # Forward
            student_emb_a, teacher_emb_a, student_hidden_a, teacher_hidden_a, ortho_loss_a = model(
                tokens_a['input_ids'], tokens_a['attention_mask']
            )
            student_emb_b, teacher_emb_b, student_hidden_b, teacher_hidden_b, ortho_loss_b = model(
                tokens_b['input_ids'], tokens_b['attention_mask']
            )
            
            # Labels for contrastive learning
            labels = torch.arange(len(student_emb_a), device=device)
            
            # Compute loss (same as train_6layer_deltanet_2.py)
            loss, contrastive_loss, distill_loss, identity_reg, layer_distill_loss, ortho_state_reg, spearman_loss, ranking_loss = compute_loss(
                student_emb_a, student_emb_b,
                teacher_emb_a, teacher_emb_b,
                student_hidden_a, student_hidden_b,
                teacher_hidden_a, teacher_hidden_b,
                labels, model, exp_config,
                ortho_loss_a=ortho_loss_a, ortho_loss_b=ortho_loss_b
            )
            
            # Backward with gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            loss_val = loss.item() * gradient_accumulation_steps  # Scale back for logging
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            loss_history.append(loss_val)
            
            # Cleanup
            del loss, contrastive_loss, distill_loss, identity_reg, layer_distill_loss, ortho_state_reg, spearman_loss, ranking_loss
            del student_emb_a, student_emb_b, teacher_emb_a, teacher_emb_b
            del student_hidden_a, student_hidden_b, teacher_hidden_a, teacher_hidden_b
            del tokens_a, tokens_b, ortho_loss_a, ortho_loss_b, labels, batch_data
            
            if step % 500 == 0:
                print(f"  Step {step}/{num_steps} | Loss: {loss_val:.4f}")

        # 4. EVALUATION (Calculate Pearson and Spearman on STS-B test set)
        print(f"\n📊 Evaluating on STS-B test set...")
        pearson_score, spearman_score = evaluate_test(model, device)
        
        duration = time.time() - start_time
        
        # 5. LOG RESULT
        param_count = sum(p.numel() for p in trainable_params)
        result_entry = {
            "experiment": exp_name,
            "pearson_score": round(float(pearson_score), 4) if pearson_score is not None else 0.0,
            "spearman_score": round(float(spearman_score), 4) if spearman_score is not None else 0.0,
            "final_loss": round(loss_history[-1], 4),
            "training_time_sec": round(duration, 2),
            "params": param_count,
            "ablation_config": exp['ablation_config'],
            "ortho_weight": exp['ortho_weight']
        }
        results.append(result_entry)
        
        print(f"✅ Completed {exp_name}")
        if pearson_score is not None:
            print(f"   Pearson: {pearson_score:.4f}")
        if spearman_score is not None:
            print(f"   Spearman: {spearman_score:.4f}")
        print(f"   Final Loss: {loss_history[-1]:.4f}")
        print(f"   Training Time: {duration:.2f}s")
        print(f"   Trainable Parameters: {param_count:,}")
        
        # Cleanup
        del model
        del optimizer
        del scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. SAVE TO JSON
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"📄 Results saved to {output_path.absolute()}")
    print(f"{'='*80}")
    print("\n📊 SUMMARY:")
    print(f"{'Experiment':<40} {'Pearson (r)':<15} {'Spearman (ρ)':<15} {'Final Loss':<15}")
    print("-" * 90)
    for r in results:
        pearson_str = f"{r['pearson_score']:.4f}" if r['pearson_score'] > 0 else "N/A"
        spearman_str = f"{r['spearman_score']:.4f}" if r['spearman_score'] > 0 else "N/A"
        print(f"{r['experiment']:<40} {pearson_str:<15} {spearman_str:<15} {r['final_loss']:<15.4f}")
    print("\n💡 Copy the pearson_score and spearman_score values directly into your LaTeX table!")


if __name__ == "__main__":
    run_ablation_suite(
        output_file="ablation_results.json",
        num_steps=5000,  # Increase for real paper (e.g., 10000-20000)
        batch_size=64,
        gradient_accumulation_steps=4,
        checkpoint_path="/workspace/LAM/best/pytorch_model.bin",
        max_length=128
    )
