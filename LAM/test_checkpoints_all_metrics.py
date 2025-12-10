#!/usr/bin/env python3
"""
Evaluate DeltaNet checkpoints with ALL similarity metrics:
- Cosine Similarity (standard - used by stsb_evaluation.py)
- Dot Product (inner product)
- Euclidean Distance (L2 distance)

Uses the EXACT same model loading logic as stsb_evaluation.py to ensure accuracy.
"""
from pathlib import Path
import torch
import torch.nn.functional as F
import sys
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine, euclidean
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import time

sys.path.append('/workspace/LAM')
from final_solution_formula import EnhancedHierarchicalDeltaNet


def compute_all_similarities(emb1, emb2):
    """Compute all three similarity metrics"""
    emb1_np = emb1.detach().cpu().numpy()
    emb2_np = emb2.detach().cpu().numpy()
    
    metrics = {}
    
    # 1. Cosine Similarity: 1 - cosine_distance
    metrics['cosine'] = 1 - np.array([cosine(a, b) for a, b in zip(emb1_np, emb2_np)])
    
    # 2. Dot Product: direct inner product (normalized embeddings)
    metrics['dot_product'] = np.array([np.dot(a, b) for a, b in zip(emb1_np, emb2_np)])
    
    # 3. Euclidean Distance: negative distance (closer = higher score)
    metrics['euclidean'] = -np.array([euclidean(a, b) for a, b in zip(emb1_np, emb2_np)])
    
    return metrics


def get_checkpoint_embeddings_and_metrics(checkpoint_path, split='validation'):
    """Get embeddings and compute all similarity metrics - uses EXACT stsb_evaluation.py logic"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ck_name = Path(checkpoint_path).name
    print(f"\n{'='*80}")
    print(f"Processing: {ck_name} ({split})")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    step = checkpoint.get('step', 0)
    
    print(f"Checkpoint step: {step}")
    
    # Load teacher model
    teacher_model = AutoModel.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2")
    d_model = teacher_model.config.hidden_size
    
    # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Build DeltaNet model - EXACT COPY from stsb_evaluation.py
    class DeltaNet(torch.nn.Module):
        def __init__(self, teacher_model):
            super().__init__()
            self.teacher_model = teacher_model
            self.embeddings = teacher_model.embeddings
            self.deltanet_layers = torch.nn.ModuleList()
            self.deltanet_norms = torch.nn.ModuleList()
            self.deltanet_ffns = torch.nn.ModuleList()
            self.ffn_norms = torch.nn.ModuleList()
            
            for i in range(6):
                self.deltanet_layers.append(
                    EnhancedHierarchicalDeltaNet(
                        d_model=d_model,
                        num_heads=config.get('num_heads', 12),
                        use_hierarchical_decay=True,
                        use_enhanced_flux=True,
                        fast_decay_init=config.get('fast_decay_init', 0.3),
                        slow_decay_init=config.get('slow_decay_init', 0.832),
                    )
                )
                self.deltanet_norms.append(teacher_model.encoder.layer[i].attention.output.LayerNorm)
                self.deltanet_ffns.append(teacher_model.encoder.layer[i].intermediate)
                self.ffn_norms.append(teacher_model.encoder.layer[i].output.LayerNorm)
            
            # Store output dense layers
            self.output_denses = torch.nn.ModuleList()
            for i in range(6):
                dense_layer = teacher_model.encoder.layer[i].output.dense
                self.output_denses.append(dense_layer)
            
            self.pooler = teacher_model.pooler
            
            # Load weights from different checkpoint formats - EXACT COPY from stsb_evaluation.py
            model_state_dict = checkpoint.get('model_state_dict', {})
            
            # First, try to load from model_state_dict (new format)
            if model_state_dict:
                deltanet_layers_dict = {}
                deltanet_norms_dict = {}
                deltanet_ffns_dict = {}
                ffn_norms_dict = {}
                embeddings_dict = {}
                
                for key, value in model_state_dict.items():
                    if key.startswith('deltanet_layers.'):
                        new_key = key.replace('deltanet_layers.', '')
                        deltanet_layers_dict[new_key] = value
                    elif key.startswith('deltanet_norms.'):
                        new_key = key.replace('deltanet_norms.', '')
                        deltanet_norms_dict[new_key] = value
                    elif key.startswith('deltanet_ffns.'):
                        new_key = key.replace('deltanet_ffns.', '')
                        deltanet_ffns_dict[new_key] = value
                    elif key.startswith('ffn_norms.'):
                        new_key = key.replace('ffn_norms.', '')
                        ffn_norms_dict[new_key] = value
                    elif key.startswith('embeddings.') and not key.startswith('teacher_model'):
                        new_key = key.replace('embeddings.', '')
                        embeddings_dict[new_key] = value
                
                # Load student embeddings
                if embeddings_dict:
                    try:
                        self.embeddings.load_state_dict(embeddings_dict, strict=False)
                    except Exception as e:
                        print(f"   ⚠️  Warning: Could not load student embeddings: {e}")
                
                # Load deltanet_layers
                if deltanet_layers_dict:
                    for i in range(6):
                        layer_state = {k[2:]: v for k, v in deltanet_layers_dict.items() if k.startswith(f'{i}.')}
                        if layer_state:
                            try:
                                self.deltanet_layers[i].load_state_dict(layer_state, strict=False)
                            except Exception as e:
                                print(f"   ⚠️  Warning: Could not load deltanet_layers[{i}]: {e}")
                
                # Load deltanet_norms
                if deltanet_norms_dict:
                    try:
                        self.deltanet_norms.load_state_dict(deltanet_norms_dict, strict=False)
                    except:
                        for i in range(6):
                            layer_state = {k[2:]: v for k, v in deltanet_norms_dict.items() if k.startswith(f'{i}.')}
                            if layer_state:
                                self.deltanet_norms[i].load_state_dict(layer_state, strict=False)
                
                # Load deltanet_ffns
                if deltanet_ffns_dict:
                    try:
                        self.deltanet_ffns.load_state_dict(deltanet_ffns_dict, strict=False)
                    except:
                        for i in range(6):
                            layer_state = {k[2:]: v for k, v in deltanet_ffns_dict.items() if k.startswith(f'{i}.')}
                            if layer_state:
                                self.deltanet_ffns[i].load_state_dict(layer_state, strict=False)
                
                # Load ffn_norms
                if ffn_norms_dict:
                    try:
                        self.ffn_norms.load_state_dict(ffn_norms_dict, strict=False)
                    except:
                        for i in range(6):
                            layer_state = {k[2:]: v for k, v in ffn_norms_dict.items() if k.startswith(f'{i}.')}
                            if layer_state:
                                self.ffn_norms[i].load_state_dict(layer_state, strict=False)
            
            # Fallback to old checkpoint formats
            elif 'deltanet_layers' in checkpoint:
                pretrained_layers = checkpoint['deltanet_layers']
                try:
                    self.deltanet_layers.load_state_dict(pretrained_layers, strict=False)
                except:
                    for i in range(6):
                        layer_state = {k[2:]: v for k, v in pretrained_layers.items() if k.startswith(f'{i}.')}
                        if not layer_state:
                            layer_state = {k: v for k, v in pretrained_layers.items() if not any(k.startswith(f'{j}.') for j in range(6))}
                        if layer_state:
                            self.deltanet_layers[i].load_state_dict(layer_state, strict=False)
            else:
                # Proper distillation checkpoints
                if 'lam_layers' in checkpoint:
                    lam_layers = checkpoint['lam_layers']
                    try:
                        self.deltanet_layers.load_state_dict(lam_layers, strict=False)
                    except:
                        for i in range(6):
                            layer_state = {k[2:]: v for k, v in lam_layers.items() if k.startswith(f'{i}.')}
                            if layer_state:
                                self.deltanet_layers[i].load_state_dict(layer_state, strict=False)
                if 'lam_norms' in checkpoint:
                    try:
                        self.deltanet_norms.load_state_dict(checkpoint['lam_norms'], strict=False)
                    except:
                        pass
                if 'deltanet_ffns' in checkpoint:
                    try:
                        self.deltanet_ffns.load_state_dict(checkpoint['deltanet_ffns'], strict=False)
                    except:
                        pass
                if 'ffn_norms' in checkpoint:
                    try:
                        self.ffn_norms.load_state_dict(checkpoint['ffn_norms'], strict=False)
                    except:
                        pass
                if 'output_dense_layers' in checkpoint:
                    try:
                        self.output_denses.load_state_dict(checkpoint['output_dense_layers'], strict=False)
                    except:
                        pass
            
            self.eval()
        
        def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
            x = self.embeddings(input_ids=input_ids)
            for i in range(6):
                residual = x
                x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)
                x = self.deltanet_norms[i](residual + x_attn)
                residual = x
                x_ffn = self.deltanet_ffns[i](x)
                x_ffn = F.gelu(x_ffn)
                x_ffn = self.output_denses[i](x_ffn)
                x = self.ffn_norms[i](residual + x_ffn)
            return {'last_hidden_state': x, 'pooler_output': self.pooler(x) if self.pooler else None}
        
        def get_sentence_embeddings(self, input_ids, attention_mask=None):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs['last_hidden_state']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
            return F.normalize(embeddings, p=2, dim=1)
    
    model = DeltaNet(teacher_model).to(device)
    model.eval()
    
    # Load dataset
    if split == 'validation':
        try:
            dataset = load_dataset("sentence-transformers/stsb", split="validation")
        except:
            dataset = load_dataset("glue", "stsb", split="validation")
    else:  # test
        try:
            dataset = load_dataset("sentence-transformers/stsb", split="test")
        except:
            dataset = load_dataset("glue", "stsb", split="test")
    
    s1 = dataset["sentence1"]
    s2 = dataset["sentence2"]
    if 'label' in dataset.column_names:
        labels = np.array(dataset["label"], dtype=float)
    else:
        labels = np.array(dataset["score"], dtype=float)
    
    print(f"Computing embeddings for {len(s1)} pairs...")
    
    # Compute embeddings using the same method as stsb_evaluation.py
    all_emb1 = []
    all_emb2 = []
    
    with torch.no_grad():
        for i in range(0, len(s1), 32):
            batch_s1 = s1[i:min(i+32, len(s1))]
            batch_s2 = s2[i:min(i+32, len(s2))]
            
            t1 = tokenizer(batch_s1, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            t2 = tokenizer(batch_s2, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            emb1 = model.get_sentence_embeddings(t1['input_ids'], t1['attention_mask'])
            emb2 = model.get_sentence_embeddings(t2['input_ids'], t2['attention_mask'])
            
            all_emb1.append(emb1.cpu())
            all_emb2.append(emb2.cpu())
    
    # Stack embeddings
    all_emb1 = torch.cat(all_emb1, dim=0)
    all_emb2 = torch.cat(all_emb2, dim=0)
    
    # Compute all similarity metrics
    all_sims = compute_all_similarities(all_emb1, all_emb2)
    
    # Calculate Pearson for each metric
    results = {}
    for metric in ['cosine', 'dot_product', 'euclidean']:
        try:
            pearson_val, _ = pearsonr(labels, all_sims[metric])
            if np.isnan(pearson_val):
                results[metric] = None
            else:
                results[metric] = pearson_val
        except Exception as e:
            results[metric] = None
    
    elapsed = time.time() - start_time
    
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"📊 Results:")
    for metric in ['cosine', 'dot_product', 'euclidean']:
        val = results.get(metric)
        if val is not None:
            print(f"    {metric:15}: {val:.4f}")
        else:
            print(f"    {metric:15}: N/A")
    
    return results


def list_checkpoints(directory, min_step=0):
    directory = Path(directory)
    if not directory.exists():
        return []
    files = [p for p in directory.iterdir() if p.suffix == '.pt']
    def step_key(p):
        name = p.stem
        if name.startswith('step'):
            try:
                step_part = name.split('_')[0]
                step_num = int(step_part[4:])
                return step_num
            except:
                return 999999
        if name == 'checkpoint_best':
            return 999998
        parts = name.split('_')
        try:
            return int(parts[-1])
        except:
            return 999999
    filtered = [f for f in files if step_key(f) >= min_step or step_key(f) >= 999998]
    return sorted(filtered, key=step_key)


def main(directory=None):
    if directory is None:
        ck_dir = "/workspace/LAM/pure_constant_lr"
    else:
        ck_dir = directory
    
    checkpoints = list_checkpoints(ck_dir, min_step=0)
    if not checkpoints:
        print(f"No .pt checkpoints found in {ck_dir}")
        return
    
    print("="*130)
    print("CHECKPOINT EVALUATION - ALL SIMILARITY METRICS")
    print("="*130)
    print(f"Testing checkpoints in {ck_dir}")
    print(f"Metrics: Cosine | Dot Product | Euclidean")
    print("="*130)
    
    results = {}
    
    for ck in checkpoints[:5]:  # Test first 5
        # Validation
        val_results = get_checkpoint_embeddings_and_metrics(str(ck), split='validation')
        
        # Test
        test_results = get_checkpoint_embeddings_and_metrics(str(ck), split='test')
        
        results[ck.name] = {
            'val': val_results,
            'test': test_results
        }
    
    # Summary tables
    print("\n" + "="*130)
    print("SUMMARY TABLE - VALIDATION RESULTS")
    print("="*130)
    print(f"{'Checkpoint':<35} {'Cosine':<14} {'Dot Product':<14} {'Euclidean':<14} {'Best':<12}")
    print("-" * 130)
    
    for name in results.keys():
        val_results = results[name]['val']
        if val_results is None:
            continue
        
        cosine_val = val_results.get('cosine')
        dot_val = val_results.get('dot_product')
        eucl_val = val_results.get('euclidean')
        
        cosine_str = f"{cosine_val:.4f}" if cosine_val is not None else "N/A"
        dot_str = f"{dot_val:.4f}" if dot_val is not None else "N/A"
        eucl_str = f"{eucl_val:.4f}" if eucl_val is not None else "N/A"
        
        valid_metrics = {
            'cosine': cosine_val,
            'dot_product': dot_val,
            'euclidean': eucl_val
        }
        valid_metrics = {k: v for k, v in valid_metrics.items() if v is not None}
        best_str = max(valid_metrics, key=valid_metrics.get) if valid_metrics else "N/A"
        
        print(f"{name:<35} {cosine_str:<14} {dot_str:<14} {eucl_str:<14} {best_str:<12}")
    
    print("\n" + "="*130)
    print("SUMMARY TABLE - TEST RESULTS")
    print("="*130)
    print(f"{'Checkpoint':<35} {'Cosine':<14} {'Dot Product':<14} {'Euclidean':<14} {'Best':<12}")
    print("-" * 130)
    
    for name in results.keys():
        test_results = results[name]['test']
        if test_results is None:
            continue
        
        cosine_val = test_results.get('cosine')
        dot_val = test_results.get('dot_product')
        eucl_val = test_results.get('euclidean')
        
        cosine_str = f"{cosine_val:.4f}" if cosine_val is not None else "N/A"
        dot_str = f"{dot_val:.4f}" if dot_val is not None else "N/A"
        eucl_str = f"{eucl_val:.4f}" if eucl_val is not None else "N/A"
        
        valid_metrics = {
            'cosine': cosine_val,
            'dot_product': dot_val,
            'euclidean': eucl_val
        }
        valid_metrics = {k: v for k, v in valid_metrics.items() if v is not None}
        best_str = max(valid_metrics, key=valid_metrics.get) if valid_metrics else "N/A"
        
        print(f"{name:<35} {cosine_str:<14} {dot_str:<14} {eucl_str:<14} {best_str:<12}")
    
    print("="*130)
    print("\n📈 KEY INSIGHTS:")
    print("  • All three metrics should give SIMILAR Pearson scores for normalized embeddings")
    print("  • Cosine and Dot Product are mathematically equivalent for unit-norm vectors")
    print("  • Euclidean uses a different scale (distance vs similarity)")
    print("="*130)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(directory=sys.argv[1])
    else:
        main()
