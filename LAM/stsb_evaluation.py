"""
STS-B Evaluation Script - Compare Checkpoints 11k and 12k
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import sys
from pathlib import Path
import time

# Import DeltaNet components
sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

def batched(iterable, n=32):
    """Batch an iterable into chunks of size n"""
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def compute_pairwise_sims(emb1, emb2):
    """Compute cosine similarity for corresponding rows"""
    emb1_np = emb1.detach().cpu().numpy()
    emb2_np = emb2.detach().cpu().numpy()
    sims = 1 - np.array([cosine(a, b) for a, b in zip(emb1_np, emb2_np)])
    return sims

def get_sentence_embeddings(model, tokenizer, sentences, device='cpu'):
    """Get sentence embeddings - OPTIMIZED with batching and encode() method"""
    # ⚡ OPTIMIZED: Use encode() method if available (much faster with batching!)
    if hasattr(model, 'encode'):
        # Use the optimized encode() method with batching
        embeddings = model.encode(
            sentences, 
            batch_size=32,  # ✅ Increased from 16 → 32 for better speedup
            convert_to_numpy=False,  # Keep as tensor
            show_progress_bar=False
        )
        return embeddings.to(device) if isinstance(embeddings, torch.Tensor) else embeddings
    else:
        # Fallback to old method (for compatibility)
        inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        embeddings = model.get_sentence_embeddings(
            inputs['input_ids'].to(device),
            inputs['attention_mask'].to(device)
        )
        return embeddings

def evaluate_checkpoint(checkpoint_path, device='cpu', split='validation'):
    """Evaluate a single checkpoint on specified split (validation or test)"""
    print(f"\n{'='*80}")
    print(f"Testing: {checkpoint_path}")
    print(f"Split: {split}")
    print(f"{'='*80}")
    
    # Load checkpoint
    loaded_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check if this is a raw state_dict (from save_pretrained) or a checkpoint dictionary
    # Raw state_dict will have keys like 'deltanet_layers.0.q_proj.weight' or 'teacher_model.embeddings.weight'
    # Checkpoint dict will have keys like 'model_state_dict', 'config', 'step', 'deltanet_layers', etc.
    is_raw_state_dict = False
    if isinstance(loaded_data, dict):
        has_checkpoint_keys = any(k in loaded_data for k in ['model_state_dict', 'deltanet_layers', 'config', 'step', 'lam_layers'])
        has_model_keys = any('deltanet_layers.' in str(k) or 'teacher_model.' in str(k) for k in loaded_data.keys())
        is_raw_state_dict = not has_checkpoint_keys and has_model_keys
    
    if is_raw_state_dict:
        # This is a raw state_dict from save_pretrained() - treat it as model_state_dict
        checkpoint = {'model_state_dict': loaded_data, 'config': {}, 'step': 0}
        print("   ℹ️  Detected raw state_dict format (from save_pretrained)")
    else:
        # This is a checkpoint dictionary
        checkpoint = loaded_data
    
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
    
    # Build DeltaNet model
    class DeltaNet(torch.nn.Module):
        def __init__(self, teacher_model):
            super().__init__()
            self.teacher_model = teacher_model  # Store reference
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
            
            # ⚡ OPTIMIZED: Store output dense layers (avoids repeated access)
            self.output_denses = torch.nn.ModuleList()
            for i in range(6):
                # Create a copy to avoid device mismatch issues
                dense_layer = teacher_model.encoder.layer[i].output.dense
                self.output_denses.append(dense_layer)
            
            self.pooler = teacher_model.pooler
            self.tokenizer_ref = None  # Will be set externally
            
            # Load weights from different checkpoint formats
            model_state_dict = checkpoint.get('model_state_dict', {})
            
            # First, try to load from model_state_dict (new format from train_6layer_optimized.py)
            if model_state_dict:
                # Extract components from model_state_dict
                deltanet_layers_dict = {}
                deltanet_norms_dict = {}
                deltanet_ffns_dict = {}
                ffn_norms_dict = {}
                embeddings_dict = {}
                
                for key, value in model_state_dict.items():
                    # Extract deltanet_layers (e.g., "deltanet_layers.0.q_proj.weight" -> "0.q_proj.weight")
                    if key.startswith('deltanet_layers.'):
                        new_key = key.replace('deltanet_layers.', '')
                        deltanet_layers_dict[new_key] = value
                    # Extract deltanet_norms (e.g., "deltanet_norms.0.weight" -> "0.weight")
                    elif key.startswith('deltanet_norms.'):
                        new_key = key.replace('deltanet_norms.', '')
                        deltanet_norms_dict[new_key] = value
                    # Extract deltanet_ffns (e.g., "deltanet_ffns.0.dense.weight" -> "0.dense.weight")
                    elif key.startswith('deltanet_ffns.'):
                        new_key = key.replace('deltanet_ffns.', '')
                        deltanet_ffns_dict[new_key] = value
                    # Extract ffn_norms (e.g., "ffn_norms.0.weight" -> "0.weight")
                    elif key.startswith('ffn_norms.'):
                        new_key = key.replace('ffn_norms.', '')
                        ffn_norms_dict[new_key] = value
                    # Extract student embeddings (e.g., "embeddings.word_embeddings.weight" -> "word_embeddings.weight")
                    elif key.startswith('embeddings.') and not key.startswith('teacher_model'):
                        new_key = key.replace('embeddings.', '')
                        embeddings_dict[new_key] = value
                
                # Load student embeddings (if trained separately from teacher)
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
            
            # ⭐ NEW: Handle student_encoder format (from train_6layer_deltanet_3.py)
            elif 'student_encoder' in checkpoint:
                print("   ℹ️  Detected student_encoder format (new LatentSemanticEncoder)")
                student_encoder_state = checkpoint['student_encoder']
                
                # Extract deltanet_layers from student_encoder
                # student_encoder.deltanet_layers.0.q_proj.weight -> 0.q_proj.weight
                deltanet_layers_dict = {}
                for key, value in student_encoder_state.items():
                    if key.startswith('deltanet_layers.'):
                        # Remove 'deltanet_layers.' prefix
                        new_key = key[len('deltanet_layers.'):]
                        deltanet_layers_dict[new_key] = value
                
                if deltanet_layers_dict:
                    # ⚡ COMPATIBILITY: Convert old resonance_flux parameters to new format
                    compatible_dict = {}
                    resonance_flux_converted = {}
                    
                    for k, v in deltanet_layers_dict.items():
                        # Convert old W_bilinear: [32, 32] -> [12, 32, 32]
                        if 'resonance_flux.W_bilinear' in k:
                            # Extract layer index from key like "0.resonance_flux.W_bilinear"
                            parts = k.split('.')
                            if len(parts) >= 2 and parts[0].isdigit():
                                layer_idx = int(parts[0])
                                old_shape = v.shape
                                if len(old_shape) == 2:  # [d_k, d_k] - old format
                                    num_heads = self.deltanet_layers[layer_idx].num_heads
                                    # Replicate for each head
                                    new_w = v.unsqueeze(0).repeat(num_heads, 1, 1)
                                    resonance_flux_converted[k] = new_w
                                    print(f"   🔄 Converted {k}: {old_shape} -> {new_w.shape}")
                                else:
                                    # Already new format
                                    compatible_dict[k] = v
                            else:
                                compatible_dict[k] = v
                        elif 'resonance_flux.flux_net' in k:
                            # Skip flux_net (structure changed significantly)
                            continue
                        else:
                            compatible_dict[k] = v
                    
                    # Add converted parameters
                    compatible_dict.update(resonance_flux_converted)
                    
                    try:
                        self.deltanet_layers.load_state_dict(compatible_dict, strict=False)
                        print(f"   ✅ Loaded {len(compatible_dict)} deltanet layer parameters")
                        if resonance_flux_converted:
                            print(f"   ✅ Converted {len(resonance_flux_converted)} resonance_flux.W_bilinear parameters")
                        if len(compatible_dict) < len(deltanet_layers_dict):
                            skipped = len(deltanet_layers_dict) - len(compatible_dict)
                            if skipped > 0:
                                print(f"   ⚠️  Skipped {skipped} incompatible parameters (flux_net structure changed)")
                    except Exception as e:
                        print(f"   ⚠️  Warning: Could not load deltanet_layers directly: {e}")
                        # Fallback: load layer by layer (with conversion)
                        for i in range(6):
                            layer_state = {}
                            for k, v in deltanet_layers_dict.items():
                                if k.startswith(f'{i}.'):
                                    new_key = k[2:]  # Remove 'i.' prefix
                                    
                                    # Convert old W_bilinear: [32, 32] -> [12, 32, 32]
                                    if new_key == 'resonance_flux.W_bilinear':
                                        old_shape = v.shape
                                        if len(old_shape) == 2:  # [d_k, d_k] - old format
                                            num_heads = self.deltanet_layers[i].num_heads
                                            new_w = v.unsqueeze(0).repeat(num_heads, 1, 1)
                                            layer_state[new_key] = new_w
                                            print(f"   🔄 Converted layer {i} W_bilinear: {old_shape} -> {new_w.shape}")
                                        else:
                                            # Already new format
                                            layer_state[new_key] = v
                                    elif 'resonance_flux.flux_net' in new_key:
                                        # Skip flux_net (structure changed)
                                        continue
                                    else:
                                        layer_state[new_key] = v
                            
                            if layer_state:
                                try:
                                    self.deltanet_layers[i].load_state_dict(layer_state, strict=False)
                                except Exception as e2:
                                    print(f"   ⚠️  Warning: Could not load deltanet_layers[{i}]: {e2}")
                
                # Try to load embeddings if present
                embeddings_dict = {}
                for key, value in student_encoder_state.items():
                    if key.startswith('embeddings.') and not key.startswith('deltanet_layers'):
                        new_key = key[len('embeddings.'):]
                        embeddings_dict[new_key] = value
                
                if embeddings_dict:
                    try:
                        self.embeddings.load_state_dict(embeddings_dict, strict=False)
                        print(f"   ✅ Loaded student embeddings")
                    except Exception as e:
                        print(f"   ⚠️  Warning: Could not load student embeddings: {e}")
            
            # Fallback to old checkpoint formats
            elif 'deltanet_layers' in checkpoint:
                # Older/simpler checkpoint format
                pretrained_layers = checkpoint['deltanet_layers']
                
                # ⚡ COMPATIBILITY: Convert old resonance_flux parameters to new format
                # Old format: W_bilinear: [32, 32], New format: [12, 32, 32] (per-head)
                compatible_layers = {}
                resonance_flux_converted = {}  # Store converted resonance_flux params per layer
                
                for k, v in pretrained_layers.items():
                    # Convert old W_bilinear: [32, 32] -> [12, 32, 32] (replicate for each head)
                    if 'resonance_flux.W_bilinear' in k:
                        # Extract layer index
                        layer_idx = None
                        for i in range(6):
                            if k.startswith(f'{i}.resonance_flux.W_bilinear'):
                                layer_idx = i
                                break
                        
                        if layer_idx is not None:
                            # Old shape: [32, 32], New shape: [num_heads, 32, 32]
                            old_shape = v.shape
                            if len(old_shape) == 2:  # [d_k, d_k]
                                num_heads = self.deltanet_layers[layer_idx].num_heads
                                # Replicate for each head: [num_heads, d_k, d_k]
                                new_w = v.unsqueeze(0).repeat(num_heads, 1, 1)
                                new_key = f'{layer_idx}.resonance_flux.W_bilinear'
                                resonance_flux_converted[new_key] = new_w
                                print(f"   🔄 Converted W_bilinear for layer {layer_idx}: {old_shape} -> {new_w.shape}")
                            else:
                                # Already in new format or unexpected shape
                                compatible_layers[k] = v
                        continue
                    
                    # Try to adapt flux_net weights if dimensions allow
                    if 'resonance_flux.flux_net' in k:
                        # Extract layer index
                        layer_idx = None
                        for i in range(6):
                            if k.startswith(f'{i}.resonance_flux.flux_net'):
                                layer_idx = i
                                break
                        
                        if layer_idx is not None:
                            # Try to extract and adapt flux_net weights
                            # Old structure might be different, so we'll try to load what we can
                            # For now, skip and let it use new initialization
                            # (The new flux_net is initialized in model, so this is acceptable)
                            continue
                    
                    # Keep all other parameters
                    compatible_layers[k] = v
                
                # Add converted resonance_flux parameters
                compatible_layers.update(resonance_flux_converted)
                
                try:
                    # Try loading with converted parameters
                    self.deltanet_layers.load_state_dict(compatible_layers, strict=False)
                    if resonance_flux_converted:
                        print("   ✅ Loaded compatible parameters (old checkpoint format)")
                        print(f"   ✅ Converted {len(resonance_flux_converted)} resonance_flux.W_bilinear parameters")
                    else:
                        print("   ✅ Loaded compatible parameters (old checkpoint format)")
                except Exception as e:
                    # Fallback: load layer by layer with conversion
                    print(f"   ⚠️  Layer-level loading due to: {e}")
                    for i in range(6):
                        layer_state = {}
                        for k, v in pretrained_layers.items():
                            # Extract layer i's state
                            if k.startswith(f'{i}.'):
                                new_key = k[2:]  # Remove 'i.' prefix
                                
                                # Convert W_bilinear if present
                                if new_key == 'resonance_flux.W_bilinear':
                                    old_shape = v.shape
                                    if len(old_shape) == 2:  # [d_k, d_k]
                                        num_heads = self.deltanet_layers[i].num_heads
                                        new_w = v.unsqueeze(0).repeat(num_heads, 1, 1)
                                        layer_state[new_key] = new_w
                                        print(f"   🔄 Converted layer {i} W_bilinear: {old_shape} -> {new_w.shape}")
                                    else:
                                        layer_state[new_key] = v
                                elif 'resonance_flux.flux_net' in new_key:
                                    # Skip flux_net (different structure)
                                    continue
                                else:
                                    layer_state[new_key] = v
                        
                        if layer_state:
                            try:
                                self.deltanet_layers[i].load_state_dict(layer_state, strict=False)
                            except Exception as layer_e:
                                print(f"   ⚠️  Layer {i} partial load: {layer_e}")
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
                # ⚡ OPTIMIZED: Use stored output dense (faster)
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
        
        def encode(self, sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True, **kwargs):
            """
            ⚡ OPTIMIZED: Batched inference for 30-50x speedup (SentenceTransformer compatible)
            """
            import numpy as np
            if isinstance(sentences, str):
                sentences = [sentences]
            
            if not sentences:
                return np.array([]) if convert_to_numpy else torch.tensor([])
            
            device = next(self.parameters()).device
            self.eval()
            
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                
                # Tokenize with dynamic padding (optimized!)
                if self.tokenizer_ref:
                    tokens = self.tokenizer_ref(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=kwargs.get('max_length', 128),
                        return_tensors='pt'
                    ).to(device)
                else:
                    # Fallback: use transformers tokenizer
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained("/workspace/LAM/all-MiniLM-L6-v2")
                    tokens = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=kwargs.get('max_length', 128),
                        return_tensors='pt'
                    ).to(device)
                
                # Forward pass
                with torch.no_grad():
                    embeddings_batch = self.get_sentence_embeddings(
                        tokens['input_ids'],
                        tokens['attention_mask']
                    )
                
                if convert_to_numpy:
                    all_embeddings.append(embeddings_batch.cpu().numpy())
                else:
                    all_embeddings.append(embeddings_batch.cpu())
            
            # Concatenate all embeddings
            if convert_to_numpy:
                if all_embeddings:
                    return np.vstack(all_embeddings)
                return np.array([])
            else:
                if all_embeddings:
                    return torch.cat(all_embeddings, dim=0)
                return torch.tensor([])
    
    model = DeltaNet(teacher_model).to(device)
    model.tokenizer_ref = tokenizer  # Store tokenizer for encode() method
    
    # Load STS-B
    if split == 'validation':
        ds = load_dataset("sentence-transformers/stsb", split="validation")
    else:  # test split
        ds = load_dataset("sentence-transformers/stsb", split="test")
    
    s1 = ds["sentence1"]
    s2 = ds["sentence2"]
    # Handle both 'label' (glue) and 'score' (sentence-transformers) column names
    if 'label' in ds.column_names:
        labels = np.array(ds["label"], dtype=float)
    else:
        labels = np.array(ds["score"], dtype=float)
    
    print(f"Testing on {len(s1)} sentence pairs...")
    
    # Get embeddings
    start_time = time.time()
    
    # ⚡ OPTIMIZED: Use encode() method for maximum speedup (handles batching internally)
    if hasattr(model, 'encode'):
        # Use encode() - processes all sentences with optimal batching
        emb_s1 = model.encode(s1, batch_size=32, convert_to_numpy=False)
        emb_s2 = model.encode(s2, batch_size=32, convert_to_numpy=False)
        
        # Ensure tensors are on correct device
        if isinstance(emb_s1, torch.Tensor):
            emb_s1 = emb_s1.to(device)
        else:
            emb_s1 = torch.tensor(emb_s1).to(device)
        if isinstance(emb_s2, torch.Tensor):
            emb_s2 = emb_s2.to(device)
        else:
            emb_s2 = torch.tensor(emb_s2).to(device)
    else:
        # Fallback to old method (batch by batch)
        emb_s1 = []
        emb_s2 = []
        for batch in batched(s1, 32):
            emb = get_sentence_embeddings(model, tokenizer, batch, device)
            emb_s1.append(emb)
        for batch in batched(s2, 32):
            emb = get_sentence_embeddings(model, tokenizer, batch, device)
            emb_s2.append(emb)
        emb_s1 = torch.cat(emb_s1, dim=0)
        emb_s2 = torch.cat(emb_s2, dim=0)
    
    # Compute correlations
    sims = compute_pairwise_sims(emb_s1, emb_s2)
    mapped = (sims + 1.0) * 2.5
    
    # Handle NaN values in correlation calculations
    try:
        pearson_cosine = pearsonr(sims, labels)[0]
        if np.isnan(pearson_cosine):
            pearson_cosine = None
    except (ValueError, RuntimeError):
        pearson_cosine = None
    
    try:
        pearson_mapped = pearsonr(mapped, labels)[0]
        if np.isnan(pearson_mapped):
            pearson_mapped = None
    except (ValueError, RuntimeError):
        pearson_mapped = None
    
    try:
        spearman_cosine = spearmanr(sims, labels)[0]
        if np.isnan(spearman_cosine):
            spearman_cosine = None
    except (ValueError, RuntimeError):
        spearman_cosine = None
    
    try:
        spearman_mapped = spearmanr(mapped, labels)[0]
        if np.isnan(spearman_mapped):
            spearman_mapped = None
    except (ValueError, RuntimeError):
        spearman_mapped = None
    
    elapsed_time = time.time() - start_time
    
    print(f"⏱️  Time: {elapsed_time:.2f}s")
    pearson_cosine_str = f"{pearson_cosine:.4f}" if pearson_cosine is not None else "N/A"
    pearson_mapped_str = f"{pearson_mapped:.4f}" if pearson_mapped is not None else "N/A"
    spearman_cosine_str = f"{spearman_cosine:.4f}" if spearman_cosine is not None else "N/A"
    spearman_mapped_str = f"{spearman_mapped:.4f}" if spearman_mapped is not None else "N/A"
    
    print(f"📊 Pearson (cosine): {pearson_cosine_str}")
    print(f"📊 Pearson (mapped): {pearson_mapped_str}")
    print(f"📊 Spearman (cosine): {spearman_cosine_str}")
    print(f"📊 Spearman (mapped): {spearman_mapped_str}")
    
    return {
        'pearson_cosine': pearson_cosine,
        'spearman_cosine': spearman_cosine,
        'time': elapsed_time
    }

def main():
    print("="*80)
    print("STS-B EVALUATION - CHECKPOINT 15000")
    print("="*80)
    
    # Use CPU for evaluation to avoid conflicts with training process
    device = 'cuda'
    #checkpoint_path = "/workspace/deltanet_minilm_6layers_FIXED_FROM_SCRATCH/checkpoint_38000.pt"
    checkpoint_path = "/workspace/proper_distillation/checkpoint_10250.pt"
    
    result = evaluate_checkpoint(checkpoint_path, device)
    
    results = {38000: result}
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Checkpoint':<15} {'Pearson':<10} {'Spearman':<10} {'Time (s)':<10}")
    print("-" * 60)
    
    for step, result in sorted(results.items()):
        print(f"{step:<15} {result['pearson_cosine']:.4f}      {result['spearman_cosine']:.4f}      {result['time']:.2f}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
