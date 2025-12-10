"""
MTEB-compatible wrapper for DeltaNet model
This wrapper allows your DeltaNet model to be evaluated using MTEB locally
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import numpy as np
from typing import Union, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Disable torch.compile for stability in evaluation
import torch._dynamo
torch._dynamo.config.disable = True

from final_solution_formula import EnhancedHierarchicalDeltaNet


class DeltaNetMTEBWrapper(nn.Module):
    """
    MTEB-compatible wrapper for DeltaNet model
    Loads checkpoint and provides encode() method for MTEB evaluation
    """
    
    def __init__(self, checkpoint_path: str, base_model_path: str = "/workspace/LAM/all-MiniLM-L6-v2", device: Optional[str] = None):
        super().__init__()
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading DeltaNet model from checkpoint: {checkpoint_path}")
        print(f"Base model: {base_model_path}")
        print(f"Device: {self.device}")
        
        # Load base model for embeddings and tokenizer
        base_path = Path(base_model_path)
        if base_path.exists() and base_path.is_dir():
            abs_path = str(base_path.resolve())
            self.teacher_model = AutoModel.from_pretrained(abs_path)
            self.tokenizer = AutoTokenizer.from_pretrained(abs_path)
        else:
            self.teacher_model = AutoModel.from_pretrained(base_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        self.d_model = self.teacher_model.config.hidden_size
        
        # Freeze teacher embeddings
        self.embeddings = self.teacher_model.embeddings
        for param in self.embeddings.parameters():
            param.requires_grad = False
        
        # Initialize DeltaNet layers (same structure as training)
        self.deltanet_layers = nn.ModuleList()
        self.deltanet_norms = nn.ModuleList()
        self.deltanet_ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        
        for i in range(6):
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=self.d_model,
                    num_heads=12,
                    use_hierarchical_decay=True,
                    use_enhanced_flux=True,
                    fast_decay_init=0.30,
                    slow_decay_init=0.85,
                )
            )
            self.deltanet_norms.append(
                self.teacher_model.encoder.layer[i].attention.output.LayerNorm
            )
            self.deltanet_ffns.append(
                self.teacher_model.encoder.layer[i].intermediate
            )
            self.ffn_norms.append(
                self.teacher_model.encoder.layer[i].output.LayerNorm
            )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Check if checkpoint is a dict with 'deltanet_layers' key (standard format)
        if isinstance(checkpoint, dict) and 'deltanet_layers' in checkpoint:
            self.deltanet_layers.load_state_dict(checkpoint['deltanet_layers'], strict=False)
            print("✅ Loaded DeltaNet layers from checkpoint")
            
            # Also load norms and FFNs if available (some checkpoints include them)
            if 'deltanet_norms' in checkpoint:
                self.deltanet_norms.load_state_dict(checkpoint['deltanet_norms'], strict=False)
                print("✅ Loaded DeltaNet norms from checkpoint")
            if 'deltanet_ffns' in checkpoint:
                self.deltanet_ffns.load_state_dict(checkpoint['deltanet_ffns'], strict=False)
                print("✅ Loaded DeltaNet FFNs from checkpoint")
            if 'ffn_norms' in checkpoint:
                self.ffn_norms.load_state_dict(checkpoint['ffn_norms'], strict=False)
                print("✅ Loaded FFN norms from checkpoint")
                
            # Print validation score if available
            if 'val_pearson' in checkpoint:
                print(f"   📊 Validation Pearson: {checkpoint['val_pearson']:.4f}")
            if 'step' in checkpoint:
                print(f"   📍 Training step: {checkpoint['step']}")
                
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Try alternative checkpoint format
            model_state = checkpoint['model_state_dict']
            # Filter for deltanet_layers
            deltanet_state = {k.replace('deltanet_layers.', ''): v 
                            for k, v in model_state.items() 
                            if 'deltanet_layers' in k}
            if deltanet_state:
                self.deltanet_layers.load_state_dict(deltanet_state, strict=False)
                print("✅ Loaded DeltaNet layers from alternative checkpoint format")
        elif isinstance(checkpoint, dict):
            # Check if it's a direct state dict with deltanet_layers keys
            checkpoint_keys = list(checkpoint.keys())
            
            # Check if any keys contain 'deltanet_layers'
            has_deltanet_keys = any('deltanet_layers' in k for k in checkpoint_keys)
            
            if has_deltanet_keys:
                # Extract deltanet_layers state dict
                deltanet_state = {}
                deltanet_norms_state = {}
                deltanet_ffns_state = {}
                ffn_norms_state = {}
                
                for k, v in checkpoint.items():
                    if k.startswith('deltanet_layers.'):
                        new_key = k.replace('deltanet_layers.', '')
                        deltanet_state[new_key] = v
                    elif k.startswith('deltanet_norms.'):
                        new_key = k.replace('deltanet_norms.', '')
                        deltanet_norms_state[new_key] = v
                    elif k.startswith('deltanet_ffns.'):
                        new_key = k.replace('deltanet_ffns.', '')
                        deltanet_ffns_state[new_key] = v
                    elif k.startswith('ffn_norms.'):
                        new_key = k.replace('ffn_norms.', '')
                        ffn_norms_state[new_key] = v
                
                if deltanet_state:
                    self.deltanet_layers.load_state_dict(deltanet_state, strict=False)
                    print("✅ Loaded DeltaNet layers from state dict format")
                if deltanet_norms_state:
                    self.deltanet_norms.load_state_dict(deltanet_norms_state, strict=False)
                    print("✅ Loaded DeltaNet norms from state dict format")
                if deltanet_ffns_state:
                    self.deltanet_ffns.load_state_dict(deltanet_ffns_state, strict=False)
                    print("✅ Loaded DeltaNet FFNs from state dict format")
                if ffn_norms_state:
                    self.ffn_norms.load_state_dict(ffn_norms_state, strict=False)
                    print("✅ Loaded FFN norms from state dict format")
            else:
                # Check if it's a full model state dict (from train_6layer_deltanet_2.py format)
                # Try to load using DeltaNetPure6Layer's state_dict structure
                print("⚠️  Warning: Could not find deltanet_layers in checkpoint")
                print(f"   Available keys (first 10): {checkpoint_keys[:10]}")
                print("   ⚠️  This checkpoint may not contain DeltaNet weights.")
                print("   ⚠️  Model will use randomly initialized DeltaNet layers.")
        else:
            print("⚠️  Warning: Unexpected checkpoint format")
            print(f"   Checkpoint type: {type(checkpoint)}")
        
        self.max_seq_length = 128  # Match training max_length (model was trained on 128 tokens)
        self.to(self.device)
        self.eval()
        print("✅ Model loaded and ready for evaluation")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward(self, input_ids, attention_mask):
        """Forward pass through DeltaNet model"""
        x = self.embeddings(input_ids=input_ids)
        
        # Pass through 6 DeltaNet layers
        for i in range(6):
            # Attention block
            residual = x
            x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)
            x = self.deltanet_norms[i](residual + x_attn)
            
            # FFN block
            residual = x
            x_ffn = self.deltanet_ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            # Use output dense from teacher model (or from checkpoint if available)
            orig_layer = self.teacher_model.encoder.layer[i]
            x_ffn = orig_layer.output.dense(x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        
        # Final embedding
        embeddings = self.mean_pooling(x, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,  # Already normalized in forward
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode sentences to embeddings (MTEB-compatible interface)
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to return numpy array or torch tensor
            normalize_embeddings: Whether to normalize (already normalized)
        
        Returns:
            Embeddings as numpy array or torch tensor
        """
        # Handle DataLoader (from MTEB)
        from torch.utils.data import DataLoader
        if isinstance(sentences, DataLoader):
            all_embeddings = []
            iterator = sentences
            if show_progress_bar:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Encoding")
            
            with torch.no_grad():
                for batch in iterator:
                    # Extract sentences from batch (could be dict or list)
                    if isinstance(batch, dict):
                        # Try common keys
                        batch_sentences = batch.get('text', batch.get('sentence', batch.get('sentences', None)))
                        if batch_sentences is None:
                            # Try to get first value that looks like text
                            for v in batch.values():
                                if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
                                    batch_sentences = v
                                    break
                    elif isinstance(batch, (list, tuple)):
                        batch_sentences = batch
                    else:
                        raise ValueError(f"Unexpected batch format: {type(batch)}")
                    
                    if batch_sentences is None:
                        continue
                    
                    # Tokenize
                    encoded = self.tokenizer(
                        batch_sentences,
                        padding='max_length',
                        max_length=self.max_seq_length,
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoded['input_ids'].to(self.device)
                    attention_mask = encoded['attention_mask'].to(self.device)
                    
                    # Forward pass
                    embeddings = self.forward(input_ids, attention_mask)
                    all_embeddings.append(embeddings.cpu())
            
            # Concatenate all embeddings
            if all_embeddings:
                all_embeddings = torch.cat(all_embeddings, dim=0)
            else:
                all_embeddings = torch.empty((0, self.d_model))
        
        else:
            # Handle string or list of strings
            if isinstance(sentences, str):
                sentences = [sentences]
            
            all_embeddings = []
            
            # Process in batches
            from tqdm import tqdm
            iterator = range(0, len(sentences), batch_size)
            if show_progress_bar:
                iterator = tqdm(iterator, desc="Encoding")
            
            with torch.no_grad():
                for start_idx in iterator:
                    end_idx = min(start_idx + batch_size, len(sentences))
                    batch_sentences = sentences[start_idx:end_idx]
                    
                    # Tokenize
                    encoded = self.tokenizer(
                        batch_sentences,
                        padding='max_length',
                        max_length=self.max_seq_length,
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoded['input_ids'].to(self.device)
                    attention_mask = encoded['attention_mask'].to(self.device)
                    
                    # Forward pass
                    embeddings = self.forward(input_ids, attention_mask)
                    
                    all_embeddings.append(embeddings.cpu())
            
            # Concatenate all embeddings
            all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return all_embeddings.numpy()
        else:
            return all_embeddings

