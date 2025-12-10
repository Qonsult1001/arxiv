#!/usr/bin/env python3
"""
Fast LAM Model Loader - Optimized for speed
Loads model once and keeps in memory for fast inference
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import os

# Disable torch.compile - causes slowdowns during compilation
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

sys.path.insert(0, str(Path(__file__).parent))
from train_6layer_deltanet_2 import DeltaNetPure6Layer

# Global model cache - load once, use many times
# Clear cache on import to pick up code changes
_MODEL_CACHE = {}

def load_lam_model_fast(model_path=None, device='cuda'):
    """
    Load LAM model with caching - MUCH faster on subsequent calls
    """
    cache_key = (model_path, device)
    
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    
    # Find model file - use the BEST checkpoint (same as test_8k_inference.py)
    if model_path is None:
        model_paths = [
            # BEST checkpoint with 0.7711 Spearman (same as test_8k_inference.py)
            Path("/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEWNEWFINAL/checkpoint_167000.pt"),
            Path("/workspace/LAM/LAM-base-v1/pytorch_model.bin"),
            Path("/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEWNEWFINAL/pytorch_model.bin"),
        ]
        for path in model_paths:
            if path.exists():
                model_path = path
                break
    
    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading LAM model from: {model_path}")
    
    config = {
        "teacher_model": "/workspace/LAM/all-MiniLM-L6-v2",
        "num_layers": 6,
        "num_heads": 12,
        "fast_decay_init": 0.30,
        "slow_decay_init": 0.85,
        "use_kernel_blending": False,
    }
    
    # Create model
    model = DeltaNetPure6Layer(
        teacher_model_name=config['teacher_model'],
        num_linear_layers=config['num_layers'],
        config=config
    ).to(device)
    
    # Load weights - handle different checkpoint formats
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # If it's a training checkpoint (has 'deltanet_layers' key), load like test_8k_inference.py
    if 'deltanet_layers' in ckpt:
        model.deltanet_layers.load_state_dict(ckpt['deltanet_layers'], strict=False)
        print("   ✅ Loaded deltanet_layers from training checkpoint")
        if 'test_spearman' in ckpt:
            print(f"   📊 Checkpoint score: {ckpt['test_spearman']:.4f} Spearman")
    else:
        # It's a full model state dict
        model.load_state_dict(ckpt, strict=False)
        print("   ✅ Loaded full model state dict")
    
    model.eval()
    
    # Cache the model
    _MODEL_CACHE[cache_key] = model
    
    print("✅ Model loaded and cached")
    return model

class FastLAMEncoder:
    """
    Fast LAM encoder - similar API to sentence-transformers
    Loads model once, uses cached version for speed
    """
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.model = load_lam_model_fast(model_path, device)
        self.tokenizer = self.model.tokenizer
        
    def encode(self, sentences, batch_size=32, convert_to_numpy=True, **kwargs):
        """
        Encode sentences - optimized with batching like sentence-transformers
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        # Process in batches (like sentence-transformers)
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Tokenize
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=kwargs.get('max_length', 512),
                return_tensors='pt'
            ).to(self.device)
            
            # Encode
            with torch.no_grad():
                embeddings = self.model.encode(
                    tokens['input_ids'],
                    tokens['attention_mask']
                )
            
            # Keep on GPU for speed (only move to CPU if convert_to_numpy=True)
            if convert_to_numpy:
                all_embeddings.append(embeddings.cpu())
            else:
                all_embeddings.append(embeddings)
        
        # Concatenate
        if convert_to_numpy:
            result = torch.cat(all_embeddings, dim=0)
            return result.numpy()
        else:
            result = torch.cat(all_embeddings, dim=0)
            return result

if __name__ == "__main__":
    # Test loading speed
    import time
    
    print("Testing fast loader...")
    
    # First load (will be slower)
    start = time.time()
    encoder = FastLAMEncoder()
    first_load = time.time() - start
    print(f"First load: {first_load:.2f}s")
    
    # Second load (cached, should be instant)
    start = time.time()
    encoder2 = FastLAMEncoder()
    second_load = time.time() - start
    print(f"Second load (cached): {second_load:.4f}s")
    print(f"Speedup: {first_load/second_load:.1f}x faster!")
    
    # Test encoding
    texts = ["Hello world", "How are you?"]
    embeddings = encoder.encode(texts)
    print(f"Encoded {len(texts)} texts, shape: {embeddings.shape}")

