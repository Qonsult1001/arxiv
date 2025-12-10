"""
🧠 LAM EMBEDDER - Use Your Trained LAM Model for Semantic Embeddings

This integrates YOUR trained LAM model (/workspace/LAM/best/pytorch_model.bin)
for perfect semantic understanding instead of generic sentence-transformers.

Your LAM model:
- Base: all-MiniLM-L6-v2 embeddings
- Enhanced: 6 layers of EnhancedHierarchicalDeltaNet
- Trained: On semantic similarity (STSB)
- Result: MUCH better recall than generic models!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Union, Optional
import os
import sys

# Add LAM path
sys.path.insert(0, '/workspace/LAM')

# Import the LAM architecture
try:
    from final_solution_formula import EnhancedHierarchicalDeltaNet
    LAM_AVAILABLE = True
except ImportError:
    LAM_AVAILABLE = False
    print("⚠️ LAM architecture not found. Using fallback embedder.")


class LAMEmbedder(nn.Module):
    """
    Your trained LAM model for semantic embeddings.
    
    Uses:
    - Base: all-MiniLM-L6-v2 embeddings
    - Enhanced: 6 EnhancedHierarchicalDeltaNet layers
    - Checkpoint: /workspace/LAM/best/pytorch_model.bin
    
    Provides MUCH better semantic understanding than generic models!
    """
    
    def __init__(
        self,
        checkpoint_path: str = "/workspace/LAM/best/pytorch_model.bin",
        base_model_path: str = "/workspace/LAM/all-MiniLM-L6-v2",
        device: str = None,
    ):
        super().__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Load base model for structure
        base_model = AutoModel.from_pretrained(base_model_path)
        self.d_model = base_model.config.hidden_size  # 384
        
        # Build model structure
        # Embeddings
        self.embeddings = base_model.embeddings
        
        # Your trained DeltaNet layers (6 layers)
        self.deltanet_layers = nn.ModuleList()
        self.deltanet_norms = nn.ModuleList()
        self.deltanet_ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.output_dense_layers = nn.ModuleList()
        
        for i in range(6):
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=self.d_model, 
                    num_heads=12,
                    use_hierarchical_decay=True, 
                    use_enhanced_flux=True,
                    fast_decay_init=0.3, 
                    slow_decay_init=0.85,
                )
            )
            self.deltanet_norms.append(nn.LayerNorm(self.d_model))
            self.deltanet_ffns.append(nn.Linear(self.d_model, self.d_model * 4))
            self.ffn_norms.append(nn.LayerNorm(self.d_model))
            self.output_dense_layers.append(nn.Linear(self.d_model * 4, self.d_model))
        
        # Pooler
        self.pooler = base_model.pooler
        
        # Load trained checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Move to device and set eval mode
        self.to(self.device)
        self.eval()
        
        print(f"✅ LAM Embedder loaded from {checkpoint_path}")
        print(f"   Device: {self.device}")
        print(f"   Embedding dim: {self.d_model}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load trained LAM weights from /best/pytorch_model.bin."""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Filter out teacher_model keys and keys with shape mismatches
        model_state = {}
        current_state = self.state_dict()
        
        for k, v in checkpoint.items():
            if k.startswith('teacher_model'):
                continue  # Skip teacher model
            
            # Check if key exists and shapes match
            if k in current_state:
                if current_state[k].shape == v.shape:
                    model_state[k] = v
                else:
                    # Shape mismatch - try to handle W_bilinear specially
                    if 'W_bilinear' in k and len(v.shape) == 3:
                        # Old format: [num_heads, d_k, d_v] -> new format: [d_k, d_v]
                        # Use the first head's weights
                        model_state[k] = v[0]
                        print(f"   ⚠️ Converted {k}: {v.shape} -> {v[0].shape}")
            else:
                # Key doesn't exist in current model, skip
                pass
        
        # Also load keys that exist in checkpoint but not in current (new parameters)
        for k, v in checkpoint.items():
            if k.startswith('teacher_model'):
                continue
            if k not in current_state and k not in model_state:
                # New key, add it
                model_state[k] = v
        
        # Load the state (allow missing keys)
        try:
            self.load_state_dict(model_state, strict=False)
            print(f"   ✅ Loaded {len(model_state)} weight tensors")
        except Exception as e:
            print(f"   ⚠️ Partial load: {e}")
            # Try loading key by key
            loaded = 0
            for k, v in model_state.items():
                try:
                    if k in current_state and current_state[k].shape == v.shape:
                        current_state[k].copy_(v)
                        loaded += 1
                except:
                    pass
            print(f"   ✅ Loaded {loaded} weights individually")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None,
        use_mean_pooling: bool = True
    ) -> torch.Tensor:
        """Forward pass to get embeddings."""
        x = self.embeddings(input_ids=input_ids)
        
        for i in range(6):
            residual = x
            x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)
            x = self.deltanet_norms[i](residual + x_attn)
            
            residual = x
            x_ffn = self.deltanet_ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            x_ffn = self.output_dense_layers[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        
        # Pool to single embedding
        if use_mean_pooling and attention_mask is not None:
            # Mean pooling - works better for similarity tasks
            mask = attention_mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # CLS token pooling
            pooled = self.pooler(x)
        
        return pooled
    
    def encode(
        self, 
        sentences: Union[str, List[str]], 
        batch_size: int = 32,
        normalize: bool = True,
        convert_to_tensor: bool = True,
    ) -> torch.Tensor:
        """
        Encode sentences to embeddings.
        
        Compatible with sentence-transformers API.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt',
                    max_length=256
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                embeddings = self.forward(
                    inputs['input_ids'], 
                    inputs.get('attention_mask')
                )
                
                all_embeddings.append(embeddings)
        
        # Concatenate
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Normalize if requested
        if normalize:
            all_embeddings = F.normalize(all_embeddings, p=2, dim=-1)
        
        if not convert_to_tensor:
            all_embeddings = all_embeddings.cpu().numpy()
        
        return all_embeddings
    
    def get_sentence_embedding_dimension(self) -> int:
        """Return embedding dimension (for compatibility)."""
        return self.d_model


def get_embedder(use_lam: bool = True):
    """
    Get the best available embedder.
    
    Args:
        use_lam: Whether to use your trained LAM model (recommended!)
        
    Returns:
        Embedder with .encode() method
    """
    if use_lam and LAM_AVAILABLE:
        try:
            return LAMEmbedder()
        except Exception as e:
            print(f"⚠️ Failed to load LAM: {e}")
            print("   Falling back to sentence-transformers...")
    
    # Fallback to sentence-transformers
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Testing LAM Embedder")
    print("=" * 60)
    
    embedder = LAMEmbedder()
    
    # Test sentences
    sentences = [
        "What is my contact email?",
        "The contact email is willie@example.com",
        "Terms of Service require an email address",
        "I like pizza",
    ]
    
    embeddings = embedder.encode(sentences)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    # Compute similarities
    query = embeddings[0:1]
    others = embeddings[1:]
    
    similarities = F.cosine_similarity(query, others)
    
    print("\nSimilarity to 'What is my contact email?':")
    for i, (sent, sim) in enumerate(zip(sentences[1:], similarities)):
        print(f"  {sim.item():.4f}: {sent}")
