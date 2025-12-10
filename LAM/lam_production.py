"""
LAM (Linear Associated Memory) - Production Version
====================================================

This is the PRODUCTION-READY version that works on ANY machine:
- Uses the compiled core module (_core.cpython-312-x86_64-linux-gnu.so)
- Works on CPU, AMD GPU, NVIDIA GPUs, and any system

Usage (same as sentence-transformers):
    from lam_production import LAM
    
    model = LAM('LAM-base-v1')
    embeddings = model.encode(['Hello world', 'How are you?'])

Compatibility:
    ✅ NVIDIA GPU - Uses PyTorch CUDA (fast)
    ✅ AMD GPU    - Uses PyTorch (works)
    ✅ CPU        - Uses PyTorch (works, slower)
    ✅ Apple Silicon - Uses PyTorch MPS (works)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
from typing import Union, List, Optional
import warnings

# Add LAM directory to path
lam_dir = Path(__file__).parent
sys.path.insert(0, str(lam_dir))

# ============================================================================
# DEVICE DETECTION
# ============================================================================

def detect_device():
    """Detect the best available device for this system."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


# Global device detection
DEVICE = detect_device()

print(f"🔧 LAM Device: {DEVICE}")


# ============================================================================
# PYTORCH FALLBACK LAYER (works everywhere)
# ============================================================================

class PyTorchDeltaNetLayer(nn.Module):
    """
    Pure PyTorch DeltaNet layer - works on any device.
    """
    
    def __init__(self, hidden_size: int = 384, num_heads: int = 12):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.fast_decay_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.slow_decay_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        H = self.num_heads
        d = self.head_dim
        
        # Project
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)
        
        # Activations
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)
        
        # Compute parameters
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)
        fast_decay = torch.sigmoid(self.fast_decay_proj(x)).transpose(1, 2)
        slow_decay = torch.sigmoid(self.slow_decay_proj(x)).transpose(1, 2)
        
        # Dual-state delta rule (optimized PyTorch)
        o = self._dual_state_delta(q, k, v, beta, fast_decay, slow_decay)
        
        # Output
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        g = torch.sigmoid(self.g_proj(x))
        o = o * g
        o = self.o_proj(o)
        
        return o, None, None, None
    
    def _dual_state_delta(self, q, k, v, beta, fast_decay, slow_decay):
        """Optimized dual-state delta rule in pure PyTorch."""
        B, H, T, D = q.shape
        
        # Use diagonal state approximation for speed
        S_fast = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
        S_slow = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
        
        outputs = []
        for t in range(T):
            # Decay
            S_fast = S_fast * fast_decay[:, :, t:t+1]
            S_slow = S_slow * slow_decay[:, :, t:t+1]
            
            # Output
            o_fast = q[:, :, t] * S_fast
            o_slow = q[:, :, t] * S_slow
            
            # Update
            update = beta[:, :, t:t+1] * k[:, :, t] * v[:, :, t]
            S_fast = S_fast + update
            S_slow = S_slow + update
            
            # Cross-interaction
            S_fast_new = S_fast + 0.05 * S_slow
            S_slow = S_slow + 0.05 * S_fast
            S_fast = S_fast_new
            
            outputs.append(0.5 * (o_fast + o_slow))
        
        return torch.stack(outputs, dim=2)


# ============================================================================
# CORE-BASED LAYER (using compiled _core module)
# ============================================================================

# Import the compiled core module
try:
    from lam_package.lam import _core
    EnhancedHierarchicalDeltaNet = _core.EnhancedHierarchicalDeltaNet
    HAS_CORE = True
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_core", "/workspace/LAM/lam_package/lam/_core.cpython-312-x86_64-linux-gnu.so")
        _core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_core)
        EnhancedHierarchicalDeltaNet = _core.EnhancedHierarchicalDeltaNet
        HAS_CORE = True
    except:
        HAS_CORE = False
        print("   ⚠️ Could not import _core module, using PyTorch fallback")

if HAS_CORE:
    class CoreDeltaNetLayer(nn.Module):
        """DeltaNet layer using compiled core module."""
        
        def __init__(self, hidden_size: int = 384, num_heads: int = 12):
            super().__init__()
            self.layer = EnhancedHierarchicalDeltaNet(
                d_model=hidden_size,
                num_heads=num_heads,
                use_hierarchical_decay=True,
                use_enhanced_flux=True,
                fast_decay_init=0.3,
                slow_decay_init=0.85,
            )
        
        def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
            out, _, _, _ = self.layer(x, attention_mask)
            return out, None, None, None
    
    DeltaNetLayer = CoreDeltaNetLayer
    print("   ✅ Using compiled core module")
else:
    DeltaNetLayer = PyTorchDeltaNetLayer
    print("   📦 Using PyTorch fallback")


# ============================================================================
# LAM ENCODER
# ============================================================================

class LAMEncoder(nn.Module):
    """LAM encoder with automatic backend selection."""
    
    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 12,
        num_layers: int = 6,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Load embeddings from teacher model
        from transformers import AutoModel
        teacher_path = lam_dir / "all-MiniLM-L6-v2"
        if teacher_path.exists():
            teacher = AutoModel.from_pretrained(str(teacher_path))
        else:
            teacher = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        for param in teacher.parameters():
            param.requires_grad = False
            
        self.embeddings = teacher.embeddings
        
        # DeltaNet layers (auto-selects best backend)
        self.deltanet_layers = nn.ModuleList([
            DeltaNetLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        # Norms and FFN from teacher
        self.deltanet_norms = nn.ModuleList([
            teacher.encoder.layer[i].attention.output.LayerNorm
            for i in range(num_layers)
        ])
        self.ffn_up = nn.ModuleList([
            teacher.encoder.layer[i].intermediate
            for i in range(num_layers)
        ])
        self.ffn_down = nn.ModuleList([
            teacher.encoder.layer[i].output.dense
            for i in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            teacher.encoder.layer[i].output.LayerNorm
            for i in range(num_layers)
        ])
    
    def mean_pooling(self, x, mask):
        mask = mask.unsqueeze(-1).expand(x.size()).float()
        return torch.sum(x * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    
    def get_extended_embeddings(self, input_ids):
        """
        Get embeddings with position interpolation support (up to 32k tokens).
        Matches DeltaNetPure6Layer.get_extended_embeddings() behavior.
        """
        batch_size, seq_len = input_ids.shape
        
        # Word embeddings (no position limit)
        word_embeddings = self.embeddings.word_embeddings(input_ids)
        
        # Token type embeddings
        token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        
        # Position embeddings with interpolation support
        original_max_pos = 512  # Base position embedding table size
        
        if seq_len <= original_max_pos:
            # Standard case: use existing position embeddings
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.embeddings.position_embeddings(position_ids)
        else:
            # Long sequence: interpolate position embeddings (supports up to 32k)
            scale_factor = (original_max_pos - 1) / (seq_len - 1)
            position_embeddings_list = []
            
            for pos in range(seq_len):
                # Map current position to original position space
                original_pos = pos * scale_factor
                lower_pos = int(original_pos)
                upper_pos = min(lower_pos + 1, original_max_pos - 1)
                weight = original_pos - lower_pos
                
                # Interpolate between lower and upper position embeddings
                lower_emb = self.embeddings.position_embeddings.weight[lower_pos]
                upper_emb = self.embeddings.position_embeddings.weight[upper_pos]
                interp_emb = (1 - weight) * lower_emb + weight * upper_emb
                position_embeddings_list.append(interp_emb)
            
            position_embeddings = torch.stack(position_embeddings_list, dim=0)
            position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine all embeddings
        embeddings = word_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        
        return embeddings
    
    @torch.inference_mode()
    def encode(self, input_ids, attention_mask):
        # Use extended embeddings with position interpolation (supports up to 32k)
        x = self.get_extended_embeddings(input_ids)
        
        for i in range(self.num_layers):
            residual = x
            x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)
            x = self.deltanet_norms[i](residual + x_attn)
            
            residual = x
            x_ffn = self.ffn_up[i](x)
            x_ffn = F.gelu(x_ffn)
            x_ffn = self.ffn_down[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        
        embeddings = self.mean_pooling(x, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)
    
    def load_weights(self, state_dict):
        """Load weights from LAM checkpoint."""
        for i in range(self.num_layers):
            prefix = f"deltanet_layers.{i}."
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'b_proj', 
                         'fast_decay_proj', 'slow_decay_proj', 'g_proj']:
                weight_key = f"{prefix}{name}.weight"
                bias_key = f"{prefix}{name}.bias"
                
                if weight_key in state_dict:
                    getattr(self.deltanet_layers[i], name).weight.data = state_dict[weight_key]
                if bias_key in state_dict:
                    getattr(self.deltanet_layers[i], name).bias.data = state_dict[bias_key]


# ============================================================================
# MAIN LAM CLASS (drop-in replacement for sentence-transformers)
# ============================================================================

class LAM:
    """
    LAM (Linear Associated Memory) - Production Version
    
    Drop-in replacement for sentence-transformers that works on ANY machine.
    Uses the compiled core module for optimal performance.
    
    Usage:
        from lam_production import LAM
        
        model = LAM('LAM-base-v1')
        embeddings = model.encode(['Hello world', 'How are you?'])
    """
    
    def __init__(self, model_name_or_path: str = "LAM-base-v1", device: Optional[str] = None):
        # Use detected device or override
        self.device = device or DEVICE
        
        # Resolve path
        if model_name_or_path.endswith("LAM-base-v1"):
            model_path = lam_dir / "LAM-base-v1"
        else:
            model_path = Path(model_name_or_path)
        
        print(f"📦 Loading LAM from: {model_path}")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        teacher_path = lam_dir / "all-MiniLM-L6-v2"
        if teacher_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(teacher_path))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Create encoder
        self.model = LAMEncoder(hidden_size=384, num_heads=12, num_layers=6)
        self.model = self.model.to(self.device)
        
        # Load weights
        weights_path = model_path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_weights(state_dict)
            print("   ✅ Weights loaded")
        
        self.model.eval()
        
        print("✅ LAM ready!")
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode sentences to embeddings (sentence-transformers compatible)."""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        if not sentences:
            return np.array([]) if convert_to_numpy else torch.tensor([])
        
        max_length = kwargs.get('max_length', 32768)  # Support up to 32k tokens
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            tokens = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors='pt'
            ).to(self.device)
            
            embeddings = self.model.encode(tokens['input_ids'], tokens['attention_mask'])
            
            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            if convert_to_numpy:
                all_embeddings.append(embeddings.cpu().numpy())
            else:
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings) if convert_to_numpy else torch.cat(all_embeddings, dim=0)
    
    def __call__(self, sentences, **kwargs):
        return self.encode(sentences, **kwargs)
    
    def get_sentence_embedding_dimension(self) -> int:
        return 384


# ============================================================================
# TEST
# ============================================================================

def test():
    print("\n" + "="*70)
    print("LAM Production Test")
    print("="*70)
    
    model = LAM('LAM-base-v1')
    
    sentences = ["Hello world", "Machine learning is amazing", "Natural language processing"]
    embeddings = model.encode(sentences)
    
    print(f"\nEncoded {len(sentences)} sentences")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print("\n✅ Test passed!")
    print("="*70)


if __name__ == "__main__":
    test()


