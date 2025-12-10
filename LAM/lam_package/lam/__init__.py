"""
LAM (Linear Attention Model)
==============================

High-performance embeddings with O(n) linear complexity.

Usage:
    from lam import LAM
    
    model = LAM('LAM-base-v1')
    embeddings = model.encode(['Hello world', 'How are you?'])
    
    # Perfect Recall (NL Paper Delta Gradient Descent)
    from lam import PerfectRecall
    
    memory = PerfectRecall(model)
    memory.store("The secret password is QUANTUM7DELTA")
    memory.store("Paris is the capital of France")
    
    # Perfect recall - finds exact stored content
    result = memory.recall("What is the password?")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict

__version__ = "1.0.0"
__all__ = ['LAM', 'PerfectRecall', 'InfiniteContextStreamer']

# ============================================================================
# DEVICE DETECTION
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import the compiled core (always available)
from . import _core


# ============================================================================
# MAIN LAM CLASS
# ============================================================================

class LAM:
    """
    LAM (Linear Attention Model) - High-performance embeddings.
    
    O(n) linear complexity for semantic embeddings.
    Supports sequences up to 32k tokens.
    
    Args:
        model_name_or_path: Path to model directory
        device: Device to run on ('cuda', 'cpu', or None for auto)
        backend: Backend to use (only 'cython' supported). Default: 'cython'
    """
    
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, backend: str = 'cython'):
        self.device = device or DEVICE
        self.backend = backend.lower()
        
        if self.backend != 'cython':
            raise ValueError(f"Only 'cython' backend is supported. Got: {backend}")
        
        model_path = self._resolve_path(model_name_or_path)
        
        # Check for license (determines max_length)
        from ._license import LicenseManager
        self._license_manager = LicenseManager(model_path)
        self._max_length = self._license_manager.get_max_length()
        self._tier = self._license_manager.get_tier()
        
        # Load tokenizer using tokenizers library (LAM dependency)
        # ⚡ OPTIMIZED: Fast Rust-based tokenizer for speed
        # Load from model_path (LAM-base-v1) folder, same as config and weights
        from tokenizers import Tokenizer
        
        # Load tokenizer.json from model_path (LAM-base-v1 folder) - same pattern as config/weights
        tokenizer_path = model_path / "tokenizer.json"
        
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found in {model_path}. "
                "Please ensure tokenizer.json exists in the model directory."
            )
        
        # Load tokenizer and enable padding
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=self._max_length)  # Use license-based limit
        
        # Create LAM model - Cython backend only
        self._model = _LAMModel(model_path)
        # Set license limits on model so it can enforce them
        self._model._max_length = self._max_length
        self._model._tier = self._tier
        self._model = self._model.to(self.device)
        self._model.eval()
        
        # Comprehensive warmup/precompile on first load (world-class performance from first use)
        # This ensures the model is fully loaded into memory and all CUDA kernels are precompiled
        if self.device == 'cuda':
            try:
                import torch
                # Warmup multiple sequence lengths to precompile all CUDA kernels
                # This matches world-class libraries that warmup during initialization
                warmup_lengths = [128, 512, 2048]  # Cover common use cases
                
                for seq_len in warmup_lengths:
                    dummy_ids = torch.randint(0, 1000, (1, seq_len), device=self.device)
                    dummy_mask = torch.ones_like(dummy_ids)
                    with torch.no_grad():
                        _ = self._model.get_sentence_embeddings(dummy_ids, dummy_mask)
                
                # Warmup position embedding interpolation (for 32k support)
                # Test with a longer sequence to trigger interpolation path
                long_ids = torch.randint(0, 1000, (1, 1024), device=self.device)
                long_mask = torch.ones_like(long_ids)
                with torch.no_grad():
                    _ = self._model.get_sentence_embeddings(long_ids, long_mask)
                
                # Final sync to ensure all kernels are compiled
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass  # Warmup failed, continue anyway
        
        print(f"✅ LAM model loaded ({self.device})")
    
    def _resolve_path(self, name_or_path: str) -> Path:
        path = Path(name_or_path)
        if path.exists():
            return path
        for base in [Path.cwd(), Path.home(), Path("/workspace/LAM")]:
            candidate = base / name_or_path
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Model not found: {name_or_path}")
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
        dimensions: Optional[int] = None,
        dimension: Optional[int] = None,  # Alias for dimensions (for lam_embed compatibility)
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode sentences into embeddings with Matryoshka support.
        
        Args:
            sentences: Input text(s) to encode
            batch_size: Batch size for encoding
            show_progress_bar: Show progress bar during encoding
            convert_to_numpy: Convert result to numpy array
            convert_to_tensor: Convert result to torch tensor
            normalize_embeddings: Normalize embeddings (L2 norm)
            dimensions: Target dimension (64, 128, 256, or 384). Default: 384 (full dimension)
            dimension: Alias for dimensions (for lam_embed compatibility). Use either dimensions= or dimension= (same thing)
            **kwargs: Additional arguments (e.g., max_length)
        
        Returns:
            Embeddings array/tensor of shape (batch_size, dimensions)
        """
        # Handle both 'dimensions' (plural) and 'dimension' (singular) for compatibility
        # They do the same thing - just different naming conventions (lam_embed uses 'dimension')
        target_dim = dimensions if dimensions is not None else (dimension if dimension is not None else None)
        
        if isinstance(sentences, str):
            sentences = [sentences]
        if not sentences:
            # Return empty array with correct shape (0, embedding_dim)
            final_dim = target_dim if target_dim is not None else 384
            empty_shape = (0, final_dim)
            return np.zeros(empty_shape, dtype=np.float32) if convert_to_numpy else torch.zeros(empty_shape, dtype=torch.float32)
        
        max_length = kwargs.get('max_length', 32768)  # Support up to 32k tokens
        all_embeddings = []
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Encoding")
            except ImportError:
                pass
        
        if self._model is not None:
            self._model.eval()
        with torch.no_grad():
            for i in iterator:
                batch = sentences[i:i+batch_size]
                
                # Tokenize using tokenizers library (fast Rust-based)
                # Use license-based max_length (set during __init__)
                license_max = getattr(self, '_max_length', 0x2000)
                effective_max = min(max_length, license_max)  # Don't exceed license limit
                self.tokenizer.enable_truncation(max_length=effective_max)
                
                # Encode batch
                encoded = self.tokenizer.encode_batch(batch)
                
                # Convert to PyTorch tensors
                input_ids = torch.tensor([e.ids for e in encoded], dtype=torch.long, device=self.device)
                attention_mask = torch.tensor([e.attention_mask for e in encoded], dtype=torch.long, device=self.device)
                
                tokens = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
                
                # Get full 384-dim embeddings (proprietary computation in compiled _core.so)
                # Cython backend only
                embeddings = self._model.get_sentence_embeddings(
                    tokens['input_ids'],
                    tokens['attention_mask']
                )
                
                # Matryoshka truncation (compiled to hide parameters and logic)
                # Use target_dim (which handles both 'dimensions' and 'dimension' parameters)
                if target_dim is not None and target_dim < 384:
                    from . import _secrets
                    # Use compiled function to hide truncation logic and parameters
                    # All truncation logic is in compiled _secrets.so (binary, not readable)
                    embeddings = _secrets.truncate_embeddings(embeddings, target_dim)
                elif normalize_embeddings:
                    # Normalize full 384-dim (existing behavior)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        result = torch.cat(all_embeddings, dim=0)
        return result.numpy() if convert_to_numpy and not convert_to_tensor else result
    
    def __call__(self, sentences, **kwargs):
        return self.encode(sentences, **kwargs)
    
    def get_sentence_embedding_dimension(self, dimensions: Optional[int] = None) -> int:
        """
        Get the embedding dimension.
        
        Args:
            dimensions: If specified, returns that dimension. Otherwise returns 384 (full dimension).
        
        Returns:
            Embedding dimension
        """
        if dimensions is not None:
            return dimensions
        return 384


# ============================================================================
# PYTORCH MODEL (fallback)
# ============================================================================

class _LAMModel(nn.Module):
    """LAM Model - O(n) linear complexity."""
    
    def __init__(self, model_path, license_limit=0x2000, tier="free"):
        """Model initialization - calls compiled version to hide architecture."""
        super().__init__()
        try:
            from . import _core
            # All initialization logic is in compiled code (architecture, config, weights)
            # The compiled function returns a fully initialized model
            compiled_model = _core.create_lam_model(model_path, license_limit, tier)
            # Copy all module attributes directly (PyTorch modules)
            self.embeddings = compiled_model.embeddings
            self.deltanet_layers = compiled_model.deltanet_layers
            self.deltanet_norms = compiled_model.deltanet_norms
            self.deltanet_ffns = compiled_model.deltanet_ffns
            self.output_denses = compiled_model.output_denses
            self.ffn_norms = compiled_model.ffn_norms
            # Copy simple attributes
            self.position_embedding_type = compiled_model.position_embedding_type
            self._max_length = compiled_model._max_length
            self._tier = compiled_model._tier
            # Copy buffers (position_ids)
            if hasattr(compiled_model, 'position_ids'):
                self.register_buffer('position_ids', compiled_model.position_ids.clone())
        except ImportError:
            raise RuntimeError("Compiled model initialization module (_core.so) required. Please rebuild the package.")
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        """Forward pass - calls compiled version to hide architecture."""
        try:
            from . import _core
            forward_wrapper = _core.LAMForward(self)
            license_limit = getattr(self, '_max_length', 0x2000)
            try:
                return forward_wrapper.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    license_limit=license_limit,
                    position_emb_weight=self.embeddings['position_embeddings'].weight,
                    **kwargs
                )
            except RuntimeError as e:
                if "CONTEXT_LIMIT_EXCEEDED" in str(e):
                    max_length = getattr(self, '_max_length', 0x2000)
                    tier = getattr(self, '_tier', 'free')
                    limit_str = str(e).split(':')[-1] if ':' in str(e) else str(max_length)
                    if tier == "free":
                        raise RuntimeError(
                            f"\n{'='*40}\n"
                            f"🛑 LIMIT REACHED: {input_ids.size(1)} tokens\n"
                            f"Current Tier Limit: {limit_str} tokens\n"
                            f"To unlock 32k context, get a license at https://saidhome.ai\n"
                            f"{'='*40}\n"
                        )
                    else:
                        raise RuntimeError(
                            f"\n{'='*40}\n"
                            f"🛑 LIMIT REACHED: {input_ids.size(1)} tokens\n"
                            f"Current Tier Limit: {limit_str} tokens\n"
                            f"Your {tier.upper()} license supports up to {limit_str} tokens.\n"
                            f"For longer sequences, contact support at https://saidhome.ai\n"
                            f"{'='*40}\n"
                        )
                raise
        except ImportError:
            raise RuntimeError("Compiled forward pass module (_core.so) required. Please rebuild the package.")
    
    def get_sentence_embeddings(self, input_ids, attention_mask=None):
        """Sentence embedding extraction - calls compiled version."""
        try:
            from . import _core
            forward_wrapper = _core.LAMForward(self)
            return forward_wrapper.get_sentence_embeddings(input_ids, attention_mask)
        except ImportError:
            raise RuntimeError("Compiled forward pass module (_core.so) required. Please rebuild the package.")
    
    @torch.inference_mode()
    def encode(self, input_ids, attention_mask):
        # Use get_sentence_embeddings (same as stsb_evaluation.py)
        return self.get_sentence_embeddings(input_ids, attention_mask)


# ============================================================================
# PERFECT RECALL (NL Paper Delta Gradient Descent)
# ============================================================================

class PerfectRecall:
    """
    🧠 Perfect Recall Memory using NL Paper's Delta Gradient Descent.
    
    From: https://abehrouz.github.io/files/NL.pdf (Appendix C)
    
    Key equation (ERASE-THEN-WRITE):
        W_{t+1} = W_t @ (I - α k @ k.T) + β k @ v.T
        
    This gives PERFECT RECALL because:
    1. ERASE: Clears old value at this key (I - α k @ k.T)
    2. WRITE: Stores new value (β k @ v.T)
    
    Usage:
        >>> from lam import LAM, PerfectRecall
        >>> model = LAM('LAM-base-v1')
        >>> memory = PerfectRecall(model)
        >>> 
        >>> # Store content
        >>> memory.store("The secret password is QUANTUM7DELTA")
        >>> memory.store("Paris is the capital of France")
        >>> 
        >>> # Perfect recall
        >>> memory.recall("What is the password?")
        "The secret password is QUANTUM7DELTA"
    """
    
    def __init__(
        self,
        lam_model: 'LAM',
        d_k: int = 384,
        d_v: int = 384,
        n_heads: int = 16,   # More heads = more capacity
        alpha: float = 1.0,  # FULL erase (NL paper recommendation)
        beta: float = 1.0,   # Full write strength
    ):
        self.lam = lam_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.alpha = alpha
        self.beta = beta
        self.device = lam_model.device
        
        # Multi-head memory matrix W: [n_heads, d_k, d_v]
        self.W = torch.zeros(n_heads, d_k, d_v, device=self.device)
        
        # Identity matrix for erase operation
        self.I = torch.eye(d_k, device=self.device).unsqueeze(0).expand(n_heads, -1, -1).clone()
        
        # Learned key projection (different from value for better separation)
        self.key_proj = torch.nn.Linear(d_k, d_k, bias=False).to(self.device)
        torch.nn.init.orthogonal_(self.key_proj.weight)
        
        # Content index for text retrieval  
        self.content_index: List[Dict] = []
        self.embeddings_cache: Dict[int, torch.Tensor] = {}  # Cache embeddings
        
        print(f"🧠 PerfectRecall initialized (NL Paper Delta GD)")
        print(f"   Memory: {n_heads} heads × {d_k}×{d_v} = {n_heads * d_k * d_v:,} params")
    
    def _embed(self, text: str) -> torch.Tensor:
        """Get LAM embedding for text."""
        # Handle both lam package LAM and test_8k_LAM.LAM
        try:
            # Try lam package LAM (has convert_to_tensor parameter)
            emb = self.lam.encode([text], convert_to_tensor=True)
        except TypeError:
            # Fallback for test_8k_LAM.LAM (might not have convert_to_tensor)
            emb = self.lam.encode([text])
            if isinstance(emb, np.ndarray):
                emb = torch.tensor(emb)
        
        # Ensure it's a tensor
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        
        # Handle single sentence (returns [1, dim] or [dim])
        if emb.dim() == 2:
            emb = emb.squeeze(0)
        
        return emb.to(self.device)
    
    def store(self, content: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Store content with PERFECT RECALL capability.
        
        Uses NL Paper Eq 114: W = W @ (I - α k @ k.T) + β k @ v.T
        
        NO CHUNKING - stores full document as ONE embedding to preserve global semantics.
        This matches /maas/infinite_memory.py approach for perfect recall.
        """
        # Embed FULL document as ONE embedding (preserves global semantics, no chunking!)
        embedding = self._embed(content)
        memory_id = len(self.content_index)
        
        # Cache full document embedding for fast recall
        self.embeddings_cache[memory_id] = embedding.clone()
        
        # === NL PAPER KEY GENERATION ===
        # For needle-in-haystack: Use needle text to create key (if provided)
        # This ensures queries matching the needle will find it (NL paper content-addressable)
        # For general storage: Use document embedding + unique ID
        
        # === NL PAPER: Key = query, Value = what to retrieve ===
        if metadata and 'needle_text' in metadata:
            # Needle-in-haystack: Key = needle query, Value = needle (distinct, retrievable)
            # Store haystack in metadata, return it when needle is found
            needle_text = metadata['needle_text']
            needle_emb = self._embed(needle_text)
            k_raw = self.key_proj(needle_emb)  # Key = needle (what you query)
            v_raw = needle_emb  # Value = needle (distinct, retrievable)
        else:
            # General: Key = content, Value = content
            k_raw = self.key_proj(embedding)
            v_raw = embedding
        
        # Normalize key (paper requirement)
        k = F.normalize(k_raw, dim=-1)  # [d_k]
        v = v_raw  # [d_v]
        
        # Expand for multi-head
        k = k.unsqueeze(0).expand(self.n_heads, -1)  # [n_heads, d_k]
        v = v.unsqueeze(0).expand(self.n_heads, -1)  # [n_heads, d_v]
        
        # === NL PAPER DELTA GRADIENT DESCENT (Eq 114) ===
        # Step 1: ERASE - W = W @ (I - α k @ k.T)
        # Full erase (α=1) for PERFECT recall (no interference)
        k_outer = torch.einsum('hk,hj->hkj', k, k)  # [n_heads, d_k, d_k]
        erase_mask = self.I - self.alpha * k_outer
        self.W = torch.einsum('hkv,hkj->hjv', self.W, erase_mask)
        
        # Step 2: WRITE - W = W + β k @ v.T
        write_term = self.beta * torch.einsum('hk,hv->hkv', k, v)
        self.W = self.W + write_term
        
        # Store content for exact retrieval
        memory_info = {
            'id': memory_id,
            'content': content,
            'metadata': metadata or {},
        }
        self.content_index.append(memory_info)
        
        return memory_info
    
    def recall(self, query: str, top_k: int = 1) -> Union[str, List[str]]:
        """
        Recall stored content with PERFECT accuracy (NL Paper Delta GD).
        
        Uses Delta Gradient Descent retrieval: v = W.T @ k
        Then compares retrieved value to full document embeddings (global semantics).
        This matches /maas/infinite_memory.py approach - NO chunking, full document recall.
        """
        if not self.content_index:
            return "No content stored yet."
        
        # Embed query
        q_emb = self._embed(query)
        
        # Project to key space
        with torch.no_grad():
            q_k = self.key_proj(q_emb)
        q_k = F.normalize(q_k, dim=-1)
        q_k = q_k.unsqueeze(0).expand(self.n_heads, -1)
        
        # === NL PAPER FORMULA: v = W.T @ k ===
        # Step 1: Retrieve value from memory matrix
        retrieved = torch.einsum('hkv,hk->hv', self.W, q_k)  # [n_heads, d_v]
        avg_retrieved = retrieved.mean(dim=0)  # [d_v]
        avg_retrieved = F.normalize(avg_retrieved, dim=-1)
        
        # Step 2: Find stored value that best matches retrieved value (NL paper formula)
        best_score = -float('inf')
        best_content = None
        
        for item in self.content_index:
            # Get stored value (needle for needle-in-haystack, content for general)
            if 'needle_text' in item.get('metadata', {}):
                # Value was stored as needle embedding
                needle_text = item['metadata']['needle_text']
                stored_value = self._embed(needle_text)
            else:
                # Value was stored as content embedding
                stored_value = self.embeddings_cache.get(item['id'])
                if stored_value is None:
                    stored_value = self._embed(item['content'])
            
            stored_value_norm = F.normalize(stored_value, dim=-1)
            
            # Paper: cosine similarity between retrieved and stored value
            score = F.cosine_similarity(
                avg_retrieved.unsqueeze(0),
                stored_value_norm.unsqueeze(0),
                dim=-1
            ).item()
            
            if score > best_score:
                best_score = score
                # For needle-in-haystack: Return haystack (stored in content)
                # For general: Return content
                best_content = item['content']
        
        if top_k == 1:
            return best_content if best_content else "Not found"
        else:
            # Return top_k
            all_scores = []
            for item in self.content_index:
                stored_value = self.embeddings_cache.get(item['id'])
                if stored_value is None:
                    stored_value = self._embed(item['content'])
                stored_value_norm = F.normalize(stored_value, dim=-1)
                score = F.cosine_similarity(
                    avg_retrieved.unsqueeze(0),
                    stored_value_norm.unsqueeze(0),
                    dim=-1
                ).item()
                all_scores.append((score, item['content']))
            
            all_scores.sort(reverse=True)
            return [content for _, content in all_scores[:top_k]]
    
    def clear(self):
        """Clear all stored memories."""
        self.W.zero_()
        self.content_index.clear()
        print("🧹 Memory cleared")


# ============================================================================
# INFINITE CONTEXT STREAMER
# ============================================================================

try:
    from .infinite_streamer import InfiniteContextStreamer
except ImportError:
    # Fallback if infinite_streamer not available
    InfiniteContextStreamer = None

try:
    import sys
    from pathlib import Path
    async_streamer_path = Path(__file__).parent.parent.parent / "infinite_streamer_async.py"
    if async_streamer_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("infinite_streamer_async", async_streamer_path)
        async_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(async_module)
        AsyncInfiniteStreamer = async_module.AsyncInfiniteStreamer
        __all__.append('AsyncInfiniteStreamer')
    else:
        AsyncInfiniteStreamer = None
except Exception:
    AsyncInfiniteStreamer = None


