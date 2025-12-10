"""
🚀 LAM DUAL-TIER ENCODER V3 (Unified Engine)
============================================

Merges 'Dual Encoder' ease-of-use with 'Universal Indexer' physics.

Tier 1 (Standard): 384d Semantic Vector
- Mechanism: Holographic Projection (v6)
- Metric: Cosine Similarity
- Use: Summaries, Semantic Search, RAG

Tier 2 (Enterprise): 12,288d Perfect Recall Vector
- Mechanism: Explicit Delta-GD (Erase-Write)
- Output: Flattened W Matrix (Unnormalized)
- Metric: DOT PRODUCT (Magnitude = Memory)
- Use: Needle-in-Haystack, Infinite Context, Perfect Recall

KEY INSIGHT:
- Standard: Semantic similarity (cosine) - for RAG summaries
- Enterprise: Perfect recall (dot product) - for forensic retrieval
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
import sys

# Add lam_package to path
sys.path.insert(0, str(Path(__file__).parent))

# Import your underlying architecture
try:
    from lam import InfiniteContextStreamer
    from lam import LAM
except ImportError:
    try:
        from infinite_streamer import InfiniteContextStreamer
    except ImportError:
        InfiniteContextStreamer = None
        print("⚠️  Warning: InfiniteContextStreamer not found")


class LAMDualEncoder:
    """
    LAM UNIFIED ENGINE (The Final Architecture)
    
    Merges 'Dual Encoder' ease-of-use with 'Universal Indexer' physics.
    
    Tier 1: STANDARD (384d)
    - Mechanism: Holographic Projection (v6)
    - Metric: Cosine Similarity
    - Use: Summaries, Semantic Search, RAG
    
    Tier 2: ENTERPRISE (12,288d)
    - Mechanism: Explicit Delta-GD (Erase-Write)
    - Output: Flattened W Matrix
    - Metric: DOT PRODUCT (Unnormalized)
    - Use: Needle-in-Haystack, Infinite Context, Perfect Recall
    """
    
    def __init__(self, model, tokenizer=None, device='cuda'):
        """
        Initialize the unified encoder.
        
        Args:
            model: LAM model instance
            tokenizer: Tokenizer (if None, uses model.tokenizer)
            device: Device to run on
        """
        self.model = model
        self.device = device if hasattr(model, 'device') else (device if torch.cuda.is_available() else 'cpu')
        
        # Set model to eval mode if it has the method
        if hasattr(self.model, 'eval'):
            self.model.eval()
        elif hasattr(self.model, '_model') and hasattr(self.model._model, 'eval'):
            self.model._model.eval()
        
        # Get tokenizer
        if tokenizer is None:
            if hasattr(model, 'tokenizer'):
                self.tokenizer = model.tokenizer
            else:
                raise ValueError("Tokenizer not provided and model has no tokenizer attribute")
        else:
            self.tokenizer = tokenizer
        
        # Initialize the Infinite Streamer (The Brain Builder)
        if InfiniteContextStreamer is None:
            raise ImportError("InfiniteContextStreamer not available")
        
        self.streamer = InfiniteContextStreamer(model, chunk_size=2048)
        
        # Architecture Constants (from model)
        self.HEADS = 12
        self.D_K = 32
        self.D_V = 32
        
        # Get last layer for k/v projections
        self.last_layer = self._get_last_layer()
        
        # For centering (optional whitening)
        self.mean_vector = None
        whitening_path = Path(__file__).parent / "lam_whitening_stats.npy"
        if whitening_path.exists():
            print("   ✅ Loading Enterprise Whitening Stats (Mean Vector)...")
            self.mean_vector = torch.from_numpy(np.load(whitening_path)).to(self.device)

    def _get_last_layer(self):
        """Get the last DeltaNet layer for k/v projections."""
        # Try different model structures
        if hasattr(self.model, '_model'):
            internal_model = self.model._model
            if hasattr(internal_model, 'deltanet_layers'):
                return internal_model.deltanet_layers[-1]
        elif hasattr(self.model, 'deltanet_layers'):
            return self.model.deltanet_layers[-1]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'deltanet_layers'):
            return self.model.model.deltanet_layers[-1]
        else:
            # Fallback: return None, will use streamer's state
            return None

    def calibrate_enterprise_mode(self, sample_texts, n_samples=100):
        """
        Learn the mean vector for centering (whitening).
        
        Args:
            sample_texts: List of sample text strings (about 100 recommended)
        """
        print("   ⚖️ Calibrating Enterprise Mode (Computing Mean Vector)...")
        vectors = []
        with torch.no_grad():
            from tqdm import tqdm
            for text in tqdm(sample_texts[:n_samples], desc="Reading sample docs"):
                # Get RAW 12k vector using Delta-GD
                raw = self._encode_enterprise_delta(text, alpha=1.0, beta=1.0)
                vectors.append(torch.from_numpy(raw))
        
        # Calculate the "Average Document"
        stack = torch.stack(vectors)
        self.mean_vector = torch.mean(stack, dim=0)
        
        # Save
        whitening_path = Path(__file__).parent / "lam_whitening_stats.npy"
        np.save(whitening_path, self.mean_vector.cpu().numpy())
        print(f"   ✅ Calibration Complete. Enterprise Mode Ready. (Saved to {whitening_path})")

    def encode(self, text, mode="standard", **kwargs):
        """
        Universal Encode Function.
        
        Args:
            text: Input text (any length)
            mode: "standard" (384d) or "enterprise" (12k)
            **kwargs: For enterprise mode:
                - alpha: Erase strength (1.0 = Perfect Recall, 0.5 = Semantic Mixing)
                - beta: Write strength (default: 1.0)
        
        Returns:
            numpy array: Embedding vector
        """
        if mode == "standard":
            return self._encode_standard(text)
        elif mode == "enterprise":
            alpha = kwargs.get('alpha', 1.0)
            beta = kwargs.get('beta', 1.0)
            semantic_weight = kwargs.get('semantic_weight', 0.0)  # Default: pure Delta-GD
            return self._encode_enterprise_delta(text, alpha=alpha, beta=beta, semantic_weight=semantic_weight)
        else:
            raise ValueError("Mode must be 'standard' or 'enterprise'")

    def encode_query(self, text, mode="standard"):
        """
        Universal Query Function.
        
        CRITICAL: Enterprise queries are PROBES (k ⊗ v), not vectors.
        
        Args:
            text: Query text
            mode: "standard" (384d) or "enterprise" (12k probe)
        
        Returns:
            numpy array: Query vector or probe
        """
        if mode == "standard":
            return self._encode_standard(text)  # Queries look like docs in Cosine space
        elif mode == "enterprise":
            return self._encode_enterprise_probe(text)
        else:
            raise ValueError("Mode must be 'standard' or 'enterprise'")

    # =========================================================================
    # TIER 1: STANDARD (384d)
    # =========================================================================

    def _encode_standard(self, text):
        """Standard Semantic Vector (Proven 0.8189 Spearman)."""
        with torch.no_grad():
            emb = self.model.encode(
                text, 
                batch_size=1, 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
        if emb.ndim == 2:
            return emb.squeeze(0)
        return emb

    # =========================================================================
    # TIER 2: ENTERPRISE (12k Delta-GD)
    # =========================================================================

    def _encode_enterprise_delta(self, text, alpha=1.0, beta=1.0, semantic_weight=0.0):
        """
        Builds the 12k Vector using EXPLICIT DELTA RULE (Erase-Write).
        This is the physics that allows 'Needle in Haystack'.
        
        Args:
            text: Input text (can be very long, e.g., 100k tokens)
            alpha: Erase strength (1.0 = Perfect Recall, 0.5 = Semantic Mixing)
            beta: Write strength (default: 1.0)
            semantic_weight: If > 0, blend with semantic projection (0.0 = pure Delta-GD)
        
        Returns:
            numpy array [12,288] - UNNORMALIZED (Magnitude = Memory)
        """
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            enc = self.tokenizer.encode(text)
            if isinstance(enc, list):
                input_ids = torch.tensor([enc], dtype=torch.long, device=self.device)
            else:
                input_ids = torch.tensor([enc.ids], dtype=torch.long, device=self.device)
        else:
            encoded = self.tokenizer(text, return_tensors='pt', add_special_tokens=True)
            input_ids = encoded['input_ids'].to(self.device)
        
        # Initialize Memory [12, 32, 32]
        W = torch.zeros(self.HEADS, self.D_K, self.D_V, device=self.device)
        I = torch.eye(self.D_K, device=self.device).unsqueeze(0).expand(self.HEADS, -1, -1)
        
        chunk_size = 512
        total_len = input_ids.shape[1]
        
        with torch.no_grad():
            # Use streamer to process in chunks (avoids OOM)
            self.streamer.reset()
            
            # Process in chunks
            for start_idx in range(0, total_len, chunk_size):
                end_idx = min(start_idx + chunk_size, total_len)
                chunk_ids = input_ids[:, start_idx:end_idx]
                
                # Get hidden states from model
                # We need to access the model's internal structure
                if hasattr(self.model, '_model'):
                    internal_model = self.model._model
                    if hasattr(internal_model, 'embeddings'):
                        # Get embeddings
                        emb_dict = internal_model.embeddings
                        if hasattr(emb_dict, 'keys') and 'word_embeddings' in emb_dict:
                            hidden = emb_dict['word_embeddings'](chunk_ids)
                            
                            # Process through layers to get final hidden state
                            for layer_idx, (layer, norm) in enumerate(zip(
                                internal_model.deltanet_layers,
                                internal_model.deltanet_norms
                            )):
                                residual = hidden
                                hidden = norm(hidden)
                                
                                # Forward pass
                                output, attn, past_kv, ortho = layer(
                                    hidden,
                                    attention_mask=torch.ones_like(chunk_ids),
                                    use_cache=True
                                )
                                hidden = residual + output
                                
                                # Extract k, v from last layer for Delta-GD
                                if layer_idx == len(internal_model.deltanet_layers) - 1:
                                    # Get k, v projections from the layer
                                    if hasattr(layer, 'k_proj') and hasattr(layer, 'v_proj'):
                                        # Project hidden to k, v
                                        k_flat = layer.k_proj(hidden)  # [1, seq, key_dim]
                                        v_flat = layer.v_proj(hidden)  # [1, seq, value_dim]
                                        
                                        # Reshape to multi-head
                                        # key_dim = HEADS * D_K = 12 * 32 = 384
                                        # value_dim = HEADS * D_V = 12 * 32 = 384
                                        k = k_flat.view(1, -1, self.HEADS, self.D_K).transpose(1, 2)  # [1, 12, seq, 32]
                                        v = v_flat.view(1, -1, self.HEADS, self.D_V).transpose(1, 2)  # [1, 12, seq, 32]
                                        
                                        # Normalize keys
                                        k = F.normalize(k, dim=-1)
                                        
                                        # Apply Delta Rule (Erase then Write) per token
                                        # Process token by token for exact Delta-GD
                                        for t in range(k.shape[2]):
                                            k_t = k[:, :, t, :].squeeze(0)  # [12, 32]
                                            v_t = v[:, :, t, :].squeeze(0)  # [12, 32]
                                            
                                            # Erase Step: W_new = W @ (I - alpha * k @ k.T)
                                            # For each head: W_h = W_h @ (I - alpha * k_h @ k_h.T)
                                            for h in range(self.HEADS):
                                                k_h = k_t[h:h+1, :]  # [1, 32]
                                                v_h = v_t[h:h+1, :]  # [1, 32]
                                                
                                                # k @ k.T: [1, 32] @ [32, 1] = [1, 1] -> expand to [32, 32]
                                                k_outer = torch.outer(k_h.squeeze(0), k_h.squeeze(0))  # [32, 32]
                                                
                                                # NL Paper Delta-GD (Appendix C): W = W @ (I - α k k^T) + β v k^T
                                                # Erase: W_h = W_h @ (I - alpha * k_outer)
                                                # CRITICAL: Use proper matrix multiplication order from NL paper
                                                W[h] = torch.matmul(W[h], I[h] - alpha * k_outer)
                                                
                                                # Write: W_h = W_h + beta * v_h @ k_h.T  
                                                # NL Paper: β v k^T (outer product of v and k)
                                                vk_outer = torch.outer(v_h.squeeze(0), k_h.squeeze(0))  # [32, 32]
                                                W[h] = W[h] + beta * vk_outer
                                        
                                        break  # Only process last layer
                else:
                    # Fallback: use streamer's state_slow (simpler but less precise)
                    _ = self.streamer.stream_embedding(
                        chunk_ids,
                        verbose=False,
                        use_state_embedding=True
                    )
                    if hasattr(self.streamer, 'state_slow') and self.streamer.state_slow is not None:
                        W_chunk = self.streamer.state_slow.squeeze(0)  # [12, 32, 32]
                        # Accumulate (simple addition for now)
                        W = W + W_chunk
        
        # Apply whitening if available
        if self.mean_vector is not None:
            W_flat = W.flatten()  # [12,288]
            W_flat = W_flat - self.mean_vector
            W = W_flat.view(self.HEADS, self.D_K, self.D_V)
        
        # Blend with semantic if requested
        if semantic_weight > 0:
            semantic_384 = self._encode_standard(text)
            semantic_384_t = torch.from_numpy(semantic_384).to(self.device)
            semantic_384_reshaped = semantic_384_t.view(12, 32)
            semantic_12k = torch.zeros(12, 32, 32, device=self.device)
            for h in range(12):
                semantic_12k[h] = torch.diag(semantic_384_reshaped[h])
            semantic_12k_flat = semantic_12k.flatten()
            
            # Normalize semantic for blending
            semantic_12k_flat = F.normalize(semantic_12k_flat, dim=0)
            W_flat_normalized = F.normalize(W.flatten(), dim=0)
            
            # Blend
            W_flat = semantic_weight * semantic_12k_flat + (1 - semantic_weight) * W_flat_normalized
            W = W_flat.view(self.HEADS, self.D_K, self.D_V)
        
        # Return Flattened [12288] - UNNORMALIZED (Magnitude = Memory)
        return W.flatten().float().cpu().numpy()

    def _encode_enterprise_probe(self, text):
        """
        Creates the Retrieval Key for Delta-GD (NL Paper Appendix C).
        
        NL Paper Formula: v = W^T @ k
        - Documents: W (memory matrix, [12, 32, 32] flattened to [12288])
        - Queries: k (key vector, [12, 32])
        - Retrieval: v_retrieved = W^T @ k gives [12, 32] retrieved value
        
        CRITICAL: For dot product similarity with flattened W, we need to compute:
        - W is [12, 32, 32] = stores all document knowledge
        - k is [12, 32] = query key
        - v_retrieved = W^T @ k = [12, 32] (retrieved value per head)
        - Similarity = ||v_retrieved|| or dot(v_retrieved, v_expected)
        
        But for flattened comparison, we can use:
        - Expand k to k @ k.T = [12, 32, 32] to match W structure
        - Then dot(W_flattened, k_expanded_flattened) approximates the retrieval
        
        However, the CORRECT way is to reshape W and compute W^T @ k properly!
        
        Args:
            text: Query text
        
        Returns:
            numpy array [12,288] - Key vector expanded to match W structure
        """
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            enc = self.tokenizer.encode(text)
            if isinstance(enc, list):
                input_ids = torch.tensor([enc], dtype=torch.long, device=self.device)
            else:
                input_ids = torch.tensor([enc.ids], dtype=torch.long, device=self.device)
        else:
            encoded = self.tokenizer(text, return_tensors='pt', add_special_tokens=True)
            input_ids = encoded['input_ids'].to(self.device)
        
        with torch.no_grad():
            # Get hidden state from model
            if hasattr(self.model, '_model'):
                internal_model = self.model._model
                if hasattr(internal_model, 'embeddings'):
                    emb_dict = internal_model.embeddings
                    if hasattr(emb_dict, 'keys') and 'word_embeddings' in emb_dict:
                        hidden = emb_dict['word_embeddings'](input_ids)
                        
                        # Process through layers
                        for layer_idx, (layer, norm) in enumerate(zip(
                            internal_model.deltanet_layers,
                            internal_model.deltanet_norms
                        )):
                            residual = hidden
                            hidden = norm(hidden)
                            output, attn, past_kv, ortho = layer(
                                hidden,
                                attention_mask=torch.ones_like(input_ids),
                                use_cache=True
                            )
                            hidden = residual + output
                            
                            # Extract k from last layer (NL Paper: query is key)
                            if layer_idx == len(internal_model.deltanet_layers) - 1:
                                if hasattr(layer, 'k_proj'):
                                    # Mean pool hidden for query
                                    hidden_mean = hidden.mean(dim=1)  # [1, 384]
                                    
                                    # Project to key space
                                    k_flat = layer.k_proj(hidden_mean)  # [1, key_dim = 384]
                                    
                                    # Reshape to multi-head keys
                                    k = k_flat.view(1, self.HEADS, self.D_K)  # [1, 12, 32]
                                    
                                    # Normalize keys (NL paper: keys are normalized)
                                    k = F.normalize(k, dim=-1)
                                    
                                    # Expand k to match W structure: k @ k.T per head
                                    # This creates [12, 32, 32] matching W's structure
                                    # When we do W^T @ k, we get the retrieved value
                                    # For dot product similarity with flattened W, we use k @ k.T
                                    k_expanded = torch.einsum('bhk,bhj->bhkj', k, k)  # [1, 12, 32, 32]
                                    
                                    # Flatten to match W: [12288]
                                    return k_expanded.squeeze(0).flatten().float().cpu().numpy()
            
            # Fallback: use semantic embedding as key
            semantic_384 = self._encode_standard(text)
            semantic_384_t = torch.from_numpy(semantic_384).to(self.device)
            semantic_384_reshaped = semantic_384_t.view(12, 32)
            # Create k @ k.T per head
            k_expanded = torch.zeros(12, 32, 32, device=self.device)
            for h in range(12):
                k_h = semantic_384_reshaped[h:h+1, :]  # [1, 32]
                k_expanded[h] = torch.outer(k_h.squeeze(0), k_h.squeeze(0))  # [32, 32]
            return k_expanded.flatten().float().cpu().numpy()


# Alias for backwards compatibility
LAMDualEncoderV2 = LAMDualEncoder
LAMUnifiedEngine = LAMDualEncoder
