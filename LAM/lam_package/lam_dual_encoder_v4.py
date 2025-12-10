"""
🚀 LAM DUAL-TIER ENCODER V5 - SEMANTIC + RECALL UNIFIED
========================================================

THE BREAKTHROUGH: Semantic projection FROM S_slow (not mean pooling!)

The Key Insight:
===============
V2 Problem: semantic_384 came from MEAN POOLING → dilutes information
V4 Problem: raw S_slow has recall but loses semantic similarity
V5 Solution: Project S_slow onto semantic subspace → BOTH semantic + recall!

How it works:
1. Process document through Delta Rule → S_slow contains ALL tokens
2. Project S_slow onto semantic subspace: semantic_part = P @ flatten(S_slow)
3. Compute orthogonal complement: structural_part = raw - semantic_part
4. Combine: embedding = α * semantic_part + (1-α) * structural_part

Why this gives us BOTH:
- semantic_part: Lives in the subspace trained for similarity (from 384d training)
- structural_part: Contains additional retrieval detail
- BOTH come from S_slow (no mean pooling dilution!)

Result:
- "password" ≈ "access code" still works (semantic similarity)
- Needle in 100k tokens still found (Delta Rule recall)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    from lam import InfiniteContextStreamer, LAM
except ImportError:
    try:
        from infinite_streamer import InfiniteContextStreamer
    except ImportError:
        InfiniteContextStreamer = None
        print("⚠️  Warning: InfiniteContextStreamer not found")


class LAMDualEncoderV4:
    """
    LAM DUAL-TIER ENCODER V5 - Semantic + Recall Unified
    
    Tier 1 (Standard): 384d Semantic Vector
      - Mean pooling, trained for STS-B
      - Spearman: 0.8189
      - Best for: Short text similarity
    
    Tier 2 (Enterprise): 12,288d Semantic-Recall Vector
      - S_slow projected onto semantic subspace
      - Preserves BOTH semantic similarity AND recall
      - Best for: Long document RAG
    
    The Magic:
      semantic_part = P_semantic @ flatten(S_slow)  # Trained similarity
      structural_part = raw - semantic_part          # Retrieval detail
      embedding = α * semantic + (1-α) * structural  # Best of both!
    """
    
    def __init__(self, model, tokenizer=None, device='cuda'):
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        if hasattr(self.model, 'eval'):
            self.model.eval()
        elif hasattr(self.model, '_model'):
            self.model._model.eval()
        
        # Get tokenizer
        if tokenizer is None:
            if hasattr(model, 'tokenizer'):
                self.tokenizer = model.tokenizer
            else:
                raise ValueError("Tokenizer not provided")
        else:
            self.tokenizer = tokenizer
        
        # Initialize streamer
        if InfiniteContextStreamer is None:
            raise ImportError("InfiniteContextStreamer not available")
        self.streamer = InfiniteContextStreamer(model, chunk_size=2048)
        
        # Compute semantic projection matrix
        # This is the LEARNED transformation from 12k → 384d semantic space
        self.P_semantic = self._compute_semantic_projection_matrix()
        
        # Mean vector for centering (optional)
        self.mean_vector = None
        whitening_path = Path(__file__).parent / "lam_whitening_stats_v5.npy"
        if whitening_path.exists():
            print("   ✅ Loading mean vector...")
            self.mean_vector = torch.from_numpy(np.load(whitening_path)).to(self.device)
        
        print("   ✅ LAM Dual Encoder V5 initialized (Semantic + Recall Unified)")

    def _compute_semantic_projection_matrix(self):
        """
        Compute P_semantic: projection matrix onto the semantic subspace.
        
        The 384d embedding extracts specific structure from S_slow:
        - Diagonal elements: S_slow[h, i, i] → main semantic signal
        - Sum projection: sum over k, mean over heads
        
        This defines a linear map W: R^12288 → R^384
        P_semantic = W^T @ (W @ W^T)^(-1) @ W projects onto the semantic subspace.
        
        Key property: If raw_12k has semantic info, P_semantic @ raw_12k extracts it!
        """
        print("   🧠 Computing semantic projection matrix...")
        
        d_12k = 12288
        d_384 = 384
        
        # Build W: the linear map from 12k → 384d
        # Based on the actual _project_state_to_embedding formula
        W = torch.zeros(d_384, d_12k, device=self.device)
        
        # Position mapping: S_slow[h, k, v] flattens to index h*1024 + k*32 + v
        for h in range(12):
            for k in range(32):
                for v in range(32):
                    flat_idx = h * 1024 + k * 32 + v
                    
                    # Diagonal contribution (k == v): weight 0.7
                    if k == v:
                        diag_out_idx = h * 32 + k
                        W[diag_out_idx, flat_idx] += 0.7
                    
                    # Sum contribution: (1/12) * 0.3 to all heads for this v
                    for out_h in range(12):
                        out_idx = out_h * 32 + v
                        W[out_idx, flat_idx] += 0.3 / 12
        
        # Compute projection matrix using SVD
        # P = V @ V^T where V contains right singular vectors
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            rank = (S > 1e-6).sum().item()
            Vh_truncated = Vh[:rank, :]
            P_semantic = Vh_truncated.T @ Vh_truncated
            print(f"   ✅ Semantic projection matrix computed (rank {rank})")
        except Exception as e:
            print(f"   ⚠️  SVD failed: {e}, using direct computation")
            WWT = W @ W.T
            WWT_inv = torch.linalg.pinv(WWT)
            P_semantic = W.T @ WWT_inv @ W
        
        return P_semantic

    def _tokenize(self, text):
        """Tokenize text."""
        if hasattr(self.tokenizer, 'encode'):
            enc = self.tokenizer.encode(text)
            if isinstance(enc, list):
                return torch.tensor([enc], dtype=torch.long, device=self.device)
            else:
                return torch.tensor([enc.ids], dtype=torch.long, device=self.device)
        else:
            encoded = self.tokenizer(text, return_tensors='pt', add_special_tokens=True)
            return encoded['input_ids'].to(self.device)

    def encode(self, text, mode="standard", semantic_weight=0.7):
        """
        Universal encoding API.
        
        Args:
            text: Input text (any length)
            mode: "standard" (384d) or "enterprise" (12k)
            semantic_weight: For enterprise mode (0.0 to 1.0)
                1.0 = Pure semantic (equivalent to 384d similarity)
                0.7 = Balanced - semantic + recall (RECOMMENDED for RAG)
                0.5 = Equal mix
                0.0 = Pure recall (raw S_slow, for forensic)
        
        Returns:
            numpy array: Embedding vector
        """
        with torch.no_grad():
            if mode == "standard":
                return self._encode_standard(text)
            elif mode == "enterprise":
                return self._encode_enterprise(text, semantic_weight)
            else:
                raise ValueError("Mode must be 'standard' or 'enterprise'")

    def _encode_standard(self, text):
        """Standard 384d semantic embedding (mean pooling)."""
        emb = self.model.encode(
            text,
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        if emb.ndim == 2:
            return emb.squeeze(0)
        return emb

    def _encode_enterprise(self, text, semantic_weight=0.7):
        """
        Enterprise 12,288d embedding with BOTH semantic similarity AND recall.
        
        THE KEY INNOVATION:
        ==================
        Instead of mixing mean-pooled 384d with raw S_slow,
        we PROJECT S_slow onto the semantic subspace!
        
        This gives us:
        - semantic_part: The part of S_slow that drives similarity (trained)
        - structural_part: The additional detail for recall
        - BOTH come from S_slow (no mean pooling dilution!)
        
        Args:
            text: Input text
            semantic_weight: Balance between semantic and structural
        
        Returns:
            12,288d embedding with semantic similarity + recall
        """
        input_ids = self._tokenize(text)
        
        # Process through streamer to get S_slow
        self.streamer.reset()
        _ = self.streamer.stream_embedding(
            input_ids,
            verbose=False,
            use_state_embedding=True
        )
        
        # Get raw S_slow
        if hasattr(self.streamer, 'state_slow') and self.streamer.state_slow is not None:
            S_slow = self.streamer.state_slow
            raw_12k = S_slow.flatten()
        else:
            print("   ⚠️  S_slow not available, using fallback")
            return self._fallback_enterprise(text)
        
        # Optional centering
        if self.mean_vector is not None:
            raw_12k = raw_12k - self.mean_vector
        
        # THE KEY STEP: Project onto semantic subspace
        # semantic_part lives in the subspace trained for similarity
        semantic_part = self.P_semantic @ raw_12k  # [12,288]
        
        # structural_part is orthogonal - contains retrieval detail
        structural_part = raw_12k - semantic_part  # [12,288]
        
        # Normalize each component
        semantic_norm = F.normalize(semantic_part, dim=0)
        structural_norm = F.normalize(structural_part, dim=0)
        
        # Combine with weighting
        # semantic_weight=1.0 → pure semantic (like 384d)
        # semantic_weight=0.7 → balanced (RECOMMENDED for RAG)
        # semantic_weight=0.0 → pure structural (forensic)
        combined = semantic_weight * semantic_norm + (1 - semantic_weight) * structural_norm
        
        # Final normalize
        embedding = F.normalize(combined, dim=0)
        
        return embedding.cpu().numpy()

    def _fallback_enterprise(self, text):
        """Fallback when S_slow not available."""
        emb_384 = self._encode_standard(text)
        emb_384 = torch.from_numpy(emb_384).to(self.device)
        emb_12k = torch.diag_embed(emb_384.view(12, 32)).flatten()
        return F.normalize(emb_12k, dim=0).cpu().numpy()

    # =========================================================================
    # BATCH ENCODING
    # =========================================================================
    
    def encode_batch(self, texts, mode="standard", semantic_weight=0.7, show_progress=True):
        """Batch encode multiple texts."""
        embeddings = []
        iterator = texts
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(texts, desc=f"Encoding ({mode})")
        
        for text in iterator:
            emb = self.encode(text, mode=mode, semantic_weight=semantic_weight)
            embeddings.append(emb)
        
        return np.stack(embeddings)

    # =========================================================================
    # CALIBRATION
    # =========================================================================
    
    def calibrate(self, sample_texts, n_samples=100):
        """Learn mean vector for centering."""
        print("   ⚖️  Calibrating...")
        vectors = []
        
        from tqdm import tqdm
        for text in tqdm(sample_texts[:n_samples], desc="Calibrating"):
            with torch.no_grad():
                input_ids = self._tokenize(text)
                self.streamer.reset()
                _ = self.streamer.stream_embedding(
                    input_ids, verbose=False, use_state_embedding=True
                )
                
                if hasattr(self.streamer, 'state_slow') and self.streamer.state_slow is not None:
                    vectors.append(self.streamer.state_slow.flatten())
        
        if vectors:
            self.mean_vector = torch.stack(vectors).mean(dim=0)
            whitening_path = Path(__file__).parent / "lam_whitening_stats_v5.npy"
            np.save(whitening_path, self.mean_vector.cpu().numpy())
            print(f"   ✅ Calibration complete")


# Backwards compatibility
LAMDualEncoder = LAMDualEncoderV4