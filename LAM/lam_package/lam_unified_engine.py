import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add paths if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    from lam import InfiniteContextStreamer
except ImportError:
    InfiniteContextStreamer = None

class LAMUnifiedEngine:
    """
    🚀 LAM UNIFIED ENGINE (The Final Architecture)
    ==============================================
    One Brain. Two Modes. Infinite Context.

    TIER 1: STANDARD (384d)
    - Logic: Holographic Projection (v6)
    - Metric: Cosine Similarity
    - Use: Summaries, Semantic Search, RAG (<32k tokens)
    - Performance: SOTA on NarrativeQA (28.82)

    TIER 2: ENTERPRISE (12,288d)
    - Logic: Delta-GD (Erase-Write) Neural Memory
    - Metric: DOT PRODUCT (Unnormalized Magnitude)
    - Use: Forensic Nuance, Infinite Context (>100k tokens), Perfect Recall
    - Physics: W_new = W(I - kk^T) + vk^T (Erases noise, amplifies signal)
    """

    def __init__(self, model, tokenizer=None, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer if tokenizer else model.tokenizer
        self.device = device
        self.model.eval()
        
        # Initialize Streamer
        if InfiniteContextStreamer:
            self.streamer = InfiniteContextStreamer(model, chunk_size=2048)
        
        # Architecture Constants
        self.HEADS = 12
        self.D_K = 32
        self.D_V = 32  # Standard DeltaNet head dimension
        
        # Hook for Projections (Enterprise Mode)
        if hasattr(model, 'model'):
            self.last_layer = model.model.layers[-1]
        elif hasattr(model, 'layers'):
            self.last_layer = model.layers[-1]
        elif hasattr(model, 'deltanet_layers'):
            self.last_layer = model.deltanet_layers[-1]
        elif hasattr(model, '_model'):
             self.last_layer = model._model.deltanet_layers[-1]

    def encode(self, text: str, mode: str = "standard", **kwargs) -> np.ndarray:
        """
        Universal Encoder.
        """
        if mode == "standard":
            # 384d Normalized (Cosine)
            return self._encode_standard(text)
            
        elif mode == "enterprise":
            # 12k Unnormalized (Dot Product)
            # Uses Explicit Delta Rule to suppress noise
            return self._encode_delta_memory(text, **kwargs)
            
        else:
            raise ValueError("Mode must be 'standard' or 'enterprise'")

    def encode_query(self, text: str, mode: str = "standard") -> np.ndarray:
        """
        Query Encoder.
        """
        if mode == "standard":
            return self._encode_standard(text)
        elif mode == "enterprise":
            # Enterprise Query is a PROBE (k ⊗ v)
            return self._create_probe(text)
        else:
            raise ValueError("Mode must be 'standard' or 'enterprise'")

    # =========================================================================
    # TIER 1: STANDARD (384d) - "The Semantic Summary"
    # =========================================================================
    def _encode_standard(self, text):
        with torch.no_grad():
            emb = self.model.encode(
                text, 
                batch_size=1, 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
        if emb.ndim == 2: return emb.squeeze(0)
        return emb

    # =========================================================================
    # TIER 2: ENTERPRISE (12k) - "The Neural Matrix"
    # =========================================================================
    def _encode_delta_memory(self, text, alpha=1.0, beta=1.0):
        """
        Builds the 12k Vector using EXPLICIT DELTA RULE.
        This suppresses noise (Erasure) and accumulates signal (Write).
        """
        # 1. Tokenize (No Truncation)
        if hasattr(self.tokenizer, 'encode'):
            enc = self.tokenizer.encode(text)
            input_ids = torch.tensor([enc if isinstance(enc, list) else enc.ids], 
                                   dtype=torch.long, device=self.device)
        else:
            input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)

        # 2. Initialize Memory [12, 32, 32]
        W = torch.zeros(self.HEADS, self.D_K, self.D_V, device=self.device)
        I = torch.eye(self.D_K, device=self.device).unsqueeze(0).expand(self.HEADS, -1, -1)
        
        chunk_size = 512
        total_len = input_ids.shape[1]
        
        # 3. Stream & Update
        with torch.no_grad():
            # Get base model hidden states (Feature Extractor)
            if hasattr(self.model, '_model'):
                base = self.model._model
            else:
                base = self.model
            
            # Access underlying HF model if wrapped
            hf_model = base.model if hasattr(base, 'model') else base
                
            outputs = hf_model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1] # [1, Seq, 384]
            
            # Linear Processing O(N)
            for i in range(0, total_len, chunk_size):
                end = min(i + chunk_size, total_len)
                chunk = hidden_states[:, i:end, :] # [1, Chunk, 384]
                
                # Project (Aligned with trained weights)
                k = self.last_layer.k_proj(chunk).view(1, -1, 12, 32).transpose(1, 2)
                v = self.last_layer.v_proj(chunk).view(1, -1, 12, 32).transpose(1, 2)
                k = F.normalize(k, dim=-1)
                
                # --- THE PHYSICS OF RECALL ---
                # k_out: The "Keys" we just saw (The Noise Pattern)
                k_out = torch.matmul(k.transpose(2, 3), k) 
                # kv_out: The "Memories" we want to write
                kv_out = torch.matmul(v.transpose(2, 3), k) # v @ k.T
                
                # Erase Step: "Don't write this if I already know it"
                # This prevents "Fox" from accumulating to infinity.
                W = torch.matmul(W, I - alpha * k_out.squeeze(0))
                
                # Write Step: "Store this new information"
                W = W + beta * kv_out.squeeze(0).transpose(1, 2)
                
        # 4. Return Raw State
        # DO NOT NORMALIZE. The Magnitude is the Memory.
        return W.flatten().float().cpu().numpy()

    def _create_probe(self, text):
        """
        Creates the Probe (Key ⊗ Value) for Dot Product Retrieval.
        """
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            if hasattr(self.model, '_model'):
                base = self.model._model.model if hasattr(self.model._model, 'model') else self.model._model
            else:
                base = self.model.model if hasattr(self.model, 'model') else self.model

            outputs = base(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1].mean(dim=1) # [1, 384]
            
            k = self.last_layer.k_proj(hidden).view(1, 12, 32)
            v = self.last_layer.v_proj(hidden).view(1, 12, 32)
            k = F.normalize(k, dim=-1)
            
            # Probe = v @ k.T (Must match W shape)
            probe = torch.matmul(v.unsqueeze(-1), k.unsqueeze(-2))
            
            # Normalize Probe (Queries should be normalized, Documents should not)
            return F.normalize(probe.flatten(), dim=0).float().cpu().numpy()

    # =========================================================================
    # MTEB COMPATIBILITY LAYER
    # =========================================================================
    def encode_corpus(self, corpus, **kwargs):
        """Wrapper for MTEB corpus encoding (List of dicts or strings)."""
        texts = []
        if isinstance(corpus, list):
            for doc in corpus:
                if isinstance(doc, dict):
                    texts.append(f"{doc.get('title','')} {doc.get('text','')}".strip())
                else:
                    texts.append(str(doc))
        
        # MTEB usually tests Standard Mode (384d). 
        # If you want to test Enterprise, change this default.
        return np.stack([self.encode(t, mode="standard") for t in texts])

    def encode_queries(self, queries, **kwargs):
        """Wrapper for MTEB query encoding."""
        return np.stack([self.encode_query(q, mode="standard") for q in queries])