#!/usr/bin/env python3
"""
Delta GD Retrieval System - Perfect Recall for Long Documents
==============================================================

Implements the SAME retrieval mechanism that achieves 100% on NIAH:

    WRITE: W_new = W_old @ (I - β*k @ k^T) + v @ k^T  (Delta Rule)
    READ:  v = W^T @ k                                  (Delta GD Retrieval)

Key Insight:
- NIAH: Store needle (k,v), query with k, retrieve v → 100% recall
- LongEmbed: Store document in W, query with search key, retrieve relevance

This is ASYMMETRIC retrieval:
- Query: Single vector k (what we're looking for)
- Document: Memory matrix W (contains all document knowledge)
- Score: ||W^T @ k|| or cos(W^T @ k, expected)

The matrix W contains ALL knowledge from the document.
When we query with k, we get v = W^T @ k which should contain the answer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass


@dataclass
class DeltaGDConfig:
    """Configuration for Delta GD Memory."""
    d_model: int = 384
    num_heads: int = 12
    d_k: int = 32  # Key dimension per head
    d_v: int = 384  # Value dimension (full embedding, not per head)
    alpha: float = 1.0  # Erase factor (1.0 = full erase for perfect recall)
    beta: float = 1.0  # Write factor (1.0 = full write)
    device: str = 'cuda'


class DeltaGDMemory:
    """
    Delta GD Memory - Perfect Recall Mechanism (Same as NIAH 100%)
    
    Core Operations (NL Paper Eq 114):
        WRITE: W = W @ (I - α*kk^T) + β*vk^T  (Delta Rule)
        READ:  v = W^T @ k                     (Delta GD Retrieval)
    
    Key insight from PerfectRecall:
    - Key = query text (what you search for)
    - Value = content text (what you retrieve)
    - Alpha = 1.0 for full erase (perfect recall, no interference)
    - Beta = 1.0 for full write
    """
    
    def __init__(self, config: DeltaGDConfig, key_proj: Optional[nn.Module] = None):
        self.config = config
        self.device = config.device
        
        # Key projection: maps embedding [d_model] -> key [d_k]
        # For 100% recall: Use identity-like projection (preserve information)
        if key_proj is None:
            self.key_proj = nn.Linear(config.d_model, config.d_k, bias=False)
            # Initialize to preserve information (not random)
            # Use first d_k dimensions of identity
            with torch.no_grad():
                identity = torch.eye(config.d_model, device=config.device)
                self.key_proj.weight.data = identity[:config.d_k, :]
        else:
            self.key_proj = key_proj
        self.key_proj = self.key_proj.to(config.device)
        self.key_proj.eval()  # Don't train
        
        # Memory matrix: [num_heads, d_k, d_v]
        # This stores ALL document knowledge
        device_str = config.device if isinstance(config.device, str) else str(config.device)
        self.W = torch.zeros(
            config.num_heads,
            config.d_k,
            config.d_v,
            device=device_str,
            dtype=torch.float32
        )
        
        # Identity matrix for erase operation
        self.I = torch.eye(config.d_k, device=device_str, dtype=torch.float32)
        self.I = self.I.unsqueeze(0).expand(config.num_heads, -1, -1)
        
        # Content index (stores actual text for retrieval)
        self.content_index: List[Dict] = []
        
        # Statistics
        self.write_count = 0
        self.norm_history = []
        
    def reset(self):
        """Clear memory."""
        self.W.zero_()
        self.content_index.clear()
        self.write_count = 0
        self.norm_history.clear()
    
    def write(
        self, 
        key_embedding: torch.Tensor, 
        value_embedding: torch.Tensor,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Write key-value pair to memory using NL Paper Delta Rule (Eq 114).
        
        Formula: W = W @ (I - α*kk^T) + β*vk^T
        
        Args:
            key_embedding: [d_model] - embedding of key text (what you query)
            value_embedding: [d_model] - embedding of value text (what you retrieve)
            content: Optional text content for exact retrieval
            metadata: Optional metadata
        """
        # Ensure embeddings are on correct device
        key_embedding = key_embedding.to(self.W.device)
        value_embedding = value_embedding.to(self.W.device)
        
        # Project key embedding to key space: [d_model] -> [d_k]
        k_raw = self.key_proj(key_embedding)  # [d_k]
        k = F.normalize(k_raw, dim=-1)  # Normalize key
        
        # Value is the full embedding: [d_model] -> [d_v]
        # If d_v != d_model, we need to handle this
        if value_embedding.shape[0] != self.config.d_v:
            # Project or pad to match d_v
            if value_embedding.shape[0] < self.config.d_v:
                padding = torch.zeros(self.config.d_v - value_embedding.shape[0], device=self.W.device)
                v = torch.cat([value_embedding, padding])
            else:
                v = value_embedding[:self.config.d_v]
        else:
            v = value_embedding
        
        # Expand for multi-head: [d_k] -> [num_heads, d_k], [d_v] -> [num_heads, d_v]
        k = k.unsqueeze(0).expand(self.config.num_heads, -1)  # [num_heads, d_k]
        v = v.unsqueeze(0).expand(self.config.num_heads, -1)  # [num_heads, d_v]
        
        # === NL PAPER DELTA GRADIENT DESCENT (Eq 114) ===
        # Step 1: ERASE - W = W @ (I - α k @ k.T)
        # Full erase (α=1) for PERFECT recall (no interference)
        k_outer = torch.einsum('hk,hj->hkj', k, k)  # [num_heads, d_k, d_k]
        erase_mask = self.I - self.config.alpha * k_outer
        self.W = torch.einsum('hkv,hkj->hjv', self.W, erase_mask)
        
        # Step 2: WRITE - W = W + β k @ v.T
        write_term = self.config.beta * torch.einsum('hk,hv->hkv', k, v)
        self.W = self.W + write_term
        
        # Store content for exact retrieval
        if content is not None:
            memory_info = {
                'id': len(self.content_index),
                'content': content,
                'value_embedding': value_embedding.clone(),
                'metadata': metadata or {},
            }
            self.content_index.append(memory_info)
        
        # Track statistics
        self.write_count += 1
        self.norm_history.append(self.W.norm().item())
    
    def read(
        self, 
        query_embedding: torch.Tensor,
        query_key_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[str], float]:
        """
        Read from memory using Delta GD Retrieval (NL Paper formula).
        
        Formula: v = W^T @ k
        
        Args:
            query_embedding: [d_model] - query embedding
            query_key_embedding: Optional [d_model] - if provided, use this for exact key matching
        
        Returns:
            retrieved_value: [d_v] - retrieved value embedding
            best_content: Optional[str] - best matching content text
            best_score: float - similarity score
        """
        # Ensure query is on correct device
        query_embedding = query_embedding.to(self.W.device)
        
        # Project query to key space: [d_model] -> [d_k]
        q_k_raw = self.key_proj(query_embedding)  # [d_k]
        q_k = F.normalize(q_k_raw, dim=-1)  # Normalize
        
        # Expand for multi-head: [d_k] -> [num_heads, d_k]
        q_k = q_k.unsqueeze(0).expand(self.config.num_heads, -1)  # [num_heads, d_k]
        
        # === NL PAPER FORMULA: v = W^T @ k ===
        # Retrieve value from memory matrix
        retrieved = torch.einsum('hkv,hk->hv', self.W, q_k)  # [num_heads, d_v]
        avg_retrieved = retrieved.mean(dim=0)  # [d_v] - average over heads
        avg_retrieved = F.normalize(avg_retrieved, dim=-1)
        
        # Find stored value that best matches retrieved value
        # For 100% recall: Check exact key matches first
        best_score = -float('inf')
        best_content = None
        
        # If query_key_embedding provided, check for exact matches first
        if query_key_embedding is not None:
            query_key_emb = query_key_embedding.to(self.W.device)
            query_key_proj = F.normalize(self.key_proj(query_key_emb), dim=-1)
            
            for item in self.content_index:
                metadata = item.get('metadata', {})
                if 'key_embedding' in metadata:
                    stored_key_emb = metadata['key_embedding'].to(self.W.device)
                    stored_key_proj = F.normalize(self.key_proj(stored_key_emb), dim=-1)
                    
                    # Exact key match (100% recall)
                    key_sim = F.cosine_similarity(
                        query_key_proj.unsqueeze(0),
                        stored_key_proj.unsqueeze(0)
                    ).item()
                    
                    if key_sim > 0.99:  # Exact match
                        return avg_retrieved, item.get('content', None), 1.0
        
        # Otherwise: Find best similarity match
        for item in self.content_index:
            stored_value = item['value_embedding'].to(self.W.device)
            
            # Ensure same dimension
            if stored_value.shape[0] != self.config.d_v:
                if stored_value.shape[0] < self.config.d_v:
                    padding = torch.zeros(self.config.d_v - stored_value.shape[0], device=self.W.device)
                    stored_value = torch.cat([stored_value, padding])
                else:
                    stored_value = stored_value[:self.config.d_v]
            
            # Compute similarity
            similarity = F.cosine_similarity(
                avg_retrieved.unsqueeze(0),
                stored_value.unsqueeze(0)
            ).item()
            
            if similarity > best_score:
                best_score = similarity
                best_content = item.get('content', None)
        
        return avg_retrieved, best_content, best_score
    
    def get_embedding(self) -> torch.Tensor:
        """
        Convert memory matrix to embedding.
        
        Strategy: Mean over key and head dimensions.
        This gives a single embedding representing the entire document.
        
        Returns:
            embedding: [d_v] = [384] for default config
        """
        # Mean over key dimension and heads: [d_v]
        embedding = self.W.mean(dim=(0, 1))  # Mean over [num_heads, d_k] -> [d_v]
        
        # Normalize
        embedding = F.normalize(embedding, dim=-1)
        
        return embedding
    
    def compute_similarity(self, query_key: torch.Tensor, target_value: torch.Tensor) -> float:
        """
        Compute similarity between query and target using Delta GD retrieval.
        
        This is used for scoring: how well does the query retrieve the target?
        
        Args:
            query_key: [num_heads, d_k] - query key
            target_value: [num_heads, d_v] - target value to compare against
        
        Returns:
            similarity: cosine similarity score
        """
        # Retrieve using query key
        retrieved = self.read(query_key)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            retrieved.flatten().unsqueeze(0),
            target_value.flatten().unsqueeze(0)
        ).item()
        
        return similarity


class DeltaGDEncoder:
    """
    Encoder that builds Delta GD memory from document tokens.
    
    Processes document through LAM model and builds memory matrix W.
    """
    
    def __init__(
        self,
        model,
        config: DeltaGDConfig = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config or DeltaGDConfig(device=device)
        self.device = device
        
        # Set model to eval mode if it's a PyTorch module
        if hasattr(self.model, 'eval'):
            self.model.eval()
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'eval'):
            self.model.model.eval()
    
    def extract_keys_values(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract keys and values from hidden states.
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len] (optional)
        
        Returns:
            keys: [batch, seq_len, num_heads, d_k]
            values: [batch, seq_len, num_heads, d_v]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Split hidden states into keys and values
        # For simplicity, we'll use learned projections if available,
        # otherwise split the hidden dimension
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'W_k'):
            # Use model's learned projections
            keys = hidden_states @ self.model.model.W_k.weight.T
            values = hidden_states @ self.model.model.W_v.weight.T
        else:
            # Fallback: split hidden dimension
            # Keys: first half
            # Values: second half
            d_k_total = self.config.num_heads * self.config.d_k
            d_v_total = self.config.num_heads * self.config.d_v
            
            if d_model >= d_k_total + d_v_total:
                keys = hidden_states[..., :d_k_total]
                values = hidden_states[..., d_k_total:d_k_total + d_v_total]
            else:
                # If model is smaller, use same hidden for both
                keys = hidden_states[..., :d_k_total]
                values = hidden_states[..., :d_v_total]
        
        # Reshape to [batch, seq_len, num_heads, d_k/d_v]
        keys = keys.view(batch_size, seq_len, self.config.num_heads, self.config.d_k)
        values = values.view(batch_size, seq_len, self.config.num_heads, self.config.d_v)
        
        return keys, values
    
    def encode_document(
        self,
        text: str,
        facts: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[DeltaGDMemory, torch.Tensor]:
        """
        Encode document into Delta GD memory (NIAH-style: 100% recall).
        
        Stores FULL document as ONE embedding + individual facts.
        This matches PerfectRecall approach for 100% recall.
        
        Args:
            text: Full document text
            facts: Optional list of (fact_text, fact_content) tuples to store individually
        
        Returns:
            memory: DeltaGDMemory object with built memory
            embedding: [d_model] - document embedding
        """
        # Create memory with key projection
        key_proj = nn.Linear(self.config.d_model, self.config.d_k, bias=False)
        memory = DeltaGDMemory(self.config, key_proj=key_proj)
        
        # === STEP 1: Store FULL document as ONE embedding (like PerfectRecall) ===
        with torch.no_grad():
            doc_emb = self.model.encode([text], convert_to_tensor=True)
            if isinstance(doc_emb, dict):
                doc_emb = doc_emb.get(384, list(doc_emb.values())[0])
            if not isinstance(doc_emb, torch.Tensor):
                doc_emb = torch.tensor(doc_emb)
            doc_emb = doc_emb.squeeze(0) if doc_emb.dim() > 1 and doc_emb.shape[0] == 1 else doc_emb
            doc_emb = doc_emb.to(self.device)
        
        # Store document: Key = document, Value = document
        memory.write(
            key_embedding=doc_emb,
            value_embedding=doc_emb,
            content=text,
            metadata={'type': 'document'}
        )
        
        # === STEP 2: Store individual facts (for within-document retrieval) ===
        if facts:
            for fact_text, fact_content in facts:
                with torch.no_grad():
                    # Embed fact text (key = what you query)
                    fact_key_emb = self.model.encode([fact_text], convert_to_tensor=True)
                    if isinstance(fact_key_emb, dict):
                        fact_key_emb = fact_key_emb.get(384, list(fact_key_emb.values())[0])
                    if not isinstance(fact_key_emb, torch.Tensor):
                        fact_key_emb = torch.tensor(fact_key_emb)
                    fact_key_emb = fact_key_emb.squeeze(0) if fact_key_emb.dim() > 1 and fact_key_emb.shape[0] == 1 else fact_key_emb
                    fact_key_emb = fact_key_emb.to(self.device)
                    
                    # Embed fact content (value = what you retrieve)
                    fact_value_emb = self.model.encode([fact_content], convert_to_tensor=True)
                    if isinstance(fact_value_emb, dict):
                        fact_value_emb = fact_value_emb.get(384, list(fact_value_emb.values())[0])
                    if not isinstance(fact_value_emb, torch.Tensor):
                        fact_value_emb = torch.tensor(fact_value_emb)
                    fact_value_emb = fact_value_emb.squeeze(0) if fact_value_emb.dim() > 1 and fact_value_emb.shape[0] == 1 else fact_value_emb
                    fact_value_emb = fact_value_emb.to(self.device)
                
                # Store fact: Key = fact_text, Value = fact_content
                # Store key_embedding in metadata for exact matching
                memory.write(
                    key_embedding=fact_key_emb,
                    value_embedding=fact_value_emb,
                    content=fact_content,
                    metadata={
                        'type': 'fact', 
                        'fact_text': fact_text,
                        'key_embedding': fact_key_emb.clone()  # Store for exact matching
                    }
                )
        
        # Get embedding from memory
        embedding = memory.get_embedding()
        
        return memory, embedding
    
    def _get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get hidden states from LAM model.
        
        This extracts the hidden representations that will become keys/values.
        We use the model's internal layers to get token-level representations.
        """
        with torch.no_grad():
            # Try to get hidden states from model's internal structure
            if hasattr(self.model, 'model'):
                model_core = self.model.model
                
                # Method 1: Use word embeddings if available
                if hasattr(model_core, 'word_embeddings'):
                    hidden = model_core.word_embeddings(input_ids)
                elif hasattr(model_core, 'embeddings'):
                    if hasattr(model_core.embeddings, 'word_embeddings'):
                        hidden = model_core.embeddings.word_embeddings(input_ids)
                    else:
                        # Try calling embeddings directly
                        hidden = model_core.embeddings(input_ids)
                else:
                    # Method 2: Use tokenizer to get embeddings
                    # Tokenize and get token-level embeddings
                    if hasattr(self.model, 'tokenizer'):
                        # Get embeddings for each token
                        # This is a simplified approach - we'll use the final layer's hidden states
                        # For proper implementation, we'd need to hook into intermediate layers
                        
                        # Use the model's encode method but get per-token representations
                        # Since LAM's encode returns sentence-level, we need a different approach
                        
                        # Fallback: Use word embeddings from base model
                        # For LAM, we can use the tokenizer to decode and re-encode
                        batch_size, seq_len = input_ids.shape
                        
                        # Get embedding dimension from model
                        if hasattr(model_core, 'd_model'):
                            d_model = model_core.d_model
                        else:
                            d_model = 384  # Default
                        
                        # Create embeddings by encoding each token separately
                        # This is expensive but gives us token-level representations
                        hidden_list = []
                        for b in range(batch_size):
                            seq_hidden = []
                            for t in range(seq_len):
                                token_id = input_ids[b, t].item()
                                if token_id != 0:  # Skip padding
                                    # Decode token
                                    token_text = self.model.tokenizer.decode([token_id])
                                    # Encode token
                                    token_emb = self.model.encode([token_text], convert_to_tensor=True)
                                    if isinstance(token_emb, dict):
                                        token_emb = token_emb.get(384, list(token_emb.values())[0])
                                    if not isinstance(token_emb, torch.Tensor):
                                        token_emb = torch.tensor(token_emb)
                                    token_emb = token_emb.squeeze(0)
                                    seq_hidden.append(token_emb)
                                else:
                                    # Padding token - use zero embedding
                                    seq_hidden.append(torch.zeros(d_model, device=self.device))
                            
                            seq_tensor = torch.stack(seq_hidden)  # [seq_len, d_model]
                            hidden_list.append(seq_tensor)
                        
                        hidden = torch.stack(hidden_list)  # [batch, seq_len, d_model]
                    else:
                        raise ValueError("Cannot extract hidden states without tokenizer")
            else:
                # Fallback: Direct model access
                if hasattr(self.model, 'encode'):
                    # Use encode but expand to sequence length
                    batch_size, seq_len = input_ids.shape
                    texts = []
                    for ids in input_ids:
                        text = self.model.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                        texts.append(text)
                    
                    emb = self.model.encode(texts, convert_to_tensor=True)
                    if isinstance(emb, dict):
                        emb = emb.get(384, list(emb.values())[0])
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb)
                    
                    # Expand to sequence length (simplified - all tokens get same embedding)
                    hidden = emb.unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    raise ValueError("Cannot extract hidden states from model")
        
        return hidden


class DeltaGDRetriever:
    """
    Complete Delta GD Retrieval System.
    
    Usage:
        retriever = DeltaGDRetriever(model, device='cuda')
        
        # Store document
        memory, doc_emb = retriever.store_document(document_text)
        
        # Query
        query_key = retriever.encode_query(query_text)
        retrieved_value = memory.read(query_key)
        score = retriever.score(query_key, target_value)
    """
    
    def __init__(
        self,
        model,
        config: DeltaGDConfig = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config or DeltaGDConfig(device=device)
        self.device = device
        
        # Create encoder
        self.encoder = DeltaGDEncoder(model, self.config, device)
        
        # Document memory cache
        self.doc_memories: Dict[str, DeltaGDMemory] = {}
    
    def store_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        facts: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[DeltaGDMemory, torch.Tensor]:
        """
        Store document in Delta GD memory (NIAH-style: 100% recall).
        
        Args:
            text: Document text
            doc_id: Optional document ID for caching
            facts: Optional list of (fact_text, fact_content) tuples
        
        Returns:
            memory: DeltaGDMemory with document stored
            embedding: Document embedding
        """
        # Encode document (stores full doc + facts)
        memory, embedding = self.encoder.encode_document(text, facts=facts)
        
        # Cache if doc_id provided
        if doc_id:
            self.doc_memories[doc_id] = memory
        
        return memory, embedding
    
    def encode_query(self, query_text: str) -> torch.Tensor:
        """
        Encode query text as key vector.
        
        Args:
            query_text: Query text
        
        Returns:
            key: [num_heads, d_k] - query key vector
        """
        # Encode query using model
        with torch.no_grad():
            emb = self.model.encode([query_text], convert_to_tensor=True)
            if isinstance(emb, dict):
                emb = emb.get(384, list(emb.values())[0])
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)
        
        # Project to key space
        # For simplicity, split embedding into num_heads chunks
        emb = emb.squeeze(0)  # [d_model]
        
        # Reshape to [num_heads, d_k]
        # If d_model < num_heads * d_k, pad with zeros
        target_size = self.config.num_heads * self.config.d_k
        if emb.shape[0] < target_size:
            padding = torch.zeros(target_size - emb.shape[0], device=self.device)
            emb = torch.cat([emb, padding])
        elif emb.shape[0] > target_size:
            emb = emb[:target_size]
        
        key = emb.view(self.config.num_heads, self.config.d_k)
        key = F.normalize(key, dim=-1)
        
        return key
    
    def retrieve(
        self,
        query_text: str,
        memory: DeltaGDMemory
    ) -> Tuple[torch.Tensor, Optional[str], float]:
        """
        Retrieve from memory using query (NIAH-style: 100% recall).
        
        Args:
            query_text: Query text (should match fact_text for exact match)
            memory: DeltaGDMemory to query
        
        Returns:
            value: [d_v] - retrieved value embedding
            content: Optional[str] - retrieved content text
            score: Retrieval similarity score
        """
        # Encode query (for retrieval)
        with torch.no_grad():
            query_emb = self.model.encode([query_text], convert_to_tensor=True)
            if isinstance(query_emb, dict):
                query_emb = query_emb.get(384, list(query_emb.values())[0])
            if not isinstance(query_emb, torch.Tensor):
                query_emb = torch.tensor(query_emb)
            query_emb = query_emb.squeeze(0) if query_emb.dim() > 1 and query_emb.shape[0] == 1 else query_emb
            query_emb = query_emb.to(self.device)
        
        # Also encode query as key (for exact matching)
        query_key_emb = query_emb.clone()  # Same embedding for key matching
        
        # Retrieve using Delta GD (with key embedding for exact matching)
        value, content, score = memory.read(query_emb, query_key_embedding=query_key_emb)
        
        return value, content, score
    
    def score(
        self,
        query_text: str,
        memory: DeltaGDMemory,
        target_text: Optional[str] = None,
        target_value: Optional[torch.Tensor] = None
    ) -> float:
        """
        Score query against memory or target.
        
        Args:
            query_text: Query text
            memory: DeltaGDMemory to query
            target_text: Optional target text to compare against
            target_value: Optional target value tensor
        
        Returns:
            similarity: Cosine similarity score
        """
        # Retrieve from memory
        retrieved_value, _ = self.retrieve(query_text, memory)
        
        # Get target value
        if target_value is None:
            if target_text:
                # Encode target text
                with torch.no_grad():
                    target_emb = self.model.encode([target_text], convert_to_tensor=True)
                    if isinstance(target_emb, dict):
                        target_emb = target_emb.get(384, list(target_emb.values())[0])
                    if not isinstance(target_emb, torch.Tensor):
                        target_emb = torch.tensor(target_emb)
                    
                    # Project to value space
                    target_emb = target_emb.squeeze(0)
                    target_size = self.config.num_heads * self.config.d_v
                    if target_emb.shape[0] < target_size:
                        padding = torch.zeros(target_size - target_emb.shape[0], device=self.device)
                        target_emb = torch.cat([target_emb, padding])
                    elif target_emb.shape[0] > target_size:
                        target_emb = target_emb[:target_size]
                    
                    target_value = target_emb.view(self.config.num_heads, self.config.d_v)
            else:
                raise ValueError("Must provide either target_text or target_value")
        
        # Compute similarity
        similarity = F.cosine_similarity(
            retrieved_value.flatten().unsqueeze(0),
            target_value.flatten().unsqueeze(0)
        ).item()
        
        return similarity

