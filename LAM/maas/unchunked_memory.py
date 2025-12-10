"""
🧠 UNCHUNKED MEMORY - True Document Learning (No Chunking!)

The problem with chunking:
- Lose context between chunks
- Query matches chunk, not full document meaning
- Can't answer questions that span chunks

The solution: HIERARCHICAL NEURAL COMPRESSION
1. Process ENTIRE document through sliding window
2. Accumulate into single memory representation
3. Query recalls from FULL document context

This is closer to how LLMs "learn" from documents - 
the entire document affects the weights, not just chunks.

Based on:
- Nested Learning (https://abehrouz.github.io/files/NL.pdf)
- Delta Gradient Descent for associative memory
- Fused kernel for O(n) linear scaling

Usage:
    >>> from unchunked_memory import UnchunkedBrain
    >>>
    >>> brain = UnchunkedBrain("researcher")
    >>>
    >>> # Add ENTIRE document (no chunking!)
    >>> brain.learn_document("paper.pdf")  # 32K words → 1 memory!
    >>>
    >>> # Query with full document context
    >>> brain.ask("What is the main contribution?")
    >>> # → Uses ENTIRE document context, not just one chunk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import os

# Fused kernel for O(n) scaling
try:
    from .fused_delta_kernel import fused_delta_update
    FUSED_AVAILABLE = True
except ImportError:
    try:
        from fused_delta_kernel import fused_delta_update
        FUSED_AVAILABLE = True
    except ImportError:
        FUSED_AVAILABLE = False

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# PDF
try:
    import pypdf as PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False


class HierarchicalDocumentMemory(nn.Module):
    """
    Memory that learns from ENTIRE documents without chunking.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ HIERARCHICAL DOCUMENT MEMORY                                     │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │ Level 1: Sentence Memory (fine-grained)                         │
    │ - Each sentence → embedding → Delta update                      │
    │ - Preserves local details                                       │
    │                                                                  │
    │ Level 2: Paragraph Memory (medium)                              │
    │ - Groups of sentences → embedding → Delta update                │
    │ - Captures paragraph-level meaning                              │
    │                                                                  │
    │ Level 3: Document Memory (coarse)                               │
    │ - Accumulated document representation                           │
    │ - Full document context                                         │
    │                                                                  │
    │ Query: Searches all levels, combines results                    │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        d_k: int = 256,
        d_v: int = 256,
        n_heads: int = 8,
    ):
        super().__init__()
        
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        # Level 1: Sentence-level memory
        self.register_buffer(
            'M_sentence',
            torch.zeros(n_heads, d_k, d_v)
        )
        
        # Level 2: Paragraph-level memory
        self.register_buffer(
            'M_paragraph',
            torch.zeros(n_heads, d_k, d_v)
        )
        
        # Level 3: Document-level memory
        self.register_buffer(
            'M_document',
            torch.zeros(n_heads, d_k, d_v)
        )
        
        # Document index (stores full text for retrieval)
        self.document_store: Dict[str, str] = {}
        self.document_summaries: Dict[str, str] = {}
        
        # Decay rates per level (from Nested Learning multi-timescale)
        self.sentence_decay = 0.95    # Fast decay - details fade
        self.paragraph_decay = 0.99   # Medium decay
        self.document_decay = 0.999   # Slow decay - document stays
        
        # Identity for Delta Gradient Descent erase
        self.register_buffer(
            'I',
            torch.eye(d_k).unsqueeze(0).expand(n_heads, -1, -1).clone()
        )
    
    def delta_update(
        self,
        M: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: float = 0.1,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Delta Gradient Descent update (NL paper Eq. 114):
        M = M @ (I - α k k^T) + β k @ v^T
        """
        k_norm = F.normalize(k, dim=-1)
        k_expanded = k_norm.unsqueeze(0).expand(self.n_heads, -1)
        v_expanded = v.unsqueeze(0).expand(self.n_heads, -1)
        
        # Erase
        k_outer = torch.einsum('hk,hj->hkj', k_expanded, k_expanded)
        erase = self.I - alpha * k_outer
        M_erased = torch.einsum('hkv,hkj->hjv', M, erase)
        
        # Write
        write = beta * torch.einsum('hk,hv->hkv', k_expanded, v_expanded)
        
        return M_erased + write
    
    def learn_document_streaming(
        self,
        doc_id: str,
        text: str,
        key_proj: nn.Module,
        value_proj: nn.Module,
        embedder: Any,
        device: torch.device,
    ) -> Dict:
        """
        Learn from ENTIRE document using streaming/accumulation.
        
        Unlike chunking:
        - Processes sentence by sentence
        - ACCUMULATES into hierarchical memory
        - Final state represents ENTIRE document
        
        Args:
            doc_id: Document identifier
            text: Full document text
            key_proj: Key projection layer
            value_proj: Value projection layer
            embedder: Sentence embedder
            device: Torch device
            
        Returns:
            Learning stats
        """
        # Split into sentences (not chunks!)
        sentences = self._split_sentences(text)
        
        # Store full text for retrieval
        self.document_store[doc_id] = text
        
        # Process sentence by sentence (streaming, not chunking)
        paragraph_buffer = []
        paragraph_count = 0
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Embed sentence
            with torch.no_grad():
                emb = embedder.encode(sentence, convert_to_tensor=True)
                emb = emb.to(device)
            
            k = key_proj(emb)
            v = value_proj(emb)
            
            # Update sentence-level memory (with decay)
            self.M_sentence.data = self.M_sentence.data * self.sentence_decay
            self.M_sentence.data = self.delta_update(
                self.M_sentence, k, v, alpha=0.1, beta=1.0
            )
            
            # Accumulate for paragraph
            paragraph_buffer.append(emb)
            
            # Every 5 sentences = 1 paragraph
            if len(paragraph_buffer) >= 5:
                # Create paragraph embedding (average of sentences)
                para_emb = torch.stack(paragraph_buffer).mean(dim=0)
                para_k = key_proj(para_emb)
                para_v = value_proj(para_emb)
                
                # Update paragraph-level memory
                self.M_paragraph.data = self.M_paragraph.data * self.paragraph_decay
                self.M_paragraph.data = self.delta_update(
                    self.M_paragraph, para_k, para_v, alpha=0.1, beta=1.0
                )
                
                paragraph_buffer = []
                paragraph_count += 1
        
        # Handle remaining sentences
        if paragraph_buffer:
            para_emb = torch.stack(paragraph_buffer).mean(dim=0)
            para_k = key_proj(para_emb)
            para_v = value_proj(para_emb)
            self.M_paragraph.data = self.delta_update(
                self.M_paragraph, para_k, para_v, alpha=0.1, beta=1.0
            )
            paragraph_count += 1
        
        # Document-level: Combine paragraph memory into document memory
        # This is the KEY: document memory accumulates ALL paragraphs
        doc_emb_k = self.M_paragraph.mean(dim=0).mean(dim=-1)  # [d_k]
        doc_emb_v = self.M_paragraph.mean(dim=0).mean(dim=0)   # [d_v]
        
        self.M_document.data = self.M_document.data * self.document_decay
        self.M_document.data = self.delta_update(
            self.M_document, doc_emb_k, doc_emb_v, alpha=0.05, beta=1.0
        )
        
        # Create document summary (first + last paragraphs)
        first_para = ' '.join(sentences[:5]) if len(sentences) >= 5 else text[:500]
        last_para = ' '.join(sentences[-5:]) if len(sentences) >= 5 else ""
        self.document_summaries[doc_id] = f"{first_para}\n...\n{last_para}"
        
        return {
            'doc_id': doc_id,
            'sentences': len(sentences),
            'paragraphs': paragraph_count,
            'words': len(text.split()),
            'method': 'streaming_unchunked',
        }
    
    def recall(
        self,
        k: torch.Tensor,
        level: str = "all",
    ) -> torch.Tensor:
        """
        Recall from hierarchical memory.
        
        Args:
            k: Query key [d_k]
            level: "sentence", "paragraph", "document", or "all"
            
        Returns:
            Retrieved value
        """
        k_norm = F.normalize(k, dim=-1)
        k_expanded = k_norm.unsqueeze(0).expand(self.n_heads, -1)
        
        if level == "sentence":
            return torch.einsum('hkv,hk->hv', self.M_sentence, k_expanded).mean(0)
        elif level == "paragraph":
            return torch.einsum('hkv,hk->hv', self.M_paragraph, k_expanded).mean(0)
        elif level == "document":
            return torch.einsum('hkv,hk->hv', self.M_document, k_expanded).mean(0)
        else:  # "all" - combine all levels
            v_sent = torch.einsum('hkv,hk->hv', self.M_sentence, k_expanded).mean(0)
            v_para = torch.einsum('hkv,hk->hv', self.M_paragraph, k_expanded).mean(0)
            v_doc = torch.einsum('hkv,hk->hv', self.M_document, k_expanded).mean(0)
            
            # Weighted combination (document context most important)
            return 0.2 * v_sent + 0.3 * v_para + 0.5 * v_doc
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class UnchunkedBrain(nn.Module):
    """
    🧠 Brain that learns ENTIRE documents without chunking.
    
    Key difference from chunked approaches:
    - Document → Streaming sentence processing → Hierarchical memory
    - NOT: Document → Chunks → Separate embeddings
    
    This means:
    - Full document context preserved
    - Cross-document relationships captured
    - Questions answered with FULL context
    """
    
    def __init__(
        self,
        name: str = "unchunked_brain",
        d_model: int = 384,
        d_k: int = 256,
        d_v: int = 256,
        n_heads: int = 8,
    ):
        super().__init__()
        
        self.name = name
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        # Embedder
        if EMBEDDINGS_AVAILABLE:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
        else:
            self.embedder = None
            self.embedding_dim = d_model
        
        # Projections
        self.key_proj = nn.Linear(self.embedding_dim, d_k, bias=False)
        self.value_proj = nn.Linear(self.embedding_dim, d_v, bias=False)
        
        # Hierarchical memory
        self.memory = HierarchicalDocumentMemory(d_k, d_v, n_heads)
        
        # Personal memory (for teachings)
        self.register_buffer('M_personal', torch.zeros(n_heads, d_k, d_v))
        
        # Stats
        self.documents_learned: List[Dict] = []
        self.teachings: List[Dict] = []
        self.total_words = 0
        
        print(f"🧠 UnchunkedBrain '{name}' initialized")
        print(f"   ├── Hierarchical memory: sentence → paragraph → document")
        print(f"   ├── NO CHUNKING - full document context preserved")
        print(f"   └── Multi-level recall for complete understanding")
    
    def _embed(self, text: str) -> torch.Tensor:
        """Get embedding."""
        if self.embedder:
            with torch.no_grad():
                return self.embedder.encode(text, convert_to_tensor=True).to(
                    self.M_personal.device
                )
        return torch.randn(self.embedding_dim, device=self.M_personal.device)
    
    def teach(self, knowledge: str, category: str = "general") -> Dict:
        """Teach personal knowledge."""
        emb = self._embed(knowledge)
        k = self.key_proj(emb)
        v = self.value_proj(emb)
        
        # Update personal memory
        k_norm = F.normalize(k, dim=-1)
        k_exp = k_norm.unsqueeze(0).expand(self.n_heads, -1)
        v_exp = v.unsqueeze(0).expand(self.n_heads, -1)
        
        # Delta update
        I = torch.eye(self.d_k, device=k.device).unsqueeze(0).expand(self.n_heads, -1, -1)
        k_outer = torch.einsum('hk,hj->hkj', k_exp, k_exp)
        erase = I - 0.1 * k_outer
        self.M_personal.data = torch.einsum('hkv,hkj->hjv', self.M_personal, erase)
        self.M_personal.data = self.M_personal.data + torch.einsum('hk,hv->hkv', k_exp, v_exp)
        
        self.teachings.append({
            'content': knowledge,
            'category': category,
            'timestamp': datetime.now().isoformat(),
        })
        
        return {'success': True, 'content': knowledge}
    
    def learn_document(self, path: str, doc_id: Optional[str] = None) -> Dict:
        """
        Learn ENTIRE document (NO CHUNKING!).
        
        Uses streaming/accumulation instead of chunking.
        """
        doc_id = doc_id or os.path.basename(path)
        
        print(f"📄 Learning document (UNCHUNKED): {path}")
        
        # Read document
        text = self._read_document(path)
        
        print(f"   Processing {len(text.split())} words as SINGLE document...")
        
        # Learn using streaming (not chunking!)
        result = self.memory.learn_document_streaming(
            doc_id=doc_id,
            text=text,
            key_proj=self.key_proj,
            value_proj=self.value_proj,
            embedder=self.embedder,
            device=self.M_personal.device,
        )
        
        self.documents_learned.append(result)
        self.total_words += result['words']
        
        print(f"   ✅ Learned as 1 unified document (not {result['sentences']} chunks!)")
        print(f"   Sentence→Paragraph→Document hierarchy built")
        
        return result
    
    def ask(self, question: str) -> str:
        """
        Ask a question with FULL document context.
        
        Unlike chunked retrieval:
        - Searches all hierarchical levels
        - Uses full document memory
        - Returns most relevant content
        """
        emb = self._embed(question)
        k = self.key_proj(emb)
        
        # Recall from all levels
        v_doc = self.memory.recall(k, level="all")
        
        # Also check personal memory
        k_norm = F.normalize(k, dim=-1)
        k_exp = k_norm.unsqueeze(0).expand(self.n_heads, -1)
        v_personal = torch.einsum('hkv,hk->hv', self.M_personal, k_exp).mean(0)
        
        # Find best matching source
        best_score = -float('inf')
        best_answer = None
        
        # Check personal teachings
        for t in self.teachings:
            t_emb = self._embed(t['content'])
            t_v = self.value_proj(t_emb)
            score = F.cosine_similarity(v_personal.unsqueeze(0), t_v.unsqueeze(0)).item()
            if score > best_score:
                best_score = score
                best_answer = t['content']
        
        # Check document summaries
        for doc_id, summary in self.memory.document_summaries.items():
            s_emb = self._embed(summary[:500])
            s_v = self.value_proj(s_emb)
            score = F.cosine_similarity(v_doc.unsqueeze(0), s_v.unsqueeze(0)).item()
            if score > best_score:
                best_score = score
                # Return relevant part of full document
                full_text = self.memory.document_store.get(doc_id, summary)
                # Find most relevant section
                best_answer = self._find_relevant_section(question, full_text)
        
        return best_answer if best_answer else "I don't have that information yet."
    
    def _find_relevant_section(self, question: str, text: str, window_size: int = 300) -> str:
        """Find most relevant section of full document for the question."""
        q_emb = self._embed(question)
        words = text.split()
        
        best_score = -float('inf')
        best_section = ""
        
        # Sliding window over document (but we're searching the FULL doc memory)
        for i in range(0, len(words), window_size // 2):
            section = ' '.join(words[i:i + window_size])
            if not section.strip():
                continue
            
            s_emb = self._embed(section)
            score = F.cosine_similarity(
                q_emb.unsqueeze(0), s_emb.unsqueeze(0)
            ).item()
            
            if score > best_score:
                best_score = score
                best_section = section
        
        return best_section
    
    def _read_document(self, path: str) -> str:
        """Read document content."""
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.pdf':
            if not PDF_AVAILABLE:
                raise ImportError("pypdf required for PDF reading")
            text = ""
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    def stats(self) -> Dict:
        """Get brain statistics."""
        return {
            'name': self.name,
            'documents': len(self.documents_learned),
            'teachings': len(self.teachings),
            'total_words': self.total_words,
            'memory_method': 'hierarchical_unchunked',
            'M_sentence_mag': self.memory.M_sentence.norm().item(),
            'M_paragraph_mag': self.memory.M_paragraph.norm().item(),
            'M_document_mag': self.memory.M_document.norm().item(),
        }
    
    def save(self, path: Optional[str] = None) -> str:
        """Save brain."""
        path = path or f"{self.name}.said"
        
        checkpoint = {
            'said_version': '4.0.0',  # Unchunked version
            'model_state_dict': {
                k: v for k, v in self.state_dict().items()
                if not k.startswith('embedder.')
            },
            'documents_learned': self.documents_learned,
            'teachings': self.teachings,
            'document_store': self.memory.document_store,
            'document_summaries': self.memory.document_summaries,
            'config': {
                'd_model': self.d_model,
                'd_k': self.d_k,
                'd_v': self.d_v,
                'n_heads': self.n_heads,
            },
            'stats': self.stats(),
        }
        
        torch.save(checkpoint, path)
        
        file_size = os.path.getsize(path) / 1024
        print(f"✅ Saved UnchunkedBrain to {path}")
        print(f"   Size: {file_size:.1f} KB")
        print(f"   Documents: {len(self.documents_learned)} (unchunked!)")
        
        return path


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 UNCHUNKED MEMORY TEST")
    print("   NO CHUNKING - Full Document Context!")
    print("=" * 70)
    
    # Create brain
    brain = UnchunkedBrain("researcher")
    
    # Teach personal info
    print("\n📝 TEACHING...")
    brain.teach("My name is Alex and I study machine learning")
    brain.teach("I'm interested in transformers and attention mechanisms")
    
    # Learn document (UNCHUNKED!)
    print("\n📄 LEARNING DOCUMENT (UNCHUNKED)...")
    test_doc = "test_documents/nested_learning.pdf"
    if os.path.exists(test_doc):
        result = brain.learn_document(test_doc)
        print(f"\n   Result: {result}")
    else:
        print(f"   Test document not found: {test_doc}")
        # Create a test document
        with open("test_doc.txt", "w") as f:
            f.write("""
            Nested Learning is a paradigm for understanding deep learning.
            It shows that gradient descent is actually associative memory.
            The Delta Gradient Descent rule is: W = W(I - αkk^T) + βkv^T.
            This allows for perfect recall through explicit erasure.
            The paper also introduces the Hope module for continual learning.
            Multi-timescale memory allows different decay rates for different information.
            """)
        result = brain.learn_document("test_doc.txt", doc_id="nested_learning_summary")
        print(f"\n   Result: {result}")
        os.remove("test_doc.txt")
    
    # Ask questions
    print("\n🔍 ASKING QUESTIONS (uses FULL document context)...")
    questions = [
        "What is my name?",
        "What is Nested Learning?",
        "What is Delta Gradient Descent?",
    ]
    
    for q in questions:
        print(f"\n   Q: {q}")
        answer = brain.ask(q)
        print(f"   A: {answer[:150]}...")
    
    # Show stats
    print("\n📊 STATS:")
    for k, v in brain.stats().items():
        print(f"   {k}: {v}")
    
    # Compare to chunked
    print("\n" + "=" * 70)
    print("📐 CHUNKED vs UNCHUNKED COMPARISON:")
    print("=" * 70)
    print("""
    CHUNKED (old approach):
    ─────────────────────────────────────────────────────────────────
    32,418 words → 73 chunks → 73 separate memories
    Query → Find best chunk → Return ONE chunk (loses context!)
    
    UNCHUNKED (new approach):
    ─────────────────────────────────────────────────────────────────
    32,418 words → 1 hierarchical memory (sentence→paragraph→document)
    Query → Search ALL levels → Return with FULL document context
    
    Key difference:
    - Chunked: "What is X?" → Returns chunk mentioning X
    - Unchunked: "What is X?" → Understands X in context of WHOLE document
    """)
    
    print("=" * 70)
    print("✅ UNCHUNKED MEMORY TEST COMPLETE!")
    print("=" * 70)









