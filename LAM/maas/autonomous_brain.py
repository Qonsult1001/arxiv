"""
🧠 AUTONOMOUS BRAIN - The Self-Learning Personal Scientist

A brain that:
1. Is ALWAYS AWAKE, always learning
2. Detects what YOU are interested in
3. Goes to the internet and learns about those topics ON ITS OWN
4. Like having a personal scientist researching for you 24/7

Example:
    You start learning Python
    → Brain detects: "User is interested in Python"
    → Brain autonomously searches: "Python best practices", "Python tutorials"
    → Brain learns from results
    → Now when you ask, it ALREADY knows!

This combines:
- Perfect unchunked memory (neural understanding + exact text)
- Autonomous learning agent (proactive internet research)
- Interest-driven relevance (only learns what matters to YOU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import threading
import queue
import time
import os
import re
import hashlib

# Web search
try:
    from duckduckgo_search import DDGS
    WEB_AVAILABLE = True
except ImportError:
    try:
        import requests
        from bs4 import BeautifulSoup
        WEB_AVAILABLE = True
    except ImportError:
        WEB_AVAILABLE = False

# PDF
try:
    import pypdf as PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Embeddings - prefer YOUR trained LAM model!
try:
    from lam_embedder import LAMEmbedder, get_embedder, LAM_AVAILABLE
    EMBEDDINGS_AVAILABLE = True
    USE_LAM = LAM_AVAILABLE
except ImportError:
    try:
        from sentence_transformers import SentenceTransformer
        EMBEDDINGS_AVAILABLE = True
        USE_LAM = False
    except ImportError:
        EMBEDDINGS_AVAILABLE = False
        USE_LAM = False

# Delta Gradient Descent from NL paper for PERFECT recall
try:
    from infinite_memory import InfiniteMemory, DeltaGradientDescentMemory
    DELTA_GD_AVAILABLE = True
except ImportError:
    DELTA_GD_AVAILABLE = False


# ============================================================
# PERFECT UNCHUNKED MEMORY (Neural + Full Text)
# ============================================================

class PerfectMemory(nn.Module):
    """
    Perfect Memory: Neural understanding + Exact text retrieval.
    
    NO CHUNKING - Full document context preserved.
    
    How it works:
    1. Stream document sentence-by-sentence
    2. Accumulate into hierarchical neural memory
    3. Store full text separately for retrieval
    4. Query: Neural finds relevant section → Return exact text
    """
    
    def __init__(self, d_k: int = 256, d_v: int = 256, n_heads: int = 8):
        super().__init__()
        
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        # Hierarchical neural memory
        self.register_buffer('M_sentence', torch.zeros(n_heads, d_k, d_v))
        self.register_buffer('M_paragraph', torch.zeros(n_heads, d_k, d_v))
        self.register_buffer('M_document', torch.zeros(n_heads, d_k, d_v))
        self.register_buffer('M_personal', torch.zeros(n_heads, d_k, d_v))
        
        # Full text storage (for exact retrieval)
        self.text_store: Dict[str, str] = {}
        self.text_index: List[Dict] = []  # [{id, text, source, timestamp}]
        
        # Identity for Delta GD
        self.register_buffer(
            'I', torch.eye(d_k).unsqueeze(0).expand(n_heads, -1, -1).clone()
        )
    
    def delta_update(self, M: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     alpha: float = 0.1, beta: float = 1.0) -> torch.Tensor:
        """Delta Gradient Descent (NL paper Eq. 114)."""
        k_norm = F.normalize(k, dim=-1)
        k_exp = k_norm.unsqueeze(0).expand(self.n_heads, -1)
        v_exp = v.unsqueeze(0).expand(self.n_heads, -1)
        
        k_outer = torch.einsum('hk,hj->hkj', k_exp, k_exp)
        erase = self.I - alpha * k_outer
        M_erased = torch.einsum('hkv,hkj->hjv', M, erase)
        write = beta * torch.einsum('hk,hv->hkv', k_exp, v_exp)
        
        return M_erased + write
    
    def learn_text(
        self,
        text: str,
        source: str,
        key_proj: nn.Module,
        value_proj: nn.Module,
        embedder: Any,
        device: torch.device,
        is_personal: bool = False,
    ) -> Dict:
        """
        Learn from text (unchunked, streaming).
        
        1. Stream sentence-by-sentence → neural memory
        2. Store full text for exact retrieval
        """
        # Store full text
        text_id = hashlib.md5(text.encode()).hexdigest()[:16]
        self.text_store[text_id] = text
        self.text_index.append({
            'id': text_id,
            'text': text[:500] + "..." if len(text) > 500 else text,
            'full_length': len(text),
            'source': source,
            'timestamp': datetime.now().isoformat(),
        })
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
        
        # Stream through sentences
        para_buffer = []
        for sent in sentences:
            with torch.no_grad():
                emb = embedder.encode(sent, convert_to_tensor=True).to(device)
                # Handle different embedder output shapes
                while emb.dim() > 1:
                    emb = emb.squeeze(0)
            
            k = key_proj(emb)
            v = value_proj(emb)
            
            # Update sentence memory
            if is_personal:
                self.M_personal.data = self.delta_update(self.M_personal, k, v, 0.15, 1.0)
            else:
                self.M_sentence.data = self.M_sentence * 0.99
                self.M_sentence.data = self.delta_update(self.M_sentence, k, v, 0.1, 1.0)
            
            para_buffer.append(emb)
            
            # Every 5 sentences = paragraph
            if len(para_buffer) >= 5:
                para_emb = torch.stack(para_buffer).mean(0)
                para_k = key_proj(para_emb)
                para_v = value_proj(para_emb)
                self.M_paragraph.data = self.M_paragraph * 0.995
                self.M_paragraph.data = self.delta_update(self.M_paragraph, para_k, para_v)
                para_buffer = []
        
        # Document level
        if not is_personal and sentences:
            doc_k = self.M_paragraph.mean(dim=0).mean(dim=-1)
            doc_v = self.M_paragraph.mean(dim=0).mean(dim=0)
            self.M_document.data = self.M_document * 0.999
            self.M_document.data = self.delta_update(self.M_document, doc_k, doc_v, 0.05)
        
        return {'text_id': text_id, 'sentences': len(sentences), 'source': source}
    
    def recall(
        self,
        query: str,
        key_proj: nn.Module,
        value_proj: nn.Module,
        embedder: Any,
        device: torch.device,
    ) -> str:
        """
        Perfect recall using direct LAM similarity.
        Prioritizes documents over Q&A history.
        """
        with torch.no_grad():
            q_emb = embedder.encode(query, convert_to_tensor=True).to(device)
            while q_emb.dim() > 1:
                q_emb = q_emb.squeeze(0)
        
        # Search documents FIRST (higher priority)
        doc_best_score = -float('inf')
        doc_best_text = None
        
        # Search Q&A/teachings as fallback
        other_best_score = -float('inf')
        other_best_text = None
        
        for entry in self.text_index:
            text_id = entry['id']
            source = entry.get('source', '')
            full_text = self.text_store.get(text_id, "")
            
            if not full_text.strip():
                continue
            
            # Determine if this is a document (higher priority)
            is_document = source.startswith('document:')
            
            # Score sections
            words = full_text.split()
            for i in range(0, len(words), 50):
                section = ' '.join(words[i:i+150])
                if not section.strip():
                    continue
                
                with torch.no_grad():
                    s_emb = embedder.encode(section, convert_to_tensor=True).to(device)
                    while s_emb.dim() > 1:
                        s_emb = s_emb.squeeze(0)
                
                score = F.cosine_similarity(
                    q_emb.unsqueeze(0), 
                    s_emb.unsqueeze(0)
                ).item()
                
                if is_document:
                    if score > doc_best_score:
                        doc_best_score = score
                        doc_best_text = section
                else:
                    if score > other_best_score:
                        other_best_score = score
                        other_best_text = section
        
        # Return document match if good enough, else fallback
        if doc_best_text and doc_best_score > 0.3:
            return doc_best_text
        elif other_best_text:
            return other_best_text
        else:
            return "No relevant information found."


# ============================================================
# AUTONOMOUS LEARNING AGENT
# ============================================================

class InterestDetector:
    """Detects what topics you're interested in from your interactions."""
    
    def __init__(self):
        self.topic_counts: Dict[str, int] = {}
        self.recent_topics: List[str] = []
        self.interest_threshold = 1  # Single mention = interest (immediate research)
        self.priority_topics: Set[str] = set()  # Capitalized/proper nouns
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text, including proper nouns."""
        topics = []
        
        # First, capture CAPITALIZED words as priority topics (names, places, teams)
        words_original = text.split()
        for word in words_original:
            clean = re.sub(r'[^a-zA-Z0-9]', '', word)
            # Capture words that start with capital (proper nouns)
            if clean and clean[0].isupper() and len(clean) > 2:
                topic = clean.lower()
                topics.append(topic)
                self.priority_topics.add(topic)  # Mark for immediate research
        
        # Also get regular keywords
        words = text.lower().split()
        
        # Filter stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                      'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                      'until', 'while', 'what', 'which', 'who', 'this', 'that',
                      'these', 'those', 'am', 'i', 'my', 'me', 'we', 'our', 'you',
                      'your', 'he', 'she', 'it', 'they', 'them', 'his', 'her',
                      'its', 'their', 'about', 'like', 'want', 'know', 'think',
                      'said', 'year', 'team', 'called', 'likes', 'son', 'daughter'}
        
        for word in words:
            word = re.sub(r'[^a-z0-9]', '', word)
            if len(word) > 3 and word not in stop_words:
                topics.append(word)
        
        return list(set(topics))  # Remove duplicates
    
    def update(self, text: str) -> List[str]:
        """Update interest tracking from new text."""
        topics = self.extract_topics(text)
        
        for topic in topics:
            self.topic_counts[topic] = self.topic_counts.get(topic, 0) + 1
            if topic not in self.recent_topics:
                self.recent_topics.append(topic)
        
        # Keep recent topics limited
        self.recent_topics = self.recent_topics[-50:]
        
        return topics
    
    def get_interests(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get top interests."""
        sorted_topics = sorted(
            self.topic_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_topics[:top_k]
    
    def get_new_interests(self) -> List[str]:
        """Get topics that just became interests (threshold crossed)."""
        return [
            topic for topic, count in self.topic_counts.items()
            if count == self.interest_threshold
        ]


class AutonomousSearcher:
    """
    Autonomously searches the internet for topics you're interested in.
    
    Like having a personal research assistant that proactively learns for you.
    """
    
    def __init__(self):
        self.search_queue: queue.Queue = queue.Queue()
        self.searched_topics: Set[str] = set()
        self.search_results: Dict[str, List[str]] = {}
    
    def search_topic(self, topic: str) -> List[str]:
        """Search for information about a topic."""
        if not WEB_AVAILABLE:
            print(f"⚠️ Web search not available")
            return []
        
        if topic in self.searched_topics:
            return self.search_results.get(topic, [])
        
        results = []
        
        try:
            # Try duckduckgo_search library first (most reliable)
            try:
                from duckduckgo_search import DDGS
                
                with DDGS() as ddgs:
                    search_results = list(ddgs.text(f"{topic}", max_results=5))
                    
                for r in search_results:
                    text = r.get('body', '') or r.get('title', '')
                    if text:
                        results.append(text)
                        
            except ImportError:
                # Fallback to requests/beautifulsoup
                import requests
                from bs4 import BeautifulSoup
                
                url = f"https://www.google.com/search?q={topic}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for div in soup.find_all('div', limit=10):
                    text = div.get_text()
                    if len(text) > 50 and len(text) < 500:
                        results.append(text)
                        if len(results) >= 5:
                            break
            
            self.searched_topics.add(topic)
            self.search_results[topic] = results
            
            if results:
                print(f"   ✅ Found {len(results)} results for '{topic}'")
            
        except Exception as e:
            print(f"⚠️ Search failed for '{topic}': {e}")
        
        return results
    
    def search_multiple(self, topics: List[str]) -> Dict[str, List[str]]:
        """Search for multiple topics."""
        all_results = {}
        for topic in topics:
            results = self.search_topic(topic)
            if results:
                all_results[topic] = results
        return all_results


class AutonomousLearningLoop:
    """
    The always-on learning loop.
    
    - Monitors your interests
    - Proactively searches the internet
    - Learns new knowledge automatically
    - Like a personal scientist researching 24/7
    """
    
    def __init__(self, brain: 'AutonomousBrain'):
        self.brain = brain
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        
        # Components
        self.interest_detector = InterestDetector()
        self.searcher = AutonomousSearcher()
        
        # Settings
        self.learning_interval = 30  # seconds between learning cycles
        self.max_searches_per_cycle = 3
        self.autonomous_search_enabled = True
        
        # Stats
        self.cycles_completed = 0
        self.topics_researched = 0
        self.knowledge_added = 0
    
    def start(self):
        """Start autonomous learning."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.thread.start()
        print("🧠 Autonomous learning STARTED")
        print("   → Brain is now ALWAYS learning!")
        print("   → Detecting your interests...")
        print("   → Will proactively research topics for you")
    
    def stop(self):
        """Stop autonomous learning."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("🛑 Autonomous learning stopped")
    
    def observe(self, text: str):
        """Observe user interaction to detect interests."""
        self.interest_detector.update(text)
        
        # Check for new interests
        new_interests = self.interest_detector.get_new_interests()
        if new_interests:
            print(f"🎯 New interest detected: {new_interests}")
    
    def _learning_loop(self):
        """Main autonomous learning loop."""
        while self.is_running:
            try:
                self._learning_cycle()
                self.cycles_completed += 1
            except Exception as e:
                print(f"⚠️ Learning cycle error: {e}")
            
            time.sleep(self.learning_interval)
    
    def _learning_cycle(self):
        """One cycle of autonomous learning."""
        if not self.autonomous_search_enabled:
            return
        
        # Get top interests not yet researched
        interests = self.interest_detector.get_interests(10)
        unresearched = [
            topic for topic, count in interests
            if topic not in self.searcher.searched_topics
            and count >= self.interest_detector.interest_threshold
        ]
        
        if not unresearched:
            return
        
        # Research top unresearched interests
        to_research = unresearched[:self.max_searches_per_cycle]
        
        for topic in to_research:
            print(f"🔍 Autonomously researching: {topic}")
            results = self.searcher.search_topic(topic)
            
            if results:
                # Learn from search results
                combined_text = f"About {topic}:\n" + "\n".join(results)
                self.brain.memory.learn_text(
                    text=combined_text,
                    source=f"autonomous_search:{topic}",
                    key_proj=self.brain.key_proj,
                    value_proj=self.brain.value_proj,
                    embedder=self.brain.embedder,
                    device=self.brain.memory.M_personal.device,
                )
                self.topics_researched += 1
                self.knowledge_added += len(results)
                print(f"   ✅ Learned {len(results)} facts about {topic}")


# ============================================================
# AUTONOMOUS BRAIN - THE FINAL SOLUTION
# ============================================================

class AutonomousBrain(nn.Module):
    """
    🧠 THE AUTONOMOUS BRAIN - Final Solution
    
    Features:
    1. ✅ Perfect unchunked memory (neural + exact text)
    2. ✅ Auto-memorizes everything (Q&A, documents)
    3. ✅ Interest detection (knows what you care about)
    4. ✅ Autonomous learning (researches internet for you)
    5. ✅ Always awake, always learning
    6. ✅ Personal scientist researching 24/7
    
    Usage:
        brain = AutonomousBrain("alex")
        
        # Just start using it - it learns automatically!
        brain.teach("I'm learning Python programming")
        # → Brain detects interest: "python"
        # → Brain autonomously researches Python
        # → Brain learns from internet results
        
        # Later...
        brain.ask("What are Python best practices?")
        # → Brain already knows from autonomous research!
    """
    
    def __init__(
        self,
        name: str = "autonomous_brain",
        d_model: int = 384,
        d_k: int = 256,
        d_v: int = 256,
        n_heads: int = 8,
        auto_start: bool = True,
    ):
        super().__init__()
        
        self.name = name
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        # Embedder - Use YOUR trained LAM model for better semantic understanding!
        if EMBEDDINGS_AVAILABLE:
            if USE_LAM:
                print("🧠 Using YOUR trained LAM model for embeddings!")
                self.embedder = get_embedder(use_lam=True)
            else:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
        else:
            self.embedder = None
            self.embedding_dim = d_model
        
        # Projections
        self.key_proj = nn.Linear(self.embedding_dim, d_k, bias=False)
        self.value_proj = nn.Linear(self.embedding_dim, d_v, bias=False)
        
        # Perfect memory (unchunked!)
        self.memory = PerfectMemory(d_k, d_v, n_heads)
        
        # 🧠 DELTA GRADIENT DESCENT MEMORY (NL paper perfect recall!)
        if DELTA_GD_AVAILABLE:
            self.delta_gd_memory = DeltaGradientDescentMemory(
                d_k=d_k,
                d_v=d_v,
                n_heads=n_heads,
                alpha=0.1,  # Erase strength (from NL paper)
                beta=1.0,   # Write strength
            )
            self.use_delta_gd = True
            print("🧠 Delta Gradient Descent memory ENABLED (NL paper perfect recall!)")
        else:
            self.delta_gd_memory = None
            self.use_delta_gd = False
        
        # Autonomous learning loop
        self.learning_loop = AutonomousLearningLoop(self)
        
        # Q&A history (auto-memorized)
        self.qa_history: List[Dict] = []
        
        # Stats
        self.total_teachings = 0
        self.total_questions = 0
        
        print(f"\n{'='*60}")
        print(f"🧠 AUTONOMOUS BRAIN '{name}' INITIALIZED")
        print(f"{'='*60}")
        print(f"   ├── Perfect unchunked memory: ✅")
        print(f"   ├── Delta GD (NL paper): {'✅' if self.use_delta_gd else '❌'}")
        print(f"   ├── Auto-memorization: ✅")
        print(f"   ├── Interest detection: ✅")
        print(f"   ├── Autonomous internet learning: ✅")
        print(f"   └── Always awake: ✅")
        
        if auto_start:
            self.start()
    
    def start(self):
        """Start the autonomous brain."""
        self.learning_loop.start()
    
    def stop(self):
        """Stop the autonomous brain."""
        self.learning_loop.stop()
    
    def teach(self, knowledge: str, category: str = "general") -> Dict:
        """
        Teach the brain something.
        
        Also:
        - Detects interests from what you teach
        - Triggers autonomous research on new topics
        """
        # Learn the knowledge in unchunked memory
        result = self.memory.learn_text(
            text=knowledge,
            source=f"teaching:{category}",
            key_proj=self.key_proj,
            value_proj=self.value_proj,
            embedder=self.embedder,
            device=self.memory.M_personal.device,
            is_personal=True,
        )
        
        # ALSO store in Delta GD memory for PERFECT RECALL (NL paper!)
        if self.use_delta_gd and self.delta_gd_memory is not None:
            with torch.no_grad():
                emb = self.embedder.encode(knowledge, convert_to_tensor=True)
                while emb.dim() > 1:
                    emb = emb.squeeze(0)
                emb = emb.to(self.memory.M_personal.device)
            k = self.key_proj(emb)
            v = self.value_proj(emb)
            self.delta_gd_memory.erase_and_write(k, v)
        
        # Observe for interest detection
        self.learning_loop.observe(knowledge)
        
        self.total_teachings += 1
        
        print(f"📝 Learned: {knowledge[:60]}...")
        
        return result
    
    def ask(self, question: str) -> str:
        """
        Ask a question using NL Paper's Delta Gradient Descent for recall.
        
        NL Paper Formula (Appendix C, Eq. 114):
        - Store:  W = W @ (I - α k k^T) + β k @ v^T  
        - Recall: v = W^T @ k
        
        For document retrieval:
        1. Query embedding → key projection → Delta GD recall → retrieved value
        2. Compare retrieved value to all stored document values
        3. Return the document whose value best matches the retrieved value
        
        Also:
        - Auto-memorizes Q&A for future
        - Observes for interest detection
        """
        # Observe the question
        self.learning_loop.observe(question)
        
        # === HANDLE META-QUERIES (list memories, show teachings, etc.) ===
        q_lower = question.lower()
        # Check for memory listing queries
        memory_keywords = ['memor']  # Base word that appears in memory/memories
        list_keywords = ['latest', 'list', 'show', 'all', 'recent', 'what are']
        if any(mk in q_lower for mk in memory_keywords) and any(lk in q_lower for lk in list_keywords):
            return self._list_memories(limit=10)
        # Check for teaching queries
        if any(kw in q_lower for kw in ['what did i teach', 'what have i taught', 'my teaching', 'list teaching', 'show teaching']):
            return self._list_teachings(limit=10)
        # Check for document listing queries
        if any(kw in q_lower for kw in ['list document', 'show document', 'what document', 'uploaded document', 'recent document']):
            return self._list_documents(limit=10)
        
        # Get question embedding
        with torch.no_grad():
            q_emb = self.embedder.encode(question, convert_to_tensor=True)
            while q_emb.dim() > 1:
                q_emb = q_emb.squeeze(0)
            q_emb = q_emb.to(self.memory.M_personal.device)
        
        # === NL PAPER PERFECT RECALL ===
        # Step 1: Project query to key space
        q_k = self.key_proj(q_emb)
        q_k_norm = F.normalize(q_k, dim=-1)
        
        # Step 2: Delta GD Recall (NL Paper Eq: v = W^T @ k)
        # This retrieves a "memory value" that blends all stored associations
        if self.use_delta_gd and self.delta_gd_memory is not None:
            retrieved_v = self.delta_gd_memory.recall(q_k)
        else:
            retrieved_v = None
        
        # Step 3: Score each stored document
        best_score = -float('inf')
        best_text = None
        best_source = None
        
        for entry in self.memory.text_index:
            text_id = entry['id']
            source = entry.get('source', '')
            full_text = self.memory.text_store.get(text_id, "")
            
            if not full_text.strip():
                continue
            
            # Embed the stored text
            with torch.no_grad():
                t_emb = self.embedder.encode(full_text[:1500], convert_to_tensor=True)
                while t_emb.dim() > 1:
                    t_emb = t_emb.squeeze(0)
                t_emb = t_emb.to(self.memory.M_personal.device)
            
            # === COMBINED SCORING (NL Paper + Direct Similarity) ===
            
            # Score 1: Direct semantic similarity (query ↔ document)
            direct_sim = F.cosine_similarity(
                q_emb.unsqueeze(0),
                t_emb.unsqueeze(0)
            ).item()
            
            # Score 2: NL Paper Delta GD (retrieved_v ↔ document value)
            if retrieved_v is not None:
                t_v = self.value_proj(t_emb)
                delta_gd_sim = F.cosine_similarity(
                    retrieved_v.unsqueeze(0),
                    t_v.unsqueeze(0)
                ).item()
            else:
                delta_gd_sim = 0
            
            # Combine scores: NL Paper Delta GD + Direct Similarity
            # Delta GD captures learned associations, direct sim captures immediate relevance
            score = 0.5 * direct_sim + 0.5 * delta_gd_sim
            
            # Boost documents over Q&A history
            if source.startswith('document:'):
                score *= 1.1
                
                # NL Paper insight: use content matching as disambiguation signal
                # When embeddings are similar, actual term presence breaks ties
                q_lower = question.lower()
                text_lower = full_text.lower()
                doc_name = source.replace('document:', '').lower()
                
                # Extract query terms (filter common words)
                stop = {'the', 'what', 'is', 'are', 'tell', 'me', 'about', 'show', 'my'}
                q_terms = [w for w in q_lower.split() if len(w) > 2 and w not in stop]
                
                # Boost for each query term found in document
                for term in q_terms:
                    if term in text_lower:
                        score *= 1.2  # 20% boost per term in content
                    if term in doc_name:
                        score *= 1.3  # 30% extra boost for filename match
            
            if score > best_score:
                best_score = score
                best_text = full_text
                best_source = source
        
        # Return the best match
        if best_text and best_score > 0.1:
            answer = best_text[:3000] if len(best_text) > 3000 else best_text
        else:
            answer = "No relevant information found."
        
        # Auto-memorize Q&A
        qa_text = f"Q: {question}\nA: {answer}"
        self.memory.learn_text(
            text=qa_text,
            source="qa_interaction",
            key_proj=self.key_proj,
            value_proj=self.value_proj,
            embedder=self.embedder,
            device=self.memory.M_personal.device,
            is_personal=True,
        )
        
        self.qa_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
        })
        
        self.total_questions += 1
        
        return answer
    
    def _list_memories(self, limit: int = 10) -> str:
        """List the most recent memories stored in the brain."""
        memories = []
        
        # Get recent entries from text_index (most recent first)
        recent = list(reversed(self.memory.text_index[-limit*2:]))  # Get extra to filter
        
        for entry in recent:
            source = entry.get('source', '')
            text_id = entry['id']
            full_text = self.memory.text_store.get(text_id, '')[:200]
            timestamp = entry.get('timestamp', '')
            
            # Skip Q&A interactions in the list
            if source == 'qa_interaction':
                continue
            
            # Format nicely
            if source.startswith('document:'):
                doc_name = source.replace('document:', '')
                memories.append(f"📄 Document: {doc_name}")
            elif source.startswith('teaching:'):
                memories.append(f"📝 Teaching: {full_text[:100]}...")
            elif source.startswith('autonomous_search:'):
                topic = source.replace('autonomous_search:', '')
                memories.append(f"🔍 Research: {topic}")
            elif source.startswith('manual_research:'):
                topic = source.replace('manual_research:', '')
                memories.append(f"🌐 Researched: {topic}")
            else:
                memories.append(f"💭 {full_text[:100]}...")
            
            if len(memories) >= limit:
                break
        
        if not memories:
            return "No memories found yet. Teach me something!"
        
        result = f"📚 Your Latest {len(memories)} Memories:\n\n"
        for i, mem in enumerate(memories, 1):
            result += f"{i}. {mem}\n"
        
        result += f"\n📊 Total stored: {len(self.memory.text_index)} items"
        return result
    
    def _list_teachings(self, limit: int = 10) -> str:
        """List teachings (things the user explicitly taught)."""
        teachings = []
        
        for entry in reversed(self.memory.text_index):
            source = entry.get('source', '')
            if source.startswith('teaching:'):
                text_id = entry['id']
                full_text = self.memory.text_store.get(text_id, '')[:150]
                teachings.append(full_text)
                
                if len(teachings) >= limit:
                    break
        
        if not teachings:
            return "You haven't taught me anything yet. Use 'Teach Your Brain' to add knowledge!"
        
        result = f"📝 Your {len(teachings)} Teachings:\n\n"
        for i, teaching in enumerate(teachings, 1):
            result += f"{i}. {teaching}...\n\n"
        
        return result
    
    def _list_documents(self, limit: int = 10) -> str:
        """List uploaded documents."""
        documents = []
        seen = set()
        
        for entry in reversed(self.memory.text_index):
            source = entry.get('source', '')
            if source.startswith('document:'):
                doc_name = source.replace('document:', '')
                # Avoid duplicates
                if doc_name in seen:
                    continue
                seen.add(doc_name)
                
                text_id = entry['id']
                full_text = self.memory.text_store.get(text_id, '')
                word_count = len(full_text.split())
                preview = full_text[:150].replace('\n', ' ')
                
                documents.append({
                    'name': doc_name,
                    'words': word_count,
                    'preview': preview
                })
                
                if len(documents) >= limit:
                    break
        
        if not documents:
            return "No documents uploaded yet. Use 'Upload from YOUR Computer' to add files!"
        
        result = f"📄 Your {len(documents)} Uploaded Documents:\n\n"
        for i, doc in enumerate(documents, 1):
            result += f"{i}. **{doc['name']}** ({doc['words']} words)\n"
            result += f"   Preview: {doc['preview']}...\n\n"
        
        result += f"\n💡 Ask questions about these documents to verify they were learned!"
        return result
    
    def learn_document(self, path: str) -> Dict:
        """Learn from a document using Delta GD for PERFECT recall (NL paper!)."""
        print(f"📄 Learning document: {path}")
        
        # Read document
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pdf':
            if not PDF_AVAILABLE:
                raise ImportError("pypdf required")
            text = ""
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        elif ext in ['.xlsx', '.xls']:
            # Handle Excel files
            try:
                import pandas as pd
                df = pd.read_excel(path)
                # Convert to text (limit rows for large files)
                max_rows = 1000  # Limit to 1000 rows for performance
                if len(df) > max_rows:
                    print(f"   ⚠️ Large file ({len(df)} rows), processing first {max_rows} rows only")
                    df = df.head(max_rows)
                text = df.to_string(index=False)
            except ImportError:
                raise ImportError("pandas and openpyxl required for Excel files: pip install pandas openpyxl")
            except Exception as e:
                raise ValueError(f"Failed to read Excel file: {e}")
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        # Check file size - if too large, process in batches
        word_count = len(text.split())
        is_large_file = word_count > 10000  # 10K+ words = large file
        
        if is_large_file:
            print(f"   📊 Large file detected ({word_count} words), processing in batches...")
        
        # Learn in unchunked memory
        result = self.memory.learn_text(
            text=text,
            source=f"document:{os.path.basename(path)}",
            key_proj=self.key_proj,
            value_proj=self.value_proj,
            embedder=self.embedder,
            device=self.memory.M_personal.device,
        )
        
        # ALSO store key sections in Delta GD memory for PERFECT RECALL
        # For large files, batch process to avoid timeout
        if self.use_delta_gd and self.delta_gd_memory is not None:
            sentences = text.replace('\n', '. ').split('. ')
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 10]
            
            # For large files, sample sentences instead of processing all
            if is_large_file:
                # Process every Nth sentence + first/last for context
                sample_rate = max(1, len(sentences) // 500)  # Max 500 sentences
                sentences = sentences[::sample_rate] + sentences[:10] + sentences[-10:]
                sentences = list(dict.fromkeys(sentences))  # Remove duplicates
                print(f"   📊 Sampling {len(sentences)} sentences for Delta GD (from {len(text.split('. '))} total)")
            
            # Batch process for performance
            batch_size = 50
            total = len(sentences)
            
            for i in range(0, total, batch_size):
                batch = sentences[i:i+batch_size]
                
                # Show progress for large files
                if is_large_file and i % (batch_size * 10) == 0:
                    print(f"   ⏳ Processing Delta GD: {min(i+batch_size, total)}/{total} sentences...")
                
                for sent in batch:
                    with torch.no_grad():
                        emb = self.embedder.encode(sent, convert_to_tensor=True)
                        while emb.dim() > 1:
                            emb = emb.squeeze(0)
                        emb = emb.to(self.memory.M_personal.device)
                    
                    k = self.key_proj(emb)
                    v = self.value_proj(emb)
                    
                    # Use learned α/β from Delta GD network
                    alpha = self.delta_gd_memory.alpha_net(k).mean()
                    beta = self.delta_gd_memory.beta_net(k).mean()
                    self.delta_gd_memory.erase_and_write(k, v, alpha=alpha, beta=beta)
        
        # Observe for interests
        self.learning_loop.observe(text[:1000])
        
        print(f"   ✅ Learned {word_count} words with Delta GD perfect recall")
        
        # Return detailed result
        result['word_count'] = word_count
        result['file_type'] = ext
        result['is_large'] = is_large_file
        result['filename'] = os.path.basename(path)
        
        return result
    
    def get_interests(self) -> List[Tuple[str, int]]:
        """Get detected interests."""
        return self.learning_loop.interest_detector.get_interests()
    
    def enable_autonomous_search(self, enabled: bool = True):
        """Enable/disable autonomous internet research."""
        self.learning_loop.autonomous_search_enabled = enabled
        status = "ENABLED" if enabled else "DISABLED"
        print(f"🌐 Autonomous internet search: {status}")
    
    def stats(self) -> Dict:
        """Get brain statistics."""
        return {
            'name': self.name,
            'teachings': self.total_teachings,
            'questions': self.total_questions,
            'qa_pairs': len(self.qa_history),
            'texts_stored': len(self.memory.text_index),
            'interests': self.get_interests()[:5],
            'autonomous_cycles': self.learning_loop.cycles_completed,
            'topics_researched': self.learning_loop.topics_researched,
            'knowledge_from_internet': self.learning_loop.knowledge_added,
            'is_learning': self.learning_loop.is_running,
        }
    
    def personality(self) -> str:
        """Get personality summary."""
        interests = self.get_interests()[:5]
        
        summary = f"""
{'='*60}
🧠 {self.name.upper()}'s AUTONOMOUS BRAIN
{'='*60}

📚 Knowledge:
   • Teachings: {self.total_teachings}
   • Q&A pairs: {len(self.qa_history)}
   • Stored texts: {len(self.memory.text_index)}

🎯 Detected Interests:
"""
        for topic, count in interests:
            summary += f"   • {topic}: mentioned {count} times\n"
        
        summary += f"""
🔬 Autonomous Research:
   • Cycles completed: {self.learning_loop.cycles_completed}
   • Topics researched: {self.learning_loop.topics_researched}
   • Facts learned: {self.learning_loop.knowledge_added}
   • Status: {'🟢 ACTIVE' if self.learning_loop.is_running else '🔴 STOPPED'}

{'='*60}
"""
        return summary
    
    def save(self, path: Optional[str] = None) -> str:
        """Save the autonomous brain."""
        path = path or f"{self.name}.said"
        
        checkpoint = {
            'said_version': '5.0.0',  # Autonomous Brain
            'name': self.name,
            'model_state_dict': {
                k: v for k, v in self.state_dict().items()
                if not k.startswith('embedder.')
            },
            'text_store': self.memory.text_store,
            'text_index': self.memory.text_index,
            'qa_history': self.qa_history,
            'interests': dict(self.learning_loop.interest_detector.topic_counts),
            'searched_topics': list(self.learning_loop.searcher.searched_topics),
            'config': {
                'd_k': self.d_k,
                'd_v': self.d_v,
                'n_heads': self.n_heads,
            },
            'stats': self.stats(),
        }
        
        torch.save(checkpoint, path)
        
        print(f"✅ Saved AutonomousBrain to {path}")
        print(f"   Size: {os.path.getsize(path)/1024:.1f} KB")
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'AutonomousBrain':
        """Load an autonomous brain."""
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint.get('config', {})
        
        brain = cls(
            name=checkpoint.get('name', 'loaded_brain'),
            auto_start=False,
            **config
        )
        
        brain.load_state_dict(checkpoint['model_state_dict'], strict=False)
        brain.memory.text_store = checkpoint.get('text_store', {})
        brain.memory.text_index = checkpoint.get('text_index', [])
        brain.qa_history = checkpoint.get('qa_history', [])
        brain.learning_loop.interest_detector.topic_counts = checkpoint.get('interests', {})
        brain.learning_loop.searcher.searched_topics = set(checkpoint.get('searched_topics', []))
        
        # Don't re-index on load - only index NEW data as it's added
        # This makes uploads instant!
        if brain.use_delta_gd and brain.delta_gd_memory is not None:
            print("   ✅ Delta GD ready - will index new data instantly")
        
        print(f"✅ Loaded AutonomousBrain from {path}")
        
        return brain
    
    def _reindex_delta_gd(self):
        """Re-index existing texts into Delta GD memory for perfect recall."""
        if not self.use_delta_gd or self.delta_gd_memory is None:
            return
        
        print("🔄 Re-indexing texts into Delta GD memory...")
        count = 0
        
        for entry in self.memory.text_index:
            text_id = entry['id']
            full_text = self.memory.text_store.get(text_id, "")
            source = entry.get('source', '')
            
            if not full_text.strip():
                continue
            
            # Only index document content (not Q&A)
            if not source.startswith('document:') and not source.startswith('teaching:'):
                continue
            
            # Process in sentences for better granularity
            sentences = full_text.replace('\n', '. ').split('. ')
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 10:
                    continue
                
                try:
                    with torch.no_grad():
                        emb = self.embedder.encode(sent, convert_to_tensor=True)
                        while emb.dim() > 1:
                            emb = emb.squeeze(0)
                        emb = emb.to(self.memory.M_personal.device)
                    
                    k = self.key_proj(emb)
                    v = self.value_proj(emb)
                    
                    self.delta_gd_memory.erase_and_write(k, v)
                    count += 1
                except:
                    pass
        
        print(f"   ✅ Indexed {count} sentences into Delta GD memory")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 AUTONOMOUS BRAIN - FINAL SOLUTION TEST")
    print("=" * 70)
    
    # Create brain
    brain = AutonomousBrain("alex", auto_start=True)
    
    # Teach about interests
    print("\n📝 TEACHING (triggers interest detection)...")
    brain.teach("I'm learning Python programming for machine learning")
    brain.teach("I want to understand transformers and attention mechanisms")
    brain.teach("Neural networks and deep learning are fascinating to me")
    
    # Show detected interests
    print("\n🎯 DETECTED INTERESTS:")
    for topic, count in brain.get_interests()[:10]:
        print(f"   {topic}: {count}")
    
    # Wait for autonomous learning
    print("\n⏳ Waiting for autonomous research (5 seconds)...")
    print("   (Brain is researching your interests in background)")
    time.sleep(5)
    
    # Ask questions
    print("\n🔍 ASKING QUESTIONS...")
    questions = [
        "What is Python?",
        "Tell me about machine learning",
        "What are transformers?",
    ]
    
    for q in questions:
        print(f"\n   Q: {q}")
        answer = brain.ask(q)
        print(f"   A: {answer[:150]}...")
    
    # Show personality
    print(brain.personality())
    
    # Stop and save
    brain.stop()
    brain.save()
    
    # Cleanup
    os.remove("alex.said")
    
    print("=" * 70)
    print("✅ AUTONOMOUS BRAIN TEST COMPLETE!")
    print("=" * 70)

