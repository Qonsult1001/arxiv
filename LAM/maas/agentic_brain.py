"""
🧠 AGENTIC BRAIN - Continuously Learning Personal AI

The brain that NEVER stops learning:
1. Auto-memorizes answers it gives you
2. Learns from documents you add
3. Continuously learns from the internet (when enabled)
4. Builds a cognizant personality over time
5. Always-on background learning (can be disabled)

This is the ALIVE version of Perfect Brain.

Usage:
    >>> from agentic_brain import AgenticBrain
    >>>
    >>> brain = AgenticBrain("alex")
    >>>
    >>> # Enable continuous learning (default: ON)
    >>> brain.start_learning()
    >>>
    >>> # It learns from everything you do
    >>> brain.teach("I love transformers")
    >>> answer = brain.ask("What architecture do I prefer?")
    >>> # → Returns answer AND stores the Q&A as new memory!
    >>>
    >>> # Add documents (learns from them automatically)
    >>> brain.add_document("path/to/paper.pdf")
    >>>
    >>> # Add a folder of documents
    >>> brain.add_documents_folder("path/to/papers/")
    >>>
    >>> # Learn from a URL
    >>> brain.learn_from_url("https://example.com/article")
    >>>
    >>> # The brain is always growing!
    >>> brain.stats()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import hashlib
import json
import os
import threading
import time
import queue

# PDF reading
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pypdf as PyPDF2
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

# Web scraping
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

# Import base brain
try:
    from .perfect_brain import (
        PerfectBrain, 
        QuantizedMemory, 
        EmbeddingCache,
        ContentAddressableMemory,
        InterestTracker,
        PreferenceNetwork,
    )
except ImportError:
    from perfect_brain import (
        PerfectBrain, 
        QuantizedMemory, 
        EmbeddingCache,
        ContentAddressableMemory,
        InterestTracker,
        PreferenceNetwork,
    )


# ============================================================
# AGENTIC LEARNING LOOP
# ============================================================

class LearningEvent:
    """A single learning event."""
    def __init__(
        self,
        content: str,
        source: str,
        event_type: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ):
        self.content = content
        self.source = source
        self.event_type = event_type  # "teach", "answer", "document", "web", "interaction"
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'source': self.source,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }


class ContinuousLearner:
    """
    Background thread that continuously processes learning events.
    
    This is the "always-on" learning loop that makes the brain ALIVE.
    """
    
    def __init__(self, brain: 'AgenticBrain'):
        self.brain = brain
        self.learning_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.events_processed = 0
        self.last_activity = datetime.now()
        
        # Learning settings
        self.auto_memorize_answers = True
        self.auto_learn_from_queries = True
        self.consolidation_interval = 60  # seconds
        
    def start(self):
        """Start the continuous learning loop."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.thread.start()
        print("🧠 Continuous learning started (background thread)")
    
    def stop(self):
        """Stop the continuous learning loop."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("🛑 Continuous learning stopped")
    
    def queue_event(self, event: LearningEvent):
        """Add a learning event to the queue."""
        self.learning_queue.put(event)
        self.last_activity = datetime.now()
    
    def _learning_loop(self):
        """Main learning loop (runs in background thread)."""
        last_consolidation = time.time()
        
        while self.is_running:
            try:
                # Process learning events
                try:
                    event = self.learning_queue.get(timeout=1.0)
                    self._process_event(event)
                    self.events_processed += 1
                except queue.Empty:
                    pass
                
                # Periodic consolidation
                if time.time() - last_consolidation > self.consolidation_interval:
                    self._consolidate()
                    last_consolidation = time.time()
                    
            except Exception as e:
                print(f"⚠️ Learning loop error: {e}")
    
    def _process_event(self, event: LearningEvent):
        """Process a single learning event."""
        # Teach the brain
        self.brain._internal_teach(
            event.content,
            category=event.event_type,
            source=event.source,
            auto_save=False,  # Don't save on every event
        )
    
    def _consolidate(self):
        """Periodic memory consolidation."""
        # This is where the brain "sleeps" and consolidates memories
        # In humans, this happens during sleep!
        pass  # Future: implement memory consolidation


class DocumentProcessor:
    """
    Processes documents into learnable chunks.
    
    Supports: PDF, TXT, MD, DOCX, etc.
    """
    
    @staticmethod
    def read_pdf(path: str) -> str:
        """Read text from PDF."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 or pypdf required for PDF reading")
        
        text = ""
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def read_text(path: str) -> str:
        """Read text file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    @staticmethod
    def read_document(path: str) -> str:
        """Read any supported document."""
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.pdf':
            return DocumentProcessor.read_pdf(path)
        elif ext in ['.txt', '.md', '.py', '.js', '.ts', '.json', '.yaml', '.yml']:
            return DocumentProcessor.read_text(path)
        else:
            # Try as text
            try:
                return DocumentProcessor.read_text(path)
            except:
                raise ValueError(f"Unsupported document type: {ext}")
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks


class WebLearner:
    """
    Learns from web pages and URLs.
    """
    
    @staticmethod
    def fetch_url(url: str) -> str:
        """Fetch and extract text from a URL."""
        if not WEB_AVAILABLE:
            raise ImportError("requests and beautifulsoup4 required for web learning")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (AgenticBrain/1.0; Learning Bot)'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts and styles
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        return text


# ============================================================
# AGENTIC BRAIN
# ============================================================

class AgenticBrain(PerfectBrain):
    """
    🧠 Agentic Brain - The ALIVE Personal AI
    
    Extends PerfectBrain with:
    - Continuous background learning
    - Auto-memorization of answers
    - Document ingestion
    - Web learning
    - Full cognizant personality building
    
    The brain NEVER stops learning!
    """
    
    def __init__(
        self,
        name: str = "my_brain",
        auto_learn: bool = True,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.auto_learn = auto_learn
        
        # Continuous learning
        self.learner = ContinuousLearner(self)
        
        # Learning history
        self.learning_history: List[Dict] = []
        self.qa_history: List[Dict] = []
        
        # Document storage
        self.documents: Dict[str, Dict] = {}
        
        # Settings
        self.auto_memorize_answers = True
        self.auto_learn_from_queries = True
        self.learn_from_internet = False  # Disabled by default for privacy
        
        print(f"🧠 AgenticBrain '{name}' initialized")
        print(f"   ├── Auto-learn: {auto_learn}")
        print(f"   ├── Auto-memorize answers: {self.auto_memorize_answers}")
        print(f"   └── Background learning: Ready to start")
        
        if auto_learn:
            self.start_learning()
    
    def start_learning(self):
        """Start continuous background learning."""
        self.learner.start()
    
    def stop_learning(self):
        """Stop continuous background learning."""
        self.learner.stop()
    
    def _internal_teach(
        self,
        knowledge: str,
        category: str = "general",
        source: Optional[str] = None,
        auto_save: bool = True,
    ) -> Dict:
        """Internal teach method (doesn't queue to learning loop)."""
        result = super().teach(knowledge, category=category, source=source)
        
        # Add to learning history
        self.learning_history.append({
            'content': knowledge,
            'category': category,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'memory_id': result.get('id'),
        })
        
        return result
    
    def teach(
        self,
        knowledge: str,
        category: str = "general",
        source: Optional[str] = None,
    ) -> Dict:
        """
        Teach the brain something.
        
        Goes through the learning loop for proper processing.
        """
        result = self._internal_teach(knowledge, category=category, source=source)
        
        # Also update interests
        self.interest_tracker.update_interest(
            self._embed(knowledge),
            strength=1.5  # Higher strength for explicit teaching
        )
        
        return result
    
    def ask(
        self,
        question: str,
        personalize: bool = True,
        memorize_answer: bool = None,
    ) -> str:
        """
        Ask the brain a question.
        
        Key difference from PerfectBrain:
        - Auto-memorizes the Q&A pair for future recall!
        - Learns from your queries to update interests
        
        Args:
            question: Your question
            personalize: Apply your preferences
            memorize_answer: Store Q&A as memory (default: auto_memorize_answers)
            
        Returns:
            The answer
        """
        # Get answer from parent
        answer = super().ask(question, personalize=personalize)
        
        # Determine if we should memorize
        should_memorize = memorize_answer if memorize_answer is not None else self.auto_memorize_answers
        
        if should_memorize and answer != "I don't know the answer to that yet. Teach me!":
            # Create Q&A memory
            qa_content = f"Q: {question}\nA: {answer}"
            
            # Queue to learning loop (processed in background)
            event = LearningEvent(
                content=qa_content,
                source="qa_interaction",
                event_type="interaction",
                metadata={'question': question, 'answer': answer},
            )
            self.learner.queue_event(event)
            
            # Store in QA history
            self.qa_history.append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
            })
        
        # Learn from query (update interests)
        if self.auto_learn_from_queries:
            self.interest_tracker.update_interest(
                self._embed(question),
                strength=0.3  # Lower strength for queries
            )
        
        return answer
    
    def add_document(
        self,
        path: str,
        doc_id: Optional[str] = None,
        chunk_size: int = 500,
    ) -> Dict:
        """
        Add a document to the brain's knowledge.
        
        Reads the document, chunks it, and learns from each chunk.
        
        Args:
            path: Path to document (PDF, TXT, MD, etc.)
            doc_id: Optional document ID
            chunk_size: Words per chunk
            
        Returns:
            Document info
        """
        doc_id = doc_id or os.path.basename(path)
        
        print(f"📄 Processing document: {path}")
        
        # Read document
        text = DocumentProcessor.read_document(path)
        
        # Chunk text
        chunks = DocumentProcessor.chunk_text(text, chunk_size=chunk_size)
        
        print(f"   ├── Extracted {len(text.split())} words")
        print(f"   └── Created {len(chunks)} chunks")
        
        # Learn from each chunk
        for i, chunk in enumerate(chunks):
            event = LearningEvent(
                content=chunk,
                source=f"document:{doc_id}:chunk_{i}",
                event_type="document",
                metadata={'doc_id': doc_id, 'chunk_id': i, 'path': path},
            )
            self.learner.queue_event(event)
        
        # Store document metadata
        self.documents[doc_id] = {
            'path': path,
            'chunks': len(chunks),
            'words': len(text.split()),
            'added': datetime.now().isoformat(),
        }
        
        print(f"✅ Document queued for learning: {doc_id}")
        
        return self.documents[doc_id]
    
    def add_documents_folder(
        self,
        folder_path: str,
        extensions: List[str] = ['.pdf', '.txt', '.md'],
    ) -> Dict:
        """
        Add all documents from a folder.
        
        Args:
            folder_path: Path to folder
            extensions: File extensions to process
            
        Returns:
            Summary of added documents
        """
        added = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    path = os.path.join(root, file)
                    try:
                        self.add_document(path)
                        added.append(path)
                    except Exception as e:
                        print(f"⚠️ Failed to process {path}: {e}")
        
        return {
            'folder': folder_path,
            'documents_added': len(added),
            'files': added,
        }
    
    def learn_from_url(self, url: str) -> Dict:
        """
        Learn from a web page.
        
        Args:
            url: URL to learn from
            
        Returns:
            Learning info
        """
        if not self.learn_from_internet:
            print("⚠️ Internet learning is disabled. Enable with brain.learn_from_internet = True")
            return {'success': False, 'reason': 'Internet learning disabled'}
        
        print(f"🌐 Learning from: {url}")
        
        try:
            text = WebLearner.fetch_url(url)
            chunks = DocumentProcessor.chunk_text(text, chunk_size=300)
            
            print(f"   ├── Extracted {len(text.split())} words")
            print(f"   └── Created {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                event = LearningEvent(
                    content=chunk,
                    source=f"web:{url}:chunk_{i}",
                    event_type="web",
                    metadata={'url': url, 'chunk_id': i},
                )
                self.learner.queue_event(event)
            
            return {
                'success': True,
                'url': url,
                'chunks': len(chunks),
                'words': len(text.split()),
            }
            
        except Exception as e:
            print(f"❌ Failed to learn from {url}: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_personality_summary(self) -> str:
        """
        Get a summary of the brain's learned personality.
        
        Based on interests, preferences, and learning history.
        """
        interests = self.get_interests(10)
        
        summary = f"🧠 {self.name}'s Personality:\n"
        summary += f"\n📚 Knowledge: {len(self.memory_index)} memories, {self.total_tokens} tokens\n"
        summary += f"📄 Documents: {len(self.documents)} processed\n"
        summary += f"💬 Interactions: {len(self.qa_history)} Q&A pairs\n"
        summary += f"\n🎯 Top Interests:\n"
        
        for topic, score in interests[:5]:
            summary += f"   • {topic}: {score:.3f}\n"
        
        summary += f"\n📊 Learning Stats:\n"
        summary += f"   • Events processed: {self.learner.events_processed}\n"
        summary += f"   • Cache hit rate: {self.embedding_cache.hit_rate()*100:.1f}%\n"
        summary += f"   • Background learning: {'Running' if self.learner.is_running else 'Stopped'}\n"
        
        return summary
    
    def stats(self) -> Dict:
        """Extended stats with agentic metrics."""
        base_stats = super().stats()
        base_stats.update({
            'documents': len(self.documents),
            'qa_interactions': len(self.qa_history),
            'learning_events_processed': self.learner.events_processed,
            'background_learning': self.learner.is_running,
            'last_activity': self.learner.last_activity.isoformat(),
        })
        return base_stats
    
    def save(self, path: Optional[str] = None) -> str:
        """Save brain with full agentic state."""
        path = path or f"{self.name}.said"
        
        # Save base brain
        super().save(path)
        
        # Add agentic state
        checkpoint = torch.load(path, weights_only=False)
        checkpoint['agentic_state'] = {
            'learning_history': self.learning_history[-1000:],  # Last 1000 events
            'qa_history': self.qa_history[-1000:],
            'documents': self.documents,
            'settings': {
                'auto_memorize_answers': self.auto_memorize_answers,
                'auto_learn_from_queries': self.auto_learn_from_queries,
                'learn_from_internet': self.learn_from_internet,
            },
        }
        torch.save(checkpoint, path)
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'AgenticBrain':
        """Load brain with full agentic state."""
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint.get('config', {})
        
        brain = cls(
            name=checkpoint.get('said_domain', 'loaded_brain').replace('.said', ''),
            auto_learn=False,  # Don't auto-start, let user decide
            **config
        )
        
        # Load base state
        state = checkpoint['model_state_dict']
        if 'W_quantized' in state:
            brain.memory.dequantize(
                state.pop('W_quantized'),
                state.pop('W_scale'),
                state.pop('W_zero')
            )
        brain.load_state_dict(state, strict=False)
        
        # Restore content memory
        cm = checkpoint.get('content_memory', {})
        brain.content_memory.hash_to_id = cm.get('hash_to_id', {})
        brain.content_memory.id_to_content = {
            int(k): v for k, v in cm.get('id_to_content', {}).items()
        }
        brain.content_memory.next_id = cm.get('next_id', 0)
        
        # Restore memory index
        brain.memory_index = checkpoint.get('memory_index', [])
        brain.step_count = checkpoint.get('stats', {}).get('step_count', 0)
        brain.total_tokens = checkpoint.get('stats', {}).get('tokens', 0)
        
        # Restore agentic state
        agentic = checkpoint.get('agentic_state', {})
        brain.learning_history = agentic.get('learning_history', [])
        brain.qa_history = agentic.get('qa_history', [])
        brain.documents = agentic.get('documents', {})
        
        settings = agentic.get('settings', {})
        brain.auto_memorize_answers = settings.get('auto_memorize_answers', True)
        brain.auto_learn_from_queries = settings.get('auto_learn_from_queries', True)
        brain.learn_from_internet = settings.get('learn_from_internet', False)
        
        print(f"✅ Loaded AgenticBrain from {path}")
        print(f"   Memories: {len(brain.memory_index)}")
        print(f"   Documents: {len(brain.documents)}")
        print(f"   Q&A history: {len(brain.qa_history)}")
        
        return brain


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 AGENTIC BRAIN TEST")
    print("   Continuous Learning + Document Ingestion")
    print("=" * 70)
    
    # Create brain with auto-learning
    brain = AgenticBrain("alex_agentic", auto_learn=True)
    
    # Teach it about yourself
    print("\n📝 TEACHING...")
    brain.teach("My name is Alex Chen and I research transformers")
    brain.teach("I prefer PyTorch over TensorFlow", category="preference")
    brain.teach("I work at Google on attention mechanisms", category="professional")
    
    # Ask questions (answers get auto-memorized!)
    print("\n🔍 ASKING QUESTIONS (auto-memorizes answers)...")
    answer1 = brain.ask("What is my name?")
    print(f"   Q: What is my name?\n   A: {answer1}")
    
    answer2 = brain.ask("What framework do I prefer?")
    print(f"   Q: What framework do I prefer?\n   A: {answer2}")
    
    # Add a document
    print("\n📄 ADDING DOCUMENT...")
    test_doc = "test_documents/nested_learning.pdf"
    if os.path.exists(test_doc):
        result = brain.add_document(test_doc)
        print(f"   Added: {result}")
    else:
        print(f"   Test document not found: {test_doc}")
    
    # Wait for background learning
    print("\n⏳ Waiting for background learning (2 seconds)...")
    time.sleep(2)
    
    # Show personality
    print("\n" + brain.get_personality_summary())
    
    # Show stats
    print("📊 STATS:")
    for k, v in brain.stats().items():
        print(f"   {k}: {v}")
    
    # Save
    print("\n💾 SAVING...")
    brain.save()
    
    # Show file size
    file_size = os.path.getsize("alex_agentic.said") / 1024
    print(f"   File size: {file_size:.1f} KB")
    
    # Stop learning and cleanup
    brain.stop_learning()
    os.remove("alex_agentic.said")
    
    print("\n" + "=" * 70)
    print("✅ AGENTIC BRAIN TEST COMPLETE!")
    print("=" * 70)









