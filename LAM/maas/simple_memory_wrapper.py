"""
Simple Memory Wrapper - Easy-to-Use Personal AI Memory (.SAID Protocol)

This wrapper makes it dead simple to use the .SAID memory system.

Just 3 commands:
  1. remember(text)  - Store a memory
  2. recall(question) - Ask a question
  3. save()/load()   - Save/load your brain

That's it!

Example Usage:
    >>> from simple_memory_wrapper import MyBrain
    >>>
    >>> brain = MyBrain()
    >>> brain.remember("I love pizza")
    >>> brain.remember("My favorite color is blue")
    >>> brain.save()
    >>>
    >>> # Later...
    >>> brain = MyBrain.load()
    >>> answer = brain.recall("What's my favorite food?")
    >>> print(answer)
    "I love pizza"

.SAID Protocol:
    Your memories are stored in .said files (e.g., "my_brain.said")
    These are portable - you can share them, back them up, or transfer them.
    
    The .said file contains:
    - S_fast (working memory) - recent context
    - S_slow (long-term memory) - permanent facts
    - memory_index - list of all stored memories
    - learned parameters - how the brain optimizes decay/importance
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Union

# Import the actual brain (relative import for package, fallback for standalone)
try:
    from .maas_enhanced import EnhancedPersonalMemoryBrain, FUSED_KERNEL_AVAILABLE
except ImportError:
    from maas_enhanced import EnhancedPersonalMemoryBrain, FUSED_KERNEL_AVAILABLE


class MyBrain:
    """
    Your Personal AI Brain - Simple Edition
    
    This is a wrapper around EnhancedPersonalMemoryBrain that makes it
    super easy to use. No need to understand the internals!
    
    Features:
    - remember(text) - Store any text as a memory
    - recall(question) - Ask questions about what you've stored
    - forget(pattern) - Remove memories matching a pattern
    - save()/load() - Persist and restore your brain
    - chat() - Interactive conversation mode
    
    Example:
        >>> brain = MyBrain("alex")
        >>> brain.remember("I was born on January 15, 1990")
        >>> brain.recall("When was I born?")
        'I was born on January 15, 1990'
    """
    
    def __init__(
        self, 
        name: str = "my_brain", 
        auto_save: bool = True, 
        domain: Optional[str] = None,
        quiet: bool = False
    ):
        """
        Create your personal AI brain (.SAID domain).
        
        Args:
            name: Name for your brain (used for save file)
            auto_save: Automatically save after each memory (default: True)
            domain: .SAID domain name (e.g., "alice.said" or "alice.said.saidhome.ai")
            quiet: Don't print initialization messages
            
        Example:
            brain = MyBrain(name="alice", domain="alice.said")
        """
        self.name = name
        self.auto_save = auto_save
        self.domain = domain or f"{name}.said"
        self.save_path = f"{name}.said"
        self.quiet = quiet
        
        # Initialize the actual brain
        if not quiet:
            print(f"🧠 Initializing your personal brain: {name}")
        
        self.brain = EnhancedPersonalMemoryBrain(
            d_k=64,
            d_v=64,
            num_heads=1,
            use_learned_decay=True,
            use_learned_importance=True,
            use_learned_consolidation=True,
        )
        
        self.memory_count = 0
        self.document_count = 0
        
        if not quiet:
            print(f"✅ Brain ready! Use brain.remember() to add memories")
    
    @classmethod
    def load(cls, name: str = "my_brain", auto_save: bool = True, quiet: bool = False):
        """
        Load a previously saved brain (.SAID file).
        
        Args:
            name: Name of the brain to load
            auto_save: Auto-save after each memory
            quiet: Don't print messages
            
        Returns:
            MyBrain instance with all memories restored
            
        Example:
            brain = MyBrain.load("alice")
        """
        save_path = f"{name}.said"
        
        if not os.path.exists(save_path):
            if not quiet:
                print(f"❌ No saved brain found at {save_path}")
                print(f"💡 Creating new brain instead...")
            return cls(name=name, auto_save=auto_save, quiet=quiet)
        
        if not quiet:
            print(f"📂 Loading brain from {save_path}...")
        
        wrapper = cls.__new__(cls)
        wrapper.name = name
        wrapper.auto_save = auto_save
        wrapper.save_path = save_path
        wrapper.domain = f"{name}.said"
        wrapper.quiet = quiet
        
        # Load the actual brain
        wrapper.brain = EnhancedPersonalMemoryBrain.load_checkpoint(save_path)
        wrapper.memory_count = len(wrapper.brain.memory_index)
        wrapper.document_count = len(wrapper.brain.documents)
        
        if not quiet:
            print(f"✅ Brain loaded!")
            print(f"   📝 {wrapper.memory_count} memories")
            print(f"   📄 {wrapper.document_count} documents")
            print(f"   💬 {wrapper.brain.total_conversation_tokens:,} conversation tokens")
        
        return wrapper
    
    def remember(self, text: str, memory_type: str = "general", quiet: bool = False) -> Dict:
        """
        Remember something.
        
        Args:
            text: What you want to remember
            memory_type: Type of memory ("personal", "preference", "professional", "fact", "general")
            quiet: Don't print messages
            
        Returns:
            Info about what was stored including:
            - id: Memory ID
            - psi: Novelty score (0=familiar, 1=novel)
            - learned_params: Decay/importance rates
            
        Example:
            brain.remember("I love pizza")
            brain.remember("My birthday is Jan 15", memory_type="personal")
        """
        result = self.brain.memorize(text, memory_type=memory_type)
        self.memory_count += 1
        
        # Compute novelty (psi) from learned importance
        # High slow_importance = familiar (low psi), Low = novel (high psi)
        learned = result.get("learned_params", {})
        slow_importance = learned.get("slow_importance", 0.5)
        psi = 1.0 - slow_importance  # Inverse of importance = novelty
        result["psi"] = psi
        
        if not quiet and not self.quiet:
            preview = text[:60] + "..." if len(text) > 60 else text
            print(f"✅ Remembered: {preview}")
            print(f"   💡 Novelty: {psi:.2f} (0=familiar, 1=novel)")
            print(f"   📊 Total memories: {self.memory_count}")
        
        # Auto-save every 10 memories
        if self.auto_save and self.memory_count % 10 == 0:
            if not quiet and not self.quiet:
                print(f"   💾 Auto-saving... ({self.memory_count} memories)")
            self.save(quiet=True)
        
        return result
    
    def recall(self, question: str, verbose: bool = True) -> str:
        """
        Ask a question and get an answer from your memories.
        
        Args:
            question: What you want to know
            verbose: Print details (default: True)
            
        Returns:
            The answer (as a string)
            
        Example:
            answer = brain.recall("What's my favorite food?")
            print(answer)  # "I love pizza"
        """
        result = self.brain.recall(question)
        
        answer = result.get('recalled_content', 'No memory found')
        confidence = result.get('confidence', 0.0)
        source = result.get('source', 'unknown')
        
        if verbose and not self.quiet:
            print(f"🔍 Question: {question}")
            preview = answer[:100] + "..." if len(answer) > 100 else answer
            print(f"💡 Answer: {preview}")
            print(f"   ✓ Confidence: {confidence:.4f}")
            print(f"   ✓ Source: {source}")
        
        return answer
    
    def recall_full(self, question: str) -> Dict:
        """
        Ask a question and get full details (not just the answer string).
        
        Args:
            question: What you want to know
            
        Returns:
            Dict with full result:
            - recalled_content: The memory content
            - confidence: Match confidence
            - source: Where it was retrieved from
            - access_count: How many times this memory was accessed
            - learned_params: The learned decay/importance for this memory
        """
        return self.brain.recall(question)
    
    def forget(self, pattern: str) -> int:
        """
        Forget memories containing a specific pattern.
        
        Args:
            pattern: Text pattern to search for (case insensitive)
            
        Returns:
            Number of memories forgotten
            
        Example:
            brain.forget("pizza")  # Forget all pizza-related memories
        """
        initial_count = len(self.brain.memory_index)
        
        # Filter out memories containing pattern
        self.brain.memory_index = [
            m for m in self.brain.memory_index
            if pattern.lower() not in m['content'].lower()
        ]
        
        forgotten = initial_count - len(self.brain.memory_index)
        self.memory_count = len(self.brain.memory_index)
        
        if forgotten > 0:
            if not self.quiet:
                print(f"🗑️  Forgot {forgotten} memories containing '{pattern}'")
            if self.auto_save:
                self.save(quiet=True)
        else:
            if not self.quiet:
                print(f"ℹ️  No memories found containing '{pattern}'")
        
        return forgotten
    
    def save(self, path: Optional[str] = None, quiet: bool = False):
        """
        Save your brain to disk (.SAID file format).
        
        Args:
            path: Where to save (uses default if not provided)
            quiet: Don't print messages
            
        Example:
            brain.save()  # Saves to "my_brain.said"
            brain.save("backup.said")  # Save to specific file
        """
        save_to = path or self.save_path
        
        if not quiet and not self.quiet:
            print(f"💾 Saving brain to {save_to}...")
        
        self.brain.save_checkpoint(save_to, domain=self.domain)
        
        if not quiet and not self.quiet:
            print(f"✅ Brain saved!")
    
    def stats(self):
        """
        Show statistics about your brain.
        
        Example:
            brain.stats()
        """
        print(f"\n{'='*60}")
        print(f"🧠 Brain Statistics: {self.name}")
        print(f"{'='*60}")
        print(f"📝 Memories stored: {self.memory_count}")
        print(f"📄 Documents stored: {self.document_count}")
        print(f"💬 Conversation tokens: {self.brain.total_conversation_tokens:,}")
        print(f"📊 S_slow magnitude: {self.brain.S_slow.norm().item():.4f}")
        print(f"   (Higher = more long-term knowledge)")
        print(f"📊 S_fast magnitude: {self.brain.S_fast.norm().item():.4f}")
        print(f"   (Higher = more recent context)")
        print(f"🔢 Processing steps: {self.brain.step_count}")
        
        if FUSED_KERNEL_AVAILABLE:
            print(f"⚡ Fused kernel: Available (use process_document_fast)")
        else:
            print(f"⚡ Fused kernel: Not available")
        
        print(f"{'='*60}\n")
    
    def list_memories(self, limit: int = 10, memory_type: Optional[str] = None):
        """
        List recent memories.
        
        Args:
            limit: How many to show
            memory_type: Filter by type (optional)
            
        Example:
            brain.list_memories(limit=5)
            brain.list_memories(memory_type="personal")
        """
        memories = self.brain.memory_index
        
        # Filter by type if specified
        if memory_type:
            memories = [m for m in memories if m.get('type') == memory_type]
        
        # Sort by step (most recent first)
        memories = sorted(memories, key=lambda m: m.get('step', 0), reverse=True)
        
        # Show recent ones
        print(f"\n📝 Recent Memories (showing {min(limit, len(memories))} of {len(memories)}):")
        print("-" * 60)
        
        for i, mem in enumerate(memories[:limit], 1):
            content = mem['content']
            mem_type = mem.get('type', 'general')
            step = mem.get('step', 0)
            access_count = mem.get('access_count', 0)
            
            # Truncate long content
            preview = content[:55] + "..." if len(content) > 55 else content
            
            print(f"{i}. [{mem_type}] {preview}")
            print(f"   Step: {step} | Accesses: {access_count}")
            
            # Show learned params if available
            learned = mem.get('learned_params', {})
            if learned:
                decay = learned.get('slow_decay', 0)
                importance = learned.get('slow_importance', 0)
                print(f"   Decay: {decay:.3f} | Importance: {importance:.3f}")
        
        print()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search through all memories by semantic similarity.
        
        Args:
            query: What to search for
            top_k: Number of results to return
            
        Returns:
            List of matching memories with scores
            
        Example:
            results = brain.search("food preferences")
            for r in results:
                print(r['content'], r['score'])
        """
        from torch.nn import functional as F
        
        K_query, _, _ = self.brain._text_to_vectors(query)
        
        results = []
        for mem in self.brain.memory_index:
            K_mem, _, _ = self.brain._text_to_vectors(mem['content'])
            score = F.cosine_similarity(
                K_query.flatten(),
                K_mem.flatten(),
                dim=0
            ).item()
            
            results.append({
                'content': mem['content'],
                'type': mem.get('type', 'general'),
                'score': score,
                'step': mem.get('step', 0),
                'access_count': mem.get('access_count', 0),
            })
        
        # Sort by score
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def process_document_fast(self, text: str, chunk_size: int = 512) -> Dict:
        """
        Process a large document using the fused Triton kernel (FAST).
        
        This is the recommended way to process large documents (5K+ tokens).
        Uses GPU-optimized kernels for 10x+ speedup.
        
        Args:
            text: The document text
            chunk_size: Tokens per chunk (default 512)
            
        Returns:
            Processing stats including tokens/second
            
        Example:
            brain.process_document_fast(long_article)
        """
        if not FUSED_KERNEL_AVAILABLE:
            print("❌ Fused kernel not available. Using standard memorize instead.")
            return self.remember(text)
        
        return self.brain.process_long_document_fused(text, chunk_size=chunk_size)
    
    def chat(self):
        """
        Start an interactive chat session.
        
        Type 'quit' or 'exit' to stop.
        Type 'stats' to see statistics.
        Type 'list' to see memories.
        Type 'search X' to search memories.
        
        Example:
            brain.chat()
        """
        print("\n" + "="*60)
        print("💬 Chat Mode - Talk to your brain!")
        print("="*60)
        print("Commands:")
        print("  'quit' or 'exit' - Stop chatting")
        print("  'stats' - Show statistics")
        print("  'list' - List recent memories")
        print("  'search <query>' - Search memories")
        print("  'save' - Save brain")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Your brain has been saved.")
                    self.save()
                    break
                
                elif user_input.lower() == 'stats':
                    self.stats()
                    continue
                
                elif user_input.lower() == 'list':
                    self.list_memories()
                    continue
                
                elif user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    results = self.search(query)
                    print(f"\n🔍 Search results for '{query}':")
                    for i, r in enumerate(results[:5], 1):
                        print(f"  {i}. [{r['type']}] {r['content'][:50]}... (score: {r['score']:.2f})")
                    print()
                    continue
                
                elif user_input.lower() == 'save':
                    self.save()
                    continue
                
                # Detect if it's a question or a statement
                question_starters = [
                    'what', 'when', 'where', 'who', 'why', 'how', 
                    'is', 'are', 'do', 'does', 'can', 'could', 'would',
                    'did', 'have', 'has', 'will', 'should', 'may', 'might'
                ]
                
                is_question = (
                    user_input.endswith('?') or 
                    any(user_input.lower().startswith(q + ' ') for q in question_starters)
                )
                
                if is_question:
                    # It's a question - recall
                    answer = self.recall(user_input, verbose=False)
                    print(f"Brain: {answer}\n")
                else:
                    # It's a statement - remember
                    self.remember(user_input)
                    print()
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Your brain has been saved.")
                self.save()
                break
                
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


# ========================================
# Simple functions for even easier usage
# ========================================

_default_brain: Optional[MyBrain] = None


def init_brain(name: str = "my_brain") -> MyBrain:
    """Initialize or load a brain (.SAID domain). Call this first!"""
    global _default_brain
    
    if os.path.exists(f"{name}.said"):
        _default_brain = MyBrain.load(name)
    else:
        _default_brain = MyBrain(name)
    
    return _default_brain


def remember(text: str, memory_type: str = "general") -> Dict:
    """Remember something."""
    global _default_brain
    
    if _default_brain is None:
        _default_brain = init_brain()
    
    return _default_brain.remember(text, memory_type=memory_type)


def recall(question: str) -> str:
    """Ask a question."""
    global _default_brain
    
    if _default_brain is None:
        _default_brain = init_brain()
    
    return _default_brain.recall(question)


def save():
    """Save the brain."""
    global _default_brain
    
    if _default_brain is None:
        print("❌ No brain to save! Call init_brain() first.")
        return
    
    _default_brain.save()


def stats():
    """Show brain statistics."""
    global _default_brain
    
    if _default_brain is None:
        print("❌ No brain initialized! Call init_brain() first.")
        return
    
    _default_brain.stats()


# ========================================
# Demo
# ========================================

if __name__ == "__main__":
    print("="*60)
    print("🧠 Simple Memory Wrapper Demo")
    print("="*60)
    
    # Create brain
    brain = MyBrain(name="demo_brain", auto_save=False)
    
    # Add some memories
    print("\n📝 Adding memories...")
    print("-" * 60)
    
    memories_to_add = [
        ("I love pizza with extra cheese", "preference"),
        ("My birthday is January 15, 1990", "personal"),
        ("I work as a software engineer", "professional"),
        ("My favorite color is blue", "preference"),
        ("I'm learning about AI and neural networks", "general"),
    ]
    
    for text, mtype in memories_to_add:
        brain.remember(text, memory_type=mtype, quiet=False)
    
    # Show stats
    brain.stats()
    
    # Ask questions
    print("\n🔍 Testing recall:")
    print("-" * 60)
    
    questions = [
        "What's my favorite food?",
        "When is my birthday?",
        "What do I do for work?",
        "What's my favorite color?",
        "What am I learning about?"
    ]
    
    for q in questions:
        answer = brain.recall(q, verbose=False)
        print(f"Q: {q}")
        print(f"A: {answer}\n")
    
    # Search test
    print("\n🔍 Testing search:")
    print("-" * 60)
    results = brain.search("work job career")
    for r in results[:3]:
        print(f"  Score {r['score']:.2f}: {r['content']}")
    
    # List memories
    print()
    brain.list_memories()
    
    # Save
    brain.save()
    
    # Show what's in the .said file
    import torch
    print("\n📂 Contents of demo_brain.said:")
    print("-" * 60)
    
    checkpoint = torch.load("demo_brain.said", weights_only=False)
    print(f"  Version: {checkpoint.get('said_version', 'unknown')}")
    print(f"  Domain: {checkpoint.get('said_domain', 'unknown')}")
    print(f"  Created: {checkpoint.get('said_created', 'unknown')}")
    print(f"  Memories: {len(checkpoint.get('memory_index', []))}")
    print(f"  Model tensors: {len(checkpoint.get('model_state_dict', {}))}")
    
    print("\n  Memory Index:")
    for i, mem in enumerate(checkpoint.get('memory_index', [])[:3]):
        print(f"    {i+1}. [{mem.get('type', 'general')}] {mem['content'][:40]}...")
        lp = mem.get('learned_params', {})
        if lp:
            print(f"       decay={lp.get('slow_decay', 0):.3f}, importance={lp.get('slow_importance', 0):.3f}")
    
    print("\n✅ Demo complete! Try:")
    print("  brain = MyBrain.load('demo_brain')")
    print("  brain.chat()  # Interactive mode")
    
    # Clean up
    os.remove("demo_brain.said")
    print("\n🧹 Cleaned up demo_brain.said")

