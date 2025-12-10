"""
🧠 Memory as a Service (MaaS) - .SAID Protocol

Personal AI Memory that learns forever and never forgets.

The .SAID file format stores:
- S_fast (Working Memory) - recent context, fast decay
- S_slow (Long-term Memory) - permanent knowledge, slow decay
- Learned Networks - self-modifying decay, importance, consolidation
- Memory Index - all stored memories with metadata

Architecture inspired by:
- Nested Learning (https://abehrouz.github.io/files/NL.pdf)
- Human hippocampus + neocortex memory systems
- Delta Rule associative memory

Usage:
    >>> from maas import MyBrain
    >>>
    >>> brain = MyBrain("alice")
    >>> brain.remember("I love pizza")
    >>> brain.save()
    >>>
    >>> # Later...
    >>> brain = MyBrain.load("alice")  # Loads alice.said
    >>> answer = brain.recall("What food do I like?")
    >>> print(answer)
    "I love pizza"

API Server:
    uvicorn maas.memory_api:app --host 0.0.0.0 --port 5000

.SAID Protocol:
    - Version: 1.1.0 (with Nested Learning enhancements)
    - File extension: .said
    - Portable: Can be shared, backed up, transferred
    - Domain naming: user.said or user.said.saidhome.ai
"""

from .maas_enhanced import (
    EnhancedPersonalMemoryBrain,
    DecayNetwork,
    ImportanceNetwork,
    ConsolidationNetwork,
    FUSED_KERNEL_AVAILABLE,
)

from .simple_memory_wrapper import (
    MyBrain,
    init_brain,
    remember,
    recall,
    save,
    stats,
)

from .infinite_memory import (
    InfiniteMemory,
    DeltaGradientDescentMemory,
)

from .perfect_brain import (
    PerfectBrain,
    QuantizedMemory,
    EmbeddingCache,
    ContentAddressableMemory,
    InterestTracker,
    PreferenceNetwork,
)

from .agentic_brain import (
    AgenticBrain,
    ContinuousLearner,
    LearningEvent,
    DocumentProcessor,
    WebLearner,
)

from .autonomous_brain import (
    AutonomousBrain,
    PerfectMemory,
    InterestDetector,
    AutonomousSearcher,
    AutonomousLearningLoop,
)

from .unchunked_memory import (
    UnchunkedBrain,
    HierarchicalDocumentMemory,
)

__version__ = "5.0.0"  # Autonomous Brain - Final Solution
__all__ = [
    # Core brain (semantic memory)
    "EnhancedPersonalMemoryBrain",
    "DecayNetwork",
    "ImportanceNetwork", 
    "ConsolidationNetwork",
    
    # Simple wrapper
    "MyBrain",
    "init_brain",
    "remember",
    "recall",
    "save",
    "stats",
    
    # Infinite Memory (perfect recall)
    "InfiniteMemory",
    "DeltaGradientDescentMemory",
    
    # Perfect Brain (complete system)
    "PerfectBrain",
    "QuantizedMemory",
    "EmbeddingCache",
    "ContentAddressableMemory",
    "InterestTracker",
    "PreferenceNetwork",
    
    # Feature flags
    "FUSED_KERNEL_AVAILABLE",
]

