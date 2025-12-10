# 🧠 Memory as a Service (MaaS)

## .SAID Protocol - Where AI Memory Lives

**Learn Forever. Never Forget.**

MaaS is a personal AI memory system that stores, learns, and recalls information using a neural associative memory architecture inspired by human cognition and the [Nested Learning paper](https://abehrouz.github.io/files/NL.pdf).

---

## 📂 Files in This Folder

| File | Purpose |
|------|---------|
| `maas_enhanced.py` | **Core Brain** - Enhanced memory with learned decay/importance |
| `simple_memory_wrapper.py` | **Easy API** - Simple `remember()`, `recall()`, `save()` interface |
| `memory_api.py` | **REST API** - FastAPI server for HTTP access |
| `fused_delta_kernel.py` | **Speed** - Triton GPU kernel for 1M+ token processing |
| `benchmark_million_tokens.py` | **Benchmark** - Test infinite context processing |
| `memory_process.md` | **Docs** - Complete flow documentation |
| `architecture_clarification.md` | **Docs** - Architecture details |

---

## 🚀 Quick Start

### Simple Usage (3 Commands)

```python
from maas import MyBrain

# Create your brain
brain = MyBrain("alice")

# Remember things
brain.remember("I love pizza")
brain.remember("My birthday is January 15", memory_type="personal")
brain.remember("I work at Google", memory_type="professional")

# Recall later
answer = brain.recall("What food do I like?")
print(answer)  # "I love pizza"

# Save to .SAID file
brain.save()  # Creates alice.said

# Load later
brain = MyBrain.load("alice")  # Loads alice.said
```

### REST API

```bash
# Start the server
cd maas
uvicorn memory_api:app --host 0.0.0.0 --port 5000

# Remember something
curl -X POST http://localhost:5000/remember \
  -H "Content-Type: application/json" \
  -d '{"text": "I love pizza"}'

# Recall
curl -X POST http://localhost:5000/recall \
  -H "Content-Type: application/json" \
  -d '{"question": "What food do I like?"}'
```

---

## 💾 .SAID File Format

The `.said` file is your portable AI memory - like a trained model checkpoint but for personal memory.

```
my_brain.said (814 KB)
├── said_version: "1.1.0"
├── said_domain: "alice.said"
├── said_created: "2025-12-06T07:17:16"
│
├── memory_index: [           ← All stored memories
│     {
│       "id": 0,
│       "content": "I love pizza",
│       "type": "preference",
│       "step": 0,            ← Temporal order (oldest=0)
│       "learned_params": {   ← Self-modifying parameters
│         "fast_decay": 0.1923,
│         "slow_decay": 0.9499,
│         "slow_importance": 0.5001,
│         "consolidation_rate": 0.05
│       },
│       "access_count": 3     ← How often recalled
│     },
│     ...
│   ]
│
├── model_state_dict: {       ← Neural memory weights (NOT the embedder)
│     "S_fast": [1, 1, 64, 64],     ← Working memory matrix
│     "S_slow": [1, 1, 64, 64],     ← Long-term memory matrix
│     "decay_network.*": ...,        ← Learned decay predictor
│     "importance_network.*": ...,   ← Learned importance predictor
│     "consolidation_network.*": ... ← Learned consolidation predictor
│   }
│
├── config: {d_k: 64, d_v: 64, use_learned_decay: true, ...}
└── stats: {total_memories: 10, total_tokens: 83, s_slow_magnitude: 0.247}
```

**File Size**: ~800KB (excludes the 87MB sentence-transformers model which is loaded at runtime)

---

## 🧬 Architecture: Learn Forever, Never Forget

### Inspired by Nested Learning

From the [Nested Learning paper](https://abehrouz.github.io/files/NL.pdf):

1. **Self-Modifying Networks** - The model learns its own:
   - Decay rates (how fast to forget)
   - Importance routing (what to remember long-term)
   - Consolidation timing (when to transfer to permanent memory)

2. **Multi-Timescale Memory**:
   ```
   S_fast (Working Memory)     ← Decays 30% per step (recent context)
   S_slow (Long-term Memory)   ← Decays 0.1% per step (permanent facts)
   .SAID File (Permanent)      ← No decay (saved to disk forever)
   ```

3. **Learn on Recall** - When you ask a question:
   - If found in S_slow → Reconsolidate (strengthen the memory)
   - If found in .SAID file → Reprocess back into S_slow
   - Access count increases → Future consolidation priority higher

### Memory Flow

```
You: "I love pizza"
         ↓
    [Encode to K,V vectors]
         ↓
    [Importance Network] → Route to S_fast or S_slow?
         ↓
    [Decay Network] → How long to remember?
         ↓
    [Delta Rule Update]
         ↓
    S_fast += K @ V.T * importance_fast
    S_slow += K @ V.T * importance_slow
         ↓
    [Save to .SAID file] ← Permanent backup

Later: "What food do I like?"
         ↓
    [Encode query to Q]
         ↓
    [Search S_slow and S_fast]
         ↓
    [Consolidation Network] → Should we strengthen this?
         ↓
    [Return "I love pizza"]
         ↓
    [Update access_count] ← For future consolidation decisions
```

---

## ⚡ Speed: 1M+ Token Processing

Using the Triton fused kernel, MaaS can process documents with 1M+ tokens:

```python
from maas import MyBrain

brain = MyBrain("alice")

# Process a massive document (e.g., entire book)
result = brain.process_document_fast(
    large_document_text,
    chunk_size=512
)

print(f"Processed {result['total_tokens']} tokens")
print(f"Speed: {result['tokens_per_second']:.0f} tokens/sec")
```

**Benchmarks**:
| Tokens | Time | Speed |
|--------|------|-------|
| 100K | 3s | 33K tok/s |
| 500K | 15s | 33K tok/s |
| 1M | 30s | 33K tok/s |

---

## 🔄 What Makes .SAID Unique

### vs. RAG (Retrieval Augmented Generation)
| Feature | RAG | .SAID |
|---------|-----|-------|
| Storage | Chunk embeddings | Full neural memory |
| Learning | None | Learns on every recall |
| Forgetting | Never | Smart decay (self-modifying) |
| Context | Chunk-level | Full document compressed |
| Personalization | None | Learns your patterns |

### vs. Fine-Tuning
| Feature | Fine-Tuning | .SAID |
|---------|-------------|-------|
| Training | GPU hours | Instant |
| Updates | Retrain all | Incremental |
| Forgetting | Catastrophic | Controlled decay |
| Size | GBs | ~1MB |
| Portability | Model weights | Single .said file |

### vs. Vector Databases
| Feature | Vector DB | .SAID |
|---------|-----------|-------|
| Storage | Key-value embeddings | Associative neural memory |
| Compression | None | Full matrix compression |
| Learning | None | Self-modifying networks |
| Query | Similarity search | Neural recall + consolidation |
| Context | Individual chunks | Full document context |

---

## 🎯 The Vision: Your Personal AI Brain

```
📂 alice.said (Your Memory Domain)
├── Everything you've ever told your AI
├── Every document it's read for you
├── Learned patterns of what's important to YOU
├── Compressed into ~1MB portable file
└── Works with ANY LLM (plug it in as context)
```

**Imagine**:
1. You talk to Claude/GPT for years
2. All memories stored in `alice.said`
3. Switch to a new AI? Just load your .said file
4. Your AI knows you from day 1

**This is the goal**: A personal memory that:
- ✅ Learns forever
- ✅ Never forgets (important things)
- ✅ Self-modifies (improves over time)
- ✅ Portable (one small file)
- ✅ Private (your data, your control)

---

## 📋 TODO for Full Vision

1. **Infinite Context** (Current: 1M tokens, Goal: Unlimited)
   - Use fused kernel for streaming updates
   - Hierarchical compression for very long documents

2. **Perfect Recall** (Current: Semantic similarity, Goal: Exact retrieval)
   - Hybrid: Neural memory + explicit key-value store
   - Content-addressable memory for exact facts

3. **Model-like Compression** (Current: 800KB, Goal: More compact)
   - Quantize memory matrices (INT8/INT4)
   - Prune low-importance memories

4. **Cross-Document Learning** (Current: Per-doc, Goal: Connected knowledge graph)
   - Link related memories across documents
   - Build knowledge graph in S_slow

---

## 🔗 Related Links

- [Nested Learning Paper](https://abehrouz.github.io/files/NL.pdf) - Theoretical foundation
- [LAM (Linear Attention Memory)](../LAM_SCIENTIFIC_OVERVIEW.md) - Our semantic architecture
- [Memory Process Doc](./memory_process.md) - Detailed flow documentation

