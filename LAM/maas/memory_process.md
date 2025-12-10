# Complete Memory Process Documentation

## What This Memory System Does

### Simple Explanation

Imagine your brain has two types of memory:
- **Short-term (S_fast)**: What you talked about today (fades quickly)
- **Long-term (S_slow)**: Important facts that stick around forever

This system does the same thing - it remembers conversations and personal facts, stores them forever in a `.pt` file, and recalls them when you ask.

## The Complete Flow: What Happens When You Use It

### 1. Starting Fresh (Initialize)

```python
brain = PersonalMemoryBrain()
```

**What happens internally:**

```
Step 1: Create empty memory storage
  ├─ S_fast  = empty matrix [1 x 1 x 64 x 64]  ← Recent memories
  ├─ S_slow  = empty matrix [1 x 1 x 64 x 64]  ← Long-term knowledge
  └─ memory_index = []                          ← List of all memories

Step 2: Load pre-trained text encoder
  ├─ Try LAM model first (if available)
  ├─ Fallback to sentence-transformers
  └─ This converts text → numbers (embeddings)

Step 3: Ready to use!
  └─ Total time: ~2-5 seconds
```

**Key Point:** The encoder is pre-trained and frozen. It never changes. Only S_fast/S_slow accumulate your memories.

---

### 2. Storing a Memory (Memorize)

```python
brain.memorize("I live in Tokyo")
```

**Complete Internal Flow:**

```
Input: "I live in Tokyo"
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 1: Convert Text to Embeddings              │
│                                                  │
│ "I live in Tokyo"                               │
│         ↓                                        │
│ LAM Encoder (frozen, no training)               │
│         ↓                                        │
│ embedding = [0.234, -0.891, ..., 0.456]        │
│             (384 numbers)                       │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 2: Project to Key/Value Vectors            │
│                                                  │
│ embedding [384 dims]                            │
│         ↓                                        │
│ Linear projections (frozen weights)             │
│         ↓                                        │
│ K = Key vector [64 dims]                        │
│ V = Value vector [64 dims]                      │
│ Q = Query vector [64 dims]                      │
│ U = Target vector [64 dims]                     │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 3: Compute Resonance Flux (Novelty)        │
│                                                  │
│ Check: How novel is this information?           │
│                                                  │
│ current_recall = K @ S_slow                     │
│ similarity = cosine(current_recall, U)          │
│ psi (ψ) = 1 - similarity                        │
│                                                  │
│ Example:                                        │
│ - First time: ψ = 0.9 (very novel!)            │
│ - 10th time: ψ = 0.2 (familiar)                │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 4: Delta Rule Update (Memory Storage)      │
│                                                  │
│ Apply decay:                                    │
│   S_fast_decayed = S_fast * 0.7                │
│   S_slow_decayed = S_slow * 0.999              │
│                                                  │
│ Add new memory:                                 │
│   update = K @ U.T  (outer product)            │
│                                                  │
│ Modulate by novelty (ψ):                       │
│   - High ψ (novel) → more to S_fast            │
│   - Low ψ (familiar) → more to S_slow          │
│                                                  │
│ Final update:                                   │
│   S_fast = S_fast_decayed + ψ * update         │
│   S_slow = S_slow_decayed + (1-ψ) * update     │
│                                                  │
│ NO BACKPROP, NO GRADIENTS - just math!         │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 5: Store Metadata                          │
│                                                  │
│ memory_index.append({                           │
│   "id": 0,                                      │
│   "content": "I live in Tokyo",                │
│   "type": "general",                           │
│   "timestamp": "2025-11-07T15:30:00",          │
│   "psi": 0.85,                                 │
│   "s_slow_magnitude": 0.13,                    │
│   "step": 1                                    │
│ })                                              │
│                                                  │
│ Note: We DON'T store embeddings here!          │
│ Embeddings are in S_slow - recompute on recall │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 6: Increment Counters                      │
│                                                  │
│ step_count += 1                                 │
│ write_count += 1                                │
│ total_conversation_tokens += ~5                 │
└─────────────────────────────────────────────────┘

Output: {
  "success": True,
  "s_slow_magnitude": 0.13,
  "psi": 0.85,
  "memory_id": 0
}

Time: ~0.001 seconds
```

**What This Means:**
- Your memory "I live in Tokyo" is now stored in S_slow
- It's available for recall immediately
- It decays very slowly (0.1% per new memory)
- Magnitude 0.13 means it's the most recent memory

---

### 3. Recalling a Memory (Recall)

```python
result = brain.recall("Where do I live?")
```

**Complete Internal Flow:**

```
Input: "Where do I live?"
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 1: Convert Query to Embedding              │
│                                                  │
│ "Where do I live?"                              │
│         ↓                                        │
│ LAM Encoder (frozen)                            │
│         ↓                                        │
│ query_embedding = [0.123, -0.456, ...]         │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 2: Project Query to Key Vector             │
│                                                  │
│ query_embedding [384]                           │
│         ↓                                        │
│ Linear projection                               │
│         ↓                                        │
│ Q_query = [64 dims]                             │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 3: Compute Read Resonance                  │
│                                                  │
│ How well does query match stored memories?      │
│                                                  │
│ score_fast = Q @ S_fast @ S_fast.T @ Q.T       │
│ score_slow = Q @ S_slow @ S_slow.T @ Q.T       │
│                                                  │
│ psi_read = compute_resonance(...)               │
│                                                  │
│ Example:                                        │
│ - score_slow = 0.87 (good match!)              │
│ - psi_read = 0.6 (moderate novelty)            │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 4: Retrieve from S_fast and S_slow         │
│                                                  │
│ Normalize for reading:                          │
│   S_fast_read = S_fast / ||S_fast||            │
│   S_slow_read = S_slow / ||S_slow||            │
│                                                  │
│ Read outputs:                                   │
│   o_fast = Q @ S_fast_read                     │
│   o_slow = Q @ S_slow_read                     │
│                                                  │
│ Blend based on psi:                             │
│   alpha = 0.5 + 0.3 * psi_read                 │
│   o_recalled = alpha * o_fast + (1-alpha) * o_slow │
│                                                  │
│ Example:                                        │
│ - psi = 0.6 → alpha = 0.68                     │
│ - Blend: 68% recent (S_fast) + 32% long-term  │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 5: Find Most Similar Memory                │
│                                                  │
│ For each memory in memory_index:                │
│   1. Get its text: "I live in Tokyo"           │
│   2. Encode it: K, U = encode(text)            │
│   3. Recall using its key: o = K @ S_slow      │
│   4. Score: similarity(o, U)                   │
│                                                  │
│ Sort by score, return highest:                  │
│   best_match = "I live in Tokyo" (score: 0.87) │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 6: Compute Semantic Score (Optional)       │
│                                                  │
│ If enabled:                                     │
│   query_emb = encoder("Where do I live?")      │
│   memory_emb = encoder("I live in Tokyo")      │
│   semantic_score = cosine(query_emb, memory_emb)│
│                                                  │
│ Example: semantic_score = 0.82                  │
└────────────────┬────────────────────────────────┘
                 ↓
Output: {
  "recalled_content": "I live in Tokyo",
  "confidence": 0.87,
  "source": "S_slow",
  "psi": 0.6,
  "semantic_score": 0.82  # if enabled
}

Time: ~0.005 seconds
```

**What This Means:**
- Query matched "I live in Tokyo" with 87% confidence
- Retrieved from long-term memory (S_slow)
- Blended recent + long-term context
- Total time: 5 milliseconds

---

### 4. Storing a Document (Store Document)

```python
brain.store_document(large_text, doc_id="my_doc")
```

**Complete Internal Flow:**

```
Input: large_text (could be 32k words)
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 1: Chunk the Document                      │
│                                                  │
│ Why? Large docs don't fit in S_slow matrices.   │
│                                                  │
│ Split into chunks:                              │
│   chunk_size = 100 words                        │
│   overlap = 20 words                            │
│                                                  │
│ Result:                                         │
│   chunks = [chunk1, chunk2, ..., chunk_N]      │
│                                                  │
│ Example: 32k words → 320 chunks                │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 2: Extract Keywords (Fast Search)          │
│                                                  │
│ For each chunk:                                 │
│   keywords = extract_keywords(chunk)            │
│                                                  │
│ Build inverted index:                           │
│   {                                             │
│     "attention": [chunk_5, chunk_17, ...],     │
│     "transformer": [chunk_6, chunk_18, ...],   │
│     "neural": [chunk_3, chunk_9, ...],         │
│     ...                                         │
│   }                                             │
│                                                  │
│ This enables instant keyword search!            │
│ Time: O(1) lookup instead of O(N) scan         │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 3: Compute Embeddings (Slow, Cached)       │
│                                                  │
│ For each chunk:                                 │
│   embedding = LAM_encoder(chunk)                │
│   embedding_cache[chunk_id] = embedding         │
│                                                  │
│ Note: This is SLOW for large docs!              │
│ - 320 chunks × 0.01s = 3.2 seconds             │
│                                                  │
│ Solution: Lazy loading (optional)               │
│ - Compute embeddings in background thread      │
│ - Keywords available instantly                 │
│ - Semantic search available after processing   │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ STEP 4: Store Document Metadata                 │
│                                                  │
│ documents[doc_id] = {                           │
│   "text": large_text,                          │
│   "chunks": chunks,                            │
│   "keyword_index": inverted_index,             │
│   "embedding_cache": {                         │
│     chunk_0: embedding_0,                      │
│     chunk_1: embedding_1,                      │
│     ...                                         │
│   },                                            │
│   "timestamp": "2025-11-07T15:35:00",          │
│   "approx_tokens": 32000                       │
│ }                                               │
│                                                  │
│ Note: Documents stored separately from S_slow   │
│ (Too large to fit in memory matrices)           │
└─────────────────────────────────────────────────┘

Output: {
  "success": True,
  "doc_id": "my_doc",
  "num_chunks": 320,
  "approx_tokens": 32000,
  "processing_time": 3.2
}
```

---

### 5. Querying a Document (Hybrid Retrieval)

```python
result = brain.query_document_hybrid("attention mechanism", doc_id="my_doc")
```

**Complete Internal Flow:**

```
Input: "attention mechanism"
    ↓
┌─────────────────────────────────────────────────┐
│ PHASE 1: Keyword Search (Instant)               │
│                                                  │
│ Extract keywords from query:                    │
│   query_keywords = ["attention", "mechanism"]   │
│                                                  │
│ Look up inverted index:                         │
│   keyword_index["attention"] → [chunk_5, ...]  │
│   keyword_index["mechanism"] → [chunk_8, ...]  │
│                                                  │
│ Find chunks with ANY keyword:                   │
│   candidates = [chunk_5, chunk_6, chunk_8, ...] │
│                                                  │
│ Score each candidate:                           │
│   score = (keywords found) / (total keywords)   │
│                                                  │
│ Example:                                        │
│   chunk_5: has "attention" → score = 0.5       │
│   chunk_8: has both → score = 1.0              │
│                                                  │
│ Time: ~0.001 seconds (instant!)                │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ PHASE 2: Semantic Re-ranking (Slower)           │
│                                                  │
│ Get query embedding:                            │
│   query_emb = LAM_encoder("attention mechanism")│
│                                                  │
│ For each keyword candidate:                     │
│   chunk_emb = embedding_cache[chunk_id]        │
│   semantic_score = cosine(query_emb, chunk_emb)│
│                                                  │
│ Example:                                        │
│   chunk_5: semantic = 0.82                     │
│   chunk_8: semantic = 0.91                     │
│                                                  │
│ Combine scores:                                 │
│   combined = 0.7 * keyword + 0.3 * semantic    │
│                                                  │
│ Example:                                        │
│   chunk_5: 0.7*0.5 + 0.3*0.82 = 0.596         │
│   chunk_8: 0.7*1.0 + 0.3*0.91 = 0.973         │
│                                                  │
│ Sort by combined score, return top-k            │
│                                                  │
│ Time: ~0.005 seconds                           │
└────────────────┬────────────────────────────────┘
                 ↓
Output: {
  "instant": {
    "results": [...],  # Keyword matches
    "total_matches": 15,
    "speed": "0.001s"
  },
  "refined": {
    "results": [      # Re-ranked by semantic
      {
        "text": "... attention mechanism ...",
        "keyword_score": 1.0,
        "semantic_score": 0.91,
        "combined_score": 0.973
      }
    ],
    "improvement": "25% better relevance"
  }
}

Total time: ~0.006 seconds
```

---

### 6. Saving Everything (Checkpoint)

```python
brain.save_checkpoint("my_memory.pt")
```

**What Gets Saved:**

```
┌─────────────────────────────────────────────────┐
│ CHECKPOINT CONTENTS                              │
│                                                  │
│ ✅ S_fast (recent memories matrix)              │
│    Size: [1 x 1 x 64 x 64] = 4,096 floats      │
│                                                  │
│ ✅ S_slow (long-term knowledge matrix)          │
│    Size: [1 x 1 x 64 x 64] = 4,096 floats      │
│                                                  │
│ ✅ W_bilinear (resonance weights)               │
│    Size: [64 x 64] = 4,096 floats              │
│                                                  │
│ ✅ memory_index (all memory metadata)           │
│    [{                                           │
│      "content": "I live in Tokyo",             │
│      "s_slow_magnitude": 0.13,                 │
│      "timestamp": "...",                       │
│      "psi": 0.85                               │
│    }, ...]                                      │
│                                                  │
│ ✅ documents (all stored documents)             │
│    {                                            │
│      "my_doc": {                               │
│        "text": "...",                          │
│        "chunks": [...],                        │
│        "keyword_index": {...},                 │
│        "embedding_cache": {...}                │
│      }                                          │
│    }                                            │
│                                                  │
│ ✅ Config & metadata                            │
│    - d_k, d_v, num_heads                       │
│    - decay rates                               │
│    - step count, conversation tokens           │
│                                                  │
│ ❌ NOT SAVED: LAM encoder weights               │
│    (Too large: 88MB. Reloaded when needed.)     │
└─────────────────────────────────────────────────┘

File size: ~1-5 MB (depends on # of memories)
Time: ~0.1 seconds
```

**What This Means:**
- All your memories are saved to disk
- S_slow contains your accumulated knowledge
- Documents are saved with embeddings cached
- You can load this later and resume exactly where you left off

---

### 7. Loading a Checkpoint

```python
brain = PersonalMemoryBrain.load_checkpoint("my_memory.pt")
```

**What Happens:**

```
Step 1: Read checkpoint file
  └─ Load .pt file from disk

Step 2: Recreate brain structure
  ├─ Create new PersonalMemoryBrain
  ├─ Reload LAM encoder (from scratch)
  └─ This takes ~2-5 seconds

Step 3: Restore saved state
  ├─ S_fast = loaded_S_fast
  ├─ S_slow = loaded_S_slow (YOUR MEMORIES!)
  ├─ memory_index = loaded_index
  └─ documents = loaded_documents

Step 4: Ready to use!
  └─ All memories intact
  └─ Can recall immediately

Time: ~5 seconds total
```

**What This Means:**
- Your brain picks up exactly where it left off
- All memories from previous sessions still there
- S_slow still contains your accumulated knowledge
- You can resume conversations seamlessly

---

## Key Concepts Explained

### Resonance Flux (ψ - psi)

**What it is:**
```python
psi = 1.0 - similarity(current_recall, new_info)
```

**What it means:**
- ψ = 0.9: Very novel! (Never seen before)
- ψ = 0.5: Somewhat familiar
- ψ = 0.1: Very familiar (seen many times)

**Why it matters:**
- High ψ → Store urgently in S_fast (important new info!)
- Low ψ → Consolidate to S_slow (already know this)
- Automatically prioritizes novel information

### S_fast vs S_slow

**S_fast (Short-term):**
- High decay (70% per new memory)
- Stores recent conversations
- Fades quickly
- Like your working memory

**S_slow (Long-term):**
- Low decay (0.1% per new memory)
- Stores important facts
- Persists across sessions
- Like your knowledge base

**Example:**
```
Memory 1: "I like coffee" → S_slow: 0.15 (remembered!)
... 100 more memories ...
Memory 101: Query "Do I like coffee?"
→ S_slow still contains coffee preference (only 10% decay)
→ S_fast forgot it (70^100 ≈ 0% remaining)
```

### Delta Rule (Not Backprop!)

**What it is:**
```python
S_slow = S_slow * decay + K @ U.T
```

**What it's NOT:**
- NOT gradient descent
- NOT backpropagation
- NOT neural network training

**What it IS:**
- Direct association storage
- Hebbian learning ("neurons that fire together wire together")
- Biologically plausible
- Interpretable (you can see what's stored)

### Pattern Separation

**What it is:**
Keeping similar but distinct memories separate.

**Example:**
```
Memory 1: "My work email is john@company.com"
Memory 2: "My personal email is john@gmail.com"

Without separation:
  → These might blend: "john@???"

With separation:
  → Forced to stay distinct
  → Query "work email" → company.com ✓
  → Query "personal email" → gmail.com ✓
```

---

## Performance Characteristics

### Speed

| Operation | Time | Why |
|-----------|------|-----|
| Memorize | 0.001s | Just matrix math (delta rule) |
| Recall | 0.005s | Scan ~100 memories, compute scores |
| Store document (32k) | 3s | Compute embeddings for 320 chunks |
| Query document (keyword) | 0.001s | Hash table lookup |
| Query document (hybrid) | 0.006s | Keyword + semantic re-rank |
| Save checkpoint | 0.1s | Write ~5MB to disk |
| Load checkpoint | 5s | Reload encoder + restore state |

### Scalability

| Metric | Limit | Reason |
|--------|-------|--------|
| Memories in S_slow | ~1,000 | Matrix size [64x64] holds ~1000 distinct patterns |
| Conversation tokens | 1,000,000 | Tracked separately, no storage limit |
| Documents | Unlimited | Stored separately with caching |
| Document size | 100k tokens | Chunks + embedding cache |
| Query speed (docs) | O(k + log n) | k = keywords, n = chunks |

### Memory Usage

| Component | Size |
|-----------|------|
| S_fast matrix | 16 KB |
| S_slow matrix | 16 KB |
| LAM encoder | 88 MB (loaded once, cached) |
| memory_index | ~1 KB per memory |
| Document embeddings | ~1.5 KB per chunk |
| Total checkpoint | 1-5 MB (typical) |

---

## Summary: The Complete Journey

```
User Input
    ↓
Encode (LAM model, frozen)
    ↓
Compute Resonance (novelty detection)
    ↓
Delta Rule Update (episodic storage)
    ↓
Store in S_fast/S_slow (accumulate)
    ↓
Save checkpoint (persist)
    ↓
Load checkpoint (restore)
    ↓
Recall (compositional retrieval)
    ↓
User Output
```

**What makes this special:**
1. **Episodic learning** - Accumulates memories, doesn't retrain
2. **Resonance-driven** - Automatically prioritizes novel info
3. **Dual-timescale** - Recent (S_fast) + Long-term (S_slow)
4. **Persistent** - Saved/loaded across sessions
5. **Fast** - No training, just storage and retrieval
6. **Interpretable** - Can see what's stored
7. **Biologically inspired** - Mirrors hippocampus + neocortex

**This is NOT:**
- Neural network training (no backprop)
- Fine-tuning (encoder stays frozen)
- Traditional RAG (has consolidation and novelty detection)

**This IS:**
- Episodic memory system (like human hippocampus)
- Personal knowledge accumulation
- Stateful AI (remembers across sessions)
- Lightweight and fast (no GPU training needed)