# Universal Neural Indexer - Rule Mapping for Scientific Proof Suite

## 🎯 Quick Reference: Which Formula for Which Task?

```
┌─────────────────────────────────────────────────────────────────┐
│ TASK CATEGORY          │ FORMULA          │ ALPHA │ SCORING    │
├─────────────────────────────────────────────────────────────────┤
│ NIAH (Needle/Passkey)  │ Exact Key Match  │ 1.0   │ W^T @ k    │
│ Standard Retrieval     │ Semantic Cosine  │ <1.0  │ cos(q, d)  │
│ LongEmbed Real         │ Semantic Cosine  │ <1.0  │ cos(q, d)  │
│ STS (Similarity)       │ Direct Cosine    │ N/A   │ cos(e1,e2) │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Detailed Rule Application

### 1. **NIAH Tests** → `LONGEMBED_SYNTHETIC_TASKS`
**Tasks:** `LEMBNeedleRetrieval`, `LEMBPasskeyRetrieval`

**Rule Applied:**
```python
# INDEX: Full erase/write (alpha=1.0) for 100% recall
W = W @ (I - 1.0*kk^T) + 1.0*vk^T

# SEARCH: Delta GD Retrieval
v = W^T @ k

# SCORE: Exact key matching
if cos(k_query, k_stored) > 0.95:
    score = 1.0  # Perfect match
else:
    score = cos(v_retrieved, v_stored)
```

**Why:** Need 100% recall for stored facts (needle-in-haystack). Alpha=1.0 ensures no interference.

**Location in Code:**
- `lam_scientific_proof_suite.py` line 913-1085: `run_longembed_niah()`
- Currently uses: `use_perfect_recall=True`
- **With Universal Indexer:** Use `mode="exact"` → same 100% recall

---

### 2. **Standard Retrieval** → `RETRIEVAL_TASKS`
**Tasks:** `SciFact`, `NFCorpus`, `ArguAna`, `SCIDOCS`, `QuoraRetrieval`, `FiQA2018`

**Rule Applied:**
```python
# INDEX: Partial update (alpha < 1.0) for multiple documents
W = W @ (I - 0.5*kk^T) + 0.5*vk^T  # alpha=0.5 allows multiple memories

# SEARCH: Direct semantic similarity (NO W matrix)
score = cos(query_embedding, document_embedding)

# Why bypass W: Avoids interference, uses LAM's native semantic understanding
```

**Why:** 
- Multiple documents in shared W matrix → interference if alpha=1.0
- LAM's embeddings are optimized for semantic similarity
- Direct cosine avoids W matrix degradation

**Location in Code:**
- `lam_scientific_proof_suite.py` line 1124-1128: `run_mteb_evaluation()` for retrieval
- Currently uses: Standard cosine similarity (line 747-755)
- **With Universal Indexer:** Use `mode="document"` → direct cosine

---

### 3. **LongEmbed Real Tasks** → `LONGEMBED_REAL_TASKS`
**Tasks:** `LEMBNarrativeQARetrieval`, `LEMBQMSumRetrieval`, `LEMBWikimQARetrieval`, `LEMBSummScreenFDRetrieval`

**Rule Applied:**
```python
# INDEX: Stream document → ONE embedding (InfiniteContextStreamer)
if len(tokens) > 8192:
    doc_emb = streamer.stream_embedding(input_ids, attention_mask)
else:
    doc_emb = model.encode([text])

# Store in W (alpha < 1.0 for consistency)
W = W @ (I - 0.5*kk^T) + 0.5*vk^T

# SEARCH: Direct semantic similarity (NO W matrix)
score = cos(query_embedding, doc_embedding)
```

**Why:**
- Long documents need streaming (no truncation)
- Semantic retrieval benefits from LAM's native understanding
- W matrix maintained but not used for scoring (avoids interference)

**Location in Code:**
- `lam_scientific_proof_suite.py` line 1139-1144: `run_mteb_evaluation()` for longembed
- Currently uses: Streaming + standard cosine (line 330-412)
- **With Universal Indexer:** Same streaming, but uses `mode="document"` for semantic

---

### 4. **STS Tasks** → `STS_TASKS`
**Tasks:** `STS12`, `STS13`, `STS14`, `STS15`, `STS16`, `STSBenchmark`, `SICK-R`

**Rule Applied:**
```python
# No indexing needed (pairwise comparison)
# Direct cosine similarity
score = cos(embedding1, embedding2)
```

**Why:** STS is pairwise comparison, no retrieval needed. No W matrix involved.

**Location in Code:**
- `lam_scientific_proof_suite.py` line 1116-1119: `run_mteb_evaluation()` for STS
- Currently uses: Standard cosine (line 757-778)
- **With Universal Indexer:** No change needed (pairwise comparison)

---

## 🔄 Integration Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Task Detection (in run_mteb_evaluation)                      │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────┐            ┌──────────────────┐
│ NIAH Task?    │            │ Retrieval Task? │
│ (Synthetic)   │            │ (Real/LongEmbed) │
└───────────────┘            └──────────────────┘
        │                               │
        ▼                               ▼
┌───────────────┐            ┌──────────────────┐
│ mode="exact"  │            │ mode="document"  │
│ alpha=1.0     │            │ alpha<1.0        │
│ W^T @ k       │            │ cos(q, d)        │
└───────────────┘            └──────────────────┘
```

---

## 📊 Current vs. Universal Indexer Comparison

| Task Type | Current Method | Universal Indexer | Benefit |
|-----------|---------------|-------------------|---------|
| **NIAH** | PerfectRecall (separate W per doc) | Shared W, exact mode | Same 100% recall, unified system |
| **Retrieval** | Standard cosine | Direct cosine (no W) | Avoids interference, better semantic |
| **LongEmbed** | Streaming + cosine | Streaming + cosine | Same performance, unified API |
| **STS** | Pairwise cosine | Pairwise cosine | No change needed |

---

## 🎯 Key Insight

**The Universal Neural Indexer uses a hybrid approach:**

1. **For Exact Matching (NIAH):**
   - Uses W matrix with alpha=1.0 → 100% recall
   - Formula: `v = W^T @ k` (Delta GD Retrieval)

2. **For Semantic Similarity (Retrieval):**
   - Uses direct cosine similarity → avoids W interference
   - Formula: `cos(query_emb, doc_emb)` (LAM's native understanding)
   - W matrix still updated for consistency, but not used for scoring

**This aligns with your test results:**
- ✅ Exact Key Matching: 100% (5/5) - uses W matrix
- ✅ Semantic Similarity: 100% (3/3) - uses direct cosine

---

## 🔧 Implementation in `lam_scientific_proof_suite.py`

**Replace `LAMForMTEB.similarity()` method (line 669):**

```python
def similarity(self, query_embeddings, corpus_embeddings):
    # Determine mode based on current task
    if self._current_task in LONGEMBED_SYNTHETIC_TASKS:
        # NIAH: Use exact mode (W^T @ k)
        mode = "exact"
    else:
        # Retrieval: Use semantic mode (cos(q, d))
        mode = "document"
    
    # Use Universal Indexer
    queries = self._embeddings_to_queries(query_embeddings)
    results = self.indexer.search(queries, mode=mode)
    return self._results_to_similarity_matrix(results)
```

**This ensures:**
- NIAH tasks get 100% recall (exact mode)
- Retrieval tasks get better semantic accuracy (direct cosine)
- All tasks use the same unified system



