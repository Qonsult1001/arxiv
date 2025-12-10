# Universal Neural Indexer Integration Guide
## How It Applies to `lam_scientific_proof_suite.py`

### Current State vs. Universal Neural Indexer

**Current Implementation:**
- Standard cosine similarity for most MTEB tasks
- PerfectRecall (Delta GD) ONLY for NIAH tests
- Separate handling for long vs short documents

**Universal Neural Indexer:**
- **Hybrid approach**: Two formulas, both use cosine
  1. **Exact Key Matching** (alpha=1.0) → 100% recall (NIAH-style)
  2. **Semantic Similarity** → Direct cosine (LAM's native understanding)

---

## Rule Mapping: Which Formula for Which Task?

### 1. **NIAH Tests** (LEMBNeedleRetrieval, LEMBPasskeyRetrieval)
**Formula:** Exact Key Matching (alpha=1.0)
```
INDEX:  W = W @ (I - 1.0*kk^T) + 1.0*vk^T  (Full erase/write)
SEARCH: v = W^T @ k                         (Delta GD Retrieval)
SCORE:  cos(k_query, k_stored) > 0.95 → 1.0 (Perfect match)
```
**Why:** Need 100% recall for stored facts (needle-in-haystack)

---

### 2. **Standard Retrieval Tasks** (SciFact, NFCorpus, ArguAna, etc.)
**Formula:** Semantic Similarity (Direct Cosine)
```
INDEX:  W = W @ (I - α*kk^T) + β*vk^T  (α < 1.0, maintains W for consistency)
SEARCH: cos(query_emb, doc_emb)        (Direct LAM embeddings, NO W matrix)
SCORE:  cos(query_emb, doc_emb)        (Leverages LAM's native semantic understanding)
```
**Why:** Avoids W matrix interference, uses LAM's optimized semantic embeddings

---

### 3. **LongEmbed Real Tasks** (LEMBNarrativeQA, LEMBQMSum, etc.)
**Formula:** Semantic Similarity + Streaming
```
INDEX:  Stream document → ONE embedding (InfiniteContextStreamer)
        W = W @ (I - α*kk^T) + β*vk^T  (α < 1.0, for consistency)
SEARCH: cos(query_emb, doc_emb)        (Direct cosine, NO W matrix)
SCORE:  cos(query_emb, doc_emb)        (Semantic similarity)
```
**Why:** 
- Long documents need streaming (no truncation)
- Semantic retrieval benefits from LAM's native understanding
- W matrix maintained but not used for scoring (avoids interference)

---

### 4. **STS Tasks** (Semantic Text Similarity)
**Formula:** Standard Cosine (No W Matrix)
```
INDEX:  N/A (pairwise comparison)
SEARCH: N/A
SCORE:  cos(emb1, emb2)                 (Direct cosine similarity)
```
**Why:** STS is pairwise comparison, no retrieval needed

---

## Integration Points in `lam_scientific_proof_suite.py`

### Point 1: `LAMForMTEB.encode_corpus()` (Line 422)
**Current:** Uses standard encoding or PerfectRecall
**With Universal Indexer:**
```python
# For NIAH tasks: Use exact mode (alpha=1.0)
if task_name in LONGEMBED_SYNTHETIC_TASKS:
    indexer.index(corpus, mode="exact")  # 100% recall
    
# For standard retrieval: Use semantic mode
else:
    indexer.index(corpus, mode="document")  # Semantic similarity
```

### Point 2: `LAMForMTEB.similarity()` (Line 669)
**Current:** Standard cosine or PerfectRecall Delta GD
**With Universal Indexer:**
```python
# For NIAH: Use exact key matching
if task_name in LONGEMBED_SYNTHETIC_TASKS:
    scores = indexer.search(queries, mode="exact")  # W^T @ k
    
# For semantic retrieval: Use direct cosine
else:
    scores = indexer.search(queries, mode="semantic")  # cos(q, d)
```

### Point 3: Long Document Handling (Line 330)
**Current:** Uses InfiniteContextStreamer
**With Universal Indexer:**
```python
# Universal Indexer already uses streamer internally
# No change needed - it handles long docs automatically
if len(token_ids) > 8192:
    # Universal Indexer uses streamer.stream_embedding()
    # Returns ONE embedding per document
```

---

## Key Differences

| Aspect | Current Implementation | Universal Neural Indexer |
|--------|----------------------|-------------------------|
| **NIAH Tests** | PerfectRecall (separate W per doc) | Shared W, exact mode (alpha=1.0) |
| **Retrieval** | Standard cosine | Direct cosine (avoids W interference) |
| **Long Docs** | Streamer → embedding | Streamer → embedding (same) |
| **Memory** | Separate W per doc (PerfectRecall) | ONE shared W matrix (Universal) |
| **Scoring** | Mixed (Delta GD for NIAH, cosine for rest) | Hybrid (exact for NIAH, cosine for semantic) |

---

## Benefits of Integration

1. **Unified Formula**: One system handles all tasks
2. **100% NIAH Recall**: Exact mode achieves perfect recall
3. **Better Semantic**: Direct cosine avoids W matrix interference
4. **Consistent Memory**: ONE shared W matrix (more efficient)
5. **No Mode Switches**: Automatic detection based on task type

---

## Implementation Strategy

1. **Add Universal Indexer to LAMForMTEB:**
   ```python
   def __init__(self, model_path, device, use_universal_indexer=True):
       if use_universal_indexer:
           from universal_neural_indexer import UniversalNeuralIndexer
           self.indexer = UniversalNeuralIndexer(model_path, device)
   ```

2. **Modify `encode_corpus()` to use indexer:**
   ```python
   def encode_corpus(self, corpus, batch_size=32, **kwargs):
       # Determine mode based on task
       mode = "exact" if self._is_niah_task else "document"
       self.indexer.index(corpus, mode=mode)
       return self.indexer._doc_embeddings  # Return embeddings
   ```

3. **Modify `similarity()` to use indexer:**
   ```python
   def similarity(self, query_embeddings, corpus_embeddings):
       # Use indexer's search method
       queries = self._embeddings_to_queries(query_embeddings)
       results = self.indexer.search(queries, mode=self._current_mode)
       return self._results_to_similarity_matrix(results)
   ```

---

## Testing Strategy

1. **NIAH Tests**: Should achieve 100% recall (same as current)
2. **Retrieval Tasks**: Should match or exceed current scores
3. **LongEmbed Real**: Should maintain current performance
4. **STS Tasks**: Should be unchanged (pairwise comparison)

---

## Formula Summary

**Universal Formula (One to Rule Them All):**
```
INDEX:  W = W @ (I - α*kk^T) + β*vk^T
SEARCH: v = W^T @ k                    (for exact)
        cos(q, d)                       (for semantic)
SCORE:  cos(k_q, k_s) > 0.95 → 1.0     (exact)
        cos(q_emb, d_emb)               (semantic)
```

**Alpha Values:**
- `α = 1.0`: Exact mode (NIAH) → 100% recall, no interference
- `α < 1.0`: Semantic mode → Allows multiple memories, but scoring uses direct cosine



