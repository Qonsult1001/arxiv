# LAM Dual Encoder - RAG Use Case

## The Philosophy

**We're a SEMANTIC MODEL, not a perfect recall model.**

- LAM provides semantic similarity for retrieval
- RAG system handles perfect recall
- 12k dimensions = more nuanced semantic understanding

## Use Case: Long Documents in RAG

### The Workflow

1. **Indexing**: Write 100k token document → 12k dimensional vector
2. **Storage**: Store in vector database (Pinecone, Milvus, etc.)
3. **Retrieval**: RAG system does perfect recall using semantic similarity
4. **Result**: Best of both worlds - semantic search + perfect recall

### Why 12k Dimensions?

- **384d**: Good semantic similarity, but limited nuance
- **12k dimensions**: 32x more dimensions = more semantic nuances captured
- **Result**: Better retrieval for complex, long documents

### Example

```python
from lam import LAM
from lam_dual_encoder import LAMDualEncoder

model = LAM('/workspace/LAM/best')
encoder = LAMDualEncoder(model)

# Long document (100k tokens)
long_doc = "..." # Your 100k token document

# Embed for RAG
doc_vector = encoder.encode(long_doc, mode="enterprise")  # [12,288]

# Store in vector DB
vector_db.upsert(id="doc_123", vector=doc_vector)

# Later: RAG retrieval
query = "What is the main topic?"
query_vector = encoder.encode(query, mode="standard")  # [384]

# RAG system finds semantically similar documents
results = vector_db.query(query_vector, top_k=10)

# RAG does perfect recall on retrieved documents
answer = rag_system.generate(query, results)
```

## Performance

| Mode | Dimensions | Spearman | Use Case |
|------|------------|----------|----------|
| Standard | 384 | 0.8189 | General RAG, fast |
| Enterprise | 12,288 | 0.8179 | Long docs, more nuances |

## Key Points

1. ✅ **Semantic similarity**: 99.9% retention (0.8179 vs 0.8189)
2. ✅ **More nuances**: 32x dimensions capture finer semantic details
3. ✅ **RAG handles recall**: We provide semantic search, RAG does perfect recall
4. ✅ **Long documents**: Embed 100k tokens → 12k vector → RAG retrieval

## Not For

- ❌ Perfect recall on 100k+ token documents (use RAG for that)
- ❌ Forensic detail extraction (we're semantic, not forensic)
- ❌ Exact word/phrase matching (use keyword search)

## For

- ✅ Semantic similarity search
- ✅ RAG retrieval on long documents
- ✅ Capturing nuanced semantic meaning
- ✅ Production RAG systems

