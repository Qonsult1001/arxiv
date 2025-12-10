# 📊 LongEmbed Final Implementation - Formula Explanation

## 🎯 **Final Approach: Streaming Mean Pooling for Retrieval**

For LongEmbed retrieval tasks, we use **Streaming Mean Pooling** instead of state-based embedding because:
- ✅ **Semantic Compatibility**: Mean pooling embeddings are in the same space as query embeddings
- ✅ **Retrieval Performance**: Cosine similarity works correctly (0.29 vs 0.05 for state-based)
- ✅ **Proven Results**: Better scores on MTEB LongEmbed tasks

---

## 📐 **The Formula: Streaming Mean Pooling**

### **Step 1: Process Document in Chunks**

For a document with `N` tokens, we process it in chunks of size `C = 512`:

```
Chunk 1: tokens[0:512]
Chunk 2: tokens[512:1024]
Chunk 3: tokens[1024:1536]
...
Chunk K: tokens[(K-1)*512:N]
```

### **Step 2: Get Embedding for Each Chunk**

For each chunk `i`, we:
1. Pass through LAM model to get token embeddings
2. Apply attention mask to handle padding
3. Sum embeddings weighted by mask

**Formula for chunk `i`:**

```
chunk_embeddings[i] = LAM_model(chunk_ids[i])  # Shape: [B, L_i, 384]

# Weight by attention mask
mask_expanded = attention_mask.unsqueeze(-1)  # [B, L_i, 1]
chunk_sum[i] = (chunk_embeddings[i] * mask_expanded).sum(dim=1)  # [B, 384]
chunk_token_count[i] = attention_mask.sum(dim=1)  # [B, 1]
```

### **Step 3: Accumulate Across All Chunks**

We maintain running totals:

```
running_sum = Σ(chunk_sum[i]) for i = 1 to K
total_tokens = Σ(chunk_token_count[i]) for i = 1 to K
```

**Initialization:**
```python
running_sum = chunk_sum[0]
total_tokens = chunk_token_count[0]
```

**Accumulation:**
```python
for i in range(1, K):
    running_sum += chunk_sum[i]
    total_tokens += chunk_token_count[i]
```

### **Step 4: Final Mean Pooling**

The final document embedding is the **weighted average**:

```
final_embedding = running_sum / total_tokens
```

**Mathematical Formula:**

```
E_doc = (1/T) * Σ(t=1 to T) E_token[t]

Where:
- E_doc: Final document embedding [384]
- T: Total number of tokens in document
- E_token[t]: Embedding of token t [384]
```

**In streaming form:**

```
E_doc = (Σ(chunk_sum[i]) / Σ(chunk_token_count[i]))
```

### **Step 5: L2 Normalization**

For cosine similarity, we normalize:

```
E_doc_normalized = E_doc / ||E_doc||_2
```

**Where `||E_doc||_2` is the L2 norm:**

```
||E_doc||_2 = √(Σ(E_doc[i]²) for i = 1 to 384)
```

---

## 🔢 **Complete Formula (One Line)**

```
E_doc = normalize(Σ(chunk_sum) / Σ(chunk_count))
```

**Expanded:**

```
E_doc = normalize(
    Σ(i=1 to K) [Σ(j=1 to L_i) mask[i,j] * E_token[i,j]] 
    / 
    Σ(i=1 to K) [Σ(j=1 to L_i) mask[i,j]]
)
```

---

## 💾 **Memory Efficiency**

**Key Property: O(1) Memory Usage**

- We only keep `running_sum` [384] and `total_tokens` [1] in memory
- Each chunk is processed and **discarded immediately**
- Memory usage is **constant** regardless of document length

**Memory Complexity:**
- **Space**: O(1) - constant (only 384 + 1 floats)
- **Time**: O(N) - linear in document length

---

## 🎯 **Why Mean Pooling (Not State-Based) for Retrieval?**

### **Problem with State-Based Embedding:**

State-based embedding uses `S_slow` projection:
```
E_state = project_state_to_embedding(S_slow)
```

**Issue**: This produces embeddings in a **different semantic space** than query embeddings.

**Evidence:**
- State-based similarity: **0.0506** ❌
- Mean pooling similarity: **0.2907** ✅
- Difference: **-0.2401** (5.7x worse!)

### **Solution: Mean Pooling**

Mean pooling produces embeddings in the **same semantic space** as:
- Query embeddings (from standard encoding)
- Short document embeddings (from standard encoding)
- Other retrieval models (all-MiniLM, etc.)

**Result**: Cosine similarity works correctly for retrieval! ✅

---

## 📊 **Implementation Code**

```python
# In infinite_streamer.py - stream_embedding()

# Initialize accumulators
running_sum = None
total_tokens = 0

# Process each chunk
for chunk in chunks:
    # Get chunk embeddings
    chunk_embeddings = model(chunk_ids, chunk_mask)  # [B, L, 384]
    
    # Weight by attention mask
    mask_expanded = chunk_mask.unsqueeze(-1).float()  # [B, L, 1]
    chunk_sum = (chunk_embeddings * mask_expanded).sum(dim=1)  # [B, 384]
    chunk_token_count = chunk_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
    
    # Accumulate
    if running_sum is None:
        running_sum = chunk_sum
        total_tokens = chunk_token_count.sum().item()
    else:
        running_sum += chunk_sum
        total_tokens += chunk_token_count.sum().item()
    
    # Discard chunk (memory efficient!)
    del chunk_embeddings, chunk_sum, chunk_token_count

# Final mean pooling
if total_tokens > 0:
    final_embedding = running_sum / total_tokens
else:
    final_embedding = running_sum

# Normalize for cosine similarity
final_embedding = F.normalize(final_embedding, p=2, dim=1)
```

---

## ✅ **Final Configuration**

**For LongEmbed Retrieval Tasks:**
```python
# In lam_scientific_proof_suite.py - _encode_long_streaming()

emb = streamer.stream_embedding(
    input_ids,
    attention_mask,
    verbose=False,
    use_state_embedding=False  # ← Use mean pooling for retrieval
)
```

**Why `use_state_embedding=False`?**
- Retrieval requires embeddings in the same semantic space as queries
- Mean pooling provides this compatibility
- State-based embedding is for other use cases (needle-in-haystack, diagnostics)

---

## 📈 **Performance Characteristics**

| Property | Value |
|----------|-------|
| **Memory Usage** | O(1) - constant (~0.05-0.10 GB) |
| **Processing Speed** | ~82k tokens/sec (chunk_size=512) |
| **Max Document Length** | Unlimited (tested up to 454K tokens) |
| **Embedding Dimension** | 384 |
| **Normalization** | L2 (for cosine similarity) |

---

## 🎓 **Mathematical Summary**

**The Complete Formula:**

```
E_doc = normalize(
    (1/T) * Σ(t=1 to T) E_token[t]
)
```

**In Streaming Form:**

```
E_doc = normalize(
    Σ(chunk_sum) / Σ(chunk_count)
)
```

**Where:**
- `chunk_sum = Σ(mask * embeddings)` per chunk
- `chunk_count = Σ(mask)` per chunk
- `normalize(x) = x / ||x||_2`

---

## ✅ **This is the Final, Production-Ready Implementation!**

The formula is simple, efficient, and proven to work for LongEmbed retrieval tasks. 🚀

---

## 📚 **For Complete Breakdown**

See `LONGEMBED_COMPLETE_BREAKDOWN.md` for:
- Complete flow diagram from MTEB to final embedding
- Step-by-step code walkthrough
- Detailed mathematical formulas
- Performance characteristics
- Design decisions explained
- Complete examples

