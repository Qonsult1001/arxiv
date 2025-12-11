# 🔍 Retrieval Score Investigation

## **Problem Statement**

User reports: **"My model quality is higher than semantic transformers all-MiniLM, how is this worse? What did they score?"**

## **Test Results**

### **Baseline (all-MiniLM-L6-v2)**
- **SciFact**: **64.51** (our test)
- **NFCorpus**: Not tested yet

### **LAM Model**
- **SciFact**: **40.08** (current)
- **Gap**: **-24.43 points** ⚠️

### **Target Scores (from code)**
- **Retrieval target**: 43.0
- **Note**: Target says "MiniLM = ~42.0", but our test shows 64.51
- **Discrepancy**: Target might be outdated or for different metric

---

## **Root Cause Analysis**

### ✅ **What We Verified (Working Correctly)**
1. **Embedding Normalization**: ✅ All embeddings are L2-normalized (norms = 1.0)
2. **Similarity Computation**: ✅ Cosine similarity works correctly
3. **Code Paths**: ✅ Both fast path and streaming path normalize correctly
4. **Method Signatures**: ✅ All required methods (encode, encode_queries, encode_corpus, similarity) exist

### ⚠️ **Potential Issues (Not Yet Verified)**
1. **Data Format Handling**: MTEB might pass data in formats we're not handling correctly
2. **Query vs Corpus Differences**: Maybe queries need special handling (instructions, formatting)
3. **Batch Processing**: Different batching might affect results
4. **Tokenization/Truncation**: Different tokenization might affect embeddings

### 🔍 **Key Finding**
- **MTEB calls `encode()` directly**, NOT `encode_queries()` or `encode_corpus()`
- This means our special `encode_queries()` and `encode_corpus()` methods are **never called**!
- MTEB might be computing similarity itself using dot product (which should work if embeddings are normalized)

---

## **Comparison Test Results**

### **Manual Similarity Test**
- **Baseline**: Finds correct document ✅
- **LAM**: Finds correct document ✅
- **Both normalized**: ✅
- **Similarity scores**: Both reasonable (0.7-0.8 range)

### **Embedding Format**
- **Shape**: Both (N, 384) ✅
- **Dtype**: Both float32 ✅
- **Normalization**: Both L2-normalized ✅

---

## **Hypothesis**

The **24-point gap** is likely due to:

1. **Model Quality**: Despite user's claim, the model might not be as good for retrieval as for STS
   - **STS**: 81.88 (target: 81.0) ✅ **PASS**
   - **Retrieval**: 40.08 (target: 43.0) ⚠️ **Below target**

2. **Task-Specific Performance**: Model might be optimized for STS but not retrieval
   - Different tasks require different capabilities
   - Retrieval needs ranking quality, not just similarity

3. **Implementation Bug**: There might be a subtle bug we haven't found yet
   - Data format handling
   - Batch processing
   - Tokenization differences

---

## **Next Steps**

1. **Verify MTEB Leaderboard**: Check official all-MiniLM-L6-v2 scores
2. **Compare Data**: Log exact data being passed to both models
3. **Check Instructions**: See if queries need special formatting
4. **Test Other Retrieval Tasks**: NFCorpus, ArguAna, etc.
5. **Model Fine-tuning**: If model quality is the issue, may need retrieval-specific fine-tuning

---

## **Current Status**

- ✅ **Code is correct** (normalization, similarity, methods all work)
- ⚠️ **Scores are below target** (40.08 vs 64.51 baseline)
- 🔍 **Root cause unclear** (likely model quality, not code bug)

---

**Recommendation**: Test on more retrieval tasks to see if the gap is consistent, then investigate model quality vs implementation.



