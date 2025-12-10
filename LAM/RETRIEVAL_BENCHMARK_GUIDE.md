# 🎯 Retrieval Benchmark Guide

## What is the Benchmark?

For **retrieval tasks**, the standard benchmark is **MTEB (Massive Text Embedding Benchmark)**.

### Main Retrieval Tasks in MTEB:

1. **MS MARCO** - Most important! Web search queries + passages
   - Metric: **nDCG@10** (Normalized Discounted Cumulative Gain at 10)
   - all-MiniLM-L6-v2 score: **~37.7 nDCG@10**

2. **Natural Questions (NQ)** - Question answering retrieval
   - Metric: **nDCG@10**
   - all-MiniLM-L6-v2 score: **~58.5 nDCG@10**

3. **Other Retrieval Tasks:**
   - ArguAna, ClimateFEVER, DBPedia, FEVER, FiQA2018, HotpotQA, NFCorpus, QuoraRetrieval, SCIDOCS, SciFact, TRECCOVID, etc.

### Average Retrieval Score:
- **all-MiniLM-L6-v2**: ~42.0 nDCG@10 (average across all retrieval tasks)

## 🎯 Your Targets to Beat all-MiniLM-L6-v2:

| Task | Metric | all-MiniLM-L6-v2 | Your Target |
|------|--------|------------------|-------------|
| **MS MARCO** | nDCG@10 | 37.7 | **>37.7** |
| **Natural Questions** | nDCG@10 | 58.5 | **>58.5** |
| **Average Retrieval** | nDCG@10 | 42.0 | **>42.0** |
| **STS-B** | Spearman | 82.0 | **>82.0** |

## 📊 How to Evaluate:

### Quick Test (STS-B):
```bash
python evaluate_retrieval_adapter.py
```

This will test:
1. **STS-B** (Semantic Textual Similarity) - Quick semantic similarity test
2. **MS MARCO** (sample) - Main retrieval benchmark

### Full MTEB Evaluation:

For comprehensive evaluation, use the MTEB library:

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Your model wrapper
class YourModel:
    def encode(self, texts, **kwargs):
        # Use adapter + streamer here
        return embeddings

# Run evaluation
evaluation = MTEB(tasks=["MSMARCO", "NQ", "ArguAna", ...])
results = evaluation.run(YourModel(), output_folder="results")
```

## 🔍 What is nDCG@10?

**nDCG@10** (Normalized Discounted Cumulative Gain at 10):
- Measures retrieval quality by ranking documents
- Considers **position** of relevant documents (top results matter more)
- Range: 0.0 to 1.0 (higher is better)
- **@10** means we only look at top 10 retrieved documents

**Example:**
- Query: "What is machine learning?"
- Retrieved docs: [relevant, irrelevant, relevant, irrelevant, ...]
- nDCG@10 measures how well relevant docs are ranked in top 10

## 🚀 How to Improve Your Adapter:

### Current Status:
- Your adapter improves raw streamer from **0.52 → 0.64** cosine similarity
- But needs **>0.90** to match teacher embeddings

### To Beat all-MiniLM-L6-v2:

1. **Train Longer:**
   ```python
   # In train_retrieval_finetune.py
   for epoch in range(20):  # Increase from 5 to 20
   ```

2. **Use More Data:**
   ```python
   # Increase dataset size
   dataset = load_dataset(..., split="train[:10000]")  # Instead of 2000
   ```

3. **Use Retrieval-Specific Training:**
   - Train on MS MARCO triplets (query, positive, negative)
   - Use contrastive loss with hard negatives
   - Focus on ranking quality, not just embedding alignment

4. **Better Loss Function:**
   ```python
   # Instead of MSE, use contrastive loss
   # This directly optimizes for retrieval ranking
   ```

## 📈 Expected Performance:

| Adapter Quality | STS-B Spearman | MS MARCO nDCG@10 | Status |
|----------------|----------------|------------------|--------|
| Current | ~64% | ~25-30 | ⚠️ Needs improvement |
| Good | >82% | >37.7 | ✅ Beats baseline |
| Excellent | >85% | >40.0 | 🎉 Competitive |

## 🎯 Next Steps:

1. **Run evaluation:**
   ```bash
   python evaluate_retrieval_adapter.py
   ```

2. **Check current scores** vs targets

3. **Improve training:**
   - More epochs
   - More data
   - Better loss function

4. **Re-evaluate** until you beat the baseline!


