# PerfectRecall Usage - Correct Implementation

## ✅ **CORRECT USAGE**

PerfectRecall (Delta GD) should **ONLY** be used for **NIAH (Needle-in-a-Haystack) tests**.

## 📋 **Usage by Test Type**

### ✅ **Standard Cosine Similarity** (Default)
Used for:
- **STS Tasks** (Semantic Textual Similarity)
- **Retrieval Tasks** (SciFact, NFCorpus, ArguAna, etc.)
- **LongEmbed Real Tasks** (NarrativeQA, QMSum, etc.)

```python
# Standard model (default)
model = LAMForMTEB('/workspace/LAM/best', device='cuda', use_perfect_recall=False)
```

### ✅ **PerfectRecall (Delta GD)** (ONLY for NIAH)
Used for:
- **NIAH Tests** (LEMBNeedleRetrieval, LEMBPasskeyRetrieval)
- **Needle-in-Haystack scenarios** (perfect recall of stored facts)

```python
# NIAH model (explicit)
model = LAMForMTEB('/workspace/LAM/best', device='cuda', use_perfect_recall=True)
```

## 🔧 **Implementation**

### In `lam_scientific_proof_suite.py`:

1. **`run_proof_suite()`**:
   - Creates standard model: `LAMForMTEB(model_path, device, use_perfect_recall=False)`
   - Uses for: STS, Retrieval, LongEmbed real tasks
   - For NIAH: Calls `run_longembed_niah()` which creates its own model with `use_perfect_recall=True`

2. **`run_longembed_niah()`**:
   - Creates NIAH model: `LAMForMTEB(model_path, device, use_perfect_recall=True)`
   - Uses PerfectRecall for: LEMBNeedleRetrieval, LEMBPasskeyRetrieval

3. **Individual tests** (CLI):
   - `--sts`, `--retrieval`, `--longembed`: Use standard model (`use_perfect_recall=False`)
   - `--niah`: Uses PerfectRecall model (`use_perfect_recall=True`)

## 🎯 **Why This Matters**

- **Standard Retrieval**: Needs semantic similarity ranking (cosine similarity)
- **NIAH Tests**: Need perfect recall of stored facts (Delta GD memory matrix)

Using PerfectRecall for standard retrieval was causing low scores (29.4 vs 43.0 target) because:
- PerfectRecall is optimized for perfect recall, not ranking
- Standard cosine similarity is better for semantic ranking

## ✅ **Status**

- ✅ **Fixed**: Default is `use_perfect_recall=False`
- ✅ **NIAH**: Uses PerfectRecall explicitly
- ✅ **All other tests**: Use standard cosine similarity

---

**Result**: Retrieval scores should now improve significantly! 🚀



