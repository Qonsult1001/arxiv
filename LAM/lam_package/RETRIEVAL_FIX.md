# 🔧 Retrieval Performance Fix

## Issue Identified

Retrieval score was **29.4** (target: 43.0) - significantly below target.

## Root Cause

**PerfectRecall was being used by default** for all retrieval tasks, but PerfectRecall is designed for:
- **Needle-in-haystack** perfect recall scenarios
- **Stored fact retrieval** with 100% recall

For **standard MTEB retrieval tasks** (SciFact, NFCorpus, ArguAna), we need:
- **Standard cosine similarity** for semantic ranking
- **Standard embedding-based retrieval** (not Delta GD memory matrix)

## Fix Applied

Changed default `use_perfect_recall=False` in `LAMForMTEB.__init__()`:

```python
# BEFORE (WRONG):
def __init__(self, model_path: str, device: str = "cuda", use_perfect_recall: bool = True):

# AFTER (FIXED):
def __init__(self, model_path: str, device: str = "cuda", use_perfect_recall: bool = False):
```

## When to Use PerfectRecall

- ✅ **Needle-in-haystack tests** (`scientific_mrl_benchmark.py` TEST 4)
- ✅ **Perfect recall scenarios** (stored facts, passwords, etc.)
- ❌ **NOT for standard MTEB retrieval** (SciFact, NFCorpus, etc.)

## When to Use Standard Cosine Similarity

- ✅ **MTEB retrieval tasks** (SciFact, NFCorpus, ArguAna, etc.)
- ✅ **Standard semantic similarity ranking**
- ✅ **RAG/search applications**

## Verification

```python
# Standard MTEB retrieval (default - uses cosine similarity)
model = LAMForMTEB('/workspace/LAM/best', device='cuda')
# use_perfect_recall=False by default

# PerfectRecall for needle-in-haystack (explicit)
model = LAMForMTEB('/workspace/LAM/best', device='cuda', use_perfect_recall=True)
```

## Expected Results

After this fix:
- **Retrieval score should improve** from 29.4 → closer to 43.0 target
- **Standard cosine similarity** will be used for ranking
- **Better semantic ranking** for MTEB retrieval tasks

---

**Status**: ✅ **FIXED** - Ready to retest


