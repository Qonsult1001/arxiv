# Why MTEB Results Differ from test_checkpoints.py

## Quick Answer

**They're actually almost identical!** The tiny differences are due to implementation details, not fundamental differences.

## Comparison Results

For checkpoint `distill_step0050_val0.8467.pt`:

### test_checkpoints.py (stsb_evaluation.py)
- **Pearson (cosine)**: 0.7825
- **Spearman (cosine)**: 0.7614
- **Dataset**: `sentence-transformers/stsb` (test split, 1379 samples)
- **Method**: Direct cosine similarity computation

### MTEB STSBenchmark
- **Pearson (cosine)**: 0.7826
- **Spearman (cosine)**: 0.7614
- **Dataset**: MTEB's standardized STSBenchmark (test split, ~1379 samples)
- **Method**: MTEB standardized evaluation pipeline

## Differences Explained

### 1. **Dataset Source** (Minor)
- `test_checkpoints.py`: Uses `sentence-transformers/stsb` dataset directly from HuggingFace
- `MTEB`: Uses MTEB's standardized version of STSBenchmark
- **Impact**: Usually the same data, but MTEB may apply minor preprocessing/normalization

### 2. **Evaluation Pipeline** (Minor)
- `test_checkpoints.py`: 
  - Direct cosine similarity computation
  - Custom batching and processing
  - Uses `scipy.stats.pearsonr` and `spearmanr`
  
- `MTEB`:
  - Standardized MTEB evaluation pipeline
  - May use different batching strategies
  - Uses MTEB's internal correlation computation
  - Applies MTEB-specific preprocessing

### 3. **Floating Point Precision** (Tiny)
- Different batching order can lead to tiny numerical differences
- Different libraries (scipy vs MTEB's internal) may have slight precision differences
- **Impact**: Differences in 4th-5th decimal place (0.7825 vs 0.7826)

## Why This Matters

1. **For Development**: `test_checkpoints.py` is faster and simpler for quick checks
2. **For Leaderboard**: MTEB is the standard - use MTEB results for official submissions
3. **For Comparison**: Both are valid, but MTEB is what the leaderboard uses

## Recommendation

- **Use `test_checkpoints.py`** for:
  - Quick checkpoint comparisons during training
  - Fast iteration and debugging
  
- **Use MTEB** for:
  - Official evaluation and leaderboard submission
  - Comparing against published results
  - Final model validation

## Conclusion

The results are **essentially identical** (difference of 0.0001 in Pearson). Both methods are valid, but MTEB is the standard for official evaluation. The tiny differences are expected and don't indicate any issues with your model or evaluation setup.

