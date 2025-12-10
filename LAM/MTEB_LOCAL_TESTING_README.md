# MTEB Local Testing Guide

This guide explains how to test your DeltaNet model locally using MTEB before submitting to the leaderboard.

## Files Created

1. **`mteb_model_wrapper.py`** - Wrapper that makes your DeltaNet model compatible with MTEB
2. **`test_mteb_local.py`** - Script to run local MTEB evaluations
3. **`MTEB_LOCAL_TESTING_README.md`** - This file

## Quick Start

### 1. Quick Test (5-10 minutes)

Run a quick test on a few STS tasks:

```bash
cd /workspace/LAM
python test_mteb_local.py
```

This will:
- Load your checkpoint from `deltanet_minilm_6layers_FIXED_FROM_SCRATCH/checkpoint_100000.pt`
- Test on 7 STS tasks (STS12, STS13, STS14, STS15, STS16, STSBenchmark, SICK-R)
- Save results to `mteb_results_local/quick_test/`

### 2. Full English Benchmark (1-2 hours)

For a complete evaluation on all English tasks:

```bash
python test_mteb_local.py --full
```

This evaluates on ~100+ English tasks including:
- Retrieval tasks (MSMARCO, NQ, etc.)
- Reranking tasks
- Clustering tasks
- Classification tasks
- STS tasks
- Bitext mining tasks

## Configuration

You can modify the checkpoint path and other settings in `test_mteb_local.py`:

```python
CHECKPOINT_PATH = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH/checkpoint_100000.pt"
BASE_MODEL_PATH = "/workspace/LAM/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("/workspace/LAM/mteb_results_local")
```

## Results

Results are saved in:
- `mteb_results_local/quick_test/` - Quick test results
- `mteb_results_local/full_english/` - Full benchmark results
- `mteb_results_local/*_results_*.json` - JSON summary files

## Next Steps

Once you're satisfied with local results:

1. **Check if you qualify for "efficient" section:**
   - The efficient section typically requires good performance on retrieval and STS tasks
   - Check your scores against the leaderboard: https://huggingface.co/spaces/mteb/leaderboard

2. **Submit to leaderboard:**
   - Follow the official MTEB submission guide: https://github.com/embeddings-benchmark/mteb
   - Use the English submission script for automatic evaluation

## Troubleshooting

### Model loading errors
- Make sure the checkpoint path is correct
- Ensure the base model path exists

### CUDA out of memory
- Reduce batch size in `mteb_model_wrapper.py` (default is 32)
- Or use CPU (slower but works): change `device='cuda'` to `device='cpu'`

### Task evaluation errors
- Some tasks may fail due to dataset download issues
- Check the error messages in the output
- Results are saved even if some tasks fail

## Notes

- This is **LOCAL TESTING ONLY** - nothing is published online
- Results are cached in `~/.cache/mteb` to speed up re-runs
- The wrapper handles both list inputs and DataLoader inputs (from MTEB)

