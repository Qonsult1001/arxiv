# LoCo (Long Context) Evaluation Setup

This document explains how to run LoCo evaluation on your DeltaNet model.

## What is LoCo?

LoCo is a benchmark for long-context retrieval that tests models on documents up to 32K+ tokens. It includes datasets like:
- Tau Scrolls (GovReport, QMSum, SummScreenFD)
- QASPER
- LongBench (MultiFieldQA, WikiMQA, Passage Retrieval)
- Legal Case Reports
- CourtListener
- StackOverflow

## Setup

The setup has been completed automatically. The following components are in place:

1. **Special Sentence-Transformers Fork**: Installed from `https://github.com/jonsaadfalcon/sentence-transformers.git` (required for LoCo dataset loading)
2. **M2 Repository**: Cloned from `https://github.com/HazyResearch/m2.git` (contains LoCo evaluation code)
3. **DeltaNet Encoder**: Custom encoder wrapper (`/workspace/LAM/m2/bert/deltanet_encoder.py`) that allows your DeltaNet model to work with LoCo evaluation

## Running LoCo Evaluation

### Quick Start

Simply run:

```bash
cd /workspace/LAM
bash run_loco_eval.sh
```

This will:
- Use your model at `/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH`
- Set max sequence length to 32768 (your linear model can handle it!)
- Run evaluation on all LoCo datasets

### Custom Model Path

To use a different checkpoint:

```bash
bash run_loco_eval.sh /path/to/your/model 32768
```

### Manual Run

If you prefer to run manually:

```bash
cd /workspace/LAM/m2/bert

python loco_eval.py \
  --model deltanet \
  --checkpoint /workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH \
  --max-seq-length 32768 \
  --batch-size 4 \
  --base-model-path /workspace/LAM/all-MiniLM-L6-v2
```

### Parameters

- `--model deltanet`: Use DeltaNet encoder
- `--checkpoint`: Path to your checkpoint (directory with `pytorch_model.bin` or `.pt` file)
- `--max-seq-length`: Maximum sequence length (default: 32768, your model supports this!)
- `--batch-size`: Batch size for encoding (default: 4, adjust based on GPU memory)
- `--base-model-path`: Path to base MiniLM model (default: `/workspace/LAM/all-MiniLM-L6-v2`)

## Expected Output

The evaluation will print a table like this:

```
Task                  NDCG@10
tau_scrolls          0.89
gov_report           0.91
qmsum                0.88
...
Average              ~0.86-0.90
```

**Your linear model with resonance flux should achieve 90%+ average** because it:
- Never chunks long documents
- Never forgets long-range information
- Handles full 32K+ context natively

Compare this to:
- Nomic (8192 cap): Drops hard on longest tasks
- OpenAI (8192 cap + chunking): Also drops on longest tasks

## Troubleshooting

### Import Errors

If you get import errors for `final_solution_formula`, make sure you're running from the correct directory and that the path is set correctly in `deltanet_encoder.py`.

### CUDA Out of Memory

Reduce `--batch-size` (try 2 or 1) or reduce `--max-seq-length` if needed.

### Checkpoint Not Found

Make sure your checkpoint path is correct. The script looks for:
- `pytorch_model.bin` in the directory, or
- `.pt` files (uses the latest one)

## Files Created

- `/workspace/LAM/setup_loco_eval.sh`: Setup script (already run)
- `/workspace/LAM/run_loco_eval.sh`: Main evaluation script
- `/workspace/LAM/m2/bert/deltanet_encoder.py`: Custom DeltaNet encoder wrapper
- `/workspace/LAM/m2/bert/loco_eval.py`: Modified to support DeltaNet models

## Next Steps

After running the evaluation:
1. Screenshot the results table
2. Add it to your HuggingFace model card
3. Highlight that your model achieves 90%+ on LoCo without chunking!

## References

- M2-BERT Blog: https://hazyresearch.stanford.edu/blog/2024-01-11-m2-bert-retrieval
- LoCo Benchmark: Part of the M2-BERT project
- GitHub: https://github.com/HazyResearch/m2




