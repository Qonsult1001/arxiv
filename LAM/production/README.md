# LAM Production Bundle

This folder contains the production-ready LAM model package for SDK/API integration.

## Structure

```
production/
├── lam-base-v1/              ← Complete LAM model package
│   ├── pytorch_model.bin     ← Base embeddings + FFN layers
│   ├── lam_checkpoint.pt     ← LAM attention layers (0.836 Pearson)
│   ├── config.json           ← Model configuration
│   ├── tokenizer files       ← Tokenization
│   └── lam_loader.py         ← Model loading wrapper
│
├── package_lam.py            ← Script to create distribution package
├── lam_wrapper.py            ← SentenceTransformer-compatible API
└── setup.py                  ← pip install setup (optional)
```

## Quick Start

### Option 1: Direct Loading (Recommended)

```python
from production.lam_wrapper import LAMEncoder

# Load model
model = LAMEncoder('production/lam-base-v1')

# Encode sentences
embeddings = model.encode([
    "This is a sentence",
    "This is another sentence"
])

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embeddings)
```

### Option 2: Package and Distribute

```bash
# Create distribution package
python production/package_lam.py

# This creates: lam-base-v1-dist.tar.gz
# Share this file with SDK/API developers
```

## Model Components

### Base Model (`pytorch_model.bin`)
- **Source**: `LAM base model/pytorch_model.bin`
- **Size**: ~90 MB
- **Contains**: Token embeddings + 6 FFN layers
- **Usage**: Frozen during inference (provides base representations)

### LAM Checkpoint (`lam_checkpoint.pt`)
- **Source**: `proper_distillation_reaccelerate/checkpoint_best_3500.pt`
- **Size**: ~15-20 MB
- **Contains**: 6 LAM attention layers
- **Achievement**: 0.836 Pearson on STS-B
- **Innovation**: O(n) linear attention replacing O(n²) transformer attention

### Tokenizer
- **Source**: `LAM base model/` tokenizer files
- **Vocabulary**: 30,522 tokens (WordPiece)
- **Max Length**: 128 tokens (configurable)

## Integration with SDK/API

### Your SDK/API should replace:

```python
# OLD (sentence-transformers)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('LAM base model')
embeddings = model.encode(texts)

# NEW (LAM)
from lam_wrapper import LAMEncoder
model = LAMEncoder('lam-base-v1')
embeddings = model.encode(texts)  # Same API!
```

### Or use as drop-in replacement:

```python
# Import LAM with SentenceTransformer alias
from lam_wrapper import LAMEncoder as SentenceTransformer

# Rest of your code stays unchanged!
model = SentenceTransformer('lam-base-v1')
```

## Performance

| Metric | Value |
|--------|-------|
| STS-B Pearson | 0.836 |
| Model Size | ~110 MB total |
| Parameters | 22M (15× smaller than E5-Large) |
| Dimensions | 384 |
| Complexity | O(n) linear |
| Max Context | 1M+ tokens (vs 128K for transformers) |

## File Sizes

- `pytorch_model.bin`: 90.9 MB
- `lam_checkpoint.pt`: ~18 MB
- `config.json`: 1 KB
- Tokenizer files: ~1 MB
- **Total**: ~110 MB

## Security Notes

⚠️ **Proprietary Components**:
- The core formula (`final_solution_formula.py`) is **NOT included**
- Only trained weights and compiled inference code are distributed
- LAM architecture details remain proprietary

✅ **What's Distributed**:
- Pre-trained model weights (checkpoint)
- Base model weights (from public LAM base model)
- Tokenizer (from public LAM base model)
- Simple loading/inference wrapper

## License

**Proprietary Commercial License**

LAM model weights and inference code are proprietary. Contact for licensing.

Base model (LAM base model) is licensed under Apache 2.0.
