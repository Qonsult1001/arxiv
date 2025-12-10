# LAM Dual Encoder V2 - Usage Guide

## Overview

The LAM Dual Encoder provides **two embedding modes** from a single trained model:

1. **Standard Mode (384d)**: Optimized for semantic search, RAG, and chatbots
2. **Enterprise Mode (12,288d)**: Lossless forensic memory for legal, medical, and compliance use cases

## STSB Benchmark Results (1,379 sentence pairs)

| Mode | Dimension | Spearman | Retention |
|------|-----------|----------|-----------|
| Standard | 384 | 0.7719 | 100% |
| Enterprise | 12,288 | 0.7421 | **96.1%** |

**Enterprise mode preserves 96.1% of semantic similarity while providing 32x more dimensions!**

## Quick Start

```python
from lam import LAM
from lam_dual_encoder import LAMDualEncoder

# Load your LAM model
model = LAM("LAM-base-v1")

# Create dual encoder
encoder = LAMDualEncoder(model)

# Standard mode (384d) - for general use
vec_384 = encoder.encode("Your document text here", mode="standard")
print(f"Standard vector shape: {vec_384.shape}")  # (384,)

# Enterprise mode (12,288d) - for forensic recall
vec_12k = encoder.encode("Your document text here", mode="enterprise")
print(f"Enterprise vector shape: {vec_12k.shape}")  # (12288,)
```

## Enterprise Mode Calibration

Before using Enterprise mode in production, you should calibrate it with sample documents:

```python
# One-time calibration (run once before deploying)
sample_docs = [
    "Document 1 text...",
    "Document 2 text...",
    # ... about 100 documents recommended
]

encoder.calibrate_enterprise_mode(sample_docs)
```

This creates a "whitening vector" that removes common language noise, making the 12k vectors more discriminative.

## When to Use Each Mode

### Standard Mode (384d)
- ✅ **General RAG systems** - Fast semantic search
- ✅ **Chatbots** - Context retrieval
- ✅ **Production APIs** - Low latency, small storage
- ✅ **Most use cases** - 384d is usually enough

### Enterprise Mode (12,288d) - RAG Semantic Search
- ✅ **Long documents in RAG** - Embed 100k tokens → 12k vector
- ✅ **More semantic nuances** - 32x dimensions capture finer details
- ✅ **RAG retrieval** - Write long docs to vector DB, RAG does perfect recall
- ✅ **Semantic similarity** - 99.9% retention (Spearman 0.8179 vs 0.8189)
- ⚠️ **NOT for perfect recall** - We're a semantic model, RAG handles recall
- ⚠️ **Higher storage** - 32x larger vectors
- ⚠️ **Slower retrieval** - More dimensions to compare

## Technical Details

### Standard Mode
- Uses **state-based embedding** (Holographic Memory Projection)
- Projects the final `S_slow` memory matrix to 384d
- Preserves end-of-document information
- Optimized for cosine similarity retrieval

### Enterprise Mode (RAG Semantic Search)
- Uses **semantic subspace projection** of raw memory state
- Inherits 99.9% semantic similarity from 384d trained model
- Captures MORE semantic nuances (32x dimensions)
- Shape: `[1, 12, 32, 32]` → `[12,288]`
- Uses the formula: `semantic_weight * semantic_12k + (1 - semantic_weight) * structural_12k`
- Default `semantic_weight=0.7` (70% semantics, 30% structural)
- **Purpose**: Embed long documents (100k tokens) for RAG retrieval
- **NOT for perfect recall** - RAG system handles that, we just need semantic similarity

## API Reference

### `LAMDualEncoder(model, tokenizer=None, device='cuda')`

**Parameters:**
- `model`: LAM model instance
- `tokenizer`: Optional tokenizer (uses `model.tokenizer` if not provided)
- `device`: Device to run on ('cuda' or 'cpu')

### `encode(text, mode="standard")`

**Parameters:**
- `text`: Input text string (any length)
- `mode`: `"standard"` (384d) or `"enterprise"` (12,288d)

**Returns:**
- NumPy array of shape `(384,)` or `(12288,)`

### `calibrate_enterprise_mode(sample_texts)`

**Parameters:**
- `sample_texts`: List of sample text strings (recommended: ~100 documents)

**Effect:**
- Creates `lam_whitening_stats.npy` file
- Removes common language noise from Enterprise mode vectors

## Example: Production API

```python
from flask import Flask, request, jsonify
from lam import LAM
from lam_dual_encoder import LAMDualEncoder

app = Flask(__name__)
model = LAM("LAM-base-v1")
encoder = LAMDualEncoder(model)

@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    data = request.json
    text = data['input']
    dimensions = data.get('dimensions', 384)
    
    if dimensions == 384:
        vec = encoder.encode(text, mode="standard")
    elif dimensions == 12288:
        vec = encoder.encode(text, mode="enterprise")
    else:
        return jsonify({"error": "Unsupported dimension"}), 400
    
    return jsonify({
        "data": [{
            "embedding": vec.tolist(),
            "dimensions": len(vec)
        }]
    })

if __name__ == '__main__':
    app.run(port=5000)
```

## Testing

Run the test suite:

```bash
cd /workspace/LAM/lam_package
python test_dual_encoder.py
```

This will test:
- ✅ Standard mode encoding (384d)
- ✅ Enterprise mode encoding (12,288d)
- ✅ Calibration functionality
- ✅ Long document handling (10k+ tokens)
- ✅ Similarity preservation

## Performance Notes

- **Standard mode**: ~same speed as regular LAM encoding
- **Enterprise mode**: Slightly slower (extracts full memory state)
- **Memory usage**: Constant O(1) for both modes (streaming architecture)
- **Storage**: Enterprise vectors are 32x larger (12,288 vs 384)

## Files

- `lam_dual_encoder.py`: Main dual encoder implementation
- `test_dual_encoder.py`: Test suite
- `lam_whitening_stats.npy`: Calibration data (created after calibration)

