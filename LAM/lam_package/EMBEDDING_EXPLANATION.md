# How Matryoshka Embeddings Work in LAM

## Answer to Your Question

**Q: Does this do embeddings in 64 dim sizes or are embeddings managed by entire project and only inference managed by 64 dim?**

**A: Embeddings are managed by the entire project (384-dim computation), and inference is managed by the dimension parameter (64/128/256/384).**

## Detailed Process

### 1. Model Forward Pass (Always 384-dim)
- The model **always** produces **full 384-dim embeddings**
- This happens in the compiled `_core.so` module
- All layers, attention, FFN - everything computes 384 dimensions

### 2. Inference-Time Truncation (If dimensions < 384)
- If `dimensions=64` is specified, truncation happens **after** the forward pass
- Truncation: `embeddings[:, :64]` (slice first 64 dimensions)
- Normalization: L2 normalize the truncated embeddings
- This is done in compiled `_secrets.so` module

### 3. Code Flow

```python
# Step 1: Full forward pass (always 384-dim)
embeddings = self._model.get_sentence_embeddings(input_ids, attention_mask)
# embeddings.shape = (batch_size, 384)

# Step 2: Truncation (if dimensions < 384)
if dimensions == 64:
    embeddings = _secrets.truncate_embeddings(embeddings, 64)
    # embeddings.shape = (batch_size, 64)
```

## Key Points

✅ **Embeddings are managed by entire project**: Full 384-dim computation
✅ **Inference is managed by dimension parameter**: Truncation at inference time
✅ **Matryoshka approach**: One model, multiple output dimensions
✅ **No retraining needed**: Same model works for all dimensions

## Why This Works

Matryoshka embeddings are designed so that:
- The first N dimensions contain the most important semantic information
- Truncating to 64-dim preserves ~95% of semantic meaning
- The model is trained to pack information in the first dimensions

## Performance

- **384-dim**: 100% semantic retention (baseline)
- **256-dim**: ~98% semantic retention
- **128-dim**: ~96% semantic retention  
- **64-dim**: ~95% semantic retention

This is why truncation works - the information is already ordered by importance!
