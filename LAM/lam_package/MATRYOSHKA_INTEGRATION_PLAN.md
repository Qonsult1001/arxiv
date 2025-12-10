# Matryoshka Projection Integration Plan

## 📋 Current State Analysis

### Where Matryoshka is Currently Used:
1. **`test_8k_LAM.py`** (lines 15-52):
   - Has `MatryoshkaProjection` class with learnable LayerNorm per dimension
   - Used in `DeltaNet.encode()` method (lines 156-185)
   - For 384-dim: Returns raw embeddings (bypasses projection)
   - For 64/128/256: Uses LayerNorm + L2 normalization

2. **`scientific_mrl_benchmark.py`**:
   - For 384-dim: Calls `self.model.model.get_sentence_embeddings()` directly (line 743)
   - For lower dims: Calls `self.model.encode(batch_s1, dimensions=dim)` (line 756)

### Current `lam/__init__.py` Structure:
- **`LAM.encode()`** (lines 123-183): User-facing API
  - Currently: Always returns 384-dim embeddings
  - NO `dimensions` parameter support
  - Calls `self._model.get_sentence_embeddings()` which returns 384-dim

- **`_LAMModel.get_sentence_embeddings()`** (lines 524-534):
  - Returns normalized 384-dim embeddings
  - No Matryoshka support

## 🎯 Integration Point: OPTION 1 (Simple Truncation)

### Recommended Approach:
**Simple truncation + normalization** (NO learnable LayerNorm needed for inference)

### Why This Works:
- Your model already has natural Matryoshka properties:
  - 97% retention at 64-dim (NO training needed)
  - 100.1% quality at 256-dim (better than 384-dim!)
- This is emergent from your architecture
- No need to load projection weights for inference

### Where to Add It:

#### **Location: `lam/__init__.py` - `LAM.encode()` method**

**Current code (lines 123-183):**
```python
def encode(
    self,
    sentences: Union[str, List[str]],
    batch_size: int = 32,
    show_progress_bar: bool = False,
    convert_to_numpy: bool = True,
    convert_to_tensor: bool = False,
    normalize_embeddings: bool = True,
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    # ... existing code ...
    embeddings = self._model.get_sentence_embeddings(
        tokens['input_ids'],
        tokens['attention_mask']
    )
    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    all_embeddings.append(embeddings.cpu())
```

**What to Add:**
1. Add `dimensions: Optional[int] = None` parameter to `encode()` signature
2. After getting 384-dim embeddings, truncate if `dimensions < 384`:
   ```python
   # Get full 384-dim embedding
   embeddings = self._model.get_sentence_embeddings(...)
   
   # Matryoshka truncation (if dimensions specified)
   if dimensions is not None and dimensions < 384:
       if dimensions not in [64, 128, 256, 384]:
           raise ValueError(f"dimensions must be one of [64, 128, 256, 384], got {dimensions}")
       # Simple truncation: slice first N dimensions
       embeddings = embeddings[:, :dimensions]
       # Normalize truncated embeddings
       embeddings = F.normalize(embeddings, p=2, dim=1)
   elif normalize_embeddings:
       # Normalize full 384-dim (existing behavior)
       embeddings = F.normalize(embeddings, p=2, dim=1)
   ```

### Why NOT in `_core.py` (compiled):
- This is **inference-only** logic (simple Python operations)
- No need to compile: slicing and normalization are fast
- Keeps the compiled core focused on the DeltaNet computation
- Easier to maintain and debug

### Why NOT in `_LAMModel.get_sentence_embeddings()`:
- `get_sentence_embeddings()` should return the full 384-dim (raw output)
- Matryoshka truncation is a **user-facing feature** (API level)
- Allows flexibility: users can get full 384-dim or truncated versions

## 📊 Implementation Details

### Supported Dimensions:
- `64`: For small databases (≤20K docs)
- `128`: For mid-sized databases (≤1.5M docs)
- `256`: For large databases (≤50M docs)
- `384`: Full dimension (default)

### Usage Example:
```python
from lam import LAM

model = LAM('LAM-base-v1')

# Full 384-dim (default)
embeddings_384 = model.encode(['Hello world'])

# Truncated to 64-dim (for small vector DBs)
embeddings_64 = model.encode(['Hello world'], dimensions=64)

# Truncated to 256-dim (for large vector DBs)
embeddings_256 = model.encode(['Hello world'], dimensions=256)
```

### Compatibility:
- **Backward compatible**: `dimensions=None` (default) returns 384-dim (existing behavior)
- **Matches `scientific_mrl_benchmark.py` usage**: `model.encode(batch, dimensions=dim)`
- **No breaking changes**: Existing code continues to work

## 🔄 Comparison with `test_8k_LAM.py`:

### `test_8k_LAM.py` (Training/Research):
- Uses learnable `MatryoshkaProjection` with LayerNorm
- Loads projection weights from checkpoint (if available)
- More complex but potentially better quality

### `lam/__init__.py` (Production/Inference):
- Simple truncation + normalization
- No weights to load
- Simpler, faster, works out-of-the-box
- Your model already has natural Matryoshka properties

## ✅ Summary

**Add to:** `lam/__init__.py` → `LAM.encode()` method (line 123)

**Implementation:** Simple truncation + normalization (OPTION 1)

**Why here:**
- User-facing API level
- Python code (no compilation needed)
- Backward compatible
- Matches usage pattern in `scientific_mrl_benchmark.py`

**NOT needed in:**
- `_core.py` (compiled) - this is inference logic, not core computation
- `_LAMModel.get_sentence_embeddings()` - should return full 384-dim






