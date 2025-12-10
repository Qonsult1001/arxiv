# Matryoshka Embeddings API

## API Difference

**lam_embed API:**
```python
from lam_embed import LAMModel

model = LAMModel.from_pretrained("saidresearch/lam-31m")
embeddings = model.encode(sentences, dimension=64)  # Note: singular "dimension"
```

**LAM API:**
```python
from lam import LAM

model = LAM("LAM-base-v1", backend='cython')  # or 'jax'
embeddings = model.encode(sentences, dimensions=64)  # Note: plural "dimensions"
```

## Key Differences

1. **Import**: `from lam import LAM` (not `LAMModel`)
2. **Model Loading**: `LAM("path/to/model")` (not `from_pretrained()`)
3. **Parameter Name**: `dimensions=` (plural) instead of `dimension=` (singular)
4. **Backend Selection**: `backend='cython'` or `backend='jax'`

## Usage Examples

### Basic Usage
```python
from lam import LAM

# Load model
model = LAM("LAM-base-v1", backend='cython')

# Encode with different dimensions
embeddings_64 = model.encode(sentences, dimensions=64)
embeddings_128 = model.encode(sentences, dimensions=128)
embeddings_256 = model.encode(sentences, dimensions=256)
embeddings_384 = model.encode(sentences, dimensions=384)  # or omit for default
```

### Semantic Similarity Test
```python
from lam import LAM
import numpy as np

model = LAM("LAM-base-v1", backend='cython')

# Test sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast animal leaps over a sleeping canine."
]

# Get embeddings
embeddings = model.encode(sentences, dimensions=128)

# Compute similarity
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)

print(f"Semantic similarity: {similarity:.4f}")
```

## Supported Dimensions

- **64**: For small databases (≤20K docs) - ~95% semantic retention
- **128**: For mid-sized databases (≤1.5M docs) - ~96% semantic retention
- **256**: For large databases (≤50M docs) - ~98% semantic retention
- **384**: Full dimension (default) - 100% semantic retention

## Testing

Run the semantic similarity test:
```bash
# Test with Cython backend
python test_semantic_matryoshka.py --backend cython

# Test with JAX backend
python test_semantic_matryoshka.py --backend jax

# Test with subset (faster)
python test_semantic_matryoshka.py --backend cython --subset 100
```

