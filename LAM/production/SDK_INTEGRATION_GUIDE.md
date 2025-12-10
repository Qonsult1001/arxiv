# LAM SDK/API Integration Guide

This guide shows how to integrate LAM into your SDK/API as a **drop-in replacement** for sentence-transformers.

---

## 🎯 Integration Strategy

Your SDK/API currently uses `sentence-transformers`. LAM provides an identical API, so integration requires minimal changes.

### Before (sentence-transformers):
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('LAM base model')
embeddings = model.encode(["text1", "text2"])
```

### After (LAM):
```python
from lam_wrapper import LAMEncoder as SentenceTransformer

model = SentenceTransformer('lam-base-v1')
embeddings = model.encode(["text1", "text2"])  # Identical API!
```

---

## 📦 Step 1: Package LAM Model

First, create the distribution package:

```bash
cd /home/user/LAM
python production/package_lam.py
```

This creates:
- `production/lam-base-v1/` - Complete model package (~110 MB)
- `production/lam-base-v1-dist.tar.gz` - Distribution archive

---

## 🔧 Step 2: Integrate into Your SDK/API

### Option A: Copy to SDK Repository

```bash
# Copy LAM package to your SDK repo
cp -r production/lam-base-v1 /path/to/your/sdk/models/

# Or extract from tarball
tar -xzf production/lam-base-v1-dist.tar.gz -C /path/to/your/sdk/models/
```

### Option B: Reference from LAM Repository

```python
# In your SDK/API code
import sys
sys.path.insert(0, '/home/user/LAM/production')

from lam_wrapper import LAMEncoder
```

---

## 💻 Step 3: Update SDK/API Code

### 3a. Create LAM Factory Function

```python
# In your SDK: models.py or similar

def load_embedding_model(model_name='lam-base-v1', use_lam=True):
    """
    Load embedding model (LAM or sentence-transformers)

    Args:
        model_name: Model name or path
        use_lam: Use LAM instead of sentence-transformers

    Returns:
        Model with .encode() method
    """
    if use_lam:
        from lam_wrapper import LAMEncoder
        return LAMEncoder(model_name)
    else:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)


# Usage in your SDK
model = load_embedding_model('models/lam-base-v1', use_lam=True)
embeddings = model.encode(texts)
```

### 3b. Environment Variable Control

```python
# Allow switching between LAM and sentence-transformers via env var

import os

USE_LAM = os.getenv('USE_LAM', 'true').lower() == 'true'

if USE_LAM:
    from lam_wrapper import LAMEncoder as SentenceTransformer
    MODEL_NAME = 'lam-base-v1'
else:
    from sentence_transformers import SentenceTransformer
    MODEL_NAME = 'LAM base model'

# Rest of code unchanged
model = SentenceTransformer(MODEL_NAME)
```

### 3c. Drop-in Replacement (Simplest)

```python
# At the top of your SDK files that use sentence-transformers

# OLD:
# from sentence_transformers import SentenceTransformer

# NEW:
from lam_wrapper import LAMEncoder as SentenceTransformer

# Rest of your code stays EXACTLY the same!
model = SentenceTransformer('lam-base-v1')
embeddings = model.encode(documents)
```

---

## 🔌 Step 4: API Endpoint Integration

### FastAPI Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from lam_wrapper import LAMEncoder

app = FastAPI()

# Load LAM model once at startup
model = LAMEncoder('models/lam-base-v1')

class EmbeddingRequest(BaseModel):
    texts: List[str]
    normalize: bool = True  # Already done by LAM

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str = "lam-base-v1"
    dimensions: int = 384

@app.post("/encode", response_model=EmbeddingResponse)
async def encode_texts(request: EmbeddingRequest):
    """Encode texts to embeddings using LAM"""

    embeddings = model.encode(
        request.texts,
        convert_to_numpy=True,
        normalize_embeddings=request.normalize
    )

    return EmbeddingResponse(
        embeddings=embeddings.tolist(),
        model="lam-base-v1",
        dimensions=384
    )

@app.get("/model/info")
async def model_info():
    """Get LAM model information"""
    return {
        "model": "lam-base-v1",
        "architecture": "Enhanced Hierarchical LAM",
        "performance": {
            "stsb_pearson": 0.836,
            "complexity": "O(n)",
            "max_context": "1M+ tokens"
        },
        "parameters": "22M",
        "dimensions": 384
    }
```

### Flask Example

```python
from flask import Flask, request, jsonify
from lam_wrapper import LAMEncoder
import numpy as np

app = Flask(__name__)

# Load LAM model
model = LAMEncoder('models/lam-base-v1')

@app.route('/encode', methods=['POST'])
def encode():
    """Encode texts to embeddings"""
    data = request.json
    texts = data.get('texts', [])

    if not texts:
        return jsonify({'error': 'No texts provided'}), 400

    embeddings = model.encode(texts, convert_to_numpy=True)

    return jsonify({
        'embeddings': embeddings.tolist(),
        'model': 'lam-base-v1',
        'dimensions': 384
    })

@app.route('/similarity', methods=['POST'])
def similarity():
    """Compute similarity between text pairs"""
    data = request.json
    text1 = data.get('text1')
    text2 = data.get('text2')

    if not text1 or not text2:
        return jsonify({'error': 'text1 and text2 required'}), 400

    embeddings = model.encode([text1, text2])

    # Cosine similarity (embeddings already normalized)
    similarity = np.dot(embeddings[0], embeddings[1])

    return jsonify({
        'similarity': float(similarity),
        'model': 'lam-base-v1'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

---

## 🧪 Step 5: Testing

### Unit Test Example

```python
import unittest
import numpy as np
from lam_wrapper import LAMEncoder

class TestLAMIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests"""
        cls.model = LAMEncoder('models/lam-base-v1')

    def test_encode_single_text(self):
        """Test encoding single text"""
        text = "This is a test"
        embeddings = self.model.encode(text)

        self.assertEqual(embeddings.shape, (1, 384))
        self.assertAlmostEqual(np.linalg.norm(embeddings[0]), 1.0, places=5)

    def test_encode_multiple_texts(self):
        """Test encoding multiple texts"""
        texts = ["First text", "Second text", "Third text"]
        embeddings = self.model.encode(texts)

        self.assertEqual(embeddings.shape, (3, 384))

    def test_similarity(self):
        """Test semantic similarity"""
        texts = [
            "The cat sits on the mat",
            "A cat is sitting on a mat",
            "The dog runs in the park"
        ]
        embeddings = self.model.encode(texts)

        # Similarity between text 0 and 1 (similar) should be higher
        sim_01 = np.dot(embeddings[0], embeddings[1])
        sim_02 = np.dot(embeddings[0], embeddings[2])

        self.assertGreater(sim_01, sim_02)

    def test_batch_size(self):
        """Test different batch sizes"""
        texts = ["Text " + str(i) for i in range(100)]

        # Small batch
        emb1 = self.model.encode(texts, batch_size=16)

        # Large batch
        emb2 = self.model.encode(texts, batch_size=64)

        # Results should be identical
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
```

---

## 📊 Step 6: Performance Comparison

### Benchmark Script

```python
import time
import numpy as np
from lam_wrapper import LAMEncoder

def benchmark_lam():
    """Benchmark LAM performance"""

    model = LAMEncoder('models/lam-base-v1')

    # Test texts
    texts = [
        "This is a test sentence for benchmarking LAM performance",
        "Another test sentence with different content",
        "The quick brown fox jumps over the lazy dog",
    ] * 100  # 300 texts total

    print("LAM Performance Benchmark")
    print("="*60)

    # Warmup
    _ = model.encode(texts[:10])

    # Benchmark encoding
    start = time.time()
    embeddings = model.encode(texts, batch_size=32)
    duration = time.time() - start

    print(f"Texts encoded: {len(texts)}")
    print(f"Total time: {duration:.2f}s")
    print(f"Throughput: {len(texts)/duration:.1f} texts/sec")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Average norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")

    # Similarity computation
    start = time.time()
    similarities = embeddings @ embeddings.T
    duration = time.time() - start

    print(f"\nSimilarity matrix: {similarities.shape}")
    print(f"Computation time: {duration:.3f}s")

    return embeddings

if __name__ == "__main__":
    benchmark_lam()
```

---

## 🚀 Step 7: Deployment

### Docker Integration

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy LAM model
COPY models/lam-base-v1 /app/models/lam-base-v1

# Copy LAM wrapper
COPY lam_wrapper.py /app/

# Copy API code
COPY api/ /app/api/

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
scikit-learn>=1.3.0
fastapi>=0.100.0
uvicorn>=0.23.0
```

---

## 🔒 Security Considerations

### What's Distributed

✅ **Included in package**:
- Pre-trained model weights (`lam_checkpoint.pt`)
- Base model weights (`pytorch_model.bin` from public LAM base model)
- Tokenizer files (from public LAM base model)
- Simple inference wrapper (`lam_wrapper.py`)

❌ **NOT included (kept private)**:
- Core formula (`final_solution_formula.py`) - **PROPRIETARY**
- Training scripts - **PROPRIETARY**
- Research documentation - **PROPRIETARY**

### License Enforcement

Add to your SDK/API:

```python
# lam_license.py

import hashlib
from datetime import datetime

def check_lam_license(license_key: str = None) -> bool:
    """
    Verify LAM license key

    Args:
        license_key: License key from environment or config

    Returns:
        bool: True if licensed, False otherwise
    """
    if license_key is None:
        import os
        license_key = os.getenv('LAM_LICENSE_KEY')

    if not license_key:
        print("⚠️  No LAM license key found. Using evaluation mode.")
        return False

    # Validate license key (implement your logic)
    # This is a placeholder - implement actual validation
    valid = validate_license_key(license_key)

    if valid:
        print("✅ LAM license validated")
    else:
        print("❌ Invalid LAM license key")

    return valid

def validate_license_key(key: str) -> bool:
    """Validate license key (implement your validation logic)"""
    # Placeholder - implement actual validation
    # Could check against database, verify signature, etc.
    return len(key) > 10  # Dummy check
```

---

## 📈 Monitoring & Logging

```python
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('lam_sdk')

def log_encoding(func):
    """Decorator to log LAM encoding operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start

        # Log metrics
        if hasattr(args[0], '__class__'):
            model_name = args[0].__class__.__name__
        else:
            model_name = 'LAMEncoder'

        texts_count = len(args[1]) if len(args) > 1 else 0

        logger.info(
            f"{model_name}.encode: {texts_count} texts, "
            f"{duration:.3f}s, "
            f"{texts_count/duration:.1f} texts/sec"
        )

        return result
    return wrapper

# Apply to LAM wrapper
from lam_wrapper import LAMEncoder
LAMEncoder.encode = log_encoding(LAMEncoder.encode)
```

---

## ✅ Integration Checklist

- [ ] Package LAM model using `package_lam.py`
- [ ] Copy LAM package to SDK repository
- [ ] Update imports to use `LAMEncoder`
- [ ] Test with existing SDK functionality
- [ ] Update API endpoints to use LAM
- [ ] Add LAM-specific configuration
- [ ] Implement license checking (if needed)
- [ ] Add logging and monitoring
- [ ] Update documentation
- [ ] Deploy and test in production environment

---

## 🆘 Troubleshooting

### Issue: Module not found

```python
# Solution: Add LAM to Python path
import sys
sys.path.insert(0, '/path/to/LAM/production')
from lam_wrapper import LAMEncoder
```

### Issue: Checkpoint not found

```python
# Solution: Verify checkpoint path
from pathlib import Path
checkpoint_path = Path('models/lam-base-v1/lam_checkpoint.pt')
assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
```

### Issue: CUDA out of memory

```python
# Solution: Use CPU or smaller batch size
model = LAMEncoder('lam-base-v1', device='cpu')
embeddings = model.encode(texts, batch_size=16)  # Smaller batch
```

---

## 📞 Support

For integration support, licensing questions, or technical issues:
- Email: [Your Email]
- Documentation: [Your Docs URL]
- Repository: https://github.com/Qonsult1001/LAM
