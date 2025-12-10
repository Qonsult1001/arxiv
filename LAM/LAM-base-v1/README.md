---
language: en
license: apache-2.0
library_name: lam
tags:
- rag
- retrieval-augmented-generation
- long-context
- infinite-context
- generative-ai
- llm
- foundation-models
- linear-attention
- transformers-alternative
- efficient-nlp
- document-analysis
- enterprise-ai
- memory-efficient
- pytorch
- deep-learning
pipeline_tag: sentence-similarity
---

# LAM-base-v1

**Infinite context (32k+) with O(N) complexity for Enterprise RAG.**

LAM-base-v1 is a high-performance embedding model that provides linear complexity attention, enabling efficient processing of long documents and sequences up to 32,768 tokens.

## Installation

### Step 1: Install LAM Package

Choose your preferred installation method:

#### Method 1: pip (Recommended)

**Windows:**
```powershell
pip install lam-attn
# Or: python -m pip install lam-attn
```

**Linux:**
```bash
pip install lam-attn
# Or: pip3 install lam-attn
# System-wide: sudo pip3 install lam-attn
```

**macOS:**
```bash
pip install lam-attn
# Or: pip3 install lam-attn
```

#### Method 2: Conda/Anaconda

For Anaconda users:

**Note:** LAM is published to PyPI only. Use pip within conda (pip works perfectly in conda environments):

```bash
# Create environment
conda create -n lam python=3.10
conda activate lam

# Install LAM using pip (pip is included in conda environments)
pip install lam-attn
```

**Why this works:** Conda environments include pip, so you can use `pip install lam-attn` just like in a regular Python environment. No separate conda package needed!

#### Method 3: Chocolatey (Windows)

**Note:** LAM is published to PyPI only. Chocolatey is used to install Python, then you use pip:

```powershell
# Install Python via Chocolatey (if needed)
choco install python

# Then install LAM using pip (pip comes with Python)
pip install lam-attn
```

**Why this works:** Chocolatey installs Python (which includes pip), then you use pip to install LAM. No separate Chocolatey package needed!

#### Method 4: Source Installation

```bash
git clone https://github.com/said-research/lam.git
cd lam/lam_package
pip install -e .
```

#### Troubleshooting

**If pip doesn't work:**
- Upgrade pip: `python -m pip install --upgrade pip`
- Use conda: `conda install pip` then `pip install lam-attn`
- Use virtual environment: `python -m venv lam_env` then activate and install

### Step 2: Download Model Files

Download this model folder (`LAM-base-v1`) which contains the model weights and configuration files.

**Model Path Examples:**
- **Windows**: `C:\Users\YourName\LAM-base-v1\` or `.\LAM-base-v1\`
- **Linux/macOS**: `~/LAM-base-v1/` or `./LAM-base-v1/`

## Quick Start

```python
from lam import LAM

# Load model (point to this LAM-base-v1 directory)
# Works on Windows, Linux, and macOS
model = LAM('LAM-base-v1')  # Relative path
# Or use full path:
# Windows: model = LAM(r'C:\Users\YourName\LAM-base-v1')
# Linux/macOS: model = LAM('/home/username/LAM-base-v1')

# Encode sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming how we process information."
]
embeddings = model.encode(sentences)

# Compute cosine similarity
import numpy as np
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)
print(f"Similarity: {similarity:.4f}")
```

## Usage

### Basic Usage

```python
from lam import LAM

# Load model (provide path to this LAM-base-v1 directory)
model = LAM('LAM-base-v1')

# Encode sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming how we process information."
]
embeddings = model.encode(sentences)

# Compute cosine similarity
import numpy as np
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)
print(f"Similarity: {similarity:.4f}")
```

### Batch Processing

```python
# Encode multiple sentences in batches
# batch_size=32 means process up to 32 sentences at a time
sentences = ["Sentence 1", "Sentence 2", "Sentence 3", ...]
embeddings = model.encode(sentences, batch_size=32)
```

### Long Documents

```python
# Process long documents up to 32k tokens
long_document = "..."  # Up to 32,768 tokens
embeddings = model.encode([long_document], max_length=32768)
```

## Features

- **O(n) Linear Complexity**: Linear scaling instead of quadratic attention
- **32K Token Context**: Process documents up to 32,768 tokens (64x longer than BERT)
- **Fast Inference**: ~14ms for 128 tokens, ~17K tokens/sec throughput
- **Enterprise Ready**: Optimized for production with fast tokenization and batching

## Model Specifications

- **Embedding Dimension**: 384
- **Max Context Length**: 32,768 tokens
- **Architecture**: 6-layer linear attention
- **Performance**: STS-B Spearman 0.7711, Pearson 0.7787

## Intended Uses

LAM-base-v1 is designed for:
- **Retrieval-Augmented Generation (RAG)**: Long-context document retrieval
- **Semantic Search**: Finding similar documents or passages
- **Document Clustering**: Grouping similar documents
- **Sentence Similarity**: Computing semantic similarity between texts
- **Enterprise Applications**: Processing long documents efficiently

## 🧠 Model Weights & Pre-training

This repository contains the **LAM Architecture** implementation (Apache 2.0). 

It allows researchers and developers to train their own linear attention models from scratch using our verified O(N) implementation.

### Accessing SOTA Weights

The pre-trained weights (Optimized for LAM, SOTA Score) are available exclusively through the **SaidHome API (.said)**.

- **Architecture:** Open Source (Apache 2.0)
- **Pre-trained Weights:** Proprietary / Licensed
- **API Access:** [saidhome.ai](https://saidhome.ai)

### Model Files

This directory contains the architecture and configuration files:

```
LAM-base-v1/
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights (~110 MB) - See licensing below
├── tokenizer.json           # Tokenizer configuration
├── tokenizer_config.json    # Tokenizer settings
├── vocab.txt                # Vocabulary file
├── special_tokens_map.json  # Special tokens mapping
└── README.md                # This file
```

**Note:** The `pytorch_model.bin` file in this repository is provided for research and development purposes. For production use of optimized pre-trained weights, please access through the SaidHome API.

## Requirements

### All Platforms

- Python >= 3.8
- PyTorch >= 2.0
- tokenizers >= 0.13.0
- numpy >= 1.20.0
- lam-attn package (install with: `pip install lam-attn`)

### Platform-Specific

#### Windows
- Python 3.8+ (from [python.org](https://www.python.org/downloads/) or Microsoft Store)
- pip (usually included with Python)
- CUDA Toolkit (optional, for GPU support)

#### Linux
- Python 3.8+ (`sudo apt install python3 python3-pip` on Ubuntu/Debian)
- pip3 (`sudo apt install python3-pip`)
- CUDA (optional, for GPU support)

#### macOS
- Python 3.8+ (from [python.org](https://www.python.org/downloads/) or Homebrew)
- pip3 (usually included)
- MPS (Metal Performance Shaders) support for Apple Silicon (automatic with PyTorch)

## Documentation

For more information, visit: https://github.com/said-research/lam

## License

**Architecture & Code:** Apache 2.0 (Open Source)

The LAM architecture implementation, code, and this repository are licensed under Apache 2.0, allowing researchers and developers to:
- Use the architecture for research and development
- Train their own models from scratch
- Modify and extend the implementation
- Commercial use (subject to Apache 2.0 terms)

**Pre-trained Weights:** Proprietary / Licensed

The pre-trained model weights (`pytorch_model.bin`) are proprietary and subject to separate licensing terms. For access to optimized SOTA weights, please contact us through [saidhome.ai](https://saidhome.ai).
