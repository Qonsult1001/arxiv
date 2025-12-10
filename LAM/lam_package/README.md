---
language: en
license: apache-2.0
library_name: lam-attn
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

# LAM (Linear Attention Model)

**Infinite context (32k+) with O(N) complexity for Enterprise RAG.**

LAM is a high-performance embedding model that provides linear complexity attention, enabling efficient processing of long documents and sequences up to 32,768 tokens.

## Installation

### Method 1: pip (Recommended)

The primary and recommended installation method for all platforms:

```bash
pip install lam-attn
```

**Platform-Specific pip Commands:**

#### Windows
```powershell
# Standard pip installation
pip install lam-attn

# If pip is not in PATH, use:
python -m pip install lam-attn

# Or with Python 3 explicitly:
python3 -m pip install lam-attn
```

#### Linux
```bash
# Using pip
pip install lam-attn

# Or using pip3
pip3 install lam-attn

# For system-wide installation (may require sudo)
sudo pip3 install lam-attn

# Using apt package manager (if available)
sudo apt update
sudo apt install python3-pip
pip3 install lam-attn
```

#### macOS
```bash
# Using pip
pip install lam-attn

# Or using pip3
pip3 install lam-attn

# Using Homebrew (if Python installed via Homebrew)
brew install python3
pip3 install lam-attn
```

### Method 2: Conda/Anaconda

For users who prefer the Anaconda distribution:

**Note:** LAM is published to PyPI only. You can use pip within conda environments (pip works perfectly in conda):

```bash
# Create and activate conda environment
conda create -n lam python=3.10
conda activate lam

# Install LAM using pip (pip works inside conda)
pip install lam-attn
```

**Why this works:** Conda environments include pip, so you can use `pip install lam-attn` just like in a regular Python environment. No separate conda package needed!

**Conda Installation Steps:**

1. **Install Anaconda or Miniconda:**
   - Download from [anaconda.com](https://www.anaconda.com/download) (Windows, Linux, macOS)
   - Or install Miniconda: [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html)

2. **Create and activate environment:**
   ```bash
   conda create -n lam python=3.10
   conda activate lam
   ```

3. **Install LAM:**
   ```bash
   pip install lam-attn
   ```

### Method 3: Chocolatey (Windows)

For Windows users who prefer Chocolatey package manager:

**Note:** LAM is published to PyPI only. Chocolatey is used here to install Python, then you use pip:

```powershell
# Install Chocolatey first (if not installed)
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Python via Chocolatey (if needed)
choco install python

# Install LAM using pip (pip comes with Python)
pip install lam-attn
```

**Why this works:** Chocolatey installs Python (which includes pip), then you use pip to install LAM. No separate Chocolatey package needed!

### Method 4: Source Installation

For development or when you need the latest version:

```bash
# Clone the repository
git clone https://github.com/said-research/lam.git
cd lam/lam_package

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Method 5: Standalone Installers (Windows)

For Windows users who prefer GUI installers:

**Note:** Standalone `.exe` or `.msi` installers are not currently available. We recommend using pip or conda. If you need a standalone installer, please contact support@saidhome.ai.

**Alternative for Non-Technical Users:**
1. Install Python from [python.org](https://www.python.org/downloads/) (check "Add Python to PATH")
2. Open Command Prompt or PowerShell
3. Run: `pip install lam-attn`

### Requirements

**All Installation Methods:**
- Python >= 3.8
- PyTorch >= 2.0
- tokenizers >= 0.13.0
- numpy >= 1.20.0
- CUDA (optional, for GPU acceleration)

**Platform-Specific Requirements:**

#### Windows
- Python 3.8+ from [python.org](https://www.python.org/downloads/) or Microsoft Store
- pip (included with Python)
- CUDA Toolkit (optional, for GPU support)
- Visual C++ Redistributable (usually included with Python)

#### Linux
- Python 3.8+ (`sudo apt install python3 python3-pip` on Ubuntu/Debian)
- pip3 (`sudo apt install python3-pip`)
- Build tools: `sudo apt install build-essential` (for compiling extensions)
- CUDA (optional, for GPU support)

#### macOS
- Python 3.8+ from [python.org](https://www.python.org/downloads/) or Homebrew
- pip3 (included with Python)
- Xcode Command Line Tools: `xcode-select --install`
- MPS (Metal Performance Shaders) support for Apple Silicon (automatic with PyTorch)

### Troubleshooting Installation

**If pip doesn't work:**

1. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Use conda instead:**
   ```bash
   conda install -c conda-forge pip
   pip install lam-attn
   ```

3. **Install from source:**
   ```bash
   git clone https://github.com/said-research/lam.git
   cd lam/lam_package
   pip install -e .
   ```

4. **Check Python version:**
   ```bash
   python --version  # Should be >= 3.8
   ```

5. **Virtual environment (recommended):**
   ```bash
   # Create virtual environment
   python -m venv lam_env
   
   # Activate (Windows)
   lam_env\Scripts\activate
   
   # Activate (Linux/macOS)
   source lam_env/bin/activate
   
   # Install
   pip install lam-attn
   ```

## Usage

### Basic Usage

```python
from lam import LAM

# Load model (provide path to your LAM-base-v1 model directory)
# Works on Windows, Linux, and macOS
model = LAM('path/to/LAM-base-v1')
# Windows example: model = LAM(r'C:\Users\YourName\LAM-base-v1')
# Linux/macOS example: model = LAM('/home/username/LAM-base-v1')

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

LAM is designed for:
- **Retrieval-Augmented Generation (RAG)**: Long-context document retrieval
- **Semantic Search**: Finding similar documents or passages
- **Document Clustering**: Grouping similar documents
- **Sentence Similarity**: Computing semantic similarity between texts
- **Enterprise Applications**: Processing long documents efficiently

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- tokenizers >= 0.13.0
- numpy >= 1.20.0

## Documentation

For more information, visit: https://github.com/said-research/lam

## License

Apache 2.0
