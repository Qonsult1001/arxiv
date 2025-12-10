#!/usr/bin/env python3
"""
LOCAL TEST: Semantic Similarity Retrieval with Long Documents (65K tokens)

This test demonstrates the challenge of semantic retrieval vs exact matching:
- Exact matching (needle-in-haystack): Works well ✅
- Semantic similarity: More challenging ⚠️

Tests with 2 real documents from LEMBNarrativeQARetrieval, encoded as full 65K token documents.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np

# Import LAM from lam package (proper import)
try:
    from lam import LAM, InfiniteContextStreamer
except ImportError:
    # Fallback: try parent directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from lam import LAM, InfiniteContextStreamer

from mteb.tasks import LEMBNarrativeQARetrieval

print("="*80)
print("SEMANTIC SIMILARITY RETRIEVAL TEST - Long Documents (65K tokens)")
print("="*80)
print("Testing: Can we retrieve documents using semantic similarity?")
print("Using: First 2 documents from LEMBNarrativeQARetrieval")
print("Using: LAM from lam package with InfiniteContextStreamer (streaming)")
print("="*80)

# Load model - use proper LAM model path
# Try common paths: /workspace/LAM/best or LAM-base-v1
model_paths = [
    "/workspace/LAM/best",
    "/workspace/LAM/LAM-base-v1",
    "LAM-base-v1"
]

model_path = None
for path in model_paths:
    from pathlib import Path
    if Path(path).exists() or path == "LAM-base-v1":
        model_path = path
        break

if model_path is None:
    model_path = "/workspace/LAM/best"  # Default fallback

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n🔧 Loading LAM from: {model_path}")
print(f"   Device: {device}")
print(f"   Using: lam package LAM with InfiniteContextStreamer (streaming)")

# Load LAM model (from lam package - proper import)
model = LAM(model_path, device=device)

# Create streamer for long content (65K tokens)
# Uses Sync-512 (production setting) - chunk_size=512
streamer = InfiniteContextStreamer(model, chunk_size=512)
print(f"   ✅ Streamer initialized (chunk_size=512 for long content)")

# Load documents
print("\n📚 Loading documents from LEMBNarrativeQARetrieval...")
task = LEMBNarrativeQARetrieval()
task.load_data()

# Get corpus and queries (handle nested structure)
corpus = task.corpus
if corpus and 'test' in corpus:
    corpus = corpus['test']

queries = task.queries
if queries and 'test' in queries:
    queries = queries['test']

# Use ONLY first 2 documents (as requested)
corpus_ids = list(corpus.keys())[:2]
print(f"   Using {len(corpus_ids)} documents for testing")

# Get document texts
documents = []
doc_titles = []
for doc_id in corpus_ids:
    doc = corpus[doc_id]
    title = doc.get('title', '')
    body = doc.get('body', doc.get('text', ''))
    full_text = f"{title} {body}".strip()
    documents.append(full_text)
    doc_titles.append(title)
    print(f"   📄 Doc {doc_id}: {len(full_text)} chars, title: '{title[:50]}...'")

# Expand documents to ~65K tokens using repetition (like test_8k_inference.py)
# Target: ~65K tokens = ~48K words (using 0.75 words/token ratio)
target_words = 48000  # ~65K tokens
print(f"\n📏 Expanding documents to ~{target_words:,} words (~65K tokens)...")

expanded_documents = []
for i, doc_text in enumerate(documents):
    # Count current words
    current_words = len(doc_text.split())
    
    # Calculate how many repetitions needed
    if current_words > 0:
        repetitions_needed = max(1, target_words // current_words)
        # Repeat document to reach target length
        expanded = (doc_text + " ") * repetitions_needed
        # Trim to exact target
        words = expanded.split()
        expanded = " ".join(words[:target_words])
    else:
        expanded = doc_text
    
    expanded_words = len(expanded.split())
    # Estimate tokens (rough: 0.75 words per token)
    estimated_tokens = int(expanded_words / 0.75)
    
    expanded_documents.append(expanded)
    print(f"   📄 Doc {i+1}: {expanded_words:,} words (~{estimated_tokens:,} tokens)")

# Encode documents as full embeddings (no chunking)
print("\n📊 Encoding documents as FULL embeddings (65K tokens each)...")
print("   Using: Standard encoding (full document as one embedding)")
print("   Note: This preserves global semantics but may have challenges with semantic similarity")

doc_embeddings = []
for i, doc_text in enumerate(expanded_documents, 1):
    print(f"\n   Encoding document {i}/{len(expanded_documents)}...")
    
    # Tokenize - handle both tokenizers library and transformers tokenizer
    try:
        # Try tokenizers library (Rust-based) - returns object with .ids
        enc = model.tokenizer.encode(doc_text)
        token_ids = enc.ids if hasattr(enc, 'ids') else enc
    except Exception:
        # Fallback: use transformers tokenizer
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, '__call__'):
            tokens = model.tokenizer(doc_text, return_tensors='pt', truncation=False)
            token_ids = tokens['input_ids'].squeeze().tolist()
        else:
            # Last resort: use encode method directly
            token_ids = model.tokenizer.encode(doc_text)
    
    print(f"      Tokens: {len(token_ids):,}")
    
    # Use streaming for long documents (>8K tokens)
    # This aligns with /lam_product/lam streaming long content approach
    if len(token_ids) > 8192:
        print(f"      Using InfiniteContextStreamer (>{8192} tokens, chunk_size=512)...")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        
        streamer.reset()
        emb = streamer.stream_embedding(
            input_ids, 
            attention_mask, 
            verbose=False
        )
        # Streamer returns [batch, dim] or [dim]
        if emb.dim() == 2:
            emb = emb.squeeze(0)  # Remove batch dimension
    else:
        # Use standard encoding for shorter documents
        print(f"      Using standard encoding (≤{8192} tokens)...")
        emb = model.encode([doc_text], convert_to_tensor=True)
        if emb.dim() == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
    
    # Handle dict returns (Matryoshka) - LAM may return dict with dimensions
    if isinstance(emb, dict):
        emb = emb.get(384, list(emb.values())[0])  # Use full dimension
    
    # Ensure tensor and move to CPU for storage
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)
    emb = emb.cpu()  # Store on CPU to save GPU memory
    
    doc_embeddings.append(emb.cpu())
    print(f"      ✅ Encoded: shape {emb.shape}")

doc_embeddings = torch.stack(doc_embeddings)  # [num_docs, emb_dim]
print(f"\n   ✅ All documents encoded: shape {doc_embeddings.shape}")

# Create semantic queries based on document content
print("\n🔍 Creating semantic queries...")
print("   These queries test SEMANTIC similarity (not exact matching)")

# Extract key topics from documents for queries
semantic_queries = []
query_doc_pairs = []  # Track which document each query should retrieve

for i, (doc_text, title) in enumerate(zip(documents, doc_titles)):
    # Create queries that are semantically related but not exact matches
    # Strategy: Extract key concepts and rephrase them
    
    # Simple approach: Use title as base for query
    if title and len(title.strip()) > 0:
        # Create a question-style query from title
        query = f"What is {title}?"
        semantic_queries.append(query)
        query_doc_pairs.append(i)  # Should retrieve document i
        print(f"   Query {len(semantic_queries)}: '{query[:60]}...' → Should retrieve Doc {i+1}")
    
    # Also create a more abstract query
    # Extract first sentence or key phrase
    first_sentence = doc_text.split('.')[0] if '.' in doc_text else doc_text[:100]
    if len(first_sentence) > 20:
        # Rephrase as question
        query = f"Tell me about {first_sentence[:50]}"
        semantic_queries.append(query)
        query_doc_pairs.append(i)
        print(f"   Query {len(semantic_queries)}: '{query[:60]}...' → Should retrieve Doc {i+1}")

# Ensure we have at least 2 queries per document
# If we don't have enough queries, create some generic ones
while len(semantic_queries) < 4:
    # Add generic queries - alternate between documents
    doc_idx = len(semantic_queries) % len(documents)
    generic_queries = [
        "What is the main topic discussed?",
        "Can you explain the key concepts?",
        "What information does this document contain?",
        "What are the important details?",
    ]
    query = generic_queries[len(semantic_queries) % len(generic_queries)]
    semantic_queries.append(query)
    query_doc_pairs.append(doc_idx)  # Assign to document based on index
    print(f"   Query {len(semantic_queries)}: '{query}' → Should retrieve Doc {doc_idx+1}")

# CRITICAL: Ensure query_doc_pairs matches semantic_queries length
assert len(query_doc_pairs) == len(semantic_queries), f"Mismatch: {len(query_doc_pairs)} pairs vs {len(semantic_queries)} queries"

print(f"\n   ✅ Created {len(semantic_queries)} semantic queries")
print(f"   ✅ Ground truth pairs: {len(query_doc_pairs)} (aligned)")

# Encode queries (short - use standard encoding)
print("\n📊 Encoding queries...")
query_embeddings = []
for i, query in enumerate(semantic_queries, 1):
    emb = model.encode([query], convert_to_tensor=True)
    if emb.dim() == 2 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    # Handle dict returns (Matryoshka)
    if isinstance(emb, dict):
        emb = emb.get(384, list(emb.values())[0])
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)
    query_embeddings.append(emb.cpu())  # Store on CPU
    if i <= 3:
        print(f"   Query {i}: '{query[:60]}...' → encoded")

query_embeddings = torch.stack(query_embeddings)  # [num_queries, emb_dim]
print(f"   ✅ All queries encoded: shape {query_embeddings.shape}")

# Normalize embeddings for cosine similarity
print("\n📐 Normalizing embeddings for cosine similarity...")
doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

# Move to device for computation (if GPU available)
if device == "cuda" and torch.cuda.is_available():
    doc_embeddings = doc_embeddings.to(device)
    query_embeddings = query_embeddings.to(device)

# Compute similarity matrix
print("\n🔍 Computing similarity matrix...")
similarities = torch.matmul(query_embeddings, doc_embeddings.T)  # [num_queries, num_docs]
similarities_np = similarities.cpu().numpy()

print(f"   Similarity matrix shape: {similarities_np.shape}")
print(f"   Similarity range: [{similarities_np.min():.4f}, {similarities_np.max():.4f}]")

# Test retrieval
print("\n" + "="*80)
print("RETRIEVAL RESULTS")
print("="*80)

correct_retrievals = 0
total_queries = len(semantic_queries)

for i, query in enumerate(semantic_queries):
    # Get similarity scores for this query
    query_sims = similarities_np[i]  # [num_docs]
    
    # Find best match
    best_doc_idx = np.argmax(query_sims)
    best_sim = query_sims[best_doc_idx]
    
    # Check if correct (we should always have ground truth now)
    if i < len(query_doc_pairs):
        expected_doc_idx = query_doc_pairs[i]
        is_correct = (best_doc_idx == expected_doc_idx)
        
        if is_correct:
            correct_retrievals += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"
    else:
        # This shouldn't happen if assertion passed, but handle gracefully
        expected_doc_idx = None
        is_correct = None
        status = "❓ UNKNOWN (no ground truth)"
    
    print(f"\n   Query {i+1}: '{query[:70]}...'")
    print(f"      Best match: Doc {best_doc_idx + 1} (similarity: {best_sim:.4f})")
    if expected_doc_idx is not None:
        print(f"      Expected: Doc {expected_doc_idx + 1}")
        print(f"      Status: {status}")
    else:
        print(f"      Status: {status}")
    
    # Show all similarities
    print(f"      All similarities: {[f'{s:.4f}' for s in query_sims]}")

# Calculate accuracy (only count queries with ground truth)
queries_with_gt = min(len(semantic_queries), len(query_doc_pairs))
if queries_with_gt > 0:
    accuracy = (correct_retrievals / queries_with_gt) * 100
    print(f"\n📊 RETRIEVAL ACCURACY: {correct_retrievals}/{queries_with_gt} ({accuracy:.1f}%)")
    if queries_with_gt < total_queries:
        print(f"   Note: {total_queries - queries_with_gt} queries without ground truth excluded")
else:
    accuracy = 0.0
    print(f"\n📊 RETRIEVAL ACCURACY: N/A (no ground truth)")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("\n🔍 Key Observations:")
print("   1. Documents are encoded as FULL embeddings (65K tokens each)")
print("   2. Queries use SEMANTIC similarity (not exact string matching)")
print("   3. This tests whether global document embeddings preserve semantic meaning")

# Check similarity distribution
all_sims = similarities_np.flatten()
print(f"\n📈 Similarity Statistics:")
print(f"   Mean: {all_sims.mean():.4f}")
print(f"   Std: {all_sims.std():.4f}")
print(f"   Min: {all_sims.min():.4f}")
print(f"   Max: {all_sims.max():.4f}")

# Compare to exact matching baseline
print(f"\n💡 Comparison:")
print(f"   - Exact matching (needle-in-haystack): Works well ✅")
print(f"   - Semantic similarity: More challenging ⚠️")
print(f"   - This test shows the difference between exact and semantic retrieval")

if accuracy >= 80:
    print(f"\n✅ GOOD: Semantic retrieval accuracy is {accuracy:.1f}%")
elif accuracy >= 50:
    print(f"\n⚠️  MODERATE: Semantic retrieval accuracy is {accuracy:.1f}%")
else:
    print(f"\n❌ POOR: Semantic retrieval accuracy is {accuracy:.1f}%")
    print(f"   This demonstrates the challenge of semantic similarity with long documents")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)

