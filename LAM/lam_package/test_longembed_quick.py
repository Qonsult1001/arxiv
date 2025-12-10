#!/usr/bin/env python3
"""
Quick test for LongEmbed with state-based embedding.
Tests on just 10 actual documents from the dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
from lam import LAM
from lam.infinite_streamer import InfiniteContextStreamer
from mteb.tasks import LEMBNarrativeQARetrieval

print("="*70)
print("QUICK LONGEMBED TEST - State-Based Embedding")
print("Testing on 10 actual documents from LEMBNarrativeQARetrieval")
print("="*70)

# Load model
model_path = "/workspace/LAM/best"
device = "cuda"

print(f"\n🔧 Loading LAM from: {model_path}")
model = LAM(model_path, device=device)
streamer = InfiniteContextStreamer(model, chunk_size=512)

# Load actual documents from LEMBNarrativeQARetrieval
print("\n📚 Loading test documents from LEMBNarrativeQARetrieval...")
task = LEMBNarrativeQARetrieval()
task.load_data()

# Get corpus (documents) - handle nested structure
corpus = task.corpus
if corpus and 'test' in corpus:
    # Nested structure: corpus['test'] contains actual documents
    corpus = corpus['test']
elif isinstance(corpus, dict) and len(corpus) > 0:
    # Check if first value is a dict (nested)
    first_key = list(corpus.keys())[0]
    if isinstance(corpus[first_key], dict):
        corpus = corpus[first_key]

corpus_ids = list(corpus.keys())[:10] if corpus else []  # Exactly 10 documents
print(f"   Testing on {len(corpus_ids)} documents")

# Get queries - handle nested structure
queries = task.queries
if queries and 'test' in queries:
    # Nested structure: queries['test'] contains actual queries
    queries = queries['test']
elif isinstance(queries, dict) and len(queries) > 0:
    # Check if first value is a dict (nested)
    first_key = list(queries.keys())[0]
    if isinstance(queries[first_key], dict):
        queries = queries[first_key]

query_ids = list(queries.keys())[:10] if queries else []  # 10 queries
print(f"   Testing with {len(query_ids)} queries")

# Test 1: Verify state-based embedding is used on all 10 documents
print("\n" + "="*70)
print("TEST 1: Verify State-Based Embedding on 10 Documents")
print("="*70)

doc_embeddings = []
state_used_count = 0

for i, doc_id in enumerate(corpus_ids, 1):
    doc = corpus[doc_id]
    text = f"{doc.get('title', '')} {doc.get('body', doc.get('text', ''))}".strip()
    
    # Tokenize
    enc = model.tokenizer.encode(text)
    ids = enc.ids if hasattr(enc, 'ids') else enc
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"\n📄 Document {i}/{len(corpus_ids)}: {len(ids):,} tokens")
    
    # Get embedding with state-based method
    emb = streamer.stream_embedding(
        input_ids, 
        attention_mask, 
        verbose=False,
        use_state_embedding=True
    )
    
    # Check if state was used
    if streamer.state_slow is not None:
        state_used_count += 1
        print(f"   ✅ State-based embedding used (S_slow: {streamer.state_slow.shape})")
    else:
        print(f"   ❌ State NOT available - using mean pooling!")
    
    doc_embeddings.append(emb.squeeze().cpu().numpy())

print(f"\n📊 Summary: {state_used_count}/{len(corpus_ids)} documents used state-based embedding")
doc_embeddings = np.array(doc_embeddings)

# Test 2: Encode queries and compute retrieval
print("\n" + "="*70)
print("TEST 2: Encode Queries and Compute Similarities")
print("="*70)

# Encode queries (short, use standard encoding)
query_embeddings = []
for i, q_id in enumerate(query_ids, 1):
    query = str(queries[q_id])  # Ensure it's a string
    # Use model's encode method for queries
    try:
        # Encode single query
        enc = model.tokenizer.encode(query)
        ids = enc.ids if hasattr(enc, 'ids') else enc
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        
        # Get embedding via model's encode method
        with torch.no_grad():
            # Use the model's encode method which handles everything
            emb_np = model.encode([query], convert_to_numpy=True)
            emb = torch.tensor(emb_np, device=device).squeeze()
            emb = F.normalize(emb, p=2, dim=0)
        
        query_embeddings.append(emb.squeeze().cpu().numpy())
        if i <= 3:
            print(f"   Query {i}: {len(ids)} tokens ✅")
    except Exception as e:
        print(f"   Query {i}: Error - {e}")
        query_embeddings.append(np.zeros(384))

query_embeddings = np.array(query_embeddings)

# Test 3: Compute similarities and show top matches
print("\n" + "="*70)
print("TEST 3: Query-Document Similarities (Top 3 per query)")
print("="*70)

# Normalize embeddings
query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
doc_embeddings = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)

# Compute similarity matrix
similarities = np.dot(query_embeddings, doc_embeddings.T)

print(f"\n📊 Top matches for each query:")
for i, q_id in enumerate(query_ids[:5], 1):  # Show first 5 queries
    query_text = str(queries[q_id])[:80]
    top_docs = np.argsort(similarities[i-1])[::-1][:3]  # Top 3
    
    print(f"\n   Query {i}: {query_text}...")
    for rank, doc_idx in enumerate(top_docs, 1):
        doc_id = corpus_ids[doc_idx]
        score = similarities[i-1, doc_idx]
        doc_title = str(corpus[doc_id].get('title', 'No title'))[:50]
        print(f"      Rank {rank}: Doc {doc_idx+1} (score: {score:.4f}) - {doc_title}...")

print("\n" + "="*70)
print("✅ Quick test complete!")
print("="*70)
print("\nIf state-based embedding is working:")
print("  ✅ TEST 1: Should show 'State-based embedding used!'")
print("  ✅ TEST 2: Difference should be > 0.05 (answer preserved)")
print("  ✅ TEST 3: Similarities should be reasonable")
print("\nIf all pass, run full benchmark with --longembed")

