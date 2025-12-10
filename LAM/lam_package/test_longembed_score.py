#!/usr/bin/env python3
"""
Compute actual MTEB score for LongEmbed test on 10 documents.
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
print("LONGEMBED SCORE CALCULATION - 10 Documents")
print("="*70)

# Load model
model_path = "/workspace/LAM/best"
device = "cuda"

print(f"\n🔧 Loading LAM from: {model_path}")
model = LAM(model_path, device=device)
streamer = InfiniteContextStreamer(model, chunk_size=512)

# Load documents
print("\n📚 Loading test documents from LEMBNarrativeQARetrieval...")
task = LEMBNarrativeQARetrieval()
task.load_data()

# Get corpus and queries (handle nested structure)
corpus = task.corpus
if corpus and 'test' in corpus:
    corpus = corpus['test']

queries = task.queries
if queries and 'test' in queries:
    queries = queries['test']

# Use exactly 10 documents and their corresponding queries
corpus_ids = list(corpus.keys())[:10]
print(f"   Testing on {len(corpus_ids)} documents")

# Get queries that have relevance judgments for these documents
qrels = task.relevant_docs if hasattr(task, 'relevant_docs') else {}
if isinstance(qrels, dict) and 'test' in qrels:
    qrels = qrels['test']

# Find queries that have relevant docs in our 10-doc subset
valid_query_ids = []
for q_id, rel_docs in qrels.items():
    if any(doc_id in corpus_ids for doc_id in rel_docs.keys()):
        valid_query_ids.append(q_id)
    if len(valid_query_ids) >= 10:
        break

if not valid_query_ids:
    # Fallback: just use first 10 queries
    valid_query_ids = list(queries.keys())[:10]

print(f"   Testing with {len(valid_query_ids)} queries")

# Encode documents
print("\n📊 Encoding documents...")
doc_embeddings = []
for i, doc_id in enumerate(corpus_ids, 1):
    doc = corpus[doc_id]
    text = f"{doc.get('title', '')} {doc.get('body', doc.get('text', ''))}".strip()
    
    enc = model.tokenizer.encode(text)
    ids = enc.ids if hasattr(enc, 'ids') else enc
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    
    emb = streamer.stream_embedding(
        input_ids, 
        attention_mask, 
        verbose=False,
        use_state_embedding=True
    )
    doc_embeddings.append(emb.squeeze().cpu().numpy())
    if i % 3 == 0:
        print(f"   Encoded {i}/{len(corpus_ids)} documents...")

doc_embeddings = np.array(doc_embeddings)
print(f"   ✅ Document embeddings shape: {doc_embeddings.shape}")

# Encode queries
print("\n📊 Encoding queries...")
query_embeddings = []
for i, q_id in enumerate(valid_query_ids, 1):
    query = str(queries[q_id])
    emb_np = model.encode([query], convert_to_numpy=True)
    query_embeddings.append(emb_np.squeeze())
    if i % 3 == 0:
        print(f"   Encoded {i}/{len(valid_query_ids)} queries...")

query_embeddings = np.array(query_embeddings)
print(f"   ✅ Query embeddings shape: {query_embeddings.shape}")

# Normalize
query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
doc_embeddings = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)

# Compute similarity matrix
similarities = np.dot(query_embeddings, doc_embeddings.T)
print(f"\n📊 Similarity matrix shape: {similarities.shape}")

# Compute MTEB metrics
print("\n" + "="*70)
print("COMPUTING MTEB METRICS")
print("="*70)

# Manual NDCG computation

# Build relevance judgments for our subset
subset_qrels = {}
for q_id in valid_query_ids:
    if q_id in qrels:
        rel_docs = {}
        for doc_id, score in qrels[q_id].items():
            if doc_id in corpus_ids:
                # Map to index in our subset
                doc_idx = corpus_ids.index(doc_id)
                rel_docs[doc_idx] = score
        if rel_docs:
            subset_qrels[q_id] = rel_docs

print(f"   Found relevance judgments for {len(subset_qrels)} queries")

# Compute NDCG@10 for each query
ndcg_scores = []
for i, q_id in enumerate(valid_query_ids):
    if q_id not in subset_qrels:
        continue
    
    # Get similarity scores for this query
    query_sims = similarities[i]  # [num_docs]
    
    # Get relevant documents
    rel_docs = subset_qrels[q_id]
    
    # Sort documents by similarity
    sorted_indices = np.argsort(query_sims)[::-1]
    
    # Build relevance list (1 if relevant, 0 otherwise)
    relevance = np.zeros(len(corpus_ids))
    for doc_idx, score in rel_docs.items():
        relevance[doc_idx] = score if isinstance(score, (int, float)) else 1.0
    
    # Compute NDCG@10
    # NDCG = DCG / IDCG
    def dcg_at_k(relevance_scores, k=10):
        """Compute DCG@k"""
        relevance_scores = np.asarray(relevance_scores)[:k]
        if len(relevance_scores) == 0:
            return 0.0
        gains = 2 ** relevance_scores - 1
        discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
        return np.sum(gains / discounts)
    
    # Get relevance in sorted order
    sorted_relevance = relevance[sorted_indices]
    
    # DCG@10
    dcg = dcg_at_k(sorted_relevance, k=10)
    
    # IDCG@10 (ideal: all relevant docs at top)
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = dcg_at_k(ideal_relevance, k=10)
    
    # NDCG@10
    ndcg = dcg / idcg if idcg > 0 else 0.0
    ndcg_scores.append(ndcg)
    
    if i < 5:  # Show first 5
        print(f"   Query {i+1}: NDCG@10 = {ndcg:.4f}")

# Average NDCG@10
avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\n📊 Average NDCG@10: {avg_ndcg:.4f}")
print(f"📊 Number of queries evaluated: {len(ndcg_scores)}")
print(f"📊 Target score: 40.0 (NDCG@10 * 100)")

# Convert to percentage (MTEB reports as 0-100)
score_percentage = avg_ndcg * 100

print(f"\n🏆 LONGEMBED SCORE: {score_percentage:.2f}")
print(f"   Target: 40.0")

if score_percentage >= 40.0:
    print(f"   ✅ PASS! (Above target)")
elif score_percentage >= 30.0:
    print(f"   🟡 CLOSE (Below target but reasonable)")
else:
    print(f"   ❌ FAIL (Below target)")

print("\n" + "="*70)
print("NOTE: This is a small sample (10 docs). Full benchmark will be more accurate.")
print("="*70)

