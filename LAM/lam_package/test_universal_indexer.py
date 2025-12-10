#!/usr/bin/env python3
"""
Test Universal Neural Indexer
==============================
Tests the "One Formula to Rule Them All" approach.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from universal_neural_indexer import UniversalNeuralIndexer
from mteb.tasks import LEMBNarrativeQARetrieval

print("="*80)
print("🚀 UNIVERSAL NEURAL INDEXER TEST")
print("="*80)
print("One Formula to Rule Them All:")
print("  INDEX: W = W @ (I - α*kk^T) + β*vk^T")
print("  SEARCH: v = W^T @ k")
print("  SCORE: cos(v_retrieved, v_query)")
print("="*80)

# Initialize
model_path = "/workspace/LAM/best"
device = "cuda"

print(f"\n🔧 Initializing Universal Neural Indexer...")
indexer = UniversalNeuralIndexer(model_path, device=device)

# Load test documents
print("\n📚 Loading test documents from LEMBNarrativeQARetrieval...")
task = LEMBNarrativeQARetrieval()
task.load_data()

corpus = task.corpus
if corpus and 'test' in corpus:
    corpus = corpus['test']

# Use first 5 documents for testing
corpus_ids = list(corpus.keys())[:5]
test_corpus = []
for doc_id in corpus_ids:
    doc = corpus[doc_id]
    text = f"{doc.get('title', '')} {doc.get('body', doc.get('text', ''))}".strip()
    test_corpus.append({'text': text, '_id': doc_id})

print(f"   Using {len(test_corpus)} documents")

# INDEX: Build Neural Index (W matrices)
print("\n" + "="*80)
print("STEP 1: INDEX - Building Neural Index (W matrices)")
print("="*80)
indexer.index(test_corpus)

# Create test queries
print("\n" + "="*80)
print("STEP 2: SEARCH - Probing Neural Index with Queries")
print("="*80)

# Create queries based on document content
queries = []
for i, doc in enumerate(test_corpus[:3]):
    text = doc['text']
    # Create query from first sentence
    first_sentence = text.split('.')[0] if '.' in text else text[:100]
    if len(first_sentence) > 20:
        query = f"Tell me about {first_sentence[:50]}"
        queries.append(query)

# Add some generic queries
queries.extend([
    "What is the main topic?",
    "Can you explain the key concepts?",
])

print(f"   Created {len(queries)} test queries")

# SEARCH: Probe the Neural Index
results = indexer.search(queries, top_k=3)

# Display results
print("\n📊 SEARCH RESULTS:")
print("="*80)
for q_id, doc_scores in results.items():
    query_idx = int(q_id)
    query_text = queries[query_idx] if query_idx < len(queries) else f"Query {q_id}"
    print(f"\n   Query: '{query_text[:60]}...'")
    print(f"   Top results:")
    for rank, (doc_id, score) in enumerate(doc_scores.items(), 1):
        doc_idx = corpus_ids.index(doc_id) if doc_id in corpus_ids else -1
        print(f"      {rank}. Doc {doc_id} (score: {score:.4f})")

# Test with document-level queries (should match)
print("\n" + "="*80)
print("STEP 3: VERIFICATION - Testing Exact Matches")
print("="*80)

# Create queries that should match specific documents
exact_queries = {}
for i, doc in enumerate(test_corpus[:3]):
    text = doc['text']
    doc_id = doc['_id']
    
    # Create query from document title or first words
    title = text.split('\n')[0][:50] if '\n' in text else text[:50]
    query = f"What is {title}?"
    exact_queries[f"q_{i}"] = query

print(f"   Testing {len(exact_queries)} exact-match queries")

exact_results = indexer.search(exact_queries, top_k=1)

correct = 0
total = len(exact_queries)

for q_id, doc_scores in exact_results.items():
    query_idx = int(q_id.split('_')[1])
    expected_doc_id = test_corpus[query_idx]['_id']
    
    if doc_scores:
        best_doc_id = list(doc_scores.keys())[0]
        is_correct = (best_doc_id == expected_doc_id)
        
        if is_correct:
            correct += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"
        
        query_text = exact_queries[q_id]
        print(f"\n   Query: '{query_text[:50]}...'")
        print(f"      Expected: Doc {expected_doc_id}")
        print(f"      Retrieved: Doc {best_doc_id} (score: {list(doc_scores.values())[0]:.4f})")
        print(f"      Status: {status}")

accuracy = (correct / total) * 100 if total > 0 else 0.0
print(f"\n📊 RETRIEVAL ACCURACY: {correct}/{total} ({accuracy:.1f}%)")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n✅ Universal Neural Indexer Results:")
print(f"   - Indexed {len(indexer._doc_ids)} documents")
if indexer._neural_index is not None:
    print(f"   - Memory size: {indexer._neural_index.numel() * 4 / 1024**2:.2f} MB (ONE shared W matrix)")
else:
    print(f"   - Memory size: N/A")
print(f"   - Retrieval accuracy: {accuracy:.1f}%")

if accuracy >= 80:
    print(f"\n   🎉 EXCELLENT: Universal formula working!")
elif accuracy >= 60:
    print(f"\n   ✅ GOOD: Universal formula functional")
else:
    print(f"\n   ⚠️  Needs tuning")

print("\n💡 The One Formula:")
print("   INDEX: W = W @ (I - α*kk^T) + β*vk^T  (Build Brain)")
print("   SEARCH: v = W^T @ k                    (Probe Brain)")
print("   SCORE: cos(v_retrieved, v_query)        (Resonance)")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)

