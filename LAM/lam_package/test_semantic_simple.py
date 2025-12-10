#!/usr/bin/env python3
"""
Simple test to verify semantic similarity is working
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from universal_neural_indexer import UniversalNeuralIndexer

print("="*80)
print("SIMPLE SEMANTIC SIMILARITY TEST")
print("="*80)

# Initialize
model_path = "/workspace/LAM/best"
device = "cuda"

print(f"\n🔧 Initializing Universal Neural Indexer...")
indexer = UniversalNeuralIndexer(model_path, device=device)

# Create simple documents with clear semantic content
doc_corpus = [
    {
        'text': 'The capital of France is Paris. Paris is a beautiful city with many museums and landmarks like the Eiffel Tower.',
        '_id': 'france_doc'
    },
    {
        'text': 'Python is a programming language. It is widely used for data science, machine learning, and web development.',
        '_id': 'python_doc'
    },
    {
        'text': 'The Great Wall of China is one of the most famous landmarks in the world. It was built to protect China from invaders.',
        '_id': 'china_doc'
    },
]

print(f"\n📚 Indexing {len(doc_corpus)} documents...")
indexer.index(doc_corpus, mode="document")

# Create semantic queries that should match
semantic_queries = [
    "What is the capital of France?",  # Should match france_doc
    "Tell me about Python programming",  # Should match python_doc
    "What is the Great Wall of China?",  # Should match china_doc
]

query_doc_pairs = ['france_doc', 'python_doc', 'china_doc']

print(f"\n🔍 Testing {len(semantic_queries)} semantic queries...")
results = indexer.search(semantic_queries, top_k=3)

correct = 0
for i, (q_id, doc_scores) in enumerate(results.items()):
    query = semantic_queries[i]
    expected_doc = query_doc_pairs[i]
    
    print(f"\n   Query: '{query}'")
    print(f"   Expected: {expected_doc}")
    
    if doc_scores:
        print(f"   Top Results:")
        for rank, (retrieved_doc, score) in enumerate(list(doc_scores.items())[:3], 1):
            marker = "✅" if retrieved_doc == expected_doc else "  "
            print(f"      {marker} Rank {rank}: {retrieved_doc} (score: {score:.4f})")
        
        retrieved_doc = list(doc_scores.keys())[0]
        if retrieved_doc == expected_doc:
            correct += 1
            print(f"   Status: ✅ CORRECT")
        else:
            print(f"   Status: ❌ WRONG")
    else:
        print(f"   Status: ❌ NO RESULTS")

accuracy = (correct / len(semantic_queries)) * 100 if semantic_queries else 0.0
print(f"\n📊 SEMANTIC SIMILARITY ACCURACY: {correct}/{len(semantic_queries)} ({accuracy:.1f}%)")

if accuracy >= 100:
    print("✅ PERFECT: Semantic similarity working correctly!")
elif accuracy >= 66:
    print("✅ GOOD: Semantic similarity mostly working")
else:
    print("❌ ISSUE: Semantic similarity needs investigation")




