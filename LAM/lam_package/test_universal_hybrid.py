#!/usr/bin/env python3
"""
Test Universal Neural Indexer - Hybrid Mode
============================================
Tests both:
1. Exact key matching (100% recall) - NIAH-style
2. Semantic similarity (document retrieval) - cosine similarity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from universal_neural_indexer import UniversalNeuralIndexer

print("="*80)
print("🚀 UNIVERSAL NEURAL INDEXER - HYBRID MODE TEST")
print("="*80)
print("Two Formulas, Both Use Cosine:")
print("  1. Exact Key Matching: alpha=1.0 → 100% recall (NIAH)")
print("  2. Semantic Similarity: cosine(query, document) → document retrieval")
print("="*80)

# Initialize
model_path = "/workspace/LAM/best"
device = "cuda"

print(f"\n🔧 Initializing Universal Neural Indexer...")
indexer = UniversalNeuralIndexer(model_path, device=device)

# TEST 1: Exact Key Matching (100% Recall)
print("\n" + "="*80)
print("TEST 1: Exact Key Matching (100% Recall - NIAH Style)")
print("="*80)

# Store exact key-value pairs (like NIAH)
exact_facts = [
    ("What is the password?", "The secret password is QUANTUM7DELTA"),
    ("What is the launch code?", "The nuclear launch code is DELTA-7-QUANTUM-9"),
    ("Who is the CEO?", "The CEO of Tesla is Elon Musk"),
    ("What is the capital?", "Paris is the capital of France"),
    ("What is the population?", "Tokyo has a population of 14 million people"),
]

exact_corpus = []
for i, (key_text, value_text) in enumerate(exact_facts):
    # Store as document with key=query, value=answer
    exact_corpus.append({
        'text': value_text,  # The answer
        '_id': f"exact_{i}",
        'key_text': key_text,  # The query
    })

print(f"   Storing {len(exact_facts)} exact key-value pairs...")
# Index in exact mode (alpha=1.0 for 100% recall)
indexer.index(exact_corpus, mode="exact")

# Test retrieval with exact queries
print(f"\n   Testing exact key matching (should be 100%):")
exact_queries = [fact[0] for fact in exact_facts]  # Use exact query keys
exact_results = indexer.search(exact_queries, top_k=1)

correct_exact = 0
for i, (q_id, doc_scores) in enumerate(exact_results.items()):
    query = exact_queries[i]
    expected_doc = f"exact_{i}"
    
    if doc_scores:
        retrieved_doc = list(doc_scores.keys())[0]
        score = list(doc_scores.values())[0]
        is_correct = (retrieved_doc == expected_doc)
        
        if is_correct:
            correct_exact += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"
        
        print(f"      Query: '{query}'")
        print(f"         Expected: {expected_doc}, Retrieved: {retrieved_doc} (score: {score:.4f})")
        print(f"         Status: {status}")

exact_accuracy = (correct_exact / len(exact_facts)) * 100
print(f"\n   📊 EXACT KEY MATCHING ACCURACY: {correct_exact}/{len(exact_facts)} ({exact_accuracy:.1f}%)")

# TEST 2: Document Retrieval (Semantic Similarity)
print("\n" + "="*80)
print("TEST 2: Document Retrieval (Semantic Similarity)")
print("="*80)

from mteb.tasks import LEMBNarrativeQARetrieval

task = LEMBNarrativeQARetrieval()
task.load_data()

corpus = task.corpus
if corpus and 'test' in corpus:
    corpus = corpus['test']

# Use first 3 documents
# Extract meaningful content for better semantic matching
corpus_ids = list(corpus.keys())[:3]
doc_corpus = []
for doc_id in corpus_ids:
    doc = corpus[doc_id]
    raw_text = f"{doc.get('title', '')} {doc.get('body', doc.get('text', ''))}".strip()
    
    # Extract meaningful semantic content from HTML
    # Strategy: Use title + description + first meaningful paragraph
    text_parts = []
    
    # Extract title
    if '<title>' in raw_text:
        title_start = raw_text.find('<title>') + 7
        title_end = raw_text.find('</title>', title_start)
        if title_end > title_start:
            title = raw_text[title_start:title_end].strip()
            text_parts.append(title)
    
    # Extract description
    if 'description' in raw_text.lower():
        desc_start = raw_text.lower().find('description')
        if desc_start > 0:
            # Find content attribute
            content_start = raw_text.find('content="', desc_start)
            if content_start > 0:
                content_start += 10
                content_end = raw_text.find('"', content_start)
                if content_end > content_start:
                    description = raw_text[content_start:content_end].strip()
                    text_parts.append(description)
    
    # Extract first meaningful text (skip HTML tags)
    clean_text = raw_text.replace('<html>', '').replace('</html>', '')
    clean_text = clean_text.replace('<head>', '').replace('</head>', '')
    clean_text = clean_text.replace('<title>', '').replace('</title>', '')
    # Get first sentence that's not HTML
    sentences = clean_text.split('.')
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 30 and '<' not in sent[:30] and sent:
            text_parts.append(sent)
            break
    
    # Combine meaningful parts
    if text_parts:
        text = ' '.join(text_parts)
    else:
        # Fallback: use raw text
        text = raw_text
    
    doc_corpus.append({'text': text, '_id': doc_id})

print(f"   Indexing {len(doc_corpus)} documents (semantic mode)...")
# Index in document mode (alpha < 1.0 for semantic similarity)
indexer.index(doc_corpus, mode="document")

# Create semantic queries that actually test semantic understanding
# For better semantic matching, use questions about the document content
semantic_queries = []
query_doc_pairs = []

for i, doc in enumerate(doc_corpus):
    text = doc['text']
    doc_id = doc['_id']
    
    # Extract meaningful content and create semantic queries
    # Strategy: Use document title/description to create a question about the document
    
    # Clean HTML tags
    clean_text = text.replace('<html>', '').replace('</html>', '').replace('<head>', '').replace('</head>', '')
    clean_text = clean_text.replace('<title>', '').replace('</title>', '')
    clean_text = clean_text.replace('<meta', '').replace('>', ' ')
    
    # Extract title for better semantic matching
    if '<title>' in text:
        title_start = text.find('<title>') + 7
        title_end = text.find('</title>', title_start)
        if title_end > title_start:
            title = text[title_start:title_end].strip()
            # Create a question about the title
            query = f"What is {title}?"
        else:
            # Use document ID as fallback
            query = f"Tell me about {doc_id}"
    elif text.startswith('ï»¿The Project Gutenberg'):
        # Gutenberg book - extract title and create question
        lines = text.split('\n')
        for line in lines[:5]:
            if 'EBook' in line and 'by' in line:
                parts = line.split('EBook of')
                if len(parts) > 1:
                    book_part = parts[1].split('by')[0].strip()
                    query = f"What is the book {book_part} about?"
                    break
        else:
            query = f"What is this Gutenberg book about?"
    else:
        # Extract first meaningful sentence and create a question
        sentences = clean_text.split('.')
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20 and '<' not in s[:20]]
        if meaningful_sentences:
            first_sentence = meaningful_sentences[0][:100]
            # Create a question about the content
            query = f"What does this document say about {first_sentence[:50]}?"
        else:
            query = f"What is in document {doc_id}?"
    
    semantic_queries.append(query)
    query_doc_pairs.append(doc_id)

print(f"   Testing {len(semantic_queries)} semantic queries...")

# DEBUG: Show document previews
print(f"\n   📄 Document Previews:")
for i, doc in enumerate(doc_corpus):
    preview = doc['text'][:150].replace('\n', ' ')
    print(f"      {doc['_id']}: {preview}...")

# DEBUG: Show queries
print(f"\n   🔍 Query Previews:")
for i, query in enumerate(semantic_queries):
    print(f"      Query {i} (expects {query_doc_pairs[i]}): {query[:80]}...")

semantic_results = indexer.search(semantic_queries, top_k=3)  # Get top 3 for debugging

correct_semantic = 0
for i, (q_id, doc_scores) in enumerate(semantic_results.items()):
    query = semantic_queries[i]
    expected_doc = query_doc_pairs[i]
    
    if doc_scores:
        # Show all top results for debugging
        print(f"\n      Query: '{query[:60]}...'")
        print(f"         Expected: {expected_doc}")
        print(f"         Top Results:")
        for rank, (retrieved_doc, score) in enumerate(list(doc_scores.items())[:3], 1):
            marker = "✅" if retrieved_doc == expected_doc else "  "
            print(f"            {marker} Rank {rank}: {retrieved_doc} (score: {score:.4f})")
        
        retrieved_doc = list(doc_scores.keys())[0]
        score = list(doc_scores.values())[0]
        is_correct = (retrieved_doc == expected_doc)
        
        if is_correct:
            correct_semantic += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"
        
        print(f"         Status: {status}")

semantic_accuracy = (correct_semantic / len(semantic_queries)) * 100 if semantic_queries else 0.0
print(f"\n   📊 SEMANTIC SIMILARITY ACCURACY: {correct_semantic}/{len(semantic_queries)} ({semantic_accuracy:.1f}%)")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✅ Results:")
print(f"   Exact Key Matching (100% recall): {exact_accuracy:.1f}%")
print(f"   Semantic Similarity (documents): {semantic_accuracy:.1f}%")

print(f"\n💡 Universal Formula: v = W^T @ k (Delta GD Retrieval)")
print(f"   WRITE: W = W @ (I - α*kk^T) + β*vk^T  (Delta Rule)")
print(f"   READ:  v = W^T @ k                     (Delta GD Retrieval)")
print(f"")
print(f"   Two Modes (Aligned with LAM's Native Capabilities):")
print(f"   1. Exact (α=1.0): 100% recall when keys match (NIAH-style)")
print(f"      - Uses W matrix retrieval: v = W^T @ k")
print(f"      - No interference, perfect recall")
print(f"   2. Semantic (α<1.0): Direct semantic similarity (LAM's native understanding)")
print(f"      - Uses LAM embeddings: cos(query_emb, doc_emb)")
print(f"      - Avoids W matrix interference, leverages LAM's semantic optimization")
print(f"      - W matrix still maintained for consistency")

if exact_accuracy >= 100:
    print(f"\n   🎉 PERFECT: Exact key matching achieves 100% recall!")
if semantic_accuracy >= 60:
    print(f"   ✅ GOOD: Semantic similarity working for document retrieval")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)

