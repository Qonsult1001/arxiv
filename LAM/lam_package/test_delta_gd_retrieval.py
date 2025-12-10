#!/usr/bin/env python3
"""
Test Delta GD Retrieval System
================================

Tests the Delta GD retrieval mechanism (same as NIAH 100% recall) on:
1. Document-level retrieval
2. Within-document retrieval (finding specific facts)

This implements:
    WRITE: W = W @ (I - β*kk^T) + vk^T  (Delta Rule)
    READ:  v = W^T @ k                   (Delta GD Retrieval)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
import re

# Import LAM
try:
    from lam import LAM, InfiniteContextStreamer
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from lam import LAM, InfiniteContextStreamer

# Import Delta GD retrieval
from delta_gd_retrieval import DeltaGDRetriever, DeltaGDConfig

from mteb.tasks import LEMBNarrativeQARetrieval

print("="*80)
print("DELTA GD RETRIEVAL TEST - Perfect Recall Mechanism")
print("="*80)
print("Testing: Delta GD retrieval (same as NIAH 100% recall)")
print("Formula: WRITE: W = W @ (I - β*kk^T) + vk^T")
print("         READ:  v = W^T @ k")
print("="*80)

# Load model
model_paths = [
    "/workspace/LAM/best",
    "/workspace/LAM/LAM-base-v1",
    "LAM-base-v1"
]

model_path = None
for path in model_paths:
    if Path(path).exists() or path == "LAM-base-v1":
        model_path = path
        break

if model_path is None:
    model_path = "/workspace/LAM/best"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n🔧 Loading LAM from: {model_path}")
print(f"   Device: {device}")

model = LAM(model_path, device=device)
print(f"   ✅ Model loaded")

# Create Delta GD retriever
config = DeltaGDConfig(device=device)
retriever = DeltaGDRetriever(model, config, device)
print(f"   ✅ Delta GD Retriever initialized")

# Load documents
print("\n📚 Loading documents from LEMBNarrativeQARetrieval...")
task = LEMBNarrativeQARetrieval()
task.load_data()

corpus = task.corpus
if corpus and 'test' in corpus:
    corpus = corpus['test']

# Use first 2 documents
corpus_ids = list(corpus.keys())[:2]
documents = []
doc_titles = []

for doc_id in corpus_ids:
    doc = corpus[doc_id]
    title = doc.get('title', '')
    body = doc.get('body', doc.get('text', ''))
    full_text = f"{title} {body}".strip()
    documents.append(full_text)
    doc_titles.append(title)
    print(f"   📄 Doc {doc_id}: {len(full_text):,} chars")

# Expand to ~65K tokens if needed
target_words = 48000
expanded_documents = []

for i, doc_text in enumerate(documents):
    current_words = len(doc_text.split())
    if current_words < target_words:
        repetitions = max(1, target_words // current_words)
        expanded = (doc_text + " ") * repetitions
        words = expanded.split()
        expanded = " ".join(words[:target_words])
    else:
        expanded = doc_text
    
    expanded_documents.append(expanded)
    expanded_words = len(expanded.split())
    estimated_tokens = int(expanded_words / 0.75)
    print(f"   📄 Doc {i+1}: {expanded_words:,} words (~{estimated_tokens:,} tokens)")

# TEST 1: Store documents using Delta GD (NIAH-style)
print("\n" + "="*80)
print("TEST 1: Store Documents in Delta GD Memory (NIAH 100% Style)")
print("="*80)

doc_memories = []
doc_embeddings = []

for i, doc_text in enumerate(expanded_documents, 1):
    print(f"\n   Storing document {i}/{len(expanded_documents)}...")
    print(f"      Length: {len(doc_text.split()):,} words")
    
    # Store in Delta GD memory (full document as ONE embedding)
    memory, embedding = retriever.store_document(doc_text, doc_id=f"doc_{i}")
    
    doc_memories.append(memory)
    doc_embeddings.append(embedding)
    
    print(f"      ✅ Stored in memory matrix W")
    print(f"      Memory norm: {memory.W.norm().item():.4f}")
    print(f"      Write count: {memory.write_count} (1 document)")
    print(f"      Embedding shape: {embedding.shape}")

doc_embeddings = torch.stack(doc_embeddings)  # [num_docs, emb_dim]

# TEST 2: Document-Level Retrieval
print("\n" + "="*80)
print("TEST 2: Document-Level Retrieval (Query → Document)")
print("="*80)
print("Question: Can a query find the correct document using Delta GD retrieval?")

# Create queries
semantic_queries = []
query_doc_pairs = []

for i, (doc_text, title) in enumerate(zip(documents, doc_titles)):
    if title and len(title.strip()) > 0:
        query = f"What is {title}?"
        semantic_queries.append(query)
        query_doc_pairs.append(i)
    
    first_sentence = doc_text.split('.')[0] if '.' in doc_text else doc_text[:100]
    if len(first_sentence) > 20:
        query = f"Tell me about {first_sentence[:50]}"
        semantic_queries.append(query)
        query_doc_pairs.append(i)

while len(semantic_queries) < 4:
    doc_idx = len(semantic_queries) % len(documents)
    generic_queries = [
        "What is the main topic discussed?",
        "Can you explain the key concepts?",
    ]
    query = generic_queries[len(semantic_queries) % len(generic_queries)]
    semantic_queries.append(query)
    query_doc_pairs.append(doc_idx)

assert len(query_doc_pairs) == len(semantic_queries)

print(f"\n   Created {len(semantic_queries)} queries")

# Test retrieval (NIAH-style: 100% recall)
correct_retrievals = 0
total_queries = len(semantic_queries)

print(f"\n   Retrieval results (Delta GD - should be 100%):")
for i, query in enumerate(semantic_queries):
    # Retrieve from each document memory
    scores = []
    for j, memory in enumerate(doc_memories):
        _, _, score = retriever.retrieve(query, memory)
        scores.append(score)
    
    # Find best match
    best_doc_idx = np.argmax(scores)
    best_score = scores[best_doc_idx]
    
    # Check correctness
    expected_doc_idx = query_doc_pairs[i]
    is_correct = (best_doc_idx == expected_doc_idx)
    
    if is_correct:
        correct_retrievals += 1
        status = "✅ CORRECT"
    else:
        status = "❌ WRONG"
    
    print(f"\n      Query {i+1}: '{query[:60]}...'")
    print(f"         Best match: Doc {best_doc_idx + 1} (score: {best_score:.4f})")
    print(f"         Expected: Doc {expected_doc_idx + 1}")
    print(f"         Status: {status}")
    print(f"         All scores: {[f'{s:.4f}' for s in scores]}")

accuracy = (correct_retrievals / total_queries) * 100
print(f"\n   📊 DOCUMENT-LEVEL ACCURACY: {correct_retrievals}/{total_queries} ({accuracy:.1f}%)")

# TEST 3: Within-Document Retrieval
print("\n" + "="*80)
print("TEST 3: Within-Document Retrieval (Query → Specific Fact)")
print("="*80)
print("Question: Can we find SPECIFIC INFORMATION within a document?")

# Use the longest document and store facts
longest_doc_idx = 0 if len(expanded_documents[0]) > len(expanded_documents[1]) else 1
longest_doc = expanded_documents[longest_doc_idx]

print(f"\n   Using document {longest_doc_idx + 1} ({len(longest_doc.split()):,} words)")

# Extract facts from document
sentences = re.split(r'[.!?]+', longest_doc)
sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

facts_list = []  # List of (fact_text, fact_content) tuples
fact_queries = []

# Extract sentences with names
name_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
for sent in sentences[:100]:
    matches = re.findall(name_pattern, sent)
    if matches and len(sent) > 30:
        fact_content = sent[:200]
        name = matches[0]
        fact_text = f"Who is {name}?"  # This is the query key
        facts_list.append((fact_text, fact_content))
        fact_queries.append((fact_text, fact_content, len(facts_list) - 1))
        if len(facts_list) >= 5:
            break

# Extract sentences with numbers/dates
number_pattern = r'\b(\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
for sent in sentences[:100]:
    matches = re.findall(number_pattern, sent)
    if matches and len(sent) > 30 and sent not in [f[1] for f in facts_list]:
        fact_content = sent[:200]
        number = matches[0]
        fact_text = f"What happened in {number}?"  # This is the query key
        facts_list.append((fact_text, fact_content))
        fact_queries.append((fact_text, fact_content, len(facts_list) - 1))
        if len(facts_list) >= 10:
            break

# If not enough facts, use long sentences
if len(facts_list) < 10:
    long_sentences = sorted(sentences, key=len, reverse=True)[:10]
    for sent in long_sentences:
        if sent not in [f[1] for f in facts_list] and len(sent) > 50:
            fact_content = sent[:200]
            words = sent.split()[:5]
            fact_text = f"What is { ' '.join(words) }?"  # This is the query key
            facts_list.append((fact_text, fact_content))
            fact_queries.append((fact_text, fact_content, len(facts_list) - 1))
            if len(facts_list) >= 10:
                break

print(f"   Extracted {len(facts_list)} facts from document")

# Re-store document WITH facts (NIAH-style: each fact stored separately)
print(f"\n   Storing document with {len(facts_list)} facts in Delta GD memory...")
longest_memory, _ = retriever.store_document(
    longest_doc, 
    doc_id=f"doc_{longest_doc_idx + 1}_with_facts",
    facts=facts_list
)
print(f"      ✅ Stored: 1 document + {len(facts_list)} facts")
print(f"      Write count: {longest_memory.write_count}")

# Test retrieval (NIAH-style: should be 100%)
print(f"\n   Testing retrieval of specific facts (should be 100%):")
correct_fact_retrievals = 0

for i, (query, expected_fact_content, expected_fact_idx) in enumerate(fact_queries):
    # Retrieve from document memory using Delta GD
    retrieved_value, retrieved_content, score = retriever.retrieve(query, longest_memory)
    
    # Check if retrieved content matches expected fact
    # For 100% recall, the retrieved content should match the fact we stored
    is_correct = (retrieved_content == expected_fact_content) or (
        retrieved_content and expected_fact_content and 
        retrieved_content[:50] == expected_fact_content[:50]
    )
    
    if is_correct:
        correct_fact_retrievals += 1
        status = "✅ CORRECT"
    else:
        status = "❌ WRONG"
    
    print(f"\n      Query {i+1}: '{query}'")
    print(f"         Expected fact: '{expected_fact_content[:50]}...'")
    if retrieved_content:
        print(f"         Retrieved: '{retrieved_content[:50]}...'")
    else:
        print(f"         Retrieved: None")
    print(f"         Score: {score:.4f}")
    print(f"         Status: {status}")

fact_accuracy = (correct_fact_retrievals / len(fact_queries)) * 100 if fact_queries else 0.0
print(f"\n   📊 WITHIN-DOCUMENT ACCURACY: {correct_fact_retrievals}/{len(fact_queries)} ({fact_accuracy:.1f}%)")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n📊 Results:")
print(f"   Document-level retrieval: {accuracy:.1f}%")
print(f"   Within-document retrieval: {fact_accuracy:.1f}%")

print(f"\n💡 Delta GD Retrieval Analysis:")
print(f"   - Uses same mechanism as NIAH (100% recall)")
print(f"   - WRITE: W = W @ (I - β*kk^T) + vk^T")
print(f"   - READ:  v = W^T @ k")
print(f"   - Memory matrix W contains ALL document knowledge")

if fact_accuracy >= 100:
    print(f"\n   ✅ PERFECT: 100% within-document retrieval achieved!")
    print(f"      Same as NIAH - Delta GD retrieval working perfectly!")
    print(f"      Key insight: Store facts with query keys for exact matching")
elif fact_accuracy >= 80:
    print(f"\n   ✅ EXCELLENT: Delta GD retrieval working well!")
elif fact_accuracy >= 60:
    print(f"\n   ⚠️  MODERATE: Some improvement needed")
else:
    print(f"\n   ❌ NEEDS WORK: May need tuning of alpha/beta or key/value extraction")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)

