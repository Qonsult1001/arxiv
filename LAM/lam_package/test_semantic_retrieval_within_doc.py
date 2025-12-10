#!/usr/bin/env python3
"""
LOCAL TEST: Semantic Similarity Retrieval WITHIN Long Documents (65K tokens)

This test is MORE CHALLENGING than document-level retrieval:
- Document-level: "Which document matches this query?" ✅ Works reasonably
- Within-document: "What specific information is in this document?" ⚠️ Much harder

The Problem:
- A 65K token document is encoded as ONE embedding
- That embedding represents the ENTIRE document (averaged/global semantics)
- Finding specific facts/information within requires semantic similarity
- This is the REAL challenge with long documents!

Tests:
1. Embed a long document (65K tokens) as ONE embedding
2. Extract specific "facts" or "information" from the document
3. Create semantic queries that should retrieve those facts
4. Test if we can distinguish between different information within the same document
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
import re

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
print("SEMANTIC SIMILARITY RETRIEVAL TEST - WITHIN Long Documents (65K tokens)")
print("="*80)
print("Testing: Can we find SPECIFIC INFORMATION inside a 65K token document?")
print("Challenge: Document is ONE embedding - can we retrieve specific facts?")
print("Using: LAM from lam package with InfiniteContextStreamer (streaming)")
print("="*80)

# Load model
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
    model_path = "/workspace/LAM/best"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n🔧 Loading LAM from: {model_path}")
print(f"   Device: {device}")

model = LAM(model_path, device=device)
streamer = InfiniteContextStreamer(model, chunk_size=512)
print(f"   ✅ Streamer initialized (chunk_size=512)")

# Load one long document
print("\n📚 Loading a long document from LEMBNarrativeQARetrieval...")
task = LEMBNarrativeQARetrieval()
task.load_data()

corpus = task.corpus
if corpus and 'test' in corpus:
    corpus = corpus['test']

# Use the LONGEST document (most challenging)
corpus_ids = list(corpus.keys())
doc_lengths = []
for doc_id in corpus_ids:
    doc = corpus[doc_id]
    text = f"{doc.get('title', '')} {doc.get('body', doc.get('text', ''))}".strip()
    doc_lengths.append((doc_id, len(text)))

# Sort by length and take the longest
doc_lengths.sort(key=lambda x: x[1], reverse=True)
longest_doc_id = doc_lengths[0][0]
doc = corpus[longest_doc_id]
full_text = f"{doc.get('title', '')} {doc.get('body', doc.get('text', ''))}".strip()

print(f"   📄 Selected document: {longest_doc_id}")
print(f"   📏 Original length: {len(full_text):,} chars")

# Expand to ~65K tokens if needed
target_words = 48000  # ~65K tokens
current_words = len(full_text.split())
if current_words < target_words:
    repetitions = max(1, target_words // current_words)
    expanded = (full_text + " ") * repetitions
    words = expanded.split()
    full_text = " ".join(words[:target_words])
    print(f"   📏 Expanded to: {len(full_text.split()):,} words (~{int(len(full_text.split())/0.75):,} tokens)")
else:
    print(f"   📏 Already long enough: {len(full_text.split()):,} words")

# Extract specific "facts" or "information" from the document
# Strategy: Find sentences/phrases that contain specific information
print("\n🔍 Extracting specific information from the document...")
print("   Looking for: Names, dates, locations, key facts, specific details")

# Extract sentences (simple approach)
sentences = re.split(r'[.!?]+', full_text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short sentences

# Find sentences with specific information patterns
facts = []
fact_queries = []

# Pattern 1: Look for sentences with names (capitalized words that might be names)
name_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
for i, sent in enumerate(sentences[:100]):  # Check first 100 sentences
    matches = re.findall(name_pattern, sent)
    if matches and len(sent) > 30:
        fact = sent[:200]  # First 200 chars
        facts.append(fact)
        # Create a query about this person
        name = matches[0]
        query = f"Who is {name}?"
        fact_queries.append((query, fact, len(facts) - 1))
        if len(facts) >= 5:
            break

# Pattern 2: Look for sentences with numbers/dates
number_pattern = r'\b(\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
for i, sent in enumerate(sentences[:100]):
    matches = re.findall(number_pattern, sent)
    if matches and len(sent) > 30 and sent not in facts:
        fact = sent[:200]
        facts.append(fact)
        # Create a query about this date/number
        number = matches[0]
        query = f"What happened in {number}?"
        fact_queries.append((query, fact, len(facts) - 1))
        if len(facts) >= 10:
            break

# Pattern 3: Look for sentences with locations
location_pattern = r'\b([A-Z][a-z]+ (?:City|State|Country|Island|Mountain|River|Lake|Ocean))\b'
for i, sent in enumerate(sentences[:100]):
    matches = re.findall(location_pattern, sent)
    if matches and len(sent) > 30 and sent not in facts:
        fact = sent[:200]
        facts.append(fact)
        location = matches[0]
        query = f"Tell me about {location}"
        fact_queries.append((query, fact, len(facts) - 1))
        if len(facts) >= 15:
            break

# Pattern 4: Extract key sentences (longer sentences often contain more information)
if len(facts) < 10:
    long_sentences = sorted(sentences, key=len, reverse=True)[:10]
    for sent in long_sentences:
        if sent not in facts and len(sent) > 50:
            fact = sent[:200]
            facts.append(fact)
            # Create generic query
            words = sent.split()[:5]
            query = f"What is { ' '.join(words) }?"
            fact_queries.append((query, fact, len(facts) - 1))
            if len(facts) >= 10:
                break

print(f"   ✅ Extracted {len(facts)} specific facts/information pieces")
for i, (query, fact, idx) in enumerate(fact_queries[:5], 1):
    print(f"      Fact {i}: '{fact[:60]}...'")
    print(f"         Query: '{query}'")

# Encode the FULL document as ONE embedding
print("\n📊 Encoding FULL document as ONE embedding (65K tokens)...")
print("   This is the challenge: ONE embedding for the entire document")

# Tokenize
try:
    enc = model.tokenizer.encode(full_text)
    token_ids = enc.ids if hasattr(enc, 'ids') else enc
except Exception:
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, '__call__'):
        tokens = model.tokenizer(full_text, return_tensors='pt', truncation=False)
        token_ids = tokens['input_ids'].squeeze().tolist()
    else:
        token_ids = model.tokenizer.encode(full_text)

print(f"   Tokens: {len(token_ids):,}")

# Use streaming for long documents
if len(token_ids) > 8192:
    print(f"   Using InfiniteContextStreamer (>{8192} tokens, chunk_size=512)...")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    
    streamer.reset()
    doc_emb = streamer.stream_embedding(
        input_ids, 
        attention_mask, 
        verbose=False
    )
    if doc_emb.dim() == 2:
        doc_emb = doc_emb.squeeze(0)
else:
    doc_emb = model.encode([full_text], convert_to_tensor=True)
    if doc_emb.dim() == 2 and doc_emb.shape[0] == 1:
        doc_emb = doc_emb.squeeze(0)

if isinstance(doc_emb, dict):
    doc_emb = doc_emb.get(384, list(doc_emb.values())[0])
if not isinstance(doc_emb, torch.Tensor):
    doc_emb = torch.tensor(doc_emb)

doc_emb = doc_emb.cpu()
print(f"   ✅ Document embedding: shape {doc_emb.shape}")

# Encode each FACT as a separate embedding
print("\n📊 Encoding individual FACTS as separate embeddings...")
print("   Each fact is encoded separately to test semantic similarity")

fact_embeddings = []
for i, fact in enumerate(facts, 1):
    emb = model.encode([fact], convert_to_tensor=True)
    if emb.dim() == 2 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    if isinstance(emb, dict):
        emb = emb.get(384, list(emb.values())[0])
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)
    fact_embeddings.append(emb.cpu())
    if i <= 3:
        print(f"      Fact {i}: '{fact[:50]}...' → encoded")

fact_embeddings = torch.stack(fact_embeddings)  # [num_facts, emb_dim]
print(f"   ✅ All facts encoded: shape {fact_embeddings.shape}")

# Encode queries
print("\n📊 Encoding semantic queries...")
query_embeddings = []
for i, (query, fact, fact_idx) in enumerate(fact_queries, 1):
    emb = model.encode([query], convert_to_tensor=True)
    if emb.dim() == 2 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    if isinstance(emb, dict):
        emb = emb.get(384, list(emb.values())[0])
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)
    query_embeddings.append(emb.cpu())
    if i <= 3:
        print(f"      Query {i}: '{query}' → encoded")

query_embeddings = torch.stack(query_embeddings)  # [num_queries, emb_dim]
print(f"   ✅ All queries encoded: shape {query_embeddings.shape}")

# Normalize
print("\n📐 Normalizing embeddings...")
doc_emb = F.normalize(doc_emb.unsqueeze(0), p=2, dim=1)  # [1, emb_dim]
fact_embeddings = F.normalize(fact_embeddings, p=2, dim=1)  # [num_facts, emb_dim]
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)  # [num_queries, emb_dim]

# Move to device if GPU
if device == "cuda" and torch.cuda.is_available():
    doc_emb = doc_emb.to(device)
    fact_embeddings = fact_embeddings.to(device)
    query_embeddings = query_embeddings.to(device)

# Test 1: Query → Document (can we find the document?)
print("\n" + "="*80)
print("TEST 1: Query → Document (Document-Level Retrieval)")
print("="*80)
print("Question: Can a semantic query find the document that contains the answer?")

query_doc_sims = torch.matmul(query_embeddings, doc_emb.T).squeeze(1)  # [num_queries]
query_doc_sims_np = query_doc_sims.cpu().numpy()

print(f"\n   Similarity scores (query → document):")
for i, (query, fact, fact_idx) in enumerate(fact_queries):
    sim = query_doc_sims_np[i]
    print(f"      Query {i+1}: '{query[:50]}...' → Document: {sim:.4f}")

avg_query_doc_sim = query_doc_sims_np.mean()
print(f"\n   📊 Average query→document similarity: {avg_query_doc_sim:.4f}")

# Test 2: Query → Facts (can we find the specific fact?)
print("\n" + "="*80)
print("TEST 2: Query → Facts (Within-Document Retrieval)")
print("="*80)
print("Question: Can a semantic query find the SPECIFIC FACT within the document?")
print("Challenge: The document is ONE embedding - can we distinguish between facts?")

query_fact_sims = torch.matmul(query_embeddings, fact_embeddings.T)  # [num_queries, num_facts]
query_fact_sims_np = query_fact_sims.cpu().numpy()

correct_retrievals = 0
total_queries = len(fact_queries)

print(f"\n   Retrieval results:")
for i, (query, fact, expected_fact_idx) in enumerate(fact_queries):
    # Get similarity scores for this query to all facts
    fact_sims = query_fact_sims_np[i]  # [num_facts]
    
    # Find best matching fact
    best_fact_idx = np.argmax(fact_sims)
    best_sim = fact_sims[best_fact_idx]
    
    # Check if correct
    is_correct = (best_fact_idx == expected_fact_idx)
    if is_correct:
        correct_retrievals += 1
    
    status = "✅ CORRECT" if is_correct else "❌ WRONG"
    
    print(f"\n      Query {i+1}: '{query}'")
    print(f"         Expected fact: {expected_fact_idx} ('{facts[expected_fact_idx][:50]}...')")
    print(f"         Best match: Fact {best_fact_idx} (similarity: {best_sim:.4f})")
    print(f"         Status: {status}")
    
    # Show top 3 matches
    top3_indices = np.argsort(fact_sims)[::-1][:3]
    print(f"         Top 3 matches:")
    for rank, idx in enumerate(top3_indices, 1):
        print(f"            {rank}. Fact {idx}: {fact_sims[idx]:.4f} ('{facts[idx][:40]}...')")

accuracy = (correct_retrievals / total_queries) * 100 if total_queries > 0 else 0.0
print(f"\n   📊 WITHIN-DOCUMENT RETRIEVAL ACCURACY: {correct_retrievals}/{total_queries} ({accuracy:.1f}%)")

# Test 3: Document → Facts (does document embedding contain fact information?)
print("\n" + "="*80)
print("TEST 3: Document → Facts (Embedding Information Content)")
print("="*80)
print("Question: Does the document embedding preserve information about individual facts?")
print("Test: Compare document embedding to each fact embedding")

doc_fact_sims = torch.matmul(doc_emb, fact_embeddings.T).squeeze(0)  # [num_facts]
doc_fact_sims_np = doc_fact_sims.cpu().numpy()

print(f"\n   Document → Facts similarity scores:")
for i, fact in enumerate(facts):
    sim = doc_fact_sims_np[i]
    print(f"      Fact {i+1}: {sim:.4f} ('{fact[:50]}...')")

avg_doc_fact_sim = doc_fact_sims_np.mean()
std_doc_fact_sim = doc_fact_sims_np.std()
print(f"\n   📊 Average document→fact similarity: {avg_doc_fact_sim:.4f}")
print(f"   📊 Std deviation: {std_doc_fact_sim:.4f}")
print(f"   📊 Range: [{doc_fact_sims_np.min():.4f}, {doc_fact_sims_np.max():.4f}]")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("\n🔍 Key Findings:")
print(f"   1. Document-level retrieval (Query → Document): {avg_query_doc_sim:.4f} avg similarity")
print(f"   2. Within-document retrieval (Query → Facts): {accuracy:.1f}% accuracy")
print(f"   3. Document embedding information content: {avg_doc_fact_sim:.4f} avg similarity to facts")

print("\n💡 Interpretation:")
if accuracy >= 80:
    print("   ✅ GOOD: Can distinguish between facts within the document")
elif accuracy >= 50:
    print("   ⚠️  MODERATE: Some ability to distinguish facts, but not perfect")
else:
    print("   ❌ POOR: Document embedding is too 'averaged' - can't distinguish facts")

if std_doc_fact_sim < 0.05:
    print("   ⚠️  WARNING: Low variance in document→fact similarities")
    print("      This suggests the document embedding is too 'averaged'")
    print("      All facts look similar to the document embedding")
else:
    print(f"   ✅ GOOD: Variance in document→fact similarities ({std_doc_fact_sim:.4f})")
    print("      The document embedding preserves some distinction between facts")

print("\n📊 Comparison:")
print("   - Document-level: 'Which document?' → Works reasonably ✅")
print("   - Within-document: 'What information is in this document?' → Much harder ⚠️")
print("   - The challenge: ONE embedding for 65K tokens may lose specific details")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\n💡 This test demonstrates the REAL challenge:")
print("   Encoding a 65K token document as ONE embedding may preserve")
print("   global semantics but lose specific information needed for retrieval.")





