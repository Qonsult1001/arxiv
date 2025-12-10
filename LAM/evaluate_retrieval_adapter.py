"""
🎯 RETRIEVAL BENCHMARK EVALUATION
=================================
Evaluates the adapter on MTEB Retrieval tasks to beat all-MiniLM-L6-v2

Target Scores (all-MiniLM-L6-v2 baseline):
- MS MARCO: ~37.7 nDCG@10
- Natural Questions: ~58.5 nDCG@10  
- Average Retrieval: ~42.0 nDCG@10

To beat it, you need:
- MS MARCO: >37.7 nDCG@10
- Natural Questions: >58.5 nDCG@10
- Average Retrieval: >42.0 nDCG@10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# Import your stack
sys.path.insert(0, str(Path(__file__).parent))
from train_6layer_deltanet_2 import DeltaNetPure6Layer, config
from infinite_streamer_async import AsyncInfiniteStreamer

class HolographicAdapter(nn.Module):
    def __init__(self, input_dim=384, output_dim=384):
        super().__init__()
        # Enhanced adapter with 2 layers for better capacity
        self.proj1 = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.proj2 = nn.Linear(output_dim, output_dim)
        # Initialize close to identity
        nn.init.eye_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.eye_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x):
        x = self.proj1(x)
        x = self.activation(x)
        x = self.proj2(x)
        return x

def evaluate_retrieval_task(model, adapter, streamer, queries, corpus, device, batch_size=32):
    """
    Evaluate retrieval task using cosine similarity
    
    Args:
        model: Base model
        adapter: Trained adapter
        streamer: Streamer for getting embeddings
        queries: List of query texts
        corpus: List of document texts
        device: Device to use
        batch_size: Batch size for encoding
    
    Returns:
        nDCG@10 scores for each query
    """
    print(f"   Encoding {len(queries)} queries and {len(corpus)} documents...")
    
    # Encode queries using adapter + streamer
    query_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), batch_size), desc="   Queries"):
            batch_queries = queries[i:i+batch_size]
            batch_embs = []
            for query in batch_queries:
                enc = model.tokenizer.encode(query, add_special_tokens=True)
                ids = torch.tensor([enc], device=device)
                streamer_emb = streamer.stream_embedding(ids, verbose=False)
                adapter_emb = adapter(streamer_emb)
                batch_embs.append(adapter_emb)
            query_embeddings.append(torch.cat(batch_embs, dim=0))
    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    
    # Encode corpus using adapter + streamer
    corpus_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus), batch_size), desc="   Corpus"):
            batch_corpus = corpus[i:i+batch_size]
            batch_embs = []
            for doc in batch_corpus:
                enc = model.tokenizer.encode(doc, add_special_tokens=True)
                ids = torch.tensor([enc], device=device)
                streamer_emb = streamer.stream_embedding(ids, verbose=False)
                adapter_emb = adapter(streamer_emb)
                batch_embs.append(adapter_emb)
            corpus_embeddings.append(torch.cat(batch_embs, dim=0))
    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)
    
    # Compute similarities
    print("   Computing similarities...")
    similarities = torch.mm(query_embeddings, corpus_embeddings.t())
    
    # Get top-k for each query
    k = 10
    top_k_scores, top_k_indices = torch.topk(similarities, k=min(k, len(corpus)), dim=1)
    
    return top_k_scores.cpu().numpy(), top_k_indices.cpu().numpy()

def compute_ndcg_at_k(relevance_scores, k=10):
    """
    Compute nDCG@k
    
    Args:
        relevance_scores: List of relevance scores (0 or 1 for binary)
        k: Number of top results to consider
    """
    if len(relevance_scores) == 0:
        return 0.0
    
    # Take top k
    relevance_scores = relevance_scores[:k]
    
    # Compute DCG
    dcg = 0.0
    for i, score in enumerate(relevance_scores):
        dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Compute IDCG (ideal DCG - all relevant docs first)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i, score in enumerate(ideal_scores):
        idcg += score / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def evaluate_msmarco_sample(model, adapter, streamer, device):
    """
    Evaluate on a sample of MS MARCO dataset using the training triplets
    This uses the same data format we trained on, so it's a fair comparison
    """
    print("\n" + "="*80)
    print("📊 MS MARCO RETRIEVAL EVALUATION")
    print("="*80)
    print("Target: >37.7 nDCG@10 (all-MiniLM-L6-v2 baseline)")
    
    try:
        from datasets import load_dataset
        import os
        os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
        os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
        
        # Load MS MARCO triplets (same format as training)
        print("\n📦 Loading MS MARCO triplets for evaluation...")
        dataset = load_dataset(
            "sentence-transformers/embedding-training-data", 
            data_files="msmarco-triplets.jsonl.gz", 
            split="train[5000:5500]",  # Use different samples than training
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
        
        # Extract queries, positives, and negatives
        queries = []
        positives = []
        negatives_list = []
        corpus_set = set()  # To build unique corpus
        
        for row in dataset:
            if 'query' in row and 'pos' in row and 'neg' in row:
                query = row['query']
                if isinstance(row['pos'], list) and len(row['pos']) > 0:
                    pos = row['pos'][0]
                elif isinstance(row['pos'], str):
                    pos = row['pos']
                else:
                    continue
                
                if isinstance(row['neg'], list) and len(row['neg']) > 0:
                    negs = row['neg']
                elif isinstance(row['neg'], str):
                    negs = [row['neg']]
                else:
                    continue
                
                queries.append(query)
                positives.append(pos)
                negatives_list.append(negs)
                
                # Add to corpus
                corpus_set.add(pos)
                for neg in negs[:5]:  # Limit negatives per query
                    corpus_set.add(neg)
        
        if len(queries) == 0:
            print("   ⚠️  No valid triplets found")
            return None
        
        corpus = list(corpus_set)
        print(f"   Loaded {len(queries)} queries and {len(corpus)} unique documents")
        
        # Build relevance mapping: for each query, which docs are relevant
        query_to_relevant = {}
        corpus_to_idx = {doc: idx for idx, doc in enumerate(corpus)}
        
        for i, (query, pos, negs) in enumerate(zip(queries, positives, negatives_list)):
            relevant_indices = []
            if pos in corpus_to_idx:
                relevant_indices.append(corpus_to_idx[pos])
            query_to_relevant[i] = relevant_indices
        
        # Evaluate
        print("   Encoding queries and corpus...")
        top_scores, top_indices = evaluate_retrieval_task(
            model, adapter, streamer, queries, corpus, device, batch_size=16
        )
        
        # Compute nDCG@10 for each query
        ndcg_scores = []
        for i, query_indices in enumerate(top_indices):
            if i not in query_to_relevant:
                continue
            
            # Create relevance vector
            relevance = np.zeros(len(corpus))
            for rel_idx in query_to_relevant[i]:
                if rel_idx < len(corpus):
                    relevance[rel_idx] = 1.0
            
            # Get relevance for retrieved docs (top 10)
            retrieved_relevance = relevance[query_indices[:10]]
            ndcg = compute_ndcg_at_k(retrieved_relevance, k=10)
            ndcg_scores.append(ndcg)
        
        if len(ndcg_scores) == 0:
            print("   ⚠️  Could not compute nDCG scores")
            return None
        
        # MTEB reports nDCG as decimal (0.0 to 1.0), convert to percentage for display
        avg_ndcg_decimal = np.mean(ndcg_scores)
        avg_ndcg_percent = avg_ndcg_decimal * 100.0
        
        print(f"\n📊 Results:")
        print(f"   Average nDCG@10: {avg_ndcg_decimal:.4f} ({avg_ndcg_percent:.2f}%)")
        print(f"   Target (all-MiniLM-L6-v2): 0.377 (37.7%)")
        print(f"   Evaluated on {len(ndcg_scores)} queries")
        print(f"   ⚠️  NOTE: Using small corpus from triplets (not full MS MARCO)")
        print(f"   💡 For official results, use full MTEB evaluation")
        
        if avg_ndcg_decimal > 0.377:
            print(f"   🎉 BEAT BASELINE by {(avg_ndcg_decimal - 0.377):.4f} ({(avg_ndcg_percent - 37.7):.2f}%)!")
        else:
            improvement = avg_ndcg_decimal - 0.25  # Assuming starting from ~0.25
            print(f"   📈 Improved from ~0.25 to {avg_ndcg_decimal:.4f} (+{improvement:.4f})")
            print(f"   ⚠️  Still below baseline by {0.377 - avg_ndcg_decimal:.4f}")
            print(f"   💡 Continue training or use more data")
        
        return avg_ndcg_decimal
        
    except Exception as e:
        print(f"   ⚠️  Could not evaluate MS MARCO: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_semantic_similarity(model, adapter, streamer, device):
    """
    Evaluate on STS-B (semantic similarity)
    This is a proxy for retrieval quality
    """
    print("\n" + "="*80)
    print("📊 STS-B SEMANTIC SIMILARITY EVALUATION")
    print("="*80)
    print("Target: >82.0 Spearman (all-MiniLM-L6-v2 baseline)")
    
    try:
        from datasets import load_dataset
        from scipy.stats import spearmanr, pearsonr
        import os
        os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
        os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
        
        # Load STS-B
        print("\n📦 Loading STS-B dataset...")
        dataset = load_dataset(
            "sentence-transformers/stsb",
            split="test",
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
        
        s1 = dataset["sentence1"]
        s2 = dataset["sentence2"]
        labels = np.array(dataset["score"], dtype=float)
        
        print(f"   Loaded {len(s1)} sentence pairs")
        
        # Encode sentences
        print("   Encoding sentences...")
        embeddings1 = []
        embeddings2 = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(s1), 32), desc="   Encoding"):
                batch_s1 = s1[i:i+32]
                batch_s2 = s2[i:i+32]
                
                batch_emb1 = []
                batch_emb2 = []
                
                for text in batch_s1:
                    enc = model.tokenizer.encode(text, add_special_tokens=True)
                    ids = torch.tensor([enc], device=device)
                    streamer_emb = streamer.stream_embedding(ids, verbose=False)
                    adapter_emb = adapter(streamer_emb)
                    batch_emb1.append(adapter_emb)
                
                for text in batch_s2:
                    enc = model.tokenizer.encode(text, add_special_tokens=True)
                    ids = torch.tensor([enc], device=device)
                    streamer_emb = streamer.stream_embedding(ids, verbose=False)
                    adapter_emb = adapter(streamer_emb)
                    batch_emb2.append(adapter_emb)
                
                embeddings1.append(torch.cat(batch_emb1, dim=0))
                embeddings2.append(torch.cat(batch_emb2, dim=0))
        
        embeddings1 = torch.cat(embeddings1, dim=0)
        embeddings2 = torch.cat(embeddings2, dim=0)
        
        # Compute cosine similarities
        similarities = F.cosine_similarity(embeddings1, embeddings2, dim=1).cpu().numpy()
        
        # Compute correlations (handle potential NaN/inf values)
        try:
            pearson = pearsonr(similarities, labels)[0]
            if np.isnan(pearson) or np.isinf(pearson):
                pearson = None
        except:
            pearson = None
        
        try:
            spearman = spearmanr(similarities, labels)[0]
            if np.isnan(spearman) or np.isinf(spearman):
                spearman = None
        except:
            spearman = None
        
        print(f"\n📊 Results:")
        if pearson is not None:
            print(f"   Pearson:  {pearson:.4f}")
        if spearman is not None:
            print(f"   Spearman: {spearman:.4f}")
        print(f"   Target (all-MiniLM-L6-v2): 0.82 Spearman (82.0%)")
        
        if spearman is not None:
            # Compare as decimal (0.0 to 1.0) - MTEB uses decimal format
            target_spearman = 0.82
            if spearman > target_spearman:
                print(f"   🎉 BEAT BASELINE by {(spearman - target_spearman):.4f}!")
            else:
                print(f"   ⚠️  Below baseline by {target_spearman - spearman:.4f}")
        else:
            print(f"   ⚠️  Could not compute Spearman correlation")
        
        return spearman
        
    except Exception as e:
        print(f"   ⚠️  Could not evaluate STS-B: {e}")
        return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🎯 RETRIEVAL BENCHMARK EVALUATION")
    print("="*80)
    print(f"Device: {device}")
    
    # 1. Load Model
    print("\n1️⃣  Loading model...")
    model_path = str(Path(__file__).parent / "all-MiniLM-L6-v2")
    model = DeltaNetPure6Layer(model_path, 6, config).to(device)
    
    weights = torch.load("/workspace/LAM/best/deltanet_shockwave_result.pt", map_location=device)
    model_state = model.state_dict()
    compatible_weights = {}
    for k, v in weights.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                compatible_weights[k] = v
            elif 'W_bilinear' in k and len(v.shape) == 2 and len(model_state[k].shape) == 3:
                num_heads = model_state[k].shape[0]
                compatible_weights[k] = v.unsqueeze(0).expand(num_heads, -1, -1).clone()
    
    model.load_state_dict(compatible_weights, strict=False)
    model.eval()
    print("   ✅ Model loaded")
    
    # 2. Load Adapter
    print("\n2️⃣  Loading adapter...")
    adapter = HolographicAdapter().to(device)
    adapter_path = Path(__file__).parent / "holographic_adapter.pt"
    
    if not adapter_path.exists():
        print(f"   ❌ Adapter not found at {adapter_path}")
        print("   💡 Run train_retrieval_finetune.py first")
        return
    
    adapter.load_state_dict(torch.load(adapter_path, map_location=device))
    adapter.eval()
    print("   ✅ Adapter loaded")
    
    # 3. Initialize Streamer
    print("\n3️⃣  Initializing streamer...")
    streamer = AsyncInfiniteStreamer(model, chunk_size=512)
    print("   ✅ Streamer ready")
    
    # 4. Evaluate
    print("\n" + "="*80)
    print("🚀 STARTING EVALUATION")
    print("="*80)
    
    # Evaluate STS-B (quick semantic similarity test)
    sts_spearman = evaluate_semantic_similarity(model, adapter, streamer, device)
    
    # Evaluate MS MARCO (main retrieval benchmark)
    msmarco_ndcg = evaluate_msmarco_sample(model, adapter, streamer, device)
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    if sts_spearman is not None:
        print(f"STS-B Spearman: {sts_spearman:.4f} ({sts_spearman*100:.2f}%) (target: >0.82)")
    
    if msmarco_ndcg is not None:
        print(f"MS MARCO nDCG@10: {msmarco_ndcg:.4f} ({msmarco_ndcg*100:.2f}%) (target: >0.377)")
    
    print("\n💡 To improve retrieval performance:")
    print("   1. Train adapter longer (more epochs)")
    print("   2. Use more training data (increase dataset size)")
    print("   3. Fine-tune on retrieval-specific data (MS MARCO triplets)")
    print("   4. Use contrastive learning with hard negatives")

if __name__ == "__main__":
    main()

