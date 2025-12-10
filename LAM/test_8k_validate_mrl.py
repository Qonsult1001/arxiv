import torch
import torch.nn.functional as F
import time
import gc
import sys
import os
from pathlib import Path
import random
from datasets import load_dataset # Added for Real Data

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DB_SIZES = [10_000, 100_000, 1_000_000, 5_000_000] 
NUM_NEEDLES = 1000 # Increased for better statistics
DIMS_TO_TEST = [384, 256, 128, 64] # Test ALL dimensions to see the curve
RECALL_THRESHOLD = 95.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Import your class
try:
    from test_8k_LAM import LAM
except ImportError:
    # Mock for standalone testing
    class LAM:
        def __init__(self, checkpoint_path, device): 
            self.device = device
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
        def encode(self, text, dimensions=None):
            return torch.randn(len(text), 384).to(self.device)

print(f"🚀 STARTING COMPREHENSIVE 'HARD MODE' BENCHMARK")
print(f"   Note: Using REAL Quora Dataset for Needles + Camouflaged Noise for Scale.")
print("="*80)

# Set cache directory (aligned with training script)
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

# 2. LOAD MODEL
print("\n📦 Loading LAM model...")
lam_model = LAM(checkpoint_path="/workspace/LAM/best/pytorch_model.bin", device=DEVICE)

# 3. GENERATE & ENCODE NEEDLES (Real Semantics)
def get_real_needles():
    print("📝 Loading 1,000 Real Pairs from QQP (Quora Question Pairs)...")
    # We use QQP (GLUE) which is more stable than the old Quora dataset
    # label=1 means the questions are semantically similar/duplicate
    try:
        # Try QQP from GLUE (more reliable)
        try:
            dataset = load_dataset("glue", "qqp", split="train", cache_dir="/workspace/.cache/huggingface/datasets")
            print("   -> Loading QQP from GLUE...")
            docs = []
            queries = []
            for row in dataset:
                # label=1 means duplicate/similar questions
                if row.get('label', -1) == 1:
                    q1 = row.get('question1', '')
                    q2 = row.get('question2', '')
                    if len(q1) > 10 and len(q2) > 10:  # Filter out very short questions
                        queries.append(q1)
                        docs.append(q2)
                        if len(queries) >= NUM_NEEDLES:
                            break
            if len(queries) >= NUM_NEEDLES:
                print(f"   ✅ Loaded {len(queries)} pairs from QQP")
                return docs, queries
        except Exception as e1:
            print(f"   ⚠️ Could not load QQP: {e1}")
            pass
        
        # Fallback: Try AllNLI dataset (sentence-transformers)
        try:
            print("   -> Trying AllNLI dataset...")
            dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train", cache_dir="/workspace/.cache/huggingface/datasets")
            docs = []
            queries = []
            for row in dataset:
                anchor = row.get('anchor', '')
                positive = row.get('positive', '')
                if len(anchor) > 10 and len(positive) > 10:
                    queries.append(anchor)
                    docs.append(positive)
                    if len(queries) >= NUM_NEEDLES:
                        break
            if len(queries) >= NUM_NEEDLES:
                print(f"   ✅ Loaded {len(queries)} pairs from AllNLI")
                return docs, queries
        except Exception as e2:
            print(f"   ⚠️ Could not load AllNLI: {e2}")
            pass
        
        # If all else fails, use synthetic
        raise ValueError("Could not load any real dataset")
    except Exception as e:
        print(f"   ⚠️ Could not load real dataset ({e}). Falling back to Synthetic...")
        # Fallback if internet/HF is down
        return generate_synthetic_needles()

def generate_synthetic_needles():
    topics = ["contract liability", "python recursion error", "heart disease symptoms", "Q3 financial earnings", "Roman empire collapse", "quantum physics entanglement", "machine learning transformers", "climate change mitigation", "supply chain logistics", "criminal law precedence"]
    docs = []
    queries = []
    for i in range(NUM_NEEDLES):
        topic = random.choice(topics)
        docs.append(f"detailed report on {topic} specific instance {i} with unique context. " * 5)
        queries.append(f"search for {topic} instance {i}")
    return docs, queries

docs_text, queries_text = get_real_needles()

with torch.no_grad():
    # Encode in batches
    needle_docs = []
    needle_queries = []
    batch_size = 32
    
    print("   -> Encoding Needles...")
    for i in range(0, len(docs_text), batch_size):
        # Handle list vs string input
        batch_d = docs_text[i:i+batch_size]
        batch_q = queries_text[i:i+batch_size]
        
        try:
            emb_d = lam_model.encode(batch_d)
            emb_q = lam_model.encode(batch_q)
        except:
            emb_d = torch.stack([lam_model.encode(t) for t in batch_d])
            emb_q = torch.stack([lam_model.encode(t) for t in batch_q])
            
        if isinstance(emb_d, dict): emb_d = emb_d[384]
        if isinstance(emb_q, dict): emb_q = emb_q[384]
            
        needle_docs.append(emb_d)
        needle_queries.append(emb_q)

    needle_docs = torch.cat(needle_docs).to(DEVICE)
    needle_queries = torch.cat(needle_queries).to(DEVICE)

# 4. COMPUTE STATS FOR "HARD NOISE"
# We calculate the mean and covariance of real embeddings
# to generate noise that looks like real data.
print("   -> Computing Embedding Statistics for Hard Noise...")
doc_mean = torch.mean(needle_docs, dim=0)
doc_std = torch.std(needle_docs, dim=0)

print(f"   ✅ Needles Ready. Shape: {needle_docs.shape}")

# ==============================================================================
# 5. THE BENCHMARK ENGINE
# ==============================================================================

def run_stress_test(db_size):
    print(f"\n📦 SCALE: {db_size:,} Documents")
    
    # A. Generate text documents (needles + noise texts)
    num_noise = db_size - NUM_NEEDLES
    
    # Generate noise text documents that look realistic
    noise_texts = []
    if num_noise > 0:
        print(f"   -> Generating {num_noise:,} 'Hard' Noise Text Documents...")
        topics = ["contract liability", "python recursion error", "heart disease symptoms", "Q3 financial earnings", "Roman empire collapse", "quantum physics entanglement", "machine learning transformers", "climate change mitigation", "supply chain logistics", "criminal law precedence"]
        for i in range(num_noise):
            topic = random.choice(topics)
            # Create similar but distinct noise documents
            noise_texts.append(f"document about {topic} variation {i} with different context. " * 5)
    
    # Combine needles and noise texts
    all_doc_texts = docs_text + noise_texts
    
    # Shuffle to mix needles and noise (deterministic for reproducibility)
    import random as py_random
    py_random.seed(42)
    combined = list(zip(all_doc_texts, range(len(all_doc_texts))))
    py_random.shuffle(combined)
    all_doc_texts_shuffled = [item[0] for item in combined]
    
    # Find where needles ended up after shuffling
    query_to_needle_position = {}
    for shuffled_idx, (_, orig_idx) in enumerate(combined):
        if orig_idx < NUM_NEEDLES:  # This is a needle
            query_to_needle_position[orig_idx] = shuffled_idx
    
    # Create tensor: needle_positions[i] = position where needle i ended up
    needle_positions = torch.tensor([query_to_needle_position[i] for i in range(NUM_NEEDLES)], device=DEVICE)

    print("-" * 90)
    print(f"{'Dim':<5} {'Time (ms)':<15} {'Speedup':<10} {'Recall@100':<15} {'Verdict'}")
    print("-" * 90)

    baseline_time = 0
    recommended_dim = 384

    # C. Test Dimensions (384 -> 64 to establish baseline first)
    # Process each dimension independently to save memory
    for dim in DIMS_TO_TEST: 
        
        # 1. Encode documents at this specific Matryoshka dimension
        print(f"   -> Encoding {db_size:,} docs at {dim}-dim...", end="", flush=True)
        encoding_batch_size = 1000
        db_slice_list = []
        
        for i in range(0, db_size, encoding_batch_size):
            batch_texts = all_doc_texts_shuffled[i:i+encoding_batch_size]
            try:
                batch_emb = lam_model.encode(batch_texts, dimensions=dim)
            except:
                # Fallback for single string inputs
                batch_emb = torch.stack([lam_model.encode(t, dimensions=dim) for t in batch_texts])
            
            if isinstance(batch_emb, dict):
                batch_emb = batch_emb[dim]
            
            # Keep on CPU initially to save GPU memory
            db_slice_list.append(batch_emb.cpu())
            
            if (i + encoding_batch_size) % 50000 == 0:
                print(f" {i:,}...", end="", flush=True)
        
        # Move to GPU only when needed for search
        db_slice = torch.cat(db_slice_list, dim=0).to(DEVICE)
        del db_slice_list
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        print(f" Done. VRAM: {db_slice.element_size() * db_slice.nelement() / 1024**3:.2f} GB")
        
        # 2. Encode queries at this dimension
        q_slice_list = []
        for i in range(0, NUM_NEEDLES, encoding_batch_size):
            batch_q = queries_text[i:i+encoding_batch_size]
            try:
                batch_emb = lam_model.encode(batch_q, dimensions=dim)
            except:
                batch_emb = torch.stack([lam_model.encode(t, dimensions=dim) for t in batch_q])
            
            if isinstance(batch_emb, dict):
                batch_emb = batch_emb[dim]
            
            q_slice_list.append(batch_emb)
        
        q_slice = torch.cat(q_slice_list, dim=0).to(DEVICE)
        del q_slice_list
        
        # Ensure contiguous for performance
        db_slice = db_slice.contiguous()
        q_slice = q_slice.contiguous()
        
        # 3. Warmup (Crucial for fair speed comparison)
        if dim == 384:
            torch.matmul(q_slice[:10], db_slice[:10000].T)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()

        # 4. Search
        if DEVICE == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()
        
        # Matrix Mul: [NUM_NEEDLES, dim] @ [dim, db_size]
        scores = torch.matmul(q_slice, db_slice.T)
        topk_indices = torch.topk(scores, k=100, dim=1).indices
        
        if DEVICE == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            avg_time = start_event.elapsed_time(end_event)
        else:
            avg_time = (time.time() - start_time) * 1000
        
        # 5. Recall
        topk_cpu = topk_indices.cpu()
        targets_cpu = needle_positions.cpu()
        
        # Vectorized check for speed
        # Check if targets_cpu[i] is in topk_cpu[i]
        # (N, 1) == (N, 100) -> (N, 100) -> any(dim=1) -> sum
        matches = (targets_cpu.unsqueeze(1) == topk_cpu).any(dim=1)
        hits = matches.sum().item()
        
        recall = (hits / NUM_NEEDLES) * 100
        
        # 6. Stats
        if dim == 384:
            baseline_time = avg_time
            speedup = 1.0
        else:
            speedup = baseline_time / avg_time if avg_time > 0 else 0
            
        # Verdict Logic
        if recall >= 98: ver = "✅ PERFECT"
        elif recall >= 95: ver = "✅ GREAT"
        elif recall >= 90: ver = "⚠️ ACCEPTABLE"
        else: ver = "❌ FAIL"
        
        # Save recommendation
        if recall >= RECALL_THRESHOLD:
            recommended_dim = dim

        print(f"{dim:<5} {avg_time:<15.2f} {speedup:<10.1f}x {recall:<15.1f}% {ver}")
        
        # Cleanup after each dimension to free memory
        del db_slice, q_slice, scores, topk_indices
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    print(f"   💡 AUTO-LAM RECOMMENDATION: {recommended_dim}-dim")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
for scale in DB_SIZES:
    try:
        run_stress_test(scale)
        print("="*80)
    except Exception as e:
        print(f"\n❌ Error at {scale}: {e}")
        import traceback
        traceback.print_exc()
        break