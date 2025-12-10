"""
🚀 RETRIEVAL-OPTIMIZED HOLOGRAPHIC ADAPTER TRAINING (PRESERVATION MODE)
======================================================================
Trains adapter with contrastive loss while PRESERVING existing 0.8190 Spearman score.
Based on successful fine-tuning approach from finetune_STSB_final_nuke_cont_6.py

Key: Low LR, careful loss balancing, continuous evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
from pathlib import Path
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Import your stack
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_6layer_deltanet_2 import DeltaNetPure6Layer, config
    from infinite_streamer_async import AsyncInfiniteStreamer
except ImportError:
    print("❌ Error: Missing LAM components.")
    sys.exit(1)

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

def contrastive_loss(query_emb, pos_emb, neg_emb, temperature=0.05):
    """InfoNCE contrastive loss for retrieval"""
    query_emb = F.normalize(query_emb, p=2, dim=1)
    pos_emb = F.normalize(pos_emb, p=2, dim=1)
    neg_emb = F.normalize(neg_emb, p=2, dim=1)
    
    pos_sim = torch.sum(query_emb * pos_emb, dim=1) / temperature
    neg_sim = torch.matmul(query_emb, neg_emb.t()) / temperature
    
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(query_emb.size(0), dtype=torch.long, device=query_emb.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss

def evaluate_stsb(model, adapter, streamer, device, split='validation', max_samples=500):
    """Evaluate on STS-B to monitor performance preservation"""
    try:
        from datasets import load_dataset
        os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
        os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
        
        # Try to load from cache first
        try:
            dataset = load_dataset(
                "sentence-transformers/stsb",
                split=split,
                cache_dir="/workspace/.cache/huggingface/datasets"
            )
        except:
            # If loading fails, skip evaluation
            print(f"   ⚠️  Could not load STS-B for evaluation, skipping...")
            return None, None
        
        s1 = dataset["sentence1"][:max_samples]
        s2 = dataset["sentence2"][:max_samples]
        labels = np.array(dataset["score"][:max_samples], dtype=float)
        
        model.eval()
        adapter.eval()
        embeddings1 = []
        embeddings2 = []
        
        with torch.no_grad():
            # Process in batches for efficiency
            for i in range(0, len(s1), 16):  # Smaller batch size for evaluation
                batch_s1 = s1[i:i+16]
                batch_s2 = s2[i:i+16]
                
                batch_emb1 = []
                batch_emb2 = []
                
                for text in batch_s1:
                    try:
                        enc = model.tokenizer.encode(text, add_special_tokens=True)
                        ids = torch.tensor([enc], device=device)
                        streamer_emb = streamer.stream_embedding(ids, verbose=False)
                        adapter_emb = adapter(streamer_emb)
                        batch_emb1.append(adapter_emb)
                    except Exception as e:
                        # Skip problematic texts
                        continue
                
                for text in batch_s2:
                    try:
                        enc = model.tokenizer.encode(text, add_special_tokens=True)
                        ids = torch.tensor([enc], device=device)
                        streamer_emb = streamer.stream_embedding(ids, verbose=False)
                        adapter_emb = adapter(streamer_emb)
                        batch_emb2.append(adapter_emb)
                    except Exception as e:
                        continue
                
                if len(batch_emb1) > 0 and len(batch_emb2) > 0:
                    embeddings1.append(torch.cat(batch_emb1, dim=0))
                    embeddings2.append(torch.cat(batch_emb2, dim=0))
        
        if len(embeddings1) == 0 or len(embeddings2) == 0:
            return None, None
        
        embeddings1 = torch.cat(embeddings1, dim=0)
        embeddings2 = torch.cat(embeddings2, dim=0)
        
        # Ensure same length
        min_len = min(len(embeddings1), len(embeddings2), len(labels))
        embeddings1 = embeddings1[:min_len]
        embeddings2 = embeddings2[:min_len]
        labels = labels[:min_len]
        
        similarities = F.cosine_similarity(embeddings1, embeddings2, dim=1).cpu().numpy()
        
        pearson = pearsonr(similarities, labels)[0]
        spearman = spearmanr(similarities, labels)[0]
        
        model.train()
        adapter.train()
        
        return pearson, spearman
    except Exception as e:
        print(f"   ⚠️  Evaluation error: {e}")
        return None, None

def train_readout():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Training Retrieval-Optimized Adapter (Preservation Mode)")
    print("="*80)
    print("🎯 Goal: Improve retrieval while preserving 0.8190 Spearman")
    print("="*80)
    
    # 1. Load Model (Frozen)
    print("\n1️⃣  Loading model...")
    model_path = str(Path(__file__).parent / "all-MiniLM-L6-v2")
    model = DeltaNetPure6Layer(model_path, 6, config).to(device)
    
    # Load weights
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
                print(f"   🔄 Converted {k}: {v.shape} -> {compatible_weights[k].shape}")
    
    model.load_state_dict(compatible_weights, strict=False)
    model.eval()
    print("   ✅ Model loaded")
    
    # 2. Initialize Adapter (Trainable)
    print("\n2️⃣  Initializing adapter...")
    adapter = HolographicAdapter().to(device)
    
    # Check if adapter exists and load it
    adapter_path = Path(__file__).parent / "holographic_adapter.pt"
    if adapter_path.exists():
        print("   📦 Loading existing adapter...")
        adapter.load_state_dict(torch.load(adapter_path, map_location=device))
        print("   ✅ Loaded existing adapter")
    
    # CONSERVATIVE LEARNING RATE (like fine-tuning script)
    # Start with very low LR to preserve existing performance
    initial_lr = 1e-5  # Much lower than before (was 2e-4)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=initial_lr, weight_decay=0.01)
    
    # Learning rate scheduler - gradual decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15, eta_min=1e-7
    )
    print(f"   ✅ Adapter initialized (LR: {initial_lr:.2e})")
    
    # 3. Load Retrieval Training Data
    print("\n3️⃣  Loading retrieval training data...")
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
    
    from datasets import load_dataset
    
    print("   📦 Loading MS MARCO triplets...")
    try:
        dataset = load_dataset(
            "sentence-transformers/embedding-training-data", 
            data_files="msmarco-triplets.jsonl.gz", 
            split="train[:5000]",  # Reduced from 10000 for faster training
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
        print(f"   ✅ Loaded {len(dataset)} triplets from cache")
        
        triplets = []
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
                    negs = row['neg'][:2]  # Use up to 2 negatives
                elif isinstance(row['neg'], str):
                    negs = [row['neg']]
                else:
                    continue
                
                triplets.append({
                    'query': query,
                    'positive': pos,
                    'negatives': negs
                })
        
        print(f"   ✅ Extracted {len(triplets)} valid triplets")
        
    except Exception as e:
        print(f"   ⚠️  Error loading MS MARCO: {e}")
        print("   💡 Falling back to STS-B pairs...")
        dataset = load_dataset(
            "sentence-transformers/stsb", 
            split="train[:3000]",
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
        triplets = []
        for row in dataset:
            triplets.append({
                'query': row['sentence1'],
                'positive': row['sentence2'],
                'negatives': []
            })
        print(f"   ✅ Loaded {len(triplets)} pairs from STS-B")
    
    streamer = AsyncInfiniteStreamer(model, chunk_size=512)
    
    # 4. Initial Evaluation
    print("\n4️⃣  Initial evaluation (baseline)...")
    initial_pearson, initial_spearman = evaluate_stsb(model, adapter, streamer, device, max_samples=500)
    if initial_spearman:
        print(f"   📊 Initial STS-B: Pearson={initial_pearson:.4f}, Spearman={initial_spearman:.4f}")
        target_spearman = max(0.8190, initial_spearman)  # Preserve at least 0.8190
        print(f"   🎯 Target: Maintain Spearman >= {target_spearman:.4f}")
    else:
        target_spearman = 0.8190
        print(f"   🎯 Target: Maintain Spearman >= {target_spearman:.4f}")
    
    # 5. Training Loop (Conservative)
    print("\n5️⃣  Starting conservative training...")
    print(f"   Epochs: 15 (reduced for stability)")
    print(f"   Learning rate: {initial_lr:.2e} (conservative)")
    print(f"   Loss: Contrastive + Preservation")
    print("="*80)
    
    adapter.train()
    best_spearman = initial_spearman if initial_spearman else 0.8190
    best_epoch = 0
    patience = 7  # Increased from 3 to 7 - give more time for recovery
    patience_counter = 0
    min_spearman_threshold = target_spearman - 0.01  # Allow small drops (0.01 = 1%)
    
    for epoch in range(15):
        total_loss = 0
        total_contrastive = 0
        total_preserve = 0
        num_batches = 0
        
        # Shuffle triplets
        indices = np.random.permutation(len(triplets))
        
        pbar = tqdm(range(0, len(triplets), 32), desc=f"Epoch {epoch+1}/15")
        for i in pbar:
            batch_triplets = [triplets[int(idx)] for idx in indices[i:i+32]]
            
            queries = [t['query'] for t in batch_triplets]
            positives = [t['positive'] for t in batch_triplets]
            
            # Get negatives
            negatives_list = []
            for j, t in enumerate(batch_triplets):
                if len(t['negatives']) > 0:
                    negatives_list.append(t['negatives'][0])
                else:
                    # In-batch negative
                    neg_idx = np.random.randint(0, len(batch_triplets))
                    while neg_idx == j:
                        neg_idx = np.random.randint(0, len(batch_triplets))
                    negatives_list.append(batch_triplets[neg_idx]['positive'])
            
            # Get Teacher Embeddings (Target)
            with torch.no_grad():
                q_encoded = model.tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to(device)
                q_teacher, _ = model.forward_teacher(q_encoded['input_ids'], q_encoded['attention_mask'])
                
                p_encoded = model.tokenizer(positives, padding=True, truncation=True, return_tensors='pt').to(device)
                p_teacher, _ = model.forward_teacher(p_encoded['input_ids'], p_encoded['attention_mask'])
                
                n_encoded = model.tokenizer(negatives_list, padding=True, truncation=True, return_tensors='pt').to(device)
                n_teacher, _ = model.forward_teacher(n_encoded['input_ids'], n_encoded['attention_mask'])
            
            # Get Streamer Embeddings (Input)
            q_streamer_embs = []
            p_streamer_embs = []
            n_streamer_embs = []
            
            with torch.no_grad():
                for query in queries:
                    enc = model.tokenizer.encode(query, add_special_tokens=True)
                    ids = torch.tensor([enc], device=device)
                    emb = streamer.stream_embedding(ids, verbose=False)
                    q_streamer_embs.append(emb)
                
                for pos in positives:
                    enc = model.tokenizer.encode(pos, add_special_tokens=True)
                    ids = torch.tensor([enc], device=device)
                    emb = streamer.stream_embedding(ids, verbose=False)
                    p_streamer_embs.append(emb)
                
                for neg in negatives_list:
                    enc = model.tokenizer.encode(neg, add_special_tokens=True)
                    ids = torch.tensor([enc], device=device)
                    emb = streamer.stream_embedding(ids, verbose=False)
                    n_streamer_embs.append(emb)
            
            q_streamer = torch.cat(q_streamer_embs, dim=0)
            p_streamer = torch.cat(p_streamer_embs, dim=0)
            n_streamer = torch.cat(n_streamer_embs, dim=0)
            
            # Apply Adapter
            optimizer.zero_grad()
            
            q_adapter = adapter(q_streamer)
            p_adapter = adapter(p_streamer)
            n_adapter = adapter(n_streamer)
            
            # Losses
            # 1. Contrastive loss (retrieval objective)
            contrastive = contrastive_loss(q_adapter, p_adapter, n_adapter, temperature=0.05)
            
            # 2. Preservation loss (maintain teacher alignment)
            # This is CRITICAL to preserve existing performance
            preserve_q = F.mse_loss(q_adapter, q_teacher)
            preserve_p = F.mse_loss(p_adapter, p_teacher)
            preserve_n = F.mse_loss(n_adapter, n_teacher)
            preservation = (preserve_q + preserve_p + preserve_n) / 3.0
            
            # Combined loss - HEAVILY weight preservation
            # This ensures we don't lose existing performance
            loss = (
                0.3 * contrastive +      # Retrieval improvement
                1.0 * preservation        # Performance preservation (HIGHER weight!)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.5)  # Conservative clipping
            optimizer.step()
            
            total_loss += loss.item()
            total_contrastive += contrastive.item()
            total_preserve += preservation.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Contrastive': f'{contrastive.item():.6f}',
                'Preserve': f'{preservation.item():.6f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        avg_contrastive = total_contrastive / num_batches
        avg_preserve = total_preserve / num_batches
        
        # Evaluate every epoch
        print(f"\nEpoch {epoch+1}/15:")
        print(f"  Loss: {avg_loss:.6f} (Contrastive: {avg_contrastive:.6f}, Preserve: {avg_preserve:.6f})")
        
        eval_pearson, eval_spearman = evaluate_stsb(model, adapter, streamer, device, max_samples=500)
        if eval_spearman:
            print(f"  📊 STS-B: Pearson={eval_pearson:.4f}, Spearman={eval_spearman:.4f}")
            
            # Check if we're maintaining performance
            # Allow small drops - only count as "drop" if significantly below threshold
            if eval_spearman >= target_spearman:
                print(f"  ✅ Performance maintained (target: {target_spearman:.4f})")
                patience_counter = 0
            elif eval_spearman >= min_spearman_threshold:
                # Small drop but still close to target - reset patience
                print(f"  ⚠️  Slightly below target by {target_spearman - eval_spearman:.4f} (but above threshold {min_spearman_threshold:.4f})")
                patience_counter = 0  # Reset - give it time to recover
            else:
                # Significant drop below threshold
                print(f"  ⚠️  Below threshold by {min_spearman_threshold - eval_spearman:.4f} (target: {target_spearman:.4f})")
                patience_counter += 1
            
            # Save best model
            if eval_spearman > best_spearman:
                best_spearman = eval_spearman
                best_epoch = epoch + 1
                torch.save(adapter.state_dict(), "holographic_adapter.pt")
                print(f"  💾 Saved best model (Spearman: {best_spearman:.4f})")
            
            # Early stopping if performance drops significantly for many epochs
            # Only stop if it's been bad for a while AND current is much worse than best
            if patience_counter >= patience:
                if eval_spearman < best_spearman - 0.005:  # Only stop if current is 0.5% worse than best
                    print(f"\n  ⚠️  Performance dropped significantly for {patience} epochs.")
                    print(f"  💡 Current: {eval_spearman:.4f}, Best: {best_spearman:.4f}")
                    print(f"  💡 Restoring best model from epoch {best_epoch}...")
                    adapter.load_state_dict(torch.load("holographic_adapter.pt", map_location=device))
                    break
                else:
                    # Still close to best - reset patience and continue
                    print(f"  💡 Performance close to best ({best_spearman:.4f}), continuing...")
                    patience_counter = 0
        else:
            print(f"  ⚠️  Could not evaluate")
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)
    print(f"Best Spearman: {best_spearman:.4f} (epoch {best_epoch})")
    print(f"Target: {target_spearman:.4f}")
    
    if best_spearman >= target_spearman:
        print("🎉 SUCCESS: Performance maintained/improved!")
    else:
        print(f"⚠️  Performance dropped by {target_spearman - best_spearman:.4f}")
        print("💡 Consider: Lower LR, more preservation weight, fewer epochs")
    
    print(f"\nAdapter saved to: holographic_adapter.pt")
    print("\n💡 Next: Run python evaluate_retrieval_adapter.py")

if __name__ == "__main__":
    train_readout()
