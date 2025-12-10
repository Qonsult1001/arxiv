"""
🚀 RETRIEVAL FINE-TUNING SCRIPT
===============================
Turns your high-scoring STS model (0.8190) into a Retrieval Beast.

Technique:
1. Load 0.8190 Checkpoint (Pre-trained on STS).
2. Load MS MARCO (Triplets: Query, Positive, Negative).
3. Train with Multiple Negatives Ranking Loss (MNRL).
   - This teaches the model: "Query should be closer to Positive than any other doc in the batch."

Goal: Fix the MTEB Retrieval gap (40 -> 50+) while keeping STS high.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Import your model definition
sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula_final import DeltaNetPure6Layer, config

# ============================================================================
# CONFIGURATION
# ============================================================================
FINE_TUNE_CONFIG = {
    # 1. PATHS
    "checkpoint_path": "/workspace/LAM/best/deltanet_shockwave_result.pt", # Your 0.8190 model
    "output_dir": "/workspace/LAM/retrieval_tuned",
    
    # 2. DATA
    # MS MARCO is the standard for retrieval training
    # We use a pre-mined triplet dataset for efficiency
    "dataset_name": "sentence-transformers/embedding-training-data", 
    "dataset_file": "msmarco-triplets.jsonl.gz", 
    "max_samples": 100_000, # Start small to prevent forgetting STS
    
    # 3. TRAINING
    "batch_size": 32,       # Larger is better for MNRL (more negatives)
    "learning_rate": 2e-5,  # Low LR to preserve existing weights
    "epochs": 1,            # 1 epoch is usually enough for adaptation
    "warmup_steps": 1000,
    "max_seq_length": 128   # Keep same as pre-training
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. DATA LOADING (MS MARCO TRIPLETS)
# ============================================================================
class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list):
        self.dataset = dataset_list

    def __getitem__(self, item):
        # Format: (Anchor/Query, Positive, Negative)
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

def load_msmarco_data(max_samples=100000):
    print(f"📚 Loading MS MARCO triplets (Max: {max_samples})...")
    # We create a simple synthetic loader for demo purposes if file missing, 
    # but in production use the real huggingface dataset
    try:
        dataset = load_dataset("sentence-transformers/embedding-training-data", data_files="msmarco-triplets.jsonl.gz", split="train", streaming=True)
        
        samples = []
        count = 0
        for row in dataset:
            # Format: [query, pos, neg]
            samples.append((row['set'][0], row['set'][1], row['set'][2]))
            count += 1
            if count >= max_samples:
                break
        print(f"✅ Loaded {len(samples)} triplets.")
        return samples
    except Exception as e:
        print(f"⚠️ Could not load specific HF dataset: {e}")
        print("   Using fallback local/synthetic data for testing...")
        # Fallback to local logic or error out
        return []

# ============================================================================
# 2. MODEL SETUP
# ============================================================================
def load_your_model():
    print(f"🔧 Initializing LAM from: {FINE_TUNE_CONFIG['checkpoint_path']}")
    
    # Initialize Architecture
    model = DeltaNetPure6Layer("microsoft/all-MiniLM-L6-v2", 6, config)
    
    # Load Weights
    state_dict = torch.load(FINE_TUNE_CONFIG['checkpoint_path'], map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    return model

# ============================================================================
# 3. MULTIPLE NEGATIVES RANKING LOSS
# ============================================================================
def multiple_negatives_ranking_loss(query_emb, pos_emb, neg_emb, scale=20.0):
    """
    Computes MNRL.
    - Query should be close to Positive.
    - Query should be far from Negative.
    - Query should be far from ALL other Positives/Negatives in the batch (In-batch negatives).
    """
    # 1. Positive Scores: (Batch, 1) -> Diagonal of Q @ P.T
    # Actually, standard MNRL implementation usually does:
    # Scores = Q @ P.T
    # Target = Range(Batch) (Diagonal)
    
    # We concatenate [Positive_Candidates, Negative_Candidates]
    # But standard MNRL often just uses (Query, Positive) pairs and treats other Positives as negatives.
    # To use Hard Negatives, we concat them.
    
    # [Batch, Dim]
    candidates = torch.cat([pos_emb, neg_emb], dim=0) # [2*Batch, Dim]
    
    # Similarity: [Batch, 2*Batch]
    scores = torch.matmul(query_emb, candidates.transpose(0, 1)) * scale
    
    # Targets: The positive for query i is at index i
    labels = torch.arange(len(query_emb), device=query_emb.device)
    
    return F.cross_entropy(scores, labels)

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================
def train():
    # 1. Setup
    model = load_your_model()
    model.train()
    
    # Load Data
    triplets = load_msmarco_data(FINE_TUNE_CONFIG['max_samples'])
    if not triplets:
        print("❌ No data loaded. Aborting.")
        return

    dataset = TripletDataset(triplets)
    dataloader = DataLoader(dataset, batch_size=FINE_TUNE_CONFIG['batch_size'], shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINE_TUNE_CONFIG['learning_rate'])
    
    print("\n🚀 STARTING RETRIEVAL FINE-TUNING...")
    print(f"   Batch Size: {FINE_TUNE_CONFIG['batch_size']}")
    print(f"   Steps: {len(dataloader)}")
    
    # 2. Loop
    for epoch in range(FINE_TUNE_CONFIG['epochs']):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            queries, positives, negatives = batch
            
            optimizer.zero_grad()
            
            # Tokenize all
            # Flatten to batch encode for speed? No, easier to separate for clarity here.
            
            # Helper to encode
            def get_emb(texts):
                tokens = model.tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
                # Use forward_student or standard forward
                emb, _, _ = model.forward_student(tokens['input_ids'], tokens['attention_mask'])
                return emb

            q_emb = get_emb(queries)
            p_emb = get_emb(positives)
            n_emb = get_emb(negatives)
            
            # Compute Loss
            loss = multiple_negatives_ranking_loss(q_emb, p_emb, n_emb)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    # 3. Save
    output_path = Path(FINE_TUNE_CONFIG['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "deltanet_retrieval_tuned.pt"
    
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Fine-tuning complete!")
    print(f"💾 Saved to: {save_path}")
    print("\n👉 ACTION: Update your MTEB script to point to this new checkpoint and run Retrieval tasks again.")

if __name__ == "__main__":
    train()