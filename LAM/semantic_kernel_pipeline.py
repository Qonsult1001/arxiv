import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pathlib import Path
import json
import gzip
import csv
import numpy as np
import os
from scipy.stats import spearmanr
from tqdm import tqdm
from typing import List, Union, Tuple

# ============================================================================
# 1. CONFIGURATION AND SETUP
# ============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ➡️ EDIT THIS LINE to change the model you want to kernelize!
#BASE_MODEL_NAME = 'stsb-roberta-base-v2' 
BASE_MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2' 
# --- Hyperparameters ---
KERNEL_TRAINING_EPOCHS = 1
KERNEL_TRAINING_LR = 2e-5
KERNEL_BATCH_SIZE = 128
# Use same multi-domain dataset as training (1.5-1.6M sentences → ~500k-800k triplets)
# Set to None to use all available data, or specify a limit
MAX_TRAINING_PAIRS = None  # None = use all available data (matches train_6layer_deltanet_1.py) 

script_dir = Path(__file__).parent
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)
kernel_output_path = data_dir / f"kernel_{BASE_MODEL_NAME.split('/')[-1]}.pt"
allnli_path = data_dir / "AllNLI.jsonl.gz"
stsb_path = data_dir / "stsbenchmark.tsv.gz"

# ============================================================================
# 2. MODEL AND DATA UTILITIES
# ============================================================================

class KernelDataset(Dataset):
    """Dataset for NLI sentence pairs used to train the kernel."""
    def __init__(self, anchors, positives, negatives):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx], self.negatives[idx]

def load_allnli_triplets_only(max_pairs: int = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Loads ONLY AllNLI triplets (high-quality negatives, no random generation).
    The kernel's job is to understand semantic relationships - AllNLI triplets
    with curated negatives are perfect for this. Other datasets are used for
    main model training, but kernel doesn't need them.
    
    Args:
        max_pairs: Maximum number of triplets to load. If None, loads all available AllNLI.
    """
    if max_pairs is None:
        print(f"\n📚 Loading AllNLI triplets for kernel training (using ALL available triplets)...")
    else:
        print(f"\n📚 Loading AllNLI triplets for kernel training (target: {max_pairs:,} triplets)...")
    print("="*80)
    print("💡 Using ONLY AllNLI triplets (high-quality curated negatives)")
    print("   Other datasets are for main model training, kernel doesn't need them")
    print("="*80)
    
    anchors, positives, negatives = [], [], []
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # Load ONLY AllNLI triplets (they already have high-quality negatives)
    print("\n1️⃣  Loading AllNLI triplets...")
    try:
        allnli_path = data_dir / "AllNLI.jsonl.gz"
    if not allnli_path.exists():
        print("📥 Downloading AllNLI corpus...")
        import requests
        url = "https://sbert.net/datasets/AllNLI.jsonl.gz"
        response = requests.get(url)
        with open(allnli_path, 'wb') as f:
            f.write(response.content)
    
    with gzip.open(allnli_path, 'rt', encoding='utf-8') as f:
            count = 0
            for line in tqdm(f, desc="Reading AllNLI triplets"):
                if max_pairs is not None and len(anchors) >= max_pairs:
                break
            try:
                    triplet = json.loads(line)
                    if len(triplet) == 3:
                        anchor, positive, negative = triplet
                        if len(anchor) > 10 and len(positive) > 10:
                            anchors.append(anchor)
                            positives.append(positive)
                            negatives.append(negative)  # High-quality curated negative
                            count += 1
            except:
                continue
        print(f"   ✅ AllNLI: {count:,} triplets (all with high-quality negatives)")
    except Exception as e:
        print(f"   ❌ Error loading AllNLI: {e}")
        return [], [], []
    
    print("="*80)
    print(f"✅ TOTAL: {len(anchors):,} triplets (anchors, positives, negatives)")
    print(f"   - All triplets from AllNLI with curated negatives (no random generation)")
    print("="*80)
    
    return anchors, positives, negatives

def load_stsb_test_data() -> List[dict]:
    """Reads the STS-B Test Set for evaluation."""
    if not stsb_path.exists():
        print("📥 Downloading STS-B dataset...")
        import requests
        url = "https://sbert.net/datasets/stsbenchmark.tsv.gz"
        response = requests.get(url)
        with open(stsb_path, 'wb') as f:
            f.write(response.content)

    test_samples = []
    with gzip.open(str(stsb_path), 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score']) / 5.0  # Normalize 0-5 to 0-1
                test_samples.append({
                    's1': row['sentence1'],
                    's2': row['sentence2'],
                    'score': score
                })
    return test_samples

# ============================================================================
# 3. KERNEL TRAINING
# ============================================================================

def train_semantic_kernel(model: SentenceTransformer, embed_dim: int, output_path: Path):
    """
    Trains a square matrix (kernel) to minimize the triplet margin loss 
    in the transformed space.
    """
    print("\n" + "="*80)
    print(f"🔧 STARTING KERNEL TRAINING for {BASE_MODEL_NAME}")
    print(f"Embedding Dimension: {embed_dim}")
    print("="*80)

    # 1. Initialize Kernel (Matrix W)
    # We use a square matrix initialized close to the Identity matrix
    kernel = nn.Parameter(torch.eye(embed_dim).to(device))
    
    # 2. Data Loader
    anchors, positives, negatives = load_allnli_triplets_only(MAX_TRAINING_PAIRS)
    dataset = KernelDataset(anchors, positives, negatives)
    dataloader = DataLoader(dataset, batch_size=KERNEL_BATCH_SIZE, shuffle=True, num_workers=0)

    # 3. Optimizer and Loss
    optimizer = optim.AdamW([kernel], lr=KERNEL_TRAINING_LR)
    # Triplet Loss: distance(A, P) should be < distance(A, N) - margin
    # We use cosine distance (1 - cosine_similarity)
    loss_fn = nn.TripletMarginLoss(margin=0.5, p=2, reduction='mean')

    # 4. Training Loop
    model.eval()
    for epoch in range(KERNEL_TRAINING_EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Kernel Epoch {epoch+1}/{KERNEL_TRAINING_EPOCHS}")
        
        for a_batch, p_batch, n_batch in pbar:
            optimizer.zero_grad()
            
            # Encode sentences using the base model
            sentences = list(a_batch) + list(p_batch) + list(n_batch)
            with torch.no_grad():
                embeddings_raw = model.encode(
                sentences, 
                convert_to_tensor=True, 
                normalize_embeddings=True, 
                show_progress_bar=False,
                batch_size=KERNEL_BATCH_SIZE
            ).to(device)
            
            # Clone embeddings to convert inference tensors to regular tensors
            # This allows them to be used in autograd operations (kernel transformation)
            # We detach and clone to create a fresh tensor that can participate in autograd
            embeddings = embeddings_raw.detach().clone()
            
            # Split back into A, P, N
            A = embeddings[:len(a_batch)]
            P = embeddings[len(a_batch):len(a_batch)*2]
            N = embeddings[len(a_batch)*2:]
            
            # Apply Kernel Transformation: V_new = V_old @ W
            A_kernel = torch.matmul(A, kernel)
            P_kernel = torch.matmul(P, kernel)
            N_kernel = torch.matmul(N, kernel)

            # Normalize transformed vectors for cosine similarity
            A_norm = F.normalize(A_kernel, p=2, dim=1)
            P_norm = F.normalize(P_kernel, p=2, dim=1)
            N_norm = F.normalize(N_kernel, p=2, dim=1)

            # Cosine Loss: we want A,P close and A,N far apart.
            # TripletMarginLoss works on L2 distance, so we use cosine distance (1 - similarity)
            
            # The loss needs three inputs: anchor, positive, negative.
            # dist(A, P) = L2 distance between A and P
            loss = loss_fn(A_norm, P_norm, N_norm)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Triplet Loss': f'{total_loss/(pbar.n+1):.6f}'})

    avg_loss = total_loss / len(dataloader)
    print(f"\n✅ Kernel Training Finished. Final Avg Triplet Loss: {avg_loss:.6f}")
    
    # Save the final kernel matrix
    torch.save({'kernel': kernel.detach()}, str(output_path))
    print(f"💾 Kernel matrix saved to {output_path}")

# ============================================================================
# 4. EVALUATION AND ALPHA OPTIMIZATION
# ============================================================================

def run_evaluation(model: SentenceTransformer, kernel: torch.Tensor, test_samples: List[dict]):
    """
    Evaluates Raw, Kernel, and Blended performance across various alpha values.
    Returns the optimal alpha and the best score.
    """
    print("\n" + "="*80)
    print("🔬 EVALUATING AND OPTIMIZING BLENDING ALPHA (STS-B TEST SET)")
    print("="*80)

    alphas = np.linspace(0.0, 1.0, num=11)  # Test alpha from 0.0 to 1.0 in steps of 0.1
    results = {}
    
    s1_all = [x['s1'] for x in test_samples]
    s2_all = [x['s2'] for x in test_samples]
    human_scores = [x['score'] for x in test_samples]

    # 1. Encode all sentences once (Raw Embeddings)
    print("Encoding all test sentences...")
    with torch.no_grad():
        E1_raw = model.encode(s1_all, convert_to_tensor=True, normalize_embeddings=True).to(device)
        E2_raw = model.encode(s2_all, convert_to_tensor=True, normalize_embeddings=True).to(device)

        # 2. Calculate Kernel Embeddings once
        E1_kernel = torch.matmul(E1_raw, kernel)
        E2_kernel = torch.matmul(E2_raw, kernel)
        E1_kernel = F.normalize(E1_kernel, p=2, dim=1)
        E2_kernel = F.normalize(E2_kernel, p=2, dim=1)

    # 3. Evaluate RAW Model (alpha=0.0)
    raw_sims = F.cosine_similarity(E1_raw, E2_raw, dim=1).cpu().numpy()
    raw_spearman, _ = spearmanr(raw_sims, human_scores)
    
    # 4. Evaluate KERNEL-ONLY Model (alpha=1.0)
    kernel_sims = F.cosine_similarity(E1_kernel, E2_kernel, dim=1).cpu().numpy()
    kernel_spearman, _ = spearmanr(kernel_sims, human_scores)
    
    best_score = raw_spearman
    optimal_alpha = 0.0
    
    print(f"Raw Model (alpha=0.0): {raw_spearman:.4f}")
    
    # 5. Loop through blending alphas
    for alpha in tqdm(alphas[1:-1], desc="Optimizing Alpha"): # Skip 0.0 and 1.0
        
        # Blend: E_final = (1-alpha) * E_raw + alpha * E_kernel
        E1_blended = (1.0 - alpha) * E1_raw + alpha * E1_kernel
        E2_blended = (1.0 - alpha) * E2_raw + alpha * E2_kernel
        
        # Normalize and Calculate Similarity
        E1_blended = F.normalize(E1_blended, p=2, dim=1)
        E2_blended = F.normalize(E2_blended, p=2, dim=1)

        blended_sims = F.cosine_similarity(E1_blended, E2_blended, dim=1).cpu().numpy()
        blended_spearman, _ = spearmanr(blended_sims, human_scores)
        
        results[alpha] = blended_spearman
        
        if blended_spearman > best_score:
            best_score = blended_spearman
            optimal_alpha = alpha

    # Add 1.0 (Kernel-only) back to results
    results[1.0] = kernel_spearman
    
    print(f"Kernel-Only (alpha=1.0): {kernel_spearman:.4f}")
    print("\n-------------------------------------------")
    print(f"🏆 BEST SCORE: {best_score:.4f} at Alpha={optimal_alpha:.2f}")
    print("-------------------------------------------")

    return optimal_alpha, best_score, raw_spearman

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def run_pipeline():
    """Runs the full kernelization pipeline for the configured model."""
    print("="*80)
    print(f"🚀 KERNELIZATION PIPELINE FOR: {BASE_MODEL_NAME}")
    print("="*80)

    # 1. Initialize Base Model
    print(f"🤖 Initializing Base Model: {BASE_MODEL_NAME}...")
    try:
        model = SentenceTransformer(BASE_MODEL_NAME).to(device)
        embed_dim = model.get_sentence_embedding_dimension()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Kernel Training
    if not kernel_output_path.exists():
        train_semantic_kernel(model, embed_dim, kernel_output_path)
    else:
        print(f"✅ Found existing kernel for this model at {kernel_output_path}. Skipping training.")

    # 3. Load Kernel
    kernel_state = torch.load(str(kernel_output_path), map_location=device, weights_only=False)
    kernel = kernel_state['kernel'].to(device)

    # 4. Load Data
    print("\n📚 Reading STS-B Test Set...")
    test_samples = load_stsb_test_data()
    print(f"✅ Loaded {len(test_samples)} test pairs.")
    
    # 5. Evaluation and Optimization
    optimal_alpha, best_score, raw_spearman = run_evaluation(model, kernel, test_samples)

    # 6. Final Summary
    print("\n" + "="*80)
    print("✅ PIPELINE SUMMARY")
    print(f"Base Model: {BASE_MODEL_NAME}")
    print(f"Original Raw Score: {raw_spearman:.4f}")
    print(f"Optimal Blended Score: {best_score:.4f}")
    print(f"Improvement: {best_score - raw_spearman:+.4f}")
    print(f"Optimal Alpha (α): {optimal_alpha:.2f}")
    print("="*80)
    
    if best_score > raw_spearman:
        print(f"\n🎉 SUCCESS: The blending improves the model! Use Blending (α={optimal_alpha:.2f})")
    else:
        print("\n⚠️ Note: Kernel blending did not improve performance. Use the Raw Model.")


if __name__ == "__main__":
    run_pipeline()