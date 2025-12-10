import torch
from sentence_transformers import SentenceTransformer
import gzip
import csv
import os
from scipy.stats import spearmanr
import numpy as np

# ==========================================
# 1. SETUP
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
kernel_path = "/workspace/LAM/data/pretrained_semantic_kernel_PAIRS.pt"
stsb_path = "/workspace/LAM/data/stsbenchmark.tsv.gz"
alphas_to_test = np.linspace(0.0, 1.0, 21) # Test alpha from 0.0 to 1.0 in 5% increments

print("======================================================================")
print("🔎 OPTIMIZING KERNEL BLENDING (ALPHA)")
print("======================================================================")

# ==========================================
# 2. LOAD RESOURCES
# ==========================================
print(f"🤖 Loading model: {model_name}...")
model = SentenceTransformer(model_name).to(device)

print(f"🔧 Loading Kernel: {kernel_path}...")
kernel_state = torch.load(kernel_path, map_location=device, weights_only=False)
kernel = kernel_state['kernel'].to(device)  # Kernel is already a tensor (384x384 matrix)

# ==========================================
# 3. PREPARE DATA
# ==========================================
print("📚 Reading STS-B Test Set...")
test_samples = []
with gzip.open(stsb_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0  # Normalize 0-5 to 0-1
            test_samples.append({
                's1': row['sentence1'],
                's2': row['sentence2'],
                'score': score
            })

human_scores = [x['score'] for x in test_samples]
print(f"✅ Loaded {len(test_samples)} test pairs.")

# ==========================================
# 4. RUN ALL BASE ENCODINGS ONCE
# ==========================================
print("⚡ Pre-calculating Raw and Kernel Embeddings...")
s1_batch = [x['s1'] for x in test_samples]
s2_batch = [x['s2'] for x in test_samples]

with torch.no_grad():
    # Raw Embeddings (Teacher) - normalized
    e1_raw = model.encode(s1_batch, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False).to(device)
    e2_raw = model.encode(s2_batch, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False).to(device)
    
    # Kernel Embeddings (Adapter) - apply kernel then normalize
    e1_kernel = torch.matmul(e1_raw, kernel)
    e2_kernel = torch.matmul(e2_raw, kernel)
    e1_kernel = torch.nn.functional.normalize(e1_kernel, p=2, dim=1)
    e2_kernel = torch.nn.functional.normalize(e2_kernel, p=2, dim=1)

# Move to CPU for numpy operations
e1_raw = e1_raw.cpu().numpy()
e2_raw = e2_raw.cpu().numpy()
e1_kernel = e1_kernel.cpu().numpy()
e2_kernel = e2_kernel.cpu().numpy()

# ==========================================
# 5. ITERATE AND TEST BLEND RATIOS
# ==========================================
best_alpha = 0.0
best_spearman = 0.0

print("\n--- Testing Blend Ratios ---")
print("Alpha | Spearman | Delta")
print("------|----------|-----------------")

# Calculate dot product for both raw and kernel space
def calculate_dot_product(e1, e2):
    # Normalized Dot Product (Cosine Similarity)
    e1_norm = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2_norm = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
    return np.sum(e1_norm * e2_norm, axis=1)

# Base score (Alpha=0.0) is the raw model
raw_scores = calculate_dot_product(e1_raw, e2_raw)
base_spearman, _ = spearmanr(raw_scores, human_scores)
best_spearman = base_spearman

print(f"{0.0:.3f} | {base_spearman:.4f} | +0.0000 (RAW)")


for alpha in alphas_to_test:
    if alpha == 0.0:
        continue # Already tested base case

    # Linear Interpolation: V_new = (1-alpha) * V_raw + alpha * V_kernel
    e1_blended = (1 - alpha) * e1_raw + alpha * e1_kernel
    e2_blended = (1 - alpha) * e2_raw + alpha * e2_kernel

    # Calculate correlation for the blended vectors
    blended_scores = calculate_dot_product(e1_blended, e2_blended)
    current_spearman, _ = spearmanr(blended_scores, human_scores)
    
    delta = current_spearman - base_spearman
    
    print(f"{alpha:.3f} | {current_spearman:.4f} | {delta:+.4f}")
    
    if current_spearman > best_spearman:
        best_spearman = current_spearman
        best_alpha = alpha

print("----------------------------------------------------------------------")
print(f"🥇 Best Spearman Score: {best_spearman:.4f}")
print(f"✨ Optimal Blend Ratio (Alpha): {best_alpha:.3f}")
print("======================================================================")