import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from pathlib import Path
import gzip
import csv
import os
from scipy.stats import spearmanr
import numpy as np
from typing import List, Union

# ==========================================
# 1. SETUP
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
script_dir = Path(__file__).parent
kernel_path = script_dir / "data" / "pretrained_semantic_kernel_PAIRS.pt"
stsb_path = script_dir / "data" / "stsbenchmark.tsv.gz"

# ==========================================
# BLENDED SEMANTIC KERNEL MODEL CLASS
# ==========================================
class BlendedSemanticKernelModel:
    """
    Inference class combining the MiniLM teacher model with the custom 
    pair-aware kernel using an optimal blending ratio (alpha).
    
    The final vector is calculated as: V_final = (1 - alpha) * V_raw + alpha * V_kernel
    """
    
    def __init__(self, model_name: str, kernel_path: str, alpha: float = 0.10):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        
        # 1. Load the Teacher (MiniLM) Model
        print(f"Loading Teacher Model: {model_name}...")
        self.teacher_model = SentenceTransformer(model_name).to(self.device)
        
        # 2. Load the Kernel Matrix
        print(f"Loading Kernel from: {kernel_path}...")
        kernel_state = torch.load(str(kernel_path), map_location=self.device, weights_only=False)
        self.kernel = kernel_state['kernel'].to(self.device)  # Kernel is already a tensor (384x384 matrix)
        
        print(f"✅ Blended Model Ready (Alpha = {self.alpha:.2f})")

    def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encodes sentences using the blended approach.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
            
    with torch.no_grad():
            # Get raw embeddings from the teacher model (V_raw) - normalized
            V_raw = self.teacher_model.encode(
                sentences, 
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).to(self.device)
            
            # Apply the kernel transformation (V_kernel)
            V_kernel = torch.matmul(V_raw, self.kernel)
            V_kernel = F.normalize(V_kernel, p=2, dim=1)
    
            # Blend the vectors: V_final = (1-alpha) * V_raw + alpha * V_kernel
            V_final = (1.0 - self.alpha) * V_raw + self.alpha * V_kernel
            
            # Normalize and return as a numpy array for easy use
            V_final = F.normalize(V_final, p=2, dim=1)
            return V_final.cpu().numpy()

print("======================================================================")
print("🧪 STS-B BENCHMARK: RAW vs. KERNEL vs. BLENDED MODEL")
print("======================================================================")

# Download STS-B if not exists
if not stsb_path.exists():
    print("📥 Downloading STS-B dataset...")
    import requests
    url = "https://sbert.net/datasets/stsbenchmark.tsv.gz"
    response = requests.get(url)
    stsb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stsb_path, 'wb') as f:
        f.write(response.content)

# ==========================================
# 2. LOAD RESOURCES
# ==========================================
print(f"🤖 Loading model: {model_name}...")
model = SentenceTransformer(model_name).to(device)

print(f"🔧 Loading Kernel: {kernel_path}...")
kernel_state = torch.load(str(kernel_path), map_location=device, weights_only=False)
kernel = kernel_state['kernel'].to(device)  # Kernel is already a tensor (384x384 matrix)

# ==========================================
# 3. PREPARE DATA
# ==========================================
print("📚 Reading STS-B Test Set...")
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

print(f"✅ Loaded {len(test_samples)} test pairs.")

# ==========================================
# 2.5. CREATE BLENDED MODEL
# ==========================================
# Use optimal alpha from semantic_optimize_blend.py (default 0.10)
BLEND_ALPHA = 0.10
print(f"\n🔀 Creating Blended Model (Alpha = {BLEND_ALPHA:.2f})...")
blended_model = BlendedSemanticKernelModel(model_name, kernel_path, alpha=BLEND_ALPHA)

# ==========================================
# 4. RUN EVALUATION
# ==========================================
raw_scores = []
kernel_scores = []
blended_scores = []
human_scores = []

print("\n⚡ Running Inference...")

# Batch processing for speed
batch_size = 64
for i in range(0, len(test_samples), batch_size):
    batch = test_samples[i:i+batch_size]
    s1_batch = [x['s1'] for x in batch]
    s2_batch = [x['s2'] for x in batch]
    labels = [x['score'] for x in batch]
    
    # 1. Encode
    with torch.no_grad():
        e1 = model.encode(s1_batch, convert_to_tensor=True, normalize_embeddings=True).to(device)
        e2 = model.encode(s2_batch, convert_to_tensor=True, normalize_embeddings=True).to(device)
        
        # 2. RAW Cosine Similarity
        curr_raw_sims = F.cosine_similarity(e1, e2, dim=1).cpu().numpy()
        
        # 3. KERNEL Similarity
        # Apply kernel: v_new = v_old @ kernel (matrix multiplication)
        k1 = torch.matmul(e1, kernel)
        k2 = torch.matmul(e2, kernel)

        # Normalize transformed vectors
        k1_norm = F.normalize(k1, p=2, dim=1)
        k2_norm = F.normalize(k2, p=2, dim=1)
        curr_kernel_sims = F.cosine_similarity(k1_norm, k2_norm, dim=1).cpu().numpy()
        
        # 4. BLENDED Similarity (using blended model)
        b1 = blended_model.encode(s1_batch)
        b2 = blended_model.encode(s2_batch)
        # Calculate cosine similarity for blended embeddings
        curr_blended_sims = np.array([np.dot(b1[j], b2[j]) for j in range(len(b1))])

    raw_scores.extend(curr_raw_sims)
    kernel_scores.extend(curr_kernel_sims)
    blended_scores.extend(curr_blended_sims)
    human_scores.extend(labels)

# ==========================================
# 5. CALCULATE METRICS (SPEARMAN)
# ==========================================
raw_spearman, _ = spearmanr(raw_scores, human_scores)
kernel_spearman, _ = spearmanr(kernel_scores, human_scores)
blended_spearman, _ = spearmanr(blended_scores, human_scores)

print("\n======================================================================")
print("📊 FINAL RESULTS (Spearman Correlation)")
print("======================================================================")
print(f"Human vs. Raw Model:      {raw_spearman:.4f}")
print(f"Human vs. Kernel Model:   {kernel_spearman:.4f}")
print(f"Human vs. Blended Model:  {blended_spearman:.4f} (Alpha={BLEND_ALPHA:.2f})")
print("----------------------------------------------------------------------")

# Compare improvements
kernel_diff = kernel_spearman - raw_spearman
blended_diff = blended_spearman - raw_spearman

print(f"\n📈 IMPROVEMENTS vs. Raw Model:")
print(f"   Kernel-only:  {kernel_diff:+.4f}")
print(f"   Blended:       {blended_diff:+.4f}")

if blended_spearman > raw_spearman and blended_spearman > kernel_spearman:
    print(f"\n✅ BLENDED MODEL IS BEST! (+{blended_diff:.4f} vs raw, +{blended_spearman - kernel_spearman:.4f} vs kernel)")
    print("   The blending approach successfully combines raw and kernel embeddings!")
elif kernel_spearman > raw_spearman:
    print(f"\n✅ Kernel improves performance (+{kernel_diff:.4f})")
    if blended_spearman < kernel_spearman:
        print(f"   ⚠️  But blending reduces performance (try different alpha?)")
else:
    print(f"\n⚠️  Kernel doesn't improve on STS-B (might be overfitted to NLI data)")
print("======================================================================")
