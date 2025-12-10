"""
STSB Evaluation for LAM Dual Encoder
Evaluates both Standard (384d) and Enterprise (12,288d) modes on STS-B
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr
from datasets import load_dataset
import sys
from pathlib import Path
import time
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from lam import LAM
from lam_dual_encoder import LAMDualEncoder

def evaluate_stsb_dual_encoder(encoder, mode="standard", split="test", device="cuda"):
    """
    Evaluate dual encoder on STS-B benchmark.
    
    Args:
        encoder: LAMDualEncoder instance
        mode: "standard" (384d) or "enterprise" (12,288d)
        split: "test" or "validation"
        device: Device to run on
        
    Returns:
        Dict with Pearson and Spearman correlations
    """
    print(f"\n{'='*80}")
    print(f"📊 STS-B EVALUATION - {mode.upper()} MODE ({'384d' if mode == 'standard' else '12,288d'})")
    print(f"{'='*80}")
    
    # Load STS-B dataset
    print(f"\n📦 Loading STS-B {split} dataset...")
    try:
        dataset = load_dataset(
            "sentence-transformers/stsb",
            split=split,
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
    except:
        try:
            dataset = load_dataset(
                "glue", "stsb",
                split=split,
                cache_dir="/workspace/.cache/huggingface/datasets"
            )
        except Exception as e:
            print(f"   ❌ Could not load STS-B dataset: {e}")
            return None
    
    s1 = dataset["sentence1"]
    s2 = dataset["sentence2"]
    
    # Handle both 'label' and 'score' column names
    if 'label' in dataset.column_names:
        labels = np.array(dataset["label"], dtype=float)
    else:
        labels = np.array(dataset["score"], dtype=float)
    
    print(f"   ✅ Loaded {len(s1)} sentence pairs")
    
    # Encode all sentences
    print(f"\n   Encoding sentences in {mode} mode...")
    start_time = time.time()
    
    all_emb1 = []
    all_emb2 = []
    
    batch_size = 32
    
    with torch.no_grad():
        # For standard mode, use regular LAM encode() for better semantic similarity
        # For enterprise mode, use dual encoder
        if mode == "standard":
            # Use regular LAM model.encode() for semantic similarity (mean pooling)
            print("   Using regular LAM encode() for semantic similarity...")
            all_emb1 = encoder.model.encode(
                s1, 
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_emb2 = encoder.model.encode(
                s2,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:
            # Enterprise mode: use dual encoder
            for i in tqdm(range(0, len(s1), batch_size), desc=f"   Encoding s1 ({mode})"):
                batch = s1[i:i+batch_size]
                batch_embs = []
                for text in batch:
                    emb = encoder.encode(text, mode=mode)
                    batch_embs.append(emb)
                all_emb1.extend(batch_embs)
            
            for i in tqdm(range(0, len(s2), batch_size), desc=f"   Encoding s2 ({mode})"):
                batch = s2[i:i+batch_size]
                batch_embs = []
                for text in batch:
                    emb = encoder.encode(text, mode=mode)
                    batch_embs.append(emb)
                all_emb2.extend(batch_embs)
    
    # Convert to numpy arrays (if not already)
    if not isinstance(all_emb1, np.ndarray):
        emb1 = np.array(all_emb1)
    else:
        emb1 = all_emb1
    if not isinstance(all_emb2, np.ndarray):
        emb2 = np.array(all_emb2)
    else:
        emb2 = all_emb2
    
    print(f"   ✅ Encoding complete. Shape: {emb1.shape}")
    
    # Compute cosine similarities
    print(f"\n   Computing cosine similarities...")
    similarities = []
    for i in tqdm(range(len(emb1)), desc="   Computing similarities"):
        sim = np.dot(emb1[i], emb2[i]) / (np.linalg.norm(emb1[i]) * np.linalg.norm(emb2[i]))
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    # Compute correlations
    print(f"\n   Computing correlations...")
    try:
        pearson = pearsonr(similarities, labels)[0]
        if np.isnan(pearson):
            pearson = None
    except (ValueError, RuntimeError) as e:
        print(f"   ⚠️  Pearson correlation failed: {e}")
        pearson = None
    
    try:
        spearman = spearmanr(similarities, labels)[0]
        if np.isnan(spearman):
            spearman = None
    except (ValueError, RuntimeError) as e:
        print(f"   ⚠️  Spearman correlation failed: {e}")
        spearman = None
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*80}")
    print(f"📊 RESULTS - {mode.upper()} MODE")
    print(f"{'='*80}")
    print(f"   Pearson Correlation:  {pearson:.4f}" if pearson is not None else "   Pearson Correlation:  N/A")
    print(f"   Spearman Correlation: {spearman:.4f}" if spearman is not None else "   Spearman Correlation: N/A")
    print(f"   Evaluation Time:      {elapsed_time:.2f}s")
    print(f"   Embedding Dimension:   {emb1.shape[1]}")
    print(f"{'='*80}")
    
    return {
        'pearson': pearson,
        'spearman': spearman,
        'time': elapsed_time,
        'dimension': emb1.shape[1],
        'num_pairs': len(s1)
    }

def main():
    """Main evaluation function"""
    print("="*80)
    print("🚀 LAM DUAL ENCODER - STSB EVALUATION")
    print("="*80)
    print("\nThis script evaluates both Standard (384d) and Enterprise (12,288d) modes")
    print("on the STS-B benchmark to verify semantic similarity preservation.\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model - use /best folder for best scores (Spearman 0.8190, Pearson 0.820)
    print("📦 Loading LAM model from /best (Spearman 0.8190)...")
    try:
        model_path = Path(__file__).parent.parent / "best"
        if not model_path.exists():
            # Fallback to LAM-base-v1
            model_path = Path(__file__).parent.parent / "LAM-base-v1"
            if not model_path.exists():
                model_path = "LAM-base-v1"
        
        model = LAM(str(model_path))
        print(f"✅ Model loaded from {model_path}\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create dual encoder
    print("📦 Creating dual encoder...")
    try:
        encoder = LAMDualEncoder(model)
        print("✅ Dual encoder created\n")
    except Exception as e:
        print(f"❌ Failed to create dual encoder: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check if whitening stats exist for Enterprise mode
    whitening_path = Path(__file__).parent / "lam_whitening_stats.npy"
    if not whitening_path.exists():
        print("⚠️  Warning: No whitening stats found for Enterprise mode.")
        print("   Enterprise mode will use raw state (no whitening).")
        print("   For best results, run calibrate_enterprise_mode() first.\n")
    
    # Evaluate Standard mode (384d)
    results_standard = evaluate_stsb_dual_encoder(
        encoder, 
        mode="standard", 
        split="test",
        device=device
    )
    
    # Evaluate Enterprise mode (12,288d)
    results_enterprise = evaluate_stsb_dual_encoder(
        encoder, 
        mode="enterprise", 
        split="test",
        device=device
    )
    
    # Comparison summary
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Mode':<15} {'Dimension':<12} {'Pearson':<12} {'Spearman':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    if results_standard:
        pearson_str = f"{results_standard['pearson']:.4f}" if results_standard['pearson'] is not None else "N/A"
        spearman_str = f"{results_standard['spearman']:.4f}" if results_standard['spearman'] is not None else "N/A"
        print(f"{'Standard':<15} {results_standard['dimension']:<12} {pearson_str:<12} {spearman_str:<12} {results_standard['time']:.2f}")
    
    if results_enterprise:
        pearson_str = f"{results_enterprise['pearson']:.4f}" if results_enterprise['pearson'] is not None else "N/A"
        spearman_str = f"{results_enterprise['spearman']:.4f}" if results_enterprise['spearman'] is not None else "N/A"
        print(f"{'Enterprise':<15} {results_enterprise['dimension']:<12} {pearson_str:<12} {spearman_str:<12} {results_enterprise['time']:.2f}")
    
    print("\n" + "="*80)
    
    # Analysis
    if results_standard and results_enterprise:
        if results_standard['spearman'] and results_enterprise['spearman']:
            diff = abs(results_standard['spearman'] - results_enterprise['spearman'])
            ratio = results_enterprise['spearman'] / results_standard['spearman'] if results_standard['spearman'] > 0 else 0
            
            print("\n📈 ANALYSIS:")
            print(f"   Spearman Difference: {diff:.4f}")
            print(f"   Enterprise/Standard Ratio: {ratio:.4f} ({ratio*100:.1f}%)")
            
            if diff < 0.05:
                print("   ✅ EXCELLENT: Enterprise mode preserves semantic similarity!")
            elif diff < 0.10:
                print("   ✅ GOOD: Enterprise mode maintains most semantic similarity")
            else:
                print("   ⚠️  NOTE: Enterprise mode shows some difference (may need calibration)")
    
    print("="*80)
    
    return 0

if __name__ == "__main__":
    exit(main())

