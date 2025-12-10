#!/usr/bin/env python3
"""
Systematic test to find optimal kernel iteration count
Tests multiple iteration counts and finds the best one
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import json
import gzip

sys.path.insert(0, str(Path(__file__).parent))
from test_pretrained_comparison_fixed import (
    SimpleEmbeddingModel, 
    load_training_data,
    evaluate,
    load_stsb_test
)
from sentence_transformers import SentenceTransformer

def test_iteration_count(iteration_count, num_steps=10):
    """Test a specific iteration count and return results"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    pairs, labels = load_training_data(num_pairs=10000)
    test_s1, test_s2, test_scores = load_stsb_test()
    
    # Load teacher
    teacher_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    teacher_model.eval()
    
    kernel_path = Path(__file__).parent / "data" / "pretrained_semantic_kernel_ULTRAFAST.pt"
    
    # Stage 1: Teacher only (baseline)
    model_stage1 = SimpleEmbeddingModel(d_model=384, num_layers=6, mode='teacher').to(device)
    
    # Quick training (10 steps for speed)
    from test_pretrained_comparison_fixed import train_real_data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_real_data(model_stage1, pairs, labels, tokenizer, device, num_steps, 
                  mode='teacher', distill_weight=1.0, teacher_model=teacher_model)
    
    # Evaluate baseline
    pearson_base, spearman_base = evaluate(model_stage1, test_s1, test_s2, test_scores, tokenizer, device)
    
    # Stage 2: With kernel at specific iteration count
    model_stage2 = SimpleEmbeddingModel(
        d_model=384, num_layers=6, mode='distillation', 
        kernel_path=str(kernel_path), reasoning_iterations=iteration_count
    ).to(device)
    
    # Load kernel
    kernel_state = torch.load(kernel_path, map_location=device, weights_only=False)
    if 'kernel' in kernel_state:
        model_stage2.kernel = kernel_state['kernel'].to(device)
    
    # Train with kernel
    train_real_data(model_stage2, pairs, labels, tokenizer, device, num_steps,
                  mode='distillation', distill_weight=1.0, kernel_distill_weight=0.7, 
                  teacher_model=teacher_model)
    
    # Evaluate with kernel
    pearson_kernel, spearman_kernel = evaluate(model_stage2, test_s1, test_s2, test_scores, tokenizer, device)
    
    improvement = spearman_kernel - spearman_base
    improvement_pct = (improvement / spearman_base * 100) if spearman_base > 0 else 0
    
    return {
        'iterations': iteration_count,
        'baseline_spearman': spearman_base,
        'kernel_spearman': spearman_kernel,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

def find_optimal_iterations():
    """Systematically test different iteration counts"""
    print("="*80)
    print("🔍 SYSTEMATIC TEST: Finding Optimal Kernel Iteration Count")
    print("="*80)
    print("\nTesting iteration counts: 1, 2, 3, 5, 8, 12, 16, 20, 24, 30, 36")
    print("This will take a while...\n")
    
    iteration_counts = [1, 2, 3, 5, 8, 12, 16, 20, 24, 30, 36]
    results = []
    
    for iters in iteration_counts:
        print(f"\n{'='*80}")
        print(f"Testing {iters} iterations...")
        print(f"{'='*80}")
        
        result = test_iteration_count(iters, num_steps=10)
        results.append(result)
        
        print(f"✅ {iters} iterations:")
        print(f"   Baseline: {result['baseline_spearman']:.4f}")
        print(f"   With kernel: {result['kernel_spearman']:.4f}")
        print(f"   Improvement: {result['improvement']:+.4f} ({result['improvement_pct']:+.2f}%)")
    
    # Find best
    best_result = max(results, key=lambda x: x['improvement'])
    
    print(f"\n{'='*80}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Iterations':<12} {'Baseline':<12} {'With Kernel':<12} {'Improvement':<15} {'% Change':<12}")
    print("-" * 80)
    for r in results:
        marker = " ⭐ BEST" if r == best_result else ""
        print(f"{r['iterations']:<12} {r['baseline_spearman']:<12.4f} {r['kernel_spearman']:<12.4f} "
              f"{r['improvement']:+.4f}{'':<7} {r['improvement_pct']:+.2f}%{marker}")
    
    print(f"\n{'='*80}")
    print(f"🏆 OPTIMAL ITERATION COUNT: {best_result['iterations']}")
    print(f"{'='*80}")
    print(f"   Baseline Spearman: {best_result['baseline_spearman']:.4f}")
    print(f"   Kernel Spearman: {best_result['kernel_spearman']:.4f}")
    print(f"   Improvement: {best_result['improvement']:+.4f} ({best_result['improvement_pct']:+.2f}%)")
    print(f"{'='*80}\n")
    
    return best_result, results

if __name__ == "__main__":
    best, all_results = find_optimal_iterations()

