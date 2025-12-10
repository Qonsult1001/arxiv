#!/usr/bin/env python3
"""
Step-by-step analysis: How many iterations until kernel matches/improves baseline?
The difference = what the kernel should learn (the added value)
"""

import torch
import sys
from pathlib import Path
from test_pretrained_comparison_fixed import (
    SimpleEmbeddingModel, 
    load_training_data,
    evaluate,
    load_stsb_test,
    train_real_data
)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

def analyze_kernel_learning_curve(num_steps=10):
    """Analyze how kernel iterations affect score relative to baseline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("📊 KERNEL LEARNING CURVE ANALYSIS")
    print("="*80)
    print("\nGoal: Find how many iterations until kernel matches/improves baseline")
    print("The difference = what the kernel should learn (added value)\n")
    
    # Load data
    pairs, labels = load_training_data(num_pairs=10000)
    test_s1, test_s2, test_scores = load_stsb_test()
    
    # Load teacher
    teacher_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    teacher_model.eval()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    kernel_path = Path(__file__).parent / "data" / "pretrained_semantic_kernel_ULTRAFAST.pt"
    
    # ========================================================================
    # STEP 1: ESTABLISH BASELINE (Normal distillation, NO kernel)
    # ========================================================================
    print("="*80)
    print("STEP 1: ESTABLISHING BASELINE (Teacher distillation, NO kernel)")
    print("="*80)
    print(f"Training for {num_steps} steps...\n")
    
    model_baseline = SimpleEmbeddingModel(d_model=384, num_layers=6, mode='teacher').to(device)
    train_real_data(model_baseline, pairs, labels, tokenizer, device, num_steps, 
                  mode='teacher', distill_weight=1.0, teacher_model=teacher_model)
    
    pearson_baseline, spearman_baseline = evaluate(model_baseline, test_s1, test_s2, test_scores, tokenizer, device)
    
    print(f"\n✅ BASELINE ESTABLISHED:")
    print(f"   Pearson:  {pearson_baseline:.4f}")
    print(f"   Spearman: {spearman_baseline:.4f} ⭐")
    print(f"   This is the target score (normal distillation)\n")
    
    # ========================================================================
    # STEP 2: TEST KERNEL WITH DIFFERENT ITERATIONS
    # ========================================================================
    print("="*80)
    print("STEP 2: TESTING KERNEL WITH DIFFERENT ITERATIONS")
    print("="*80)
    print("Finding: How many iterations to match/improve baseline?\n")
    
    iteration_counts = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
    results = []
    
    for iters in iteration_counts:
        print(f"Testing {iters} iteration(s)...")
        
        # Train model with kernel at this iteration count
        model_kernel = SimpleEmbeddingModel(
            d_model=384, num_layers=6, mode='distillation', 
            kernel_path=str(kernel_path), reasoning_iterations=iters
        ).to(device)
        
        # Load kernel
        kernel_state = torch.load(kernel_path, map_location=device, weights_only=False)
        if 'kernel' in kernel_state:
            model_kernel.kernel = kernel_state['kernel'].to(device)
        
        # Train with kernel
        train_real_data(model_kernel, pairs, labels, tokenizer, device, num_steps,
                      mode='distillation', distill_weight=1.0, kernel_distill_weight=0.7, 
                      teacher_model=teacher_model)
        
        # Evaluate
        pearson_kernel, spearman_kernel = evaluate(model_kernel, test_s1, test_s2, test_scores, tokenizer, device)
        
        # Calculate difference (what kernel adds)
        diff = spearman_kernel - spearman_baseline
        diff_pct = (diff / spearman_baseline * 100) if spearman_baseline > 0 else 0
        
        # Status
        if spearman_kernel >= spearman_baseline:
            status = "✅ MATCHES/IMPROVES"
        else:
            status = "❌ BELOW"
        
        results.append({
            'iterations': iters,
            'spearman': spearman_kernel,
            'difference': diff,
            'difference_pct': diff_pct,
            'status': status
        })
        
        print(f"   Spearman: {spearman_kernel:.4f} | Diff: {diff:+.4f} ({diff_pct:+.2f}%) | {status}\n")
    
    # ========================================================================
    # STEP 3: ANALYSIS
    # ========================================================================
    print("="*80)
    print("STEP 3: ANALYSIS - WHAT THE KERNEL ADDS")
    print("="*80)
    
    print(f"\n📊 BASELINE (Normal Distillation):")
    print(f"   Spearman: {spearman_baseline:.4f}")
    print(f"   This is what we get WITHOUT kernel\n")
    
    print(f"{'Iterations':<12} {'Kernel Score':<15} {'Difference':<15} {'% Change':<12} {'Status':<20}")
    print("-" * 80)
    
    # Find milestones
    first_match = None
    best_improvement = None
    best_result = None
    
    for r in results:
        marker = ""
        if first_match is None and r['spearman'] >= spearman_baseline:
            first_match = r['iterations']
            marker = " ⭐ FIRST MATCH"
        if best_improvement is None or r['difference'] > best_improvement['difference']:
            best_improvement = r
            best_result = r
        
        print(f"{r['iterations']:<12} {r['spearman']:<15.4f} {r['difference']:+.4f}{'':<7} "
              f"{r['difference_pct']:+.2f}%{'':<5} {r['status']}{marker}")
    
    print("\n" + "="*80)
    print("🎯 KEY FINDINGS")
    print("="*80)
    
    if first_match:
        print(f"\n✅ FIRST ITERATION TO MATCH BASELINE: {first_match}")
        print(f"   At {first_match} iterations, kernel score = baseline score")
        print(f"   This is the minimum needed for kernel to be useful\n")
    else:
        print("\n⚠️  Kernel never matches baseline in tested range\n")
    
    if best_result:
        print(f"🏆 BEST IMPROVEMENT: {best_result['iterations']} iterations")
        print(f"   Kernel Score: {best_result['spearman']:.4f}")
        print(f"   Added Value: {best_result['difference']:+.4f} ({best_result['difference_pct']:+.2f}%)")
        print(f"   This is what the kernel should learn: {best_result['difference']:+.4f} improvement\n")
    
    print("="*80)
    print("💡 INTERPRETATION")
    print("="*80)
    print(f"""
The kernel's job is to ADD value on top of normal distillation.

Baseline (no kernel):     {spearman_baseline:.4f}
Best with kernel:          {best_result['spearman']:.4f} ({best_result['iterations']} iterations)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kernel's added value:     {best_result['difference']:+.4f} ({best_result['difference_pct']:+.2f}%)

This {best_result['difference']:+.4f} improvement is what the kernel should learn.
It represents the semantic refinement the kernel provides beyond normal distillation.

Optimal iterations: {best_result['iterations']} (gives maximum added value)
""")
    
    return {
        'baseline': spearman_baseline,
        'first_match': first_match,
        'best': best_result,
        'all_results': results
    }

if __name__ == "__main__":
    results = analyze_kernel_learning_curve(num_steps=10)

