"""
Quick comparison of validation vs test results from saved JSON files
"""
import json
from pathlib import Path

def load_results(split):
    """Load saved test results"""
    results_file = Path(__file__).parent / "results" / f"pearson_score_validation_{split}.json"
    if not results_file.exists():
        print(f"⚠️  {split} results not found at {results_file}")
        return None
    with open(results_file, 'r') as f:
        return json.load(f)

def main():
    print("="*70)
    print("VALIDATION vs TEST RESULTS COMPARISON")
    print("="*70)

    # Load both results
    val_results = load_results("validation")
    test_results = load_results("test")

    if not val_results or not test_results:
        print("\n⚠️  Missing results files. Run both tests first:")
        print("  python test_pearson_score.py --model ../LAM-base-v1 --split validation")
        print("  python test_pearson_score.py --model ../LAM-base-v1 --split test")
        return

    # Extract key metrics
    val_lam = val_results['lam_results']['pearson']
    val_baseline = val_results['baseline_results']['pearson']
    val_samples = val_results['dataset']['num_pairs']

    test_lam = test_results['lam_results']['pearson']
    test_baseline = test_results['baseline_results']['pearson']
    test_samples = test_results['dataset']['num_pairs']

    # Calculate drops
    lam_drop = test_lam - val_lam
    baseline_drop = test_baseline - val_baseline
    lam_drop_pct = (lam_drop / val_lam) * 100
    baseline_drop_pct = (baseline_drop / val_baseline) * 100

    # Print comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)

    print(f"\n{'Metric':<30} {'Validation':<15} {'Test':<15} {'Drop':>12}")
    print("-" * 70)
    print(f"{'LAM Pearson':<30} {val_lam:<15.4f} {test_lam:<15.4f} {lam_drop:>12.4f} ({lam_drop_pct:>+6.2f}%)")
    print(f"{'Baseline Pearson':<30} {val_baseline:<15.4f} {test_baseline:<15.4f} {baseline_drop:>12.4f} ({baseline_drop_pct:>+6.2f}%)")

    print(f"\n{'Dataset Size':<30} {val_samples:<15,} {test_samples:<15,}")

    # Error analysis comparison
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)

    val_mae = val_results['lam_results']['error_analysis']['mean_absolute_error']
    val_rmse = val_results['lam_results']['error_analysis']['root_mean_squared_error']
    test_mae = test_results['lam_results']['error_analysis']['mean_absolute_error']
    test_rmse = test_results['lam_results']['error_analysis']['root_mean_squared_error']

    print(f"\n{'Metric':<30} {'Validation':<15} {'Test':<15} {'Change':>12}")
    print("-" * 70)
    print(f"{'LAM MAE':<30} {val_mae:<15.4f} {test_mae:<15.4f} {(test_mae - val_mae):>12.4f}")
    print(f"{'LAM RMSE':<30} {val_rmse:<15.4f} {test_rmse:<15.4f} {(test_rmse - val_rmse):>12.4f}")

    # Per-range analysis
    if 'range_analysis' in val_results['lam_results'] and 'range_analysis' in test_results['lam_results']:
        print("\n" + "="*70)
        print("PER-RANGE PEARSON SCORES")
        print("="*70)

        val_ranges = val_results['lam_results']['range_analysis']
        test_ranges = test_results['lam_results']['range_analysis']

        print(f"\n{'Range':<25} {'Val Pearson':<15} {'Test Pearson':<15} {'Drop':>12}")
        print("-" * 70)

        for range_name in val_ranges.keys():
            val_score = val_ranges[range_name]['pearson']
            test_score = test_ranges.get(range_name, {}).get('pearson', 0.0)
            drop = test_score - val_score
            print(f"{range_name:<25} {val_score:<15.4f} {test_score:<15.4f} {drop:>12.4f}")

    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    if abs(lam_drop) > 0.05:
        print("\n🔴 SIGNIFICANT OVERFITTING DETECTED (>5 point drop)")
        print(f"\n   LAM dropped {abs(lam_drop):.4f} points ({lam_drop_pct:+.1f}%)")
        print(f"   Baseline dropped {abs(baseline_drop):.4f} points ({baseline_drop_pct:+.1f}%)")
        print(f"   LAM's drop is {abs(lam_drop - baseline_drop):.4f} points worse than baseline")

        print("\n   Root Causes:")
        print("   • Model overfit to validation set patterns during checkpoint selection")
        print("   • Test set has different distribution or harder examples")
        print("   • LAM may be memorizing instead of learning robust features")

    elif abs(lam_drop) > 0.02:
        print("\n🟡 MODERATE OVERFITTING (2-5 point drop)")
        print(f"\n   LAM dropped {abs(lam_drop):.4f} points ({lam_drop_pct:+.1f}%)")
        print(f"   Baseline dropped {abs(baseline_drop):.4f} points ({baseline_drop_pct:+.1f}%)")
        print(f"   Difference: {abs(lam_drop - baseline_drop):.4f} points")

    else:
        print("\n🟢 GOOD GENERALIZATION (<2 point drop)")
        print(f"\n   LAM dropped {abs(lam_drop):.4f} points - within normal variation")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if abs(lam_drop) > 0.05:
        print("\n1. DO NOT train on test set - that would be scientific fraud")
        print("\n2. Instead, improve training with MORE and DIVERSE data:")
        print("   • Add NLI dataset (SNLI, MultiNLI)")
        print("   • Add other STS datasets (STS-2012 through STS-2017)")
        print("   • Add paraphrase datasets (PAWS, QQP)")
        print("   • Use data augmentation (back-translation, paraphrasing)")

        print("\n3. Add regularization:")
        print("   • Increase dropout (try 0.1, 0.2)")
        print("   • Add weight decay")
        print("   • Use gradient clipping")
        print("   • Try early stopping")

        print("\n4. Training improvements:")
        print("   • Use multiple validation sets (cross-validation)")
        print("   • Don't select checkpoint based solely on validation performance")
        print("   • Monitor generalization gap during training")

        print("\n5. Architecture improvements:")
        print("   • Add layer normalization")
        print("   • Experiment with different pooling strategies")
        print("   • Try ensemble methods")

    elif abs(lam_drop) > 0.02:
        print("\n1. Train on additional datasets (NLI, more STS)")
        print("2. Increase regularization slightly")
        print("3. Consider ensemble methods")

    else:
        print("\nYour model generalizes well! The drop is within normal variation.")

    # Publication guidance
    print("\n" + "="*70)
    print("FOR PUBLICATION")
    print("="*70)

    print("\nReport BOTH scores in your paper:")
    print(f"  • Validation: {val_lam:.4f} (used for checkpoint selection)")
    print(f"  • Test: {test_lam:.4f} (official held-out benchmark)")

    if abs(lam_drop) > 0.02:
        print("\nBe transparent about the gap:")
        print("  • Acknowledge the performance drop")
        print("  • Explain it shows honest evaluation (not cherry-picked results)")
        print("  • Compare to baseline's drop to provide context")
        print("  • Discuss what you learned and future improvements")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
