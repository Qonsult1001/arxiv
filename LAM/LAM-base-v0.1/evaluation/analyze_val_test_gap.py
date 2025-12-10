"""
Analyze Validation vs Test Set Performance Gap

This script identifies what's different about test set pairs that LAM struggles with.
"""
import json
from pathlib import Path
import sys

# Install dependencies if needed
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datasets import load_dataset
    from scipy.stats import pearsonr
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "numpy", "matplotlib", "seaborn", "datasets", "scipy", "scikit-learn"])
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datasets import load_dataset
    from scipy.stats import pearsonr

# Import LAM encoder
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "production"))
from lam_wrapper import LAMEncoder
from sentence_transformers import SentenceTransformer

def load_results(split):
    """Load saved test results"""
    results_file = Path(__file__).parent / "results" / f"pearson_score_validation_{split}.json"
    if not results_file.exists():
        return None
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_sentence_characteristics(sentences1, sentences2, labels, predictions, split_name):
    """Analyze characteristics of sentences and their errors"""

    errors = np.abs(predictions - labels)

    # Calculate sentence lengths
    lengths1 = [len(s.split()) for s in sentences1]
    lengths2 = [len(s.split()) for s in sentences2]
    avg_lengths = [(l1 + l2) / 2 for l1, l2 in zip(lengths1, lengths2)]

    # Calculate length difference
    length_diffs = [abs(l1 - l2) for l1, l2 in zip(lengths1, lengths2)]

    # Analyze by similarity range
    similarity_ranges = {
        'Very Dissimilar (0-1)': (0, 1),
        'Dissimilar (1-2)': (1, 2),
        'Somewhat Similar (2-3)': (2, 3),
        'Similar (3-4)': (3, 4),
        'Very Similar (4-5)': (4, 5)
    }

    range_stats = {}
    for range_name, (low, high) in similarity_ranges.items():
        mask = (labels >= low) & (labels < high)
        if np.sum(mask) > 0:
            range_stats[range_name] = {
                'count': int(np.sum(mask)),
                'mean_error': float(np.mean(errors[mask])),
                'mean_prediction': float(np.mean(predictions[mask])),
                'mean_label': float(np.mean(labels[mask])),
                'mean_length': float(np.mean([avg_lengths[i] for i in range(len(mask)) if mask[i]])),
            }

    # Analyze by sentence length bins
    length_bins = {
        'Short (0-10 words)': (0, 10),
        'Medium (10-20 words)': (10, 20),
        'Long (20-30 words)': (20, 30),
        'Very Long (30+ words)': (30, 1000)
    }

    length_stats = {}
    for bin_name, (low, high) in length_bins.items():
        mask = np.array([(l >= low and l < high) for l in avg_lengths])
        if np.sum(mask) > 0:
            length_stats[bin_name] = {
                'count': int(np.sum(mask)),
                'mean_error': float(np.mean(errors[mask])),
                'mean_prediction': float(np.mean(predictions[mask])),
                'mean_label': float(np.mean(labels[mask])),
            }

    # Analyze by length difference
    length_diff_bins = {
        'Similar Length (0-2)': (0, 2),
        'Moderate Diff (2-5)': (2, 5),
        'Large Diff (5-10)': (5, 10),
        'Very Large Diff (10+)': (10, 1000)
    }

    length_diff_stats = {}
    for bin_name, (low, high) in length_diff_bins.items():
        mask = np.array([(d >= low and d < high) for d in length_diffs])
        if np.sum(mask) > 0:
            length_diff_stats[bin_name] = {
                'count': int(np.sum(mask)),
                'mean_error': float(np.mean(errors[mask])),
                'mean_prediction': float(np.mean(predictions[mask])),
                'mean_label': float(np.mean(labels[mask])),
            }

    return {
        'split': split_name,
        'total_samples': len(labels),
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'range_stats': range_stats,
        'length_stats': length_stats,
        'length_diff_stats': length_diff_stats,
        'avg_sentence_length': float(np.mean(avg_lengths)),
        'avg_length_diff': float(np.mean(length_diffs)),
    }

def get_predictions(model, sentences1, sentences2):
    """Get model predictions"""
    print(f"  Computing embeddings for {len(sentences1)} pairs...")

    # Encode sentences
    embeddings1 = model.encode(sentences1, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    embeddings2 = model.encode(sentences2, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    # Compute cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    predictions = []
    for emb1, emb2 in zip(embeddings1, embeddings2):
        sim = cosine_similarity([emb1], [emb2])[0][0]
        predictions.append(sim * 5.0)  # Scale to [0, 5]

    return np.array(predictions)

def find_worst_examples(sentences1, sentences2, labels, predictions, n=10):
    """Find examples with largest errors"""
    errors = np.abs(predictions - labels)
    worst_indices = np.argsort(errors)[-n:][::-1]

    examples = []
    for idx in worst_indices:
        examples.append({
            'sentence1': sentences1[idx],
            'sentence2': sentences2[idx],
            'true_label': float(labels[idx]),
            'prediction': float(predictions[idx]),
            'error': float(errors[idx]),
        })

    return examples

def visualize_comparison(val_analysis, test_analysis, output_path):
    """Create comparison visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Validation vs Test Set Analysis - LAM Performance Gap', fontsize=16, fontweight='bold')

    # 1. Error by similarity range
    ax = axes[0, 0]
    ranges = list(val_analysis['range_stats'].keys())
    val_errors = [val_analysis['range_stats'][r]['mean_error'] for r in ranges]
    test_errors = [test_analysis['range_stats'].get(r, {'mean_error': 0})['mean_error'] for r in ranges]

    x = np.arange(len(ranges))
    width = 0.35
    ax.bar(x - width/2, val_errors, width, label='Validation', color='#2ecc71')
    ax.bar(x + width/2, test_errors, width, label='Test', color='#e74c3c')
    ax.set_xlabel('Similarity Range')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error by Similarity Range')
    ax.set_xticks(x)
    ax.set_xticklabels([r.split('(')[0].strip() for r in ranges], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Error by sentence length
    ax = axes[0, 1]
    length_bins = list(val_analysis['length_stats'].keys())
    val_errors = [val_analysis['length_stats'][b]['mean_error'] for b in length_bins]
    test_errors = [test_analysis['length_stats'].get(b, {'mean_error': 0})['mean_error'] for b in length_bins]

    x = np.arange(len(length_bins))
    ax.bar(x - width/2, val_errors, width, label='Validation', color='#2ecc71')
    ax.bar(x + width/2, test_errors, width, label='Test', color='#e74c3c')
    ax.set_xlabel('Average Sentence Length')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error by Sentence Length')
    ax.set_xticks(x)
    ax.set_xticklabels([b.split('(')[0].strip() for b in length_bins], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Error by length difference
    ax = axes[0, 2]
    diff_bins = list(val_analysis['length_diff_stats'].keys())
    val_errors = [val_analysis['length_diff_stats'][b]['mean_error'] for b in diff_bins]
    test_errors = [test_analysis['length_diff_stats'].get(b, {'mean_error': 0})['mean_error'] for b in diff_bins]

    x = np.arange(len(diff_bins))
    ax.bar(x - width/2, val_errors, width, label='Validation', color='#2ecc71')
    ax.bar(x + width/2, test_errors, width, label='Test', color='#e74c3c')
    ax.set_xlabel('Length Difference Between Sentences')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error by Length Difference')
    ax.set_xticks(x)
    ax.set_xticklabels([b.split('(')[0].strip() for b in diff_bins], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Sample distribution by similarity
    ax = axes[1, 0]
    val_counts = [val_analysis['range_stats'][r]['count'] for r in ranges]
    test_counts = [test_analysis['range_stats'].get(r, {'count': 0})['count'] for r in ranges]

    x = np.arange(len(ranges))
    ax.bar(x - width/2, val_counts, width, label='Validation', color='#2ecc71')
    ax.bar(x + width/2, test_counts, width, label='Test', color='#e74c3c')
    ax.set_xlabel('Similarity Range')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Distribution by Similarity')
    ax.set_xticks(x)
    ax.set_xticklabels([r.split('(')[0].strip() for r in ranges], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Overall statistics comparison
    ax = axes[1, 1]
    metrics = ['Mean Error', 'Median Error', 'Avg Length', 'Avg Length Diff']
    val_values = [
        val_analysis['mean_error'],
        val_analysis['median_error'],
        val_analysis['avg_sentence_length'] / 10,  # Scale for visibility
        val_analysis['avg_length_diff']
    ]
    test_values = [
        test_analysis['mean_error'],
        test_analysis['median_error'],
        test_analysis['avg_sentence_length'] / 10,
        test_analysis['avg_length_diff']
    ]

    x = np.arange(len(metrics))
    ax.bar(x - width/2, val_values, width, label='Validation', color='#2ecc71')
    ax.bar(x + width/2, test_values, width, label='Test', color='#e74c3c')
    ax.set_ylabel('Value')
    ax.set_title('Overall Statistics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Performance summary
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
PERFORMANCE GAP ANALYSIS
========================

Validation Set:
  Samples: {val_analysis['total_samples']}
  Mean Error: {val_analysis['mean_error']:.4f}
  Median Error: {val_analysis['median_error']:.4f}
  Avg Length: {val_analysis['avg_sentence_length']:.1f} words

Test Set:
  Samples: {test_analysis['total_samples']}
  Mean Error: {test_analysis['mean_error']:.4f}
  Median Error: {test_analysis['median_error']:.4f}
  Avg Length: {test_analysis['avg_sentence_length']:.1f} words

Performance Drop:
  Error Increase: {test_analysis['mean_error'] - val_analysis['mean_error']:.4f}
  Relative Change: {((test_analysis['mean_error'] / val_analysis['mean_error']) - 1) * 100:.1f}%

Key Findings:
  • Test set has {'MORE' if test_analysis['avg_sentence_length'] > val_analysis['avg_sentence_length'] else 'FEWER'} words on average
  • Error is {'HIGHER' if test_analysis['mean_error'] > val_analysis['mean_error'] else 'LOWER'} on test set
  • Suggests {'overfitting' if test_analysis['mean_error'] > val_analysis['mean_error'] else 'good generalization'}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

def main():
    print("="*70)
    print("VALIDATION vs TEST SET GAP ANALYSIS")
    print("="*70)

    # Load model
    model_path = Path(__file__).parent.parent
    print(f"\nLoading LAM model from: {model_path}")
    model = LAMEncoder(str(model_path))

    # Load datasets
    print("\n" + "="*70)
    print("LOADING VALIDATION SET")
    print("="*70)
    val_dataset = load_dataset("sentence-transformers/stsb", split="validation")
    val_sentences1 = val_dataset['sentence1']
    val_sentences2 = val_dataset['sentence2']
    val_labels = np.array(val_dataset['score'])

    print("\n" + "="*70)
    print("LOADING TEST SET")
    print("="*70)
    test_dataset = load_dataset("sentence-transformers/stsb", split="test")
    test_sentences1 = test_dataset['sentence1']
    test_sentences2 = test_dataset['sentence2']
    test_labels = np.array(test_dataset['score'])

    # Get predictions
    print("\n" + "="*70)
    print("COMPUTING VALIDATION PREDICTIONS")
    print("="*70)
    val_predictions = get_predictions(model, val_sentences1, val_sentences2)
    val_pearson, _ = pearsonr(val_predictions, val_labels)
    print(f"✓ Validation Pearson: {val_pearson:.4f}")

    print("\n" + "="*70)
    print("COMPUTING TEST PREDICTIONS")
    print("="*70)
    test_predictions = get_predictions(model, test_sentences1, test_sentences2)
    test_pearson, _ = pearsonr(test_predictions, test_labels)
    print(f"✓ Test Pearson: {test_pearson:.4f}")

    # Analyze characteristics
    print("\n" + "="*70)
    print("ANALYZING VALIDATION SET CHARACTERISTICS")
    print("="*70)
    val_analysis = analyze_sentence_characteristics(
        val_sentences1, val_sentences2, val_labels, val_predictions, 'validation'
    )

    print("\n" + "="*70)
    print("ANALYZING TEST SET CHARACTERISTICS")
    print("="*70)
    test_analysis = analyze_sentence_characteristics(
        test_sentences1, test_sentences2, test_labels, test_predictions, 'test'
    )

    # Find worst examples from test set
    print("\n" + "="*70)
    print("TOP 10 WORST TEST SET EXAMPLES")
    print("="*70)
    worst_examples = find_worst_examples(
        test_sentences1, test_sentences2, test_labels, test_predictions, n=10
    )

    for i, ex in enumerate(worst_examples, 1):
        print(f"\n{i}. Error: {ex['error']:.4f}")
        print(f"   True: {ex['true_label']:.2f} | Predicted: {ex['prediction']:.2f}")
        print(f"   S1: {ex['sentence1'][:80]}...")
        print(f"   S2: {ex['sentence2'][:80]}...")

    # Save analysis
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    analysis_results = {
        'validation': val_analysis,
        'test': test_analysis,
        'performance_gap': {
            'pearson_drop': float(test_pearson - val_pearson),
            'error_increase': float(test_analysis['mean_error'] - val_analysis['mean_error']),
            'validation_pearson': float(val_pearson),
            'test_pearson': float(test_pearson),
        },
        'worst_test_examples': worst_examples
    }

    analysis_path = results_dir / "val_test_gap_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n✅ Analysis saved to: {analysis_path}")

    # Create visualization
    viz_dir = Path(__file__).parent / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    viz_path = viz_dir / "val_test_gap_analysis.png"
    visualize_comparison(val_analysis, test_analysis, viz_path)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nValidation Set Performance:")
    print(f"  Pearson: {val_pearson:.4f}")
    print(f"  Mean Error: {val_analysis['mean_error']:.4f}")
    print(f"  Avg Sentence Length: {val_analysis['avg_sentence_length']:.1f} words")

    print(f"\nTest Set Performance:")
    print(f"  Pearson: {test_pearson:.4f}")
    print(f"  Mean Error: {test_analysis['mean_error']:.4f}")
    print(f"  Avg Sentence Length: {test_analysis['avg_sentence_length']:.1f} words")

    print(f"\nPerformance Gap:")
    print(f"  Pearson Drop: {test_pearson - val_pearson:.4f} ({((test_pearson / val_pearson) - 1) * 100:.1f}%)")
    print(f"  Error Increase: {test_analysis['mean_error'] - val_analysis['mean_error']:.4f}")

    print(f"\n{'⚠️  OVERFITTING DETECTED' if test_pearson < val_pearson - 0.02 else '✓ Reasonable generalization'}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if test_pearson < val_pearson - 0.05:
        print("\n🔴 SIGNIFICANT OVERFITTING (>5 point drop)")
        print("\nRecommendations:")
        print("  1. Train on larger, more diverse dataset (NLI, SNLI, multi-genre)")
        print("  2. Add regularization (dropout, weight decay)")
        print("  3. Use data augmentation (paraphrasing, back-translation)")
        print("  4. Reduce model capacity or training epochs")
        print("  5. Use cross-validation across multiple STS datasets")
    elif test_pearson < val_pearson - 0.02:
        print("\n🟡 MODERATE OVERFITTING (2-5 point drop)")
        print("\nRecommendations:")
        print("  1. Add more diverse training data")
        print("  2. Increase regularization slightly")
        print("  3. Consider early stopping based on test set performance")
    else:
        print("\n🟢 GOOD GENERALIZATION (<2 point drop)")
        print("\nThis is normal variation between splits.")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
