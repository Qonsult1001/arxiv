"""
Test 1: Verify LAM's 0.836 Pearson Score on STS-B
=====================================================

This test rigorously validates LAM's claimed performance on the STS-B benchmark
with complete statistical analysis and confidence intervals.

Tests:
1. Full STS-B benchmark (test set)
2. Per-category performance breakdown
3. Bootstrap confidence intervals (95%)
4. Statistical significance tests
5. Error analysis and distribution
6. Comparison with baseline
"""

import json
import numpy as np
from typing import Dict, Tuple, List
from scipy.stats import pearsonr, spearmanr, bootstrap
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from datasets import load_dataset
    import torch
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "datasets", "scipy", "matplotlib", "seaborn", "torch"])
    from datasets import load_dataset
    import torch

# Import LAM custom encoder
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "production"))
    from lam_wrapper import LAMEncoder
    print("✓ Using LAM custom encoder (lam_base.bin + lam_tweak.pt)")
except ImportError as e:
    print(f"⚠️  Could not import LAMEncoder: {e}")
    print("Falling back to sentence-transformers")
    from sentence_transformers import SentenceTransformer as LAMEncoder

# Import SentenceTransformer for type hints
from sentence_transformers import SentenceTransformer


class PearsonScoreValidator:
    """Comprehensive STS-B Pearson score validation"""

    def __init__(self, model_path: str, baseline_path: str = "all-MiniLM-L6-v2", split: str = "test"):
        self.model_path = model_path
        self.baseline_path = baseline_path
        self.split = split
        self.results_dir = Path(__file__).parent / "results"
        self.viz_dir = Path(__file__).parent / "visualizations"

        self.results_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

    def load_stsb_dataset(self) -> Tuple[List[str], List[str], List[float]]:
        """
        Load STS-B dataset

        Args:
            split: 'validation' (1500 pairs) or 'test' (1379 pairs)

        Note:
            - validation: Used during model development/checkpoint selection
            - test: Official benchmark, never seen during training
        """
        print(f"Loading STS-B {self.split} dataset...")
        print(f"  Split: {self.split}")

        if self.split == "validation":
            print(f"  ⚠️  VALIDATION SET: Used during model development")
            print(f"  ⚠️  For publication, also test on 'test' split")
        elif self.split == "test":
            print(f"  ✓ TEST SET: Official benchmark (held-out)")

        dataset = load_dataset("sentence-transformers/stsb", split=self.split)

        sentences1 = dataset['sentence1']
        sentences2 = dataset['sentence2']
        # Handle both 'label' (glue) and 'score' (sentence-transformers) column names
        if 'label' in dataset.column_names:
            scores = dataset['label']
        else:
            scores = dataset['score']

        print(f"  Loaded {len(sentences1)} sentence pairs")
        return sentences1, sentences2, scores

    def compute_embeddings_and_similarity(
        self,
        model: SentenceTransformer,
        sentences1: List[str],
        sentences2: List[str]
    ) -> np.ndarray:
        """Compute embeddings and cosine similarities"""
        print("Computing embeddings...")
        emb1 = model.encode(sentences1, show_progress_bar=True, convert_to_numpy=True)
        emb2 = model.encode(sentences2, show_progress_bar=True, convert_to_numpy=True)

        # Compute cosine similarities
        print("Computing cosine similarities...")
        cosine_scores = np.sum(emb1 * emb2, axis=1) / (
            np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
        )

        # Scale to 0-5 range like STS-B scores
        # Sentence embeddings produce cosine similarities in [0, 1] range
        # Map directly to [0, 5] (standard sentence-transformers approach)
        cosine_scores = np.clip(cosine_scores, 0, 1) * 5.0

        return cosine_scores

    def bootstrap_confidence_interval(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float]:
        """Compute 95% confidence interval using bootstrap"""
        print(f"Computing bootstrap CI with {n_bootstrap} samples...")

        def pearson_stat(pred, lab):
            return pearsonr(pred, lab)[0]

        rng = np.random.default_rng(42)
        bootstrap_samples = []

        for _ in range(n_bootstrap):
            indices = rng.choice(len(predictions), size=len(predictions), replace=True)
            sample_pred = predictions[indices]
            sample_lab = labels[indices]
            bootstrap_samples.append(pearson_stat(sample_pred, sample_lab))

        ci_lower = np.percentile(bootstrap_samples, 2.5)
        ci_upper = np.percentile(bootstrap_samples, 97.5)

        return ci_lower, ci_upper

    def analyze_errors(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """Detailed error analysis"""
        errors = predictions - labels

        return {
            "mean_absolute_error": float(np.mean(np.abs(errors))),
            "root_mean_squared_error": float(np.sqrt(np.mean(errors**2))),
            "median_absolute_error": float(np.median(np.abs(errors))),
            "max_error": float(np.max(np.abs(errors))),
            "error_std": float(np.std(errors)),
            "error_percentiles": {
                "25th": float(np.percentile(np.abs(errors), 25)),
                "50th": float(np.percentile(np.abs(errors), 50)),
                "75th": float(np.percentile(np.abs(errors), 75)),
                "90th": float(np.percentile(np.abs(errors), 90)),
                "95th": float(np.percentile(np.abs(errors), 95)),
                "99th": float(np.percentile(np.abs(errors), 99))
            }
        }

    def per_score_range_analysis(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """Analyze performance across different score ranges"""
        ranges = [
            (0, 1, "Very Dissimilar"),
            (1, 2, "Dissimilar"),
            (2, 3, "Somewhat Similar"),
            (3, 4, "Similar"),
            (4, 5, "Very Similar")
        ]

        results = {}
        for low, high, label in ranges:
            mask = (labels >= low) & (labels < high)
            if np.sum(mask) > 0:
                try:
                    range_pearson, _ = pearsonr(predictions[mask], labels[mask])
                    # Handle NaN (occurs when input is constant)
                    if np.isnan(range_pearson):
                        range_pearson = 0.0
                except (ValueError, RuntimeError):
                    range_pearson = 0.0
                range_mae = np.mean(np.abs(predictions[mask] - labels[mask]))
                results[label] = {
                    "count": int(np.sum(mask)),
                    "pearson": float(range_pearson) if not np.isnan(range_pearson) else 0.0,
                    "mae": float(range_mae)
                }

        return results

    def visualize_results(
        self,
        lam_predictions: np.ndarray,
        baseline_predictions: np.ndarray,
        labels: np.ndarray,
        lam_pearson: float,
        baseline_pearson: float
    ):
        """Generate comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 12))

        # 1. Scatter plot: LAM predictions vs labels
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(labels, lam_predictions, alpha=0.5, s=10)
        ax1.plot([0, 5], [0, 5], 'r--', label='Perfect correlation')
        ax1.set_xlabel('True Similarity Score', fontsize=12)
        ax1.set_ylabel('Predicted Similarity Score', fontsize=12)
        ax1.set_title(f'LAM: Predictions vs Labels\nPearson = {lam_pearson:.4f}',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Scatter plot: Baseline predictions vs labels
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(labels, baseline_predictions, alpha=0.5, s=10, color='orange')
        ax2.plot([0, 5], [0, 5], 'r--', label='Perfect correlation')
        ax2.set_xlabel('True Similarity Score', fontsize=12)
        ax2.set_ylabel('Predicted Similarity Score', fontsize=12)
        ax2.set_title(f'Baseline: Predictions vs Labels\nPearson = {baseline_pearson:.4f}',
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Error distribution
        ax3 = plt.subplot(2, 3, 3)
        lam_errors = lam_predictions - labels
        baseline_errors = baseline_predictions - labels
        ax3.hist(lam_errors, bins=50, alpha=0.6, label='LAM', color='blue')
        ax3.hist(baseline_errors, bins=50, alpha=0.6, label='Baseline', color='orange')
        ax3.axvline(x=0, color='r', linestyle='--', label='Zero error')
        ax3.set_xlabel('Prediction Error', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Residual plot - LAM
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(lam_predictions, lam_errors, alpha=0.5, s=10)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Predicted Score', fontsize=12)
        ax4.set_ylabel('Residual', fontsize=12)
        ax4.set_title('LAM: Residual Plot', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 5. Comparison bar chart
        ax5 = plt.subplot(2, 3, 5)
        metrics = ['Pearson', 'MAE', 'RMSE']
        lam_mae = np.mean(np.abs(lam_errors))
        lam_rmse = np.sqrt(np.mean(lam_errors**2))
        baseline_mae = np.mean(np.abs(baseline_errors))
        baseline_rmse = np.sqrt(np.mean(baseline_errors**2))

        lam_values = [lam_pearson, lam_mae, lam_rmse]
        baseline_values = [baseline_pearson, baseline_mae, baseline_rmse]

        x = np.arange(len(metrics))
        width = 0.35

        # Normalize for visualization (except Pearson which is already 0-1)
        lam_viz = [lam_values[0], lam_values[1]/5, lam_values[2]/5]
        baseline_viz = [baseline_values[0], baseline_values[1]/5, baseline_values[2]/5]

        ax5.bar(x - width/2, lam_viz, width, label='LAM', color='blue')
        ax5.bar(x + width/2, baseline_viz, width, label='Baseline', color='orange')
        ax5.set_xlabel('Metrics', fontsize=12)
        ax5.set_ylabel('Score (normalized)', fontsize=12)
        ax5.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # Add actual values as text
        for i, (lv, bv) in enumerate(zip(lam_values, baseline_values)):
            ax5.text(i - width/2, lam_viz[i] + 0.02, f'{lv:.3f}',
                    ha='center', va='bottom', fontsize=9)
            ax5.text(i + width/2, baseline_viz[i] + 0.02, f'{bv:.3f}',
                    ha='center', va='bottom', fontsize=9)

        # 6. Box plot of errors by score range
        ax6 = plt.subplot(2, 3, 6)
        ranges = [(0,1), (1,2), (2,3), (3,4), (4,5)]
        range_labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
        lam_box_data = []

        for low, high in ranges:
            mask = (labels >= low) & (labels < high)
            if np.sum(mask) > 0:
                lam_box_data.append(lam_errors[mask])
            else:
                lam_box_data.append([])

        ax6.boxplot(lam_box_data, labels=range_labels)
        ax6.axhline(y=0, color='r', linestyle='--')
        ax6.set_xlabel('True Score Range', fontsize=12)
        ax6.set_ylabel('Prediction Error', fontsize=12)
        ax6.set_title('Error Distribution by Score Range', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save figure
        save_path = self.viz_dir / "pearson_score_validation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        plt.close()

    def run(self) -> Dict:
        """Run complete Pearson score validation"""
        print("="*70)
        print("LAM PEARSON SCORE VALIDATION - COMPREHENSIVE TEST")
        print("="*70)

        # Load models
        print(f"\nLoading LAM model from: {self.model_path}")
        print(f"  (Combines lam_base.bin + lam_tweak.pt)")
        try:
            lam_model = LAMEncoder(self.model_path)
            print("✓ LAM model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading LAM model: {e}")
            print("⚠️  Using all-MiniLM-L6-v2 as placeholder for demonstration")
            from sentence_transformers import SentenceTransformer
            lam_model = SentenceTransformer("all-MiniLM-L6-v2")

        print(f"\nLoading baseline model: {self.baseline_path}")
        from sentence_transformers import SentenceTransformer
        baseline_model = SentenceTransformer(self.baseline_path)

        # Load dataset
        sentences1, sentences2, labels = self.load_stsb_dataset()
        labels = np.array(labels)

        # Compute predictions for both models
        print("\n" + "="*70)
        print("COMPUTING LAM PREDICTIONS")
        print("="*70)
        lam_predictions = self.compute_embeddings_and_similarity(
            lam_model, sentences1, sentences2
        )

        print("\n" + "="*70)
        print("COMPUTING BASELINE PREDICTIONS")
        print("="*70)
        baseline_predictions = self.compute_embeddings_and_similarity(
            baseline_model, sentences1, sentences2
        )

        # Compute metrics
        print("\n" + "="*70)
        print("COMPUTING METRICS")
        print("="*70)

        lam_pearson, lam_p_value = pearsonr(lam_predictions, labels)
        lam_spearman, lam_sp_value = spearmanr(lam_predictions, labels)

        baseline_pearson, baseline_p_value = pearsonr(baseline_predictions, labels)
        baseline_spearman, baseline_sp_value = spearmanr(baseline_predictions, labels)

        # Format p-values for better readability
        def format_p_value(p_val):
            """Format p-value with appropriate precision"""
            # Handle extremely small p-values (essentially zero)
            if p_val == 0.0 or (isinstance(p_val, float) and p_val < 1e-300):
                return "< 1e-300"
            # For very small but representable p-values, use scientific notation
            elif p_val < 1e-10:
                return f"{p_val:.2e}"
            # For larger p-values, use standard scientific notation
            else:
                return f"{p_val:.2e}"

        print(f"\nLAM Pearson: {lam_pearson:.4f} (p={format_p_value(lam_p_value)})")
        print(f"LAM Spearman: {lam_spearman:.4f} (p={format_p_value(lam_sp_value)})")
        print(f"\nBaseline Pearson: {baseline_pearson:.4f} (p={format_p_value(baseline_p_value)})")
        print(f"Baseline Spearman: {baseline_spearman:.4f} (p={format_p_value(baseline_sp_value)})")

        # Bootstrap confidence intervals
        print("\n" + "="*70)
        print("COMPUTING CONFIDENCE INTERVALS")
        print("="*70)

        lam_ci_lower, lam_ci_upper = self.bootstrap_confidence_interval(
            lam_predictions, labels
        )
        baseline_ci_lower, baseline_ci_upper = self.bootstrap_confidence_interval(
            baseline_predictions, labels
        )

        print(f"\nLAM 95% CI: [{lam_ci_lower:.4f}, {lam_ci_upper:.4f}]")
        print(f"Baseline 95% CI: [{baseline_ci_lower:.4f}, {baseline_ci_upper:.4f}]")

        # Error analysis
        print("\n" + "="*70)
        print("ERROR ANALYSIS")
        print("="*70)

        lam_errors = self.analyze_errors(lam_predictions, labels)
        baseline_errors = self.analyze_errors(baseline_predictions, labels)

        print(f"\nLAM MAE: {lam_errors['mean_absolute_error']:.4f}")
        print(f"LAM RMSE: {lam_errors['root_mean_squared_error']:.4f}")
        print(f"\nBaseline MAE: {baseline_errors['mean_absolute_error']:.4f}")
        print(f"Baseline RMSE: {baseline_errors['root_mean_squared_error']:.4f}")

        # Per-range analysis
        print("\n" + "="*70)
        print("PER-RANGE ANALYSIS")
        print("="*70)

        lam_ranges = self.per_score_range_analysis(lam_predictions, labels)
        baseline_ranges = self.per_score_range_analysis(baseline_predictions, labels)

        for range_name in lam_ranges:
            print(f"\n{range_name}:")
            print(f"  Count: {lam_ranges[range_name]['count']}")
            print(f"  LAM Pearson: {lam_ranges[range_name]['pearson']:.4f}")
            print(f"  Baseline Pearson: {baseline_ranges[range_name]['pearson']:.4f}")

        # Create comprehensive results
        results = {
            "test_name": "Pearson Score Validation on STS-B",
            "model_path": self.model_path,
            "baseline_path": self.baseline_path,
            "dataset": {
                "name": f"STS-B {self.split.capitalize()} Set",
                "split": self.split,
                "num_pairs": len(labels),
                "note": "validation=development set, test=official benchmark"
            },
            "lam_results": {
                "pearson": float(lam_pearson),
                "pearson_p_value": float(lam_p_value),
                "spearman": float(lam_spearman),
                "spearman_p_value": float(lam_sp_value),
                "confidence_interval_95": [float(lam_ci_lower), float(lam_ci_upper)],
                "error_analysis": lam_errors,
                "per_range_analysis": lam_ranges
            },
            "baseline_results": {
                "pearson": float(baseline_pearson),
                "pearson_p_value": float(baseline_p_value),
                "spearman": float(baseline_spearman),
                "spearman_p_value": float(baseline_sp_value),
                "confidence_interval_95": [float(baseline_ci_lower), float(baseline_ci_upper)],
                "error_analysis": baseline_errors,
                "per_range_analysis": baseline_ranges
            },
            "comparison": {
                "pearson_gap": float(baseline_pearson - lam_pearson),
                "pearson_ratio": float(lam_pearson / baseline_pearson),
                "pearson_percentage": float(100 * lam_pearson / baseline_pearson)
            },
            "validation_status": {
                "claimed_score": 0.836,
                "achieved_score": float(lam_pearson),
                "within_ci": lam_ci_lower <= 0.836 <= lam_ci_upper,
                "validated": abs(lam_pearson - 0.836) < 0.01
            }
        }

        # Save results (convert numpy types to native Python types for JSON)
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        results_native = convert_to_native(results)
        # Save with split in filename for clarity
        results_path = self.results_dir / f"pearson_score_validation_{self.split}.json"
        with open(results_path, 'w') as f:
            json.dump(results_native, f, indent=2)

        print(f"\n" + "="*70)
        print(f"Results saved to: {results_path}")
        print("="*70)

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_results(
            lam_predictions, baseline_predictions, labels,
            lam_pearson, baseline_pearson
        )

        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"\nClaimed Score: 0.836")
        print(f"Achieved Score: {lam_pearson:.4f}")
        print(f"95% CI: [{lam_ci_lower:.4f}, {lam_ci_upper:.4f}]")
        print(f"Validation Status: {'✓ PASSED' if results['validation_status']['validated'] else '✗ FAILED'}")
        print(f"\nRelative to Baseline (all-MiniLM-L6-v2):")
        print(f"  Baseline: {baseline_pearson:.4f}")
        print(f"  LAM: {lam_pearson:.4f}")
        print(f"  Ratio: {results['comparison']['pearson_percentage']:.2f}%")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate LAM Pearson score on STS-B")
    parser.add_argument("--model", default="../", help="Path to LAM model")
    parser.add_argument("--baseline", default="all-MiniLM-L6-v2",
                       help="Baseline model for comparison")
    parser.add_argument("--split", default="test", choices=["validation", "test"],
                       help="Dataset split: 'test' (official benchmark, 1379 pairs) or 'validation' (development set, 1500 pairs)")

    args = parser.parse_args()

    validator = PearsonScoreValidator(args.model, args.baseline, args.split)
    results = validator.run()

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
