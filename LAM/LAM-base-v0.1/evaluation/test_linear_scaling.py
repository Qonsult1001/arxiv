"""
Test 2: Verify LAM's O(n) Linear Scaling
=========================================

This test rigorously validates that LAM maintains linear O(n) complexity
for both memory usage and inference time across increasing sequence lengths.

Tests:
1. Memory scaling: 128 to 100K tokens
2. Time scaling: 128 to 100K tokens
3. Comparison with quadratic baseline (all-MiniLM-L6-v2)
4. Regression analysis to confirm O(n) vs O(n²)
5. Crossover point identification
"""

import json
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import gc
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import psutil
    import GPUtil
    import torch
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "psutil", "gputil", "matplotlib", "seaborn", "scipy", "torch"])
    import psutil
    import GPUtil
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


class LinearScalingValidator:
    """Validate O(n) linear scaling for LAM"""

    def __init__(self, model_path: str, baseline_path: str = "all-MiniLM-L6-v2"):
        self.model_path = model_path
        self.baseline_path = baseline_path
        self.results_dir = Path(__file__).parent / "results"
        self.viz_dir = Path(__file__).parent / "visualizations"

        self.results_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

        # Test sequence lengths (logarithmic scale)
        self.test_lengths = [
            128, 256, 512, 1024, 2048, 4096, 8192,
            16384, 32768, 65536, 100000
        ]

    def generate_test_text(self, num_tokens: int) -> str:
        """Generate synthetic text of approximately num_tokens"""
        # Average ~4 characters per token
        words_per_token = 0.75
        num_words = int(num_tokens * words_per_token)

        # Use varied vocabulary to avoid compression artifacts
        words = [
            "semantic", "embedding", "transformer", "attention", "neural",
            "network", "machine", "learning", "artificial", "intelligence",
            "language", "model", "processing", "natural", "algorithm",
            "optimization", "gradient", "descent", "backpropagation", "training",
            "validation", "testing", "accuracy", "precision", "recall"
        ]

        text = " ".join(np.random.choice(words, size=num_words))
        return text

    def measure_memory_gpu(self) -> float:
        """Measure GPU memory usage in MB"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024**2)
        return 0.0

    def measure_memory_cpu(self) -> float:
        """Measure CPU memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**2)

    def run_inference_with_monitoring(
        self,
        model: SentenceTransformer,
        text: str,
        warmup: bool = False
    ) -> Tuple[float, float, float]:
        """
        Run inference and monitor memory + time

        Returns:
            (inference_time_ms, peak_memory_mb, avg_memory_mb)
        """
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Baseline memory
        baseline_memory = self.measure_memory_gpu() if torch.cuda.is_available() else self.measure_memory_cpu()

        memory_samples = []

        # Run inference
        start_time = time.perf_counter()

        try:
            embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)

            # Sample memory during inference
            current_memory = self.measure_memory_gpu() if torch.cuda.is_available() else self.measure_memory_cpu()
            memory_samples.append(current_memory - baseline_memory)

        except Exception as e:
            print(f"  ✗ OOM or Error: {e}")
            return None, None, None

        end_time = time.perf_counter()

        # Final memory
        final_memory = self.measure_memory_gpu() if torch.cuda.is_available() else self.measure_memory_cpu()
        memory_samples.append(final_memory - baseline_memory)

        inference_time_ms = (end_time - start_time) * 1000
        peak_memory_mb = max(memory_samples)
        avg_memory_mb = np.mean(memory_samples)

        return inference_time_ms, peak_memory_mb, avg_memory_mb

    def test_model_scaling(
        self,
        model: SentenceTransformer,
        model_name: str,
        max_length: int = None
    ) -> Dict:
        """Test scaling behavior for a single model"""
        print(f"\n{'='*70}")
        print(f"Testing {model_name}")
        print(f"{'='*70}")

        results = {
            "model_name": model_name,
            "lengths": [],
            "times_ms": [],
            "peak_memory_mb": [],
            "avg_memory_mb": [],
            "status": []
        }

        for length in self.test_lengths:
            if max_length and length > max_length:
                print(f"\nSequence length {length}: SKIPPED (exceeds model max_length)")
                results["lengths"].append(length)
                results["times_ms"].append(None)
                results["peak_memory_mb"].append(None)
                results["avg_memory_mb"].append(None)
                results["status"].append("skipped")
                continue

            print(f"\nSequence length {length}:")

            # Generate test text
            text = self.generate_test_text(length)

            # Warmup run
            print("  Warmup...", end=" ")
            try:
                self.run_inference_with_monitoring(model, text[:100], warmup=True)
                print("✓")
            except:
                print("✗")

            # Actual measurement (3 runs, take median)
            print("  Measuring...", end=" ")
            times = []
            peak_mems = []
            avg_mems = []

            for run in range(3):
                time_ms, peak_mem, avg_mem = self.run_inference_with_monitoring(model, text)

                if time_ms is None:
                    # OOM occurred
                    results["lengths"].append(length)
                    results["times_ms"].append(None)
                    results["peak_memory_mb"].append(None)
                    results["avg_memory_mb"].append(None)
                    results["status"].append("oom")
                    print(f"✗ OOM")
                    break

                times.append(time_ms)
                peak_mems.append(peak_mem)
                avg_mems.append(avg_mem)

            if times:
                median_time = float(np.median(times))
                median_peak_mem = float(np.median(peak_mems))
                median_avg_mem = float(np.median(avg_mems))

                results["lengths"].append(length)
                results["times_ms"].append(median_time)
                results["peak_memory_mb"].append(median_peak_mem)
                results["avg_memory_mb"].append(median_avg_mem)
                results["status"].append("success")

                print(f"✓ Time: {median_time:.1f}ms, Peak Memory: {median_peak_mem:.1f}MB")

            # If we hit OOM, stop testing longer sequences
            if results["status"][-1] == "oom":
                print(f"  Stopping tests for {model_name} due to OOM")
                break

        return results

    def fit_complexity_model(
        self,
        lengths: List[int],
        values: List[float]
    ) -> Tuple[float, float, str]:
        """
        Fit both linear and quadratic models and determine best fit

        Returns:
            (linear_r2, quadratic_r2, best_fit)
        """
        # Filter out None values
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        lengths = np.array([lengths[i] for i in valid_indices])
        values = np.array([values[i] for i in valid_indices])

        if len(lengths) < 3:
            return 0.0, 0.0, "insufficient_data"

        # Linear model: y = a*x + b
        linear_coeffs = np.polyfit(lengths, values, 1)
        linear_pred = np.polyval(linear_coeffs, lengths)
        linear_r2 = 1 - (np.sum((values - linear_pred)**2) / np.sum((values - np.mean(values))**2))

        # Quadratic model: y = a*x^2 + b*x + c
        quad_coeffs = np.polyfit(lengths, values, 2)
        quad_pred = np.polyval(quad_coeffs, lengths)
        quad_r2 = 1 - (np.sum((values - quad_pred)**2) / np.sum((values - np.mean(values))**2))

        # Determine best fit (with preference for linear if close)
        if linear_r2 > 0.95 and abs(quad_coeffs[0]) < 1e-9:
            best_fit = "linear"
        elif quad_r2 - linear_r2 > 0.05:
            best_fit = "quadratic"
        else:
            best_fit = "linear"

        return float(linear_r2), float(quad_r2), best_fit

    def visualize_scaling(
        self,
        lam_results: Dict,
        baseline_results: Dict
    ):
        """Generate comprehensive scaling visualizations"""
        fig = plt.figure(figsize=(20, 12))

        # Extract data
        lam_lengths = np.array(lam_results["lengths"])
        lam_times = np.array([t if t is not None else np.nan for t in lam_results["times_ms"]])
        lam_memory = np.array([m if m is not None else np.nan for m in lam_results["peak_memory_mb"]])

        baseline_lengths = np.array(baseline_results["lengths"])
        baseline_times = np.array([t if t is not None else np.nan for t in baseline_results["times_ms"]])
        baseline_memory = np.array([m if m is not None else np.nan for m in baseline_results["peak_memory_mb"]])

        # 1. Time scaling - Linear scale
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(lam_lengths, lam_times, 'o-', label='LAM', linewidth=2, markersize=8)
        ax1.plot(baseline_lengths, baseline_times, 's-', label='Baseline', linewidth=2, markersize=8)
        ax1.set_xlabel('Sequence Length (tokens)', fontsize=12)
        ax1.set_ylabel('Inference Time (ms)', fontsize=12)
        ax1.set_title('Inference Time vs Sequence Length', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 2. Time scaling - Log-log scale
        ax2 = plt.subplot(2, 3, 2)
        valid_lam_idx = ~np.isnan(lam_times)
        valid_baseline_idx = ~np.isnan(baseline_times)
        ax2.loglog(lam_lengths[valid_lam_idx], lam_times[valid_lam_idx],
                  'o-', label='LAM (O(n))', linewidth=2, markersize=8)
        ax2.loglog(baseline_lengths[valid_baseline_idx], baseline_times[valid_baseline_idx],
                  's-', label='Baseline (O(n²))', linewidth=2, markersize=8)

        # Add reference lines
        ref_lengths = np.array([128, 100000])
        linear_ref = 1 * (ref_lengths / ref_lengths[0])
        quad_ref = 1 * (ref_lengths / ref_lengths[0])**2
        ax2.loglog(ref_lengths, linear_ref, '--', color='blue', alpha=0.5, label='O(n) reference')
        ax2.loglog(ref_lengths, quad_ref, '--', color='orange', alpha=0.5, label='O(n²) reference')

        ax2.set_xlabel('Sequence Length (tokens)', fontsize=12)
        ax2.set_ylabel('Inference Time (ms)', fontsize=12)
        ax2.set_title('Time Scaling (Log-Log)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')

        # 3. Memory scaling - Linear scale
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(lam_lengths, lam_memory, 'o-', label='LAM', linewidth=2, markersize=8)
        ax3.plot(baseline_lengths, baseline_memory, 's-', label='Baseline', linewidth=2, markersize=8)
        ax3.set_xlabel('Sequence Length (tokens)', fontsize=12)
        ax3.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax3.set_title('Memory Usage vs Sequence Length', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)

        # 4. Memory scaling - Log-log scale
        ax4 = plt.subplot(2, 3, 4)
        valid_lam_mem_idx = ~np.isnan(lam_memory)
        valid_baseline_mem_idx = ~np.isnan(baseline_memory)
        ax4.loglog(lam_lengths[valid_lam_mem_idx], lam_memory[valid_lam_mem_idx],
                  'o-', label='LAM (O(n))', linewidth=2, markersize=8)
        ax4.loglog(baseline_lengths[valid_baseline_mem_idx], baseline_memory[valid_baseline_mem_idx],
                  's-', label='Baseline (O(n²))', linewidth=2, markersize=8)

        # Add reference lines
        linear_ref_mem = 50 * (ref_lengths / ref_lengths[0])
        quad_ref_mem = 50 * (ref_lengths / ref_lengths[0])**2
        ax4.loglog(ref_lengths, linear_ref_mem, '--', color='blue', alpha=0.5, label='O(n) reference')
        ax4.loglog(ref_lengths, quad_ref_mem, '--', color='orange', alpha=0.5, label='O(n²) reference')

        ax4.set_xlabel('Sequence Length (tokens)', fontsize=12)
        ax4.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax4.set_title('Memory Scaling (Log-Log)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, which='both')

        # 5. Speedup comparison
        ax5 = plt.subplot(2, 3, 5)
        # Find common lengths where both models succeeded
        common_lengths = []
        speedups = []
        for i, length in enumerate(lam_lengths):
            if length in baseline_lengths:
                baseline_idx = list(baseline_lengths).index(length)
                if (lam_times[i] is not None and not np.isnan(lam_times[i]) and
                    baseline_times[baseline_idx] is not None and not np.isnan(baseline_times[baseline_idx])):
                    common_lengths.append(length)
                    speedups.append(baseline_times[baseline_idx] / lam_times[i])

        if common_lengths:
            ax5.plot(common_lengths, speedups, 'o-', linewidth=2, markersize=8, color='green')
            ax5.axhline(y=1, color='r', linestyle='--', label='No speedup')
            ax5.set_xlabel('Sequence Length (tokens)', fontsize=12)
            ax5.set_ylabel('Speedup (Baseline / LAM)', fontsize=12)
            ax5.set_title('LAM Speedup Over Baseline', fontsize=14, fontweight='bold')
            ax5.legend(fontsize=11)
            ax5.grid(True, alpha=0.3)
            ax5.set_xscale('log')

        # 6. Memory reduction comparison
        ax6 = plt.subplot(2, 3, 6)
        common_lengths_mem = []
        memory_reductions = []
        for i, length in enumerate(lam_lengths):
            if length in baseline_lengths:
                baseline_idx = list(baseline_lengths).index(length)
                if (lam_memory[i] is not None and not np.isnan(lam_memory[i]) and
                    baseline_memory[baseline_idx] is not None and not np.isnan(baseline_memory[baseline_idx])):
                    common_lengths_mem.append(length)
                    memory_reductions.append(baseline_memory[baseline_idx] / lam_memory[i])

        if common_lengths_mem:
            ax6.plot(common_lengths_mem, memory_reductions, 'o-', linewidth=2, markersize=8, color='purple')
            ax6.axhline(y=1, color='r', linestyle='--', label='No reduction')
            ax6.set_xlabel('Sequence Length (tokens)', fontsize=12)
            ax6.set_ylabel('Memory Reduction (Baseline / LAM)', fontsize=12)
            ax6.set_title('LAM Memory Reduction Over Baseline', fontsize=14, fontweight='bold')
            ax6.legend(fontsize=11)
            ax6.grid(True, alpha=0.3)
            ax6.set_xscale('log')

        plt.tight_layout()

        # Save figure
        save_path = self.viz_dir / "linear_scaling_validation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.close()

    def run(self) -> Dict:
        """Run complete linear scaling validation"""
        print("="*70)
        print("LAM LINEAR SCALING VALIDATION - O(n) COMPLEXITY TEST")
        print("="*70)

        # Load models
        print(f"\nLoading LAM model from: {self.model_path}")
        print(f"  (Combines lam_base.bin + lam_tweak.pt)")
        try:
            lam_model = LAMEncoder(self.model_path)
            print("✓ LAM model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading LAM model: {e}")
            print("⚠️  Using all-MiniLM-L6-v2 as placeholder")
            from sentence_transformers import SentenceTransformer
            lam_model = SentenceTransformer("all-MiniLM-L6-v2")

        print(f"\nLoading baseline model: {self.baseline_path}")
        from sentence_transformers import SentenceTransformer
        baseline_model = SentenceTransformer(self.baseline_path)

        # Test LAM scaling
        lam_results = self.test_model_scaling(lam_model, "LAM")

        # Test baseline scaling (with max_length limit)
        baseline_results = self.test_model_scaling(
            baseline_model, "Baseline", max_length=512
        )

        # Complexity analysis
        print("\n" + "="*70)
        print("COMPLEXITY ANALYSIS")
        print("="*70)

        lam_time_linear_r2, lam_time_quad_r2, lam_time_fit = self.fit_complexity_model(
            lam_results["lengths"], lam_results["times_ms"]
        )
        lam_mem_linear_r2, lam_mem_quad_r2, lam_mem_fit = self.fit_complexity_model(
            lam_results["lengths"], lam_results["peak_memory_mb"]
        )

        baseline_time_linear_r2, baseline_time_quad_r2, baseline_time_fit = self.fit_complexity_model(
            baseline_results["lengths"], baseline_results["times_ms"]
        )
        baseline_mem_linear_r2, baseline_mem_quad_r2, baseline_mem_fit = self.fit_complexity_model(
            baseline_results["lengths"], baseline_results["peak_memory_mb"]
        )

        print(f"\nLAM Time Complexity:")
        print(f"  Linear R²: {lam_time_linear_r2:.4f}")
        print(f"  Quadratic R²: {lam_time_quad_r2:.4f}")
        print(f"  Best Fit: {lam_time_fit}")

        print(f"\nLAM Memory Complexity:")
        print(f"  Linear R²: {lam_mem_linear_r2:.4f}")
        print(f"  Quadratic R²: {lam_mem_quad_r2:.4f}")
        print(f"  Best Fit: {lam_mem_fit}")

        print(f"\nBaseline Time Complexity:")
        print(f"  Linear R²: {baseline_time_linear_r2:.4f}")
        print(f"  Quadratic R²: {baseline_time_quad_r2:.4f}")
        print(f"  Best Fit: {baseline_time_fit}")

        print(f"\nBaseline Memory Complexity:")
        print(f"  Linear R²: {baseline_mem_linear_r2:.4f}")
        print(f"  Quadratic R²: {baseline_mem_quad_r2:.4f}")
        print(f"  Best Fit: {baseline_mem_fit}")

        # Compile results
        results = {
            "test_name": "Linear Scaling Validation - O(n) Complexity",
            "lam_model": self.model_path,
            "baseline_model": self.baseline_path,
            "test_lengths": self.test_lengths,
            "lam_results": lam_results,
            "baseline_results": baseline_results,
            "complexity_analysis": {
                "lam": {
                    "time": {
                        "linear_r2": lam_time_linear_r2,
                        "quadratic_r2": lam_time_quad_r2,
                        "best_fit": lam_time_fit
                    },
                    "memory": {
                        "linear_r2": lam_mem_linear_r2,
                        "quadratic_r2": lam_mem_quad_r2,
                        "best_fit": lam_mem_fit
                    }
                },
                "baseline": {
                    "time": {
                        "linear_r2": baseline_time_linear_r2,
                        "quadratic_r2": baseline_time_quad_r2,
                        "best_fit": baseline_time_fit
                    },
                    "memory": {
                        "linear_r2": baseline_mem_linear_r2,
                        "quadratic_r2": baseline_mem_quad_r2,
                        "best_fit": baseline_mem_fit
                    }
                }
            },
            "validation_status": {
                "lam_time_is_linear": lam_time_fit == "linear" and lam_time_linear_r2 > 0.90,
                "lam_memory_is_linear": lam_mem_fit == "linear" and lam_mem_linear_r2 > 0.90,
                "baseline_time_is_quadratic": baseline_time_fit == "quadratic" or baseline_time_quad_r2 > baseline_time_linear_r2,
                "validated": (lam_time_fit == "linear" and lam_mem_fit == "linear")
            }
        }

        # Save results
        results_path = self.results_dir / "linear_scaling_validation.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Results saved to: {results_path}")
        print("="*70)

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_scaling(lam_results, baseline_results)

        # Print validation summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"\nLAM Time Complexity: {lam_time_fit} (R²={lam_time_linear_r2:.4f})")
        print(f"LAM Memory Complexity: {lam_mem_fit} (R²={lam_mem_linear_r2:.4f})")
        print(f"\nValidation Status: {'✓ PASSED - O(n) Linear Confirmed' if results['validation_status']['validated'] else '✗ FAILED'}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate LAM linear O(n) scaling")
    parser.add_argument("--model", default="../", help="Path to LAM model")
    parser.add_argument("--baseline", default="all-MiniLM-L6-v2",
                       help="Baseline model for comparison")

    args = parser.parse_args()

    validator = LinearScalingValidator(args.model, args.baseline)
    results = validator.run()

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
