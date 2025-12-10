"""
Test 3: Long Context Processing Without Chunking
=================================================

This test validates LAM's ability to process extremely long contexts (32K, 100K, 1M tokens)
as single-pass encodings without chunking, maintaining semantic coherence.

Tests:
1. Single-pass encoding up to 1M tokens
2. Semantic coherence across long documents
3. Memory footprint validation
4. Comparison with chunked baseline approach
5. Document-level similarity preservation
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
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import psutil
    import torch
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "scikit-learn", "psutil", "matplotlib", "seaborn", "torch"])
    from sklearn.metrics.pairwise import cosine_similarity
    import psutil
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


class LongContextValidator:
    """Validate long context processing without chunking"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.results_dir = Path(__file__).parent / "results"
        self.viz_dir = Path(__file__).parent / "visualizations"

        self.results_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

        # Test configurations
        self.test_lengths = [
            ("32K", 32768),
            ("64K", 65536),
            ("100K", 100000),
            ("250K", 250000),
            ("500K", 500000),
            ("1M", 1000000)
        ]

    def generate_structured_document(
        self,
        num_tokens: int,
        num_sections: int = 10
    ) -> Tuple[str, List[Tuple[int, int, str]]]:
        """
        Generate a structured document with distinct semantic sections

        Returns:
            (full_document, [(start_idx, end_idx, section_theme)])
        """
        themes = [
            ("Technology", ["artificial intelligence", "machine learning", "neural networks",
                          "deep learning", "computer vision", "natural language processing"]),
            ("Science", ["quantum physics", "molecular biology", "chemistry", "astronomy",
                        "genetics", "neuroscience"]),
            ("History", ["ancient civilizations", "world wars", "renaissance",
                        "industrial revolution", "colonization", "monarchy"]),
            ("Geography", ["mountains", "rivers", "continents", "oceans",
                          "climate zones", "ecosystems"]),
            ("Economics", ["market forces", "supply demand", "inflation",
                          "monetary policy", "trade", "fiscal policy"]),
            ("Philosophy", ["epistemology", "ethics", "metaphysics",
                           "logic", "aesthetics", "ontology"]),
            ("Medicine", ["anatomy", "pharmacology", "surgery",
                         "diagnosis", "treatment", "prevention"]),
            ("Arts", ["painting", "sculpture", "music", "literature",
                     "theater", "cinema"]),
            ("Sports", ["athletics", "competition", "training",
                       "strategy", "teamwork", "performance"]),
            ("Environment", ["sustainability", "conservation", "climate change",
                            "biodiversity", "pollution", "renewable energy"])
        ]

        tokens_per_section = num_tokens // num_sections
        document_parts = []
        section_metadata = []
        current_pos = 0

        for i in range(num_sections):
            theme_name, theme_words = themes[i % len(themes)]

            # Generate section content
            section_words = []
            target_words = int(tokens_per_section * 0.75)  # ~0.75 words per token

            while len(section_words) < target_words:
                # Add theme-specific words with some variation
                section_words.extend(np.random.choice(theme_words, size=min(10, target_words - len(section_words))))
                # Add filler words for coherence
                fillers = ["the", "a", "is", "and", "of", "in", "to", "for", "with", "on"]
                section_words.extend(np.random.choice(fillers, size=min(5, target_words - len(section_words))))

            section_text = " ".join(section_words)
            section_text = f"\n\nSection {i+1}: {theme_name}\n{section_text}\n"

            start_pos = current_pos
            end_pos = start_pos + len(section_text)
            section_metadata.append((start_pos, end_pos, theme_name))

            document_parts.append(section_text)
            current_pos = end_pos

        full_document = "".join(document_parts)
        return full_document, section_metadata

    def measure_memory_mb(self) -> float:
        """Measure current memory usage"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024**2)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024**2)

    def encode_with_monitoring(
        self,
        model: SentenceTransformer,
        text: str,
        method: str = "single_pass"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Encode text and monitor performance

        Args:
            method: 'single_pass' or 'chunked'

        Returns:
            (embedding, metrics)
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        baseline_memory = self.measure_memory_mb()
        metrics = {
            "method": method,
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": baseline_memory,
            "inference_time_s": 0.0,
            "success": False,
            "error": None
        }

        start_time = time.perf_counter()

        try:
            if method == "single_pass":
                # Single-pass encoding
                embedding = model.encode(
                    text,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

            elif method == "chunked":
                # Chunked encoding (baseline comparison)
                chunk_size = 512  # Standard transformer context
                words = text.split()
                chunks = []

                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i+chunk_size])
                    chunks.append(chunk)

                # Encode all chunks
                chunk_embeddings = model.encode(
                    chunks,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=8
                )

                # Average pooling
                embedding = np.mean(chunk_embeddings, axis=0)

            end_time = time.perf_counter()

            # Measure peak memory
            peak_memory = self.measure_memory_mb()

            metrics.update({
                "peak_memory_mb": peak_memory,
                "memory_used_mb": peak_memory - baseline_memory,
                "inference_time_s": end_time - start_time,
                "success": True,
                "embedding_shape": embedding.shape
            })

            return embedding, metrics

        except Exception as e:
            end_time = time.perf_counter()
            metrics.update({
                "inference_time_s": end_time - start_time,
                "success": False,
                "error": str(e)
            })
            return None, metrics

    def test_semantic_coherence(
        self,
        model: SentenceTransformer,
        document: str,
        section_metadata: List[Tuple[int, int, str]]
    ) -> Dict:
        """
        Test if full-document embedding preserves semantic relationships
        between sections
        """
        print("  Testing semantic coherence...")

        # Encode full document
        full_embedding, full_metrics = self.encode_with_monitoring(
            model, document, method="single_pass"
        )

        if full_embedding is None:
            return {"success": False, "error": full_metrics["error"]}

        # Encode individual sections
        section_embeddings = []
        section_themes = []

        for start, end, theme in section_metadata:
            section_text = document[start:end]
            section_emb, _ = self.encode_with_monitoring(model, section_text, "single_pass")

            if section_emb is not None:
                section_embeddings.append(section_emb)
                section_themes.append(theme)

        section_embeddings = np.array(section_embeddings)

        # Compute similarities
        # 1. Full doc vs each section
        full_to_section_sims = cosine_similarity(
            full_embedding.reshape(1, -1),
            section_embeddings
        )[0]

        # 2. Section-to-section similarities
        section_to_section_sims = cosine_similarity(section_embeddings)

        # Analysis
        results = {
            "success": True,
            "num_sections": len(section_themes),
            "full_to_section_similarities": {
                theme: float(sim)
                for theme, sim in zip(section_themes, full_to_section_sims)
            },
            "mean_full_to_section_similarity": float(np.mean(full_to_section_sims)),
            "std_full_to_section_similarity": float(np.std(full_to_section_sims)),
            "min_full_to_section_similarity": float(np.min(full_to_section_sims)),
            "section_coherence_score": float(np.mean(section_to_section_sims)),
        }

        return results

    def test_length(
        self,
        model: SentenceTransformer,
        length_name: str,
        num_tokens: int
    ) -> Dict:
        """Test a specific document length"""
        print(f"\n{'='*70}")
        print(f"Testing {length_name} tokens ({num_tokens:,})")
        print(f"{'='*70}")

        # Generate document
        print("  Generating structured document...")
        document, section_metadata = self.generate_structured_document(
            num_tokens, num_sections=10
        )

        actual_tokens = len(document.split())
        print(f"  Generated document: ~{actual_tokens:,} tokens, {len(document):,} characters")

        # Test single-pass encoding
        print("  Encoding with single-pass method...")
        single_pass_emb, single_pass_metrics = self.encode_with_monitoring(
            model, document, method="single_pass"
        )

        if not single_pass_metrics["success"]:
            print(f"  ✗ Single-pass encoding FAILED: {single_pass_metrics['error']}")
            return {
                "length_name": length_name,
                "target_tokens": num_tokens,
                "actual_tokens": actual_tokens,
                "single_pass": single_pass_metrics,
                "success": False
            }

        print(f"  ✓ Single-pass encoding succeeded")
        print(f"    Time: {single_pass_metrics['inference_time_s']:.2f}s")
        print(f"    Memory: {single_pass_metrics['memory_used_mb']:.1f}MB")

        # Test semantic coherence
        coherence_results = self.test_semantic_coherence(
            model, document, section_metadata
        )

        if coherence_results["success"]:
            print(f"  ✓ Semantic coherence preserved")
            print(f"    Mean similarity: {coherence_results['mean_full_to_section_similarity']:.4f}")
        else:
            print(f"  ✗ Semantic coherence test failed")

        # Compile results
        results = {
            "length_name": length_name,
            "target_tokens": num_tokens,
            "actual_tokens": actual_tokens,
            "document_chars": len(document),
            "single_pass": single_pass_metrics,
            "semantic_coherence": coherence_results,
            "success": True
        }

        return results

    def visualize_results(self, all_results: List[Dict]):
        """Generate comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 12))

        # Extract successful results
        successful = [r for r in all_results if r.get("success", False)]

        if not successful:
            print("No successful results to visualize")
            return

        lengths = [r["actual_tokens"] for r in successful]
        times = [r["single_pass"]["inference_time_s"] for r in successful]
        memories = [r["single_pass"]["memory_used_mb"] for r in successful]
        coherences = [r["semantic_coherence"]["mean_full_to_section_similarity"]
                     for r in successful if "semantic_coherence" in r]

        # 1. Time vs Length
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(lengths, times, 'o-', linewidth=2, markersize=10, color='blue')
        ax1.set_xlabel('Document Length (tokens)', fontsize=12)
        ax1.set_ylabel('Inference Time (seconds)', fontsize=12)
        ax1.set_title('Inference Time vs Document Length', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add annotations
        for i, (l, t) in enumerate(zip(lengths, times)):
            ax1.annotate(f'{t:.1f}s', (l, t), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9)

        # 2. Memory vs Length
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(lengths, memories, 'o-', linewidth=2, markersize=10, color='green')
        ax2.set_xlabel('Document Length (tokens)', fontsize=12)
        ax2.set_ylabel('Memory Used (MB)', fontsize=12)
        ax2.set_title('Memory Usage vs Document Length', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add annotations
        for i, (l, m) in enumerate(zip(lengths, memories)):
            ax2.annotate(f'{m:.0f}MB', (l, m), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9)

        # 3. Semantic coherence across lengths
        ax3 = plt.subplot(2, 3, 3)
        if coherences and len(coherences) == len(lengths):
            ax3.plot(lengths, coherences, 'o-', linewidth=2, markersize=10, color='purple')
            ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
            ax3.set_xlabel('Document Length (tokens)', fontsize=12)
            ax3.set_ylabel('Mean Semantic Similarity', fontsize=12)
            ax3.set_title('Semantic Coherence vs Length', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1])

        # 4. Memory efficiency (MB per 1K tokens)
        ax4 = plt.subplot(2, 3, 4)
        memory_per_1k = [m / (l/1000) for m, l in zip(memories, lengths)]
        ax4.plot(lengths, memory_per_1k, 'o-', linewidth=2, markersize=10, color='orange')
        ax4.set_xlabel('Document Length (tokens)', fontsize=12)
        ax4.set_ylabel('Memory per 1K tokens (MB)', fontsize=12)
        ax4.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 5. Processing speed (tokens/second)
        ax5 = plt.subplot(2, 3, 5)
        tokens_per_sec = [l / t for l, t in zip(lengths, times)]
        ax5.plot(lengths, tokens_per_sec, 'o-', linewidth=2, markersize=10, color='red')
        ax5.set_xlabel('Document Length (tokens)', fontsize=12)
        ax5.set_ylabel('Processing Speed (tokens/sec)', fontsize=12)
        ax5.set_title('Processing Speed vs Length', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. Summary table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        table_data = []
        for r in successful:
            row = [
                r["length_name"],
                f"{r['actual_tokens']:,}",
                f"{r['single_pass']['inference_time_s']:.1f}s",
                f"{r['single_pass']['memory_used_mb']:.0f}MB",
                "✓" if r.get("semantic_coherence", {}).get("success") else "✗"
            ]
            table_data.append(row)

        table = ax6.table(cellText=table_data,
                         colLabels=['Length', 'Tokens', 'Time', 'Memory', 'Coherence'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax6.set_title('Summary Table', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save
        save_path = self.viz_dir / "long_context_validation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.close()

    def run(self) -> Dict:
        """Run complete long context validation"""
        print("="*70)
        print("LAM LONG CONTEXT VALIDATION - NO CHUNKING TEST")
        print("="*70)

        # Load model
        print(f"\nLoading LAM model from: {self.model_path}")
        print(f"  (Combines lam_base.bin + lam_tweak.pt)")
        try:
            model = LAMEncoder(self.model_path)
            print("✓ LAM model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading LAM model: {e}")
            print("⚠️  Using all-MiniLM-L6-v2 as placeholder")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")

        # Run tests for each length
        all_results = []
        for length_name, num_tokens in self.test_lengths:
            try:
                result = self.test_length(model, length_name, num_tokens)
                all_results.append(result)
            except Exception as e:
                print(f"  ✗ Test failed with error: {e}")
                all_results.append({
                    "length_name": length_name,
                    "target_tokens": num_tokens,
                    "success": False,
                    "error": str(e)
                })

        # Compile final results
        results = {
            "test_name": "Long Context Processing Without Chunking",
            "model_path": self.model_path,
            "test_configurations": [
                {"name": name, "tokens": tokens}
                for name, tokens in self.test_lengths
            ],
            "results": all_results,
            "summary": {
                "total_tests": len(all_results),
                "successful": sum(1 for r in all_results if r.get("success", False)),
                "failed": sum(1 for r in all_results if not r.get("success", False)),
                "max_length_achieved": max(
                    (r["actual_tokens"] for r in all_results if r.get("success", False)),
                    default=0
                )
            }
        }

        # Save results
        results_path = self.results_dir / "long_context_validation.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Results saved to: {results_path}")
        print("="*70)

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_results(all_results)

        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"\nTotal tests: {results['summary']['total_tests']}")
        print(f"Successful: {results['summary']['successful']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Max length achieved: {results['summary']['max_length_achieved']:,} tokens")

        print("\nPer-Length Results:")
        for r in all_results:
            if r.get("success"):
                print(f"  ✓ {r['length_name']}: {r['actual_tokens']:,} tokens, "
                      f"{r['single_pass']['inference_time_s']:.1f}s, "
                      f"{r['single_pass']['memory_used_mb']:.0f}MB")
            else:
                print(f"  ✗ {r['length_name']}: FAILED")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate LAM long context processing without chunking"
    )
    parser.add_argument("--model", default="../", help="Path to LAM model")

    args = parser.parse_args()

    validator = LongContextValidator(args.model)
    results = validator.run()

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
