"""
Test Different Pooling Strategies and Similarity Metrics

This script tests whether Mean Pooling + Cosine Similarity is causing the val→test drop.

Hypothesis from research:
- Mean pooling dilutes signal on long sequences
- Cosine similarity unstable with anisotropic embeddings
- Max-Mean pooling (concatenate mean + max) is more robust

We'll test:
1. Mean Pooling + Cosine Similarity (current)
2. Mean Pooling + Dot Product
3. Mean Pooling + Euclidean Distance
4. Max Pooling + Cosine Similarity
5. Max-Mean Pooling + Cosine Similarity (SOTA approach)
6. Max-Mean Pooling + Dot Product
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "production"))

import torch
import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from typing import Tuple, List
from lam_wrapper import LAMEncoder

class PoolingTester:
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"Loading LAM model from: {model_path}")
        self.model = LAMEncoder(str(model_path))

    def get_token_embeddings(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """Get token-level embeddings (before pooling)"""
        print(f"  Encoding {len(sentences)} sentences...")

        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Get tokenized inputs
            inputs = self.model.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            device = next(self.model.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get token embeddings (before pooling)
            with torch.no_grad():
                outputs = self.model.model(**inputs, output_hidden_states=True)
                # Last hidden state shape: (batch_size, seq_len, hidden_dim)
                token_embeddings = outputs.last_hidden_state

            all_embeddings.append(token_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Standard mean pooling"""
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings and divide by number of tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def max_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Max pooling - take element-wise maximum across sequence"""
        # Set padding tokens to large negative value
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings = token_embeddings.clone()
        token_embeddings[input_mask_expanded == 0] = -1e9

        # Max pooling
        max_embeddings = torch.max(token_embeddings, dim=1)[0]

        return max_embeddings

    def max_mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Max-Mean pooling - concatenate mean and max pooled vectors"""
        mean_pooled = self.mean_pooling(token_embeddings, attention_mask)
        max_pooled = self.max_pooling(token_embeddings, attention_mask)

        # Concatenate
        return torch.cat([mean_pooled, max_pooled], dim=1)

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Cosine similarity between pairs of embeddings"""
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = []
        for e1, e2 in zip(emb1, emb2):
            sim = cosine_similarity([e1], [e2])[0][0]
            similarities.append(sim)

        return np.array(similarities)

    def dot_product(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Dot product between pairs of embeddings"""
        return np.sum(emb1 * emb2, axis=1)

    def euclidean_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Negative Euclidean distance (so higher = more similar)"""
        return -np.linalg.norm(emb1 - emb2, axis=1)

    def test_strategy(self,
                     sentences1: List[str],
                     sentences2: List[str],
                     labels: np.ndarray,
                     pooling_method: str,
                     similarity_metric: str) -> Tuple[float, float]:
        """Test a specific pooling + similarity combination"""

        print(f"\n  Testing: {pooling_method} + {similarity_metric}")

        # Get tokenized inputs for attention masks
        inputs1 = self.model.tokenizer(
            sentences1,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs2 = self.model.tokenizer(
            sentences2,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Get token embeddings
        token_emb1 = self.get_token_embeddings(sentences1)
        token_emb2 = self.get_token_embeddings(sentences2)

        # Apply pooling
        if pooling_method == "mean":
            emb1 = self.mean_pooling(token_emb1, inputs1['attention_mask'])
            emb2 = self.mean_pooling(token_emb2, inputs2['attention_mask'])
        elif pooling_method == "max":
            emb1 = self.max_pooling(token_emb1, inputs1['attention_mask'])
            emb2 = self.max_pooling(token_emb2, inputs2['attention_mask'])
        elif pooling_method == "max-mean":
            emb1 = self.max_mean_pooling(token_emb1, inputs1['attention_mask'])
            emb2 = self.max_mean_pooling(token_emb2, inputs2['attention_mask'])
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

        # Convert to numpy
        emb1 = emb1.numpy()
        emb2 = emb2.numpy()

        # Normalize for cosine similarity
        if similarity_metric == "cosine":
            from sklearn.preprocessing import normalize
            emb1 = normalize(emb1)
            emb2 = normalize(emb2)

        # Compute similarity
        if similarity_metric == "cosine":
            similarities = self.cosine_similarity(emb1, emb2)
        elif similarity_metric == "dot":
            similarities = self.dot_product(emb1, emb2)
        elif similarity_metric == "euclidean":
            similarities = self.euclidean_distance(emb1, emb2)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

        # Scale to [0, 5] range for STS-B (if needed)
        if similarity_metric == "cosine":
            predictions = similarities * 5.0
        else:
            # For dot product and euclidean, scale to [0, 5]
            min_sim = similarities.min()
            max_sim = similarities.max()
            predictions = (similarities - min_sim) / (max_sim - min_sim) * 5.0

        # Compute Pearson correlation
        pearson, _ = pearsonr(predictions, labels)
        spearman, _ = spearmanr(predictions, labels)

        return pearson, spearman

    def run_comprehensive_test(self, split: str = "validation"):
        """Test all combinations on a dataset split"""

        print("="*70)
        print(f"POOLING STRATEGY COMPREHENSIVE TEST - {split.upper()} SET")
        print("="*70)

        # Load dataset
        print(f"\nLoading STS-B {split} dataset...")
        dataset = load_dataset("sentence-transformers/stsb", split=split)
        sentences1 = dataset['sentence1']
        sentences2 = dataset['sentence2']
        labels = np.array(dataset['score'])
        print(f"  Loaded {len(labels)} sentence pairs")

        # Test all combinations
        pooling_methods = ["mean", "max", "max-mean"]
        similarity_metrics = ["cosine", "dot", "euclidean"]

        results = {}

        for pooling in pooling_methods:
            for similarity in similarity_metrics:
                key = f"{pooling}+{similarity}"
                try:
                    pearson, spearman = self.test_strategy(
                        sentences1, sentences2, labels,
                        pooling, similarity
                    )
                    results[key] = {
                        'pearson': pearson,
                        'spearman': spearman
                    }
                    print(f"    Pearson: {pearson:.4f}, Spearman: {spearman:.4f}")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results[key] = {
                        'pearson': 0.0,
                        'spearman': 0.0,
                        'error': str(e)
                    }

        return results


def main():
    print("="*70)
    print("TESTING POOLING STRATEGIES HYPOTHESIS")
    print("="*70)

    print("\nHypothesis:")
    print("  • Mean pooling dilutes signal on varying sequence lengths")
    print("  • Cosine similarity unstable with anisotropic embeddings")
    print("  • Max-Mean pooling should be more robust")

    model_path = Path(__file__).parent.parent
    tester = PoolingTester(str(model_path))

    # Test on validation set
    print("\n" + "="*70)
    print("VALIDATION SET RESULTS")
    print("="*70)
    val_results = tester.run_comprehensive_test("validation")

    # Test on test set
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    test_results = tester.run_comprehensive_test("test")

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON: VALIDATION vs TEST")
    print("="*70)

    print(f"\n{'Strategy':<25} {'Val Pearson':<15} {'Test Pearson':<15} {'Drop':<12} {'Stable?'}")
    print("-" * 80)

    for key in val_results.keys():
        val_p = val_results[key]['pearson']
        test_p = test_results[key]['pearson']
        drop = test_p - val_p
        stable = "✓" if abs(drop) < 0.02 else "✗"

        print(f"{key:<25} {val_p:<15.4f} {test_p:<15.4f} {drop:<12.4f} {stable}")

    # Find best strategy
    print("\n" + "="*70)
    print("BEST STRATEGIES")
    print("="*70)

    # Best on test set
    best_test = max(test_results.items(), key=lambda x: x[1]['pearson'])
    print(f"\nBest Test Performance: {best_test[0]}")
    print(f"  Pearson: {best_test[1]['pearson']:.4f}")

    # Most stable (smallest drop)
    drops = {k: test_results[k]['pearson'] - val_results[k]['pearson']
             for k in val_results.keys()}
    most_stable = min(drops.items(), key=lambda x: abs(x[1]))
    print(f"\nMost Stable Strategy: {most_stable[0]}")
    print(f"  Drop: {most_stable[1]:.4f}")
    print(f"  Val: {val_results[most_stable[0]]['pearson']:.4f}")
    print(f"  Test: {test_results[most_stable[0]]['pearson']:.4f}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if abs(drops["mean+cosine"]) > 0.05:
        print("\n🔴 CONFIRMED: Mean+Cosine is unstable!")

        if best_test[1]['pearson'] > val_results["mean+cosine"]['pearson']:
            print(f"\n✅ SOLUTION FOUND: {best_test[0]} achieves {best_test[1]['pearson']:.4f}")
            print(f"   This is {(best_test[1]['pearson'] - test_results['mean+cosine']['pearson']):.4f} points higher!")
            print("\n   Implementation:")
            print(f"   1. Replace mean pooling with {best_test[0].split('+')[0]} pooling")
            print(f"   2. Use {best_test[0].split('+')[1]} instead of cosine similarity")
            print("   3. Re-run all evaluations with new strategy")

        if abs(most_stable[1]) < 0.02:
            print(f"\n🟢 STABLE ALTERNATIVE: {most_stable[0]}")
            print(f"   Val→Test drop is only {most_stable[1]:.4f} (very stable!)")
            print(f"   Test score: {test_results[most_stable[0]]['pearson']:.4f}")

    # Save results
    import json
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        'validation': val_results,
        'test': test_results,
        'drops': drops,
        'best_test': best_test[0],
        'most_stable': most_stable[0]
    }

    output_path = results_dir / "pooling_strategy_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
