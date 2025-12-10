"""
Compare test_checkpoints.py evaluation vs MTEB evaluation
"""
from mteb_model_wrapper import DeltaNetMTEBWrapper
from stsb_evaluation import evaluate_checkpoint
import torch

CHECKPOINT = "pure_constant_lr/distill_step0050_val0.8467.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*80)
print("COMPARING EVALUATION METHODS")
print("="*80)

# Method 1: test_checkpoints.py (stsb_evaluation.py)
print("\n1. test_checkpoints.py method (stsb_evaluation.py):")
print("-" * 80)
res_stsb = evaluate_checkpoint(CHECKPOINT, device=device, split='test')
print(f"   Dataset: sentence-transformers/stsb (test split)")
print(f"   Samples: 1379")
print(f"   Pearson (cosine):  {res_stsb.get('pearson_cosine', 'N/A'):.6f}")
print(f"   Spearman (cosine): {res_stsb.get('spearman_cosine', 'N/A'):.6f}")

# Method 2: MTEB
print("\n2. MTEB STSBenchmark task:")
print("-" * 80)
import mteb
model = DeltaNetMTEBWrapper(CHECKPOINT, device=device)
task = mteb.get_task('STSBenchmark')
result = mteb.evaluate(model, task, show_progress_bar=False, overwrite_strategy='always')

# Extract scores
if hasattr(result, 'task_results') and result.task_results:
    task_res = result.task_results[0]
    if hasattr(task_res, 'scores') and isinstance(task_res.scores, dict):
        scores = task_res.scores
        print(f"   Dataset: MTEB STSBenchmark")
        # Try to get sample count
        try:
            data = task.load_data()
            if data and 'test' in data:
                print(f"   Samples: {len(data['test'])}")
            else:
                print(f"   Samples: ~1379 (standard STS-B test set)")
        except:
            print(f"   Samples: ~1379 (standard STS-B test set)")
        
        pearson_val = scores.get('cosine_pearson') or scores.get('pearson')
        spearman_val = scores.get('cosine_spearman') or scores.get('spearman')
        main_score = scores.get('main_score')
        
        if pearson_val is not None:
            print(f"   Pearson (cosine):  {pearson_val:.6f}")
        if spearman_val is not None:
            print(f"   Spearman (cosine): {spearman_val:.6f}")
        if main_score is not None:
            print(f"   Main score:        {main_score:.6f}")

print("\n" + "="*80)
print("DIFFERENCES:")
print("="*80)
print("1. Dataset source:")
print("   - test_checkpoints.py: Uses 'sentence-transformers/stsb' dataset")
print("   - MTEB: Uses MTEB's standardized STSBenchmark dataset")
print("\n2. Evaluation method:")
print("   - test_checkpoints.py: Direct cosine similarity computation")
print("   - MTEB: Standardized MTEB evaluation pipeline")
print("\n3. Score reporting:")
print("   - test_checkpoints.py: Reports cosine Pearson/Spearman")
print("   - MTEB: Reports multiple metrics, main_score is cosine_spearman")
print("="*80)

