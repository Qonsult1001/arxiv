"""
Local MTEB evaluation script - tests your model + baseline comparison
This script evaluates both your DeltaNet model and MiniLM-L6-H384-uncased (base pretrained) for direct comparison
"""
import mteb
from pathlib import Path
import torch
import json
import re
from datetime import datetime
from mteb_model_wrapper import DeltaNetMTEBWrapper
from sentence_transformers import SentenceTransformer

# Configuration
CHECKPOINT_PATH = "/workspace/LAM/best/deltanet_shockwave_result.pt"
BASE_MODEL_PATH = "/workspace/LAM/all-MiniLM-L6-v2"
BASELINE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # all-MiniLM-L6-v2 (trained model)
OUTPUT_DIR = Path("/workspace/LAM/mteb_results_local")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Models will be loaded later
deltanet_model = None
baseline_model = None

# Quick test tasks (STS tasks for semantic similarity)
QUICK_TEST_TASKS = [
    "STS12",
    "STS13", 
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
    "SICK-R",
]

# Full English benchmark
FULL_ENGLISH_TASKS = [
    # Retrieval
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
    
    # Reranking
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
    
    # Clustering
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
    
    # Pair Classification
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
    
    # Classification
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    
    # Summarization
    "SummEval",
    
    # STS
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SICK-R",
    
    # Bitext Mining
    "BUCC",
    "Tatoeba",
]

def extract_score(task_result):
    """Extract main score from task result object"""
    if isinstance(task_result, dict) and 'error' in task_result:
        return None
    
    if isinstance(task_result, dict):
        score = task_result.get('main_score') or task_result.get('score')
        if score is not None:
            if hasattr(score, 'item'):
                score = score.item()
            if isinstance(score, (int, float)):
                return float(score)
        return None
    
    if isinstance(task_result, str):
        return None
    
    score = None
    try:
        if hasattr(task_result, 'task_results') and task_result.task_results:
            task_res = task_result.task_results[0]
            if hasattr(task_res, 'scores'):
                scores = task_res.scores
                
                if isinstance(scores, dict):
                    score_dict = None
                    
                    for split_key in ['test', 'validation', 'dev', 'default']:
                        if split_key in scores:
                            split_scores = scores[split_key]
                            if isinstance(split_scores, list) and len(split_scores) > 0:
                                score_dict = split_scores[0]
                                break
                            elif isinstance(split_scores, dict):
                                score_dict = split_scores
                                break
                    
                    if score_dict is None:
                        for val in scores.values():
                            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                                score_dict = val[0]
                                break
                            elif isinstance(val, dict):
                                score_dict = val
                                break
                    
                    if score_dict and isinstance(score_dict, dict):
                        score = (score_dict.get('main_score') or 
                                score_dict.get('cosine_spearman') or 
                                score_dict.get('spearman') or 
                                score_dict.get('cosine_pearson') or 
                                score_dict.get('pearson'))
                    else:
                        score = scores.get('main_score') or scores.get('spearman') or scores.get('cosine_spearman')
                elif hasattr(scores, '__dict__'):
                    score = getattr(scores, 'main_score', None) or getattr(scores, 'spearman', None)
                elif hasattr(scores, '__getitem__'):
                    try:
                        score = scores.get('main_score') if hasattr(scores, 'get') else scores[0] if len(scores) > 0 else None
                    except:
                        pass
        elif hasattr(task_result, 'main_score'):
            score = task_result.main_score
    except Exception as e:
        pass
    
    if score is not None:
        if hasattr(score, 'item'):
            score = score.item()
        if isinstance(score, (int, float)):
            return float(score)
    
    return None

def print_comparison_table(deltanet_results, baseline_results):
    """
    Print a side-by-side comparison table
    
    Args:
        deltanet_results: Dict of DeltaNet results {task_name: result}
        baseline_results: Dict of baseline results {task_name: result}
    """
    print("\n" + "="*120)
    print("DIRECT COMPARISON: DeltaNet vs MiniLM-L6-H384-uncased (Base Pretrained)")
    print("="*120)
    
    # Get all task names
    all_tasks = sorted(set(list(deltanet_results.keys()) + list(baseline_results.keys())))
    
    # Print header
    header = f"{'Task':<35} {'DeltaNet':>12} {'Baseline':>12} {'Diff':>10} {'Status':<20}"
    print(header)
    print("-" * 120)
    
    deltanet_scores = []
    baseline_scores = []
    
    for task_name in all_tasks:
        dn_result = deltanet_results.get(task_name)
        bl_result = baseline_results.get(task_name)
        
        dn_score = extract_score(dn_result) if dn_result else None
        bl_score = extract_score(bl_result) if bl_result else None
        
        row = f"{task_name:<35}"
        
        if dn_score is not None:
            row += f" {dn_score:>12.4f}"
            deltanet_scores.append(dn_score)
        else:
            row += f" {'-':>12}"
        
        if bl_score is not None:
            row += f" {bl_score:>12.4f}"
            baseline_scores.append(bl_score)
        else:
            row += f" {'-':>12}"
        
        # Calculate difference
        if dn_score is not None and bl_score is not None:
            diff = dn_score - bl_score
            diff_pct = (diff / bl_score * 100) if bl_score != 0 else 0
            row += f" {diff:>+10.4f}"
            
            # Status with emoji
            if diff > 0.01:
                status = f"✅ Better ({diff_pct:+.1f}%)"
            elif diff > -0.01:
                status = "≈ Equal"
            else:
                status = f"❌ Behind ({diff_pct:+.1f}%)"
            
            row += f" {status:<20}"
            
            # Highlight STSBenchmark
            if task_name == "STSBenchmark":
                print(f"⭐ {row}")
            else:
                print(f"   {row}")
        else:
            row += f" {'-':>10} {'N/A':<20}"
            print(f"   {row}")
    
    # Print statistics
    print("\n" + "-" * 120)
    print("AGGREGATE STATISTICS")
    print("-" * 120)
    
    if deltanet_scores and baseline_scores:
        dn_avg = sum(deltanet_scores) / len(deltanet_scores)
        bl_avg = sum(baseline_scores) / len(baseline_scores)
        diff_avg = dn_avg - bl_avg
        diff_pct = (diff_avg / bl_avg * 100) if bl_avg != 0 else 0
        
        print(f"{'Metric':<35} {'DeltaNet':>12} {'Baseline':>12} {'Diff':>10} {'Status':<20}")
        print("-" * 120)
        print(f"{'Average Score':<35} {dn_avg:>12.4f} {bl_avg:>12.4f} {diff_avg:>+10.4f} {diff_pct:>+9.1f}%")
        print(f"{'Tasks Evaluated':<35} {len(deltanet_scores):>12} {len(baseline_scores):>12}")
        
        # STSBenchmark specific
        stsb_dn = extract_score(deltanet_results.get('STSBenchmark'))
        stsb_bl = extract_score(baseline_results.get('STSBenchmark'))
        
        if stsb_dn is not None and stsb_bl is not None:
            stsb_diff = stsb_dn - stsb_bl
            stsb_pct = (stsb_diff / stsb_bl * 100) if stsb_bl != 0 else 0
            print(f"{'STSBenchmark (Key Metric)':<35} {stsb_dn:>12.4f} {stsb_bl:>12.4f} {stsb_diff:>+10.4f} {stsb_pct:>+9.1f}%")
            
            print("\n" + "="*120)
            print("VERDICT")
            print("="*120)
            
            if stsb_dn >= 0.82:
                print("🎯 EXCELLENT: DeltaNet performance is excellent (≥0.82)")
            elif stsb_dn >= stsb_bl:
                print(f"✅ COMPETITIVE: DeltaNet matches or exceeds baseline by {stsb_pct:+.1f}%")
                if stsb_dn >= 0.78:
                    print("   Ready for efficient architecture submission!")
                else:
                    print(f"   Gap to 0.78 threshold: {0.78 - stsb_dn:.4f}")
            else:
                print(f"⚠️  BEHIND BASELINE: DeltaNet is {abs(stsb_pct):.1f}% behind baseline")
                print(f"   Baseline score: {stsb_bl:.4f}")
                print(f"   Your score: {stsb_dn:.4f}")
                print(f"   Gap: {abs(stsb_diff):.4f}")
                print("\n   Possible improvements:")
                print("   1. Check if model converged properly")
                print("   2. Try longer training")
                print("   3. Verify distillation is working")
                print("   4. Check resonance flux parameters")
    
    print("="*120)

def evaluate_model(model, model_name, tasks_list, task_type="quick_test"):
    """
    Evaluate a model on specified tasks
    
    Args:
        model: Model to evaluate (either DeltaNet wrapper or SentenceTransformer)
        model_name: Name for saving results
        tasks_list: List of task names to evaluate
        task_type: Type of evaluation (for naming output files)
    """
    print("\n" + "="*80)
    print(f"Starting MTEB Evaluation: {model_name} ({task_type})")
    print(f"Tasks: {len(tasks_list)}")
    print("="*80)
    
    # Get tasks using new API
    tasks = mteb.get_tasks(tasks=tasks_list)
    
    # Run evaluation on each task
    all_results = {}
    
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Evaluating: {task.metadata.name}")
        try:
            result = mteb.evaluate(
                model,
                task,
                show_progress_bar=True,
                overwrite_strategy="always"
            )
            all_results[task.metadata.name] = result
            
            # Extract and print score immediately
            score = extract_score(result)
            if score is not None:
                print(f"✅ {task.metadata.name} complete - Score: {score:.4f}")
            else:
                print(f"✅ {task.metadata.name} complete")
        except Exception as e:
            print(f"❌ Error evaluating {task.metadata.name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task.metadata.name] = {"error": str(e)}
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = OUTPUT_DIR / f"{model_name}_{task_type}_results_{timestamp}.json"
    
    results_to_save = {}
    for task_name, task_result in all_results.items():
        if isinstance(task_result, dict) and 'error' in task_result:
            results_to_save[task_name] = task_result
        else:
            score = extract_score(task_result)
            if score is not None:
                results_to_save[task_name] = {'main_score': score}
            else:
                results_to_save[task_name] = str(task_result)
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    print(f"\n✅ {model_name} evaluation complete!")
    print(f"   Results saved to: {results_file}")
    
    return all_results

if __name__ == "__main__":
    import sys
    
    # Check for custom checkpoint path
    checkpoint_path = CHECKPOINT_PATH
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--checkpoint" and i + 1 < len(sys.argv):
                checkpoint_path = sys.argv[i + 1]
                print(f"Using custom checkpoint: {checkpoint_path}")
            elif arg == "--checkpoint":
                print("Error: --checkpoint requires a path argument")
                sys.exit(1)
    
    CHECKPOINT_PATH = checkpoint_path
    
    # Determine which tasks to run
    if "--full" in sys.argv:
        print("Running FULL English benchmark evaluation (this will take 2-4 hours)")
        print("Press Ctrl+C to cancel, or wait 5 seconds...")
        import time
        time.sleep(5)
        tasks_to_run = FULL_ENGLISH_TASKS
        task_type = "full_english"
    else:
        print("Running QUICK test evaluation (10-20 minutes for both models)")
        print("Use --full flag for complete evaluation")
        tasks_to_run = QUICK_TEST_TASKS
        task_type = "quick_test"
    
    print("\n" + "="*80)
    print("COMPARATIVE MTEB EVALUATION")
    print("="*80)
    print(f"Tasks: {len(tasks_to_run)}")
    print(f"Models: 2 (DeltaNet + Baseline)")
    print("="*80)
    
    # Load models
    print("\n📦 Loading models...")
    
    print("\n1️⃣  Loading DeltaNet model...")
    deltanet_model = DeltaNetMTEBWrapper(
        checkpoint_path=CHECKPOINT_PATH,
        base_model_path=BASE_MODEL_PATH,
        device=device
    )
    print("   ✅ DeltaNet loaded")
    
    print("\n2️⃣  Loading baseline model (MiniLM-L6-H384-uncased - base pretrained) from HuggingFace...")
    # Load directly from HuggingFace using transformers + simple wrapper
    from transformers import AutoModel, AutoTokenizer
    import torch.nn.functional as F
    
    class BaselineWrapper:
        """Simple wrapper for baseline model to work with MTEB"""
        def __init__(self, model_name, device):
            print(f"   Loading from HuggingFace: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
            self.max_seq_length = 128  # Match DeltaNet training
            self.device = device
            
        def mean_pooling(self, token_embeddings, attention_mask):
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        
        def encode(self, sentences, batch_size=32, show_progress_bar=False, 
                   convert_to_numpy=True, normalize_embeddings=False, **kwargs):
            # Handle DataLoader (from MTEB)
            from torch.utils.data import DataLoader
            if isinstance(sentences, DataLoader):
                all_embeddings = []
                iterator = sentences
                if show_progress_bar:
                    from tqdm import tqdm
                    iterator = tqdm(iterator, desc="Encoding")
                
                with torch.no_grad():
                    for batch in iterator:
                        # Extract sentences from batch (could be dict or list)
                        if isinstance(batch, dict):
                            # Try common keys
                            batch_sentences = batch.get('text', batch.get('sentence', batch.get('sentences', None)))
                            if batch_sentences is None:
                                # Try to get first value that looks like text
                                for v in batch.values():
                                    if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
                                        batch_sentences = v
                                        break
                        elif isinstance(batch, (list, tuple)):
                            batch_sentences = batch
                        else:
                            raise ValueError(f"Unexpected batch format: {type(batch)}")
                        
                        if batch_sentences is None:
                            continue
                        
                        encoded = self.tokenizer(
                            batch_sentences,
                            padding='max_length',
                            max_length=self.max_seq_length,
                            truncation=True,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        outputs = self.model(**encoded)
                        embeddings = self.mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
                        if normalize_embeddings:
                            embeddings = F.normalize(embeddings, p=2, dim=1)
                        
                        all_embeddings.append(embeddings.cpu())
                
                # Concatenate all embeddings
                if all_embeddings:
                    all_embeddings = torch.cat(all_embeddings, dim=0)
                else:
                    all_embeddings = torch.empty((0, 384))  # all-MiniLM-L6-v2 has 384 dims
                
                if convert_to_numpy:
                    return all_embeddings.numpy()
                return all_embeddings
            
            # Handle string or list of strings
            if isinstance(sentences, str):
                sentences = [sentences]
            
            all_embeddings = []
            from tqdm import tqdm
            iterator = range(0, len(sentences), batch_size)
            if show_progress_bar:
                iterator = tqdm(iterator, desc="Encoding")
            
            with torch.no_grad():
                for start_idx in iterator:
                    end_idx = min(start_idx + batch_size, len(sentences))
                    batch_sentences = sentences[start_idx:end_idx]
                    
                    encoded = self.tokenizer(
                        batch_sentences,
                        padding='max_length',
                        max_length=self.max_seq_length,
                        truncation=True,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    outputs = self.model(**encoded)
                    embeddings = self.mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
                    if normalize_embeddings:
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    all_embeddings.append(embeddings.cpu())
            
            all_embeddings = torch.cat(all_embeddings, dim=0)
            if convert_to_numpy:
                return all_embeddings.numpy()
            return all_embeddings
    
    baseline_model = BaselineWrapper(BASELINE_MODEL_NAME, device=device)
    print(f"   ✅ Baseline loaded (max_seq_length: {baseline_model.max_seq_length} - matching DeltaNet training)")
    
    # Run evaluations
    try:
        print("\n" + "="*80)
        print("PHASE 1: Evaluating DeltaNet")
        print("="*80)
        deltanet_results = evaluate_model(
            deltanet_model, 
            "deltanet", 
            tasks_to_run, 
            task_type=task_type
        )
        
        print("\n" + "="*80)
        print("PHASE 2: Evaluating Baseline")
        print("="*80)
        baseline_results = evaluate_model(
            baseline_model, 
            "baseline", 
            tasks_to_run, 
            task_type=task_type
        )
        
        # Print comparison
        print_comparison_table(deltanet_results, baseline_results)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("\nNext steps:")
        print("  • If DeltaNet ≥ baseline: Great! Your architecture works")
        print("  • If DeltaNet < baseline: Check training/architecture")
        print("  • For full evaluation: python test_mteb_local_with_baseline.py --full")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        print("Partial results may be available in the output folder")
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()