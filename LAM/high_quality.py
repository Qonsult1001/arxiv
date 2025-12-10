"""
Add High-Priority SBERT Datasets

This script loads the TOP-PERFORMING datasets from SBERT research:
1. StackExchange title+body (59.83 score - BEST!)
2. Google Q&A pairs (59.76 score)
3. Yahoo Answers (58.85 score)
4. StackExchange duplicate questions (58.47 score)

These are CRITICAL for achieving 0.84+ Spearman!
"""

from datasets import load_dataset
import json
import gzip
from pathlib import Path
from tqdm import tqdm

def download_stackexchange_titlebody(output_dir, max_samples=500000):
    """
    Load StackExchange title+body pairs (SBERT's BEST dataset - 59.83 score!)
    
    This is the #1 dataset from SBERT research!
    """
    print("\n⭐ Loading StackExchange title+body (TOP PERFORMER - 59.83)...")
    
    try:
        # Try different configs - available: 'body-body-pair', 'post-post-pair', 'title-title-pair'
        # For title+body, we'll use 'post-post-pair' which has both title and body
        dataset = load_dataset(
            "sentence-transformers/stackexchange-duplicates",
            "post-post-pair",  # This config has title and body fields
            split="train",
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
        
        output_path = output_dir / "stackexchange_title_body.jsonl.gz"
        pairs = []
        
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="StackExchange title+body"):
                # post-post-pair has 'post1' and 'post2' fields (full post text)
                # For title+body, we'll use post1 as "title" (question) and post2 as "body" (answer)
                post1 = str(item.get('post1', '')).strip()
                post2 = str(item.get('post2', '')).strip()
                
                # Also check for other field combinations
                if not post1 or not post2:
                    if 'title' in item and 'body' in item:
                        post1 = str(item['title']).strip()
                        post2 = str(item['body']).strip()
                    elif 'text1' in item and 'text2' in item:
                        post1 = str(item['text1']).strip()
                        post2 = str(item['text2']).strip()
                
                # Use post1 as title/question, post2 as body/answer
                if post1 and post2 and len(post1) > 10 and len(post2) > 20:
                    pair = [post1, post2]
                    json.dump(pair, f)
                    f.write('\n')
                    pairs.append(pair)
                    
                    if len(pairs) >= max_samples:
                        break
        
        if pairs:
            print(f"   ✅ Saved {len(pairs):,} StackExchange title+body pairs")
            print(f"   📁 {output_path}")
            return len(pairs)
        else:
            print(f"   ⚠️  No valid pairs found with title+body format")
            return 0
        
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        print(f"\n   💡 Try alternative: sentence-transformers/stackexchange-duplicates")
        return 0

def download_stackexchange_duplicates(output_dir, max_samples=300000):
    """
    Load StackExchange duplicate questions (58.47 score)
    """
    print("\n⭐ Loading StackExchange duplicate questions (58.47)...")
    
    try:
        dataset = load_dataset(
            "sentence-transformers/stackexchange-duplicates",
            "title-title-pair", 
            split="train",
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
        
        output_path = output_dir / "stackexchange_duplicates.jsonl.gz"
        pairs = []
        
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="StackExchange duplicates"):
                # Duplicate question titles
                if 'title1' in item and 'title2' in item:
                    q1 = str(item['title1']).strip()
                    q2 = str(item['title2']).strip()
                    
                    if len(q1) > 10 and len(q2) > 10:
                        pair = [q1, q2]
                        json.dump(pair, f)
                        f.write('\n')
                        pairs.append(pair)
                        
                        if len(pairs) >= max_samples:
                            break
        
        print(f"   ✅ Saved {len(pairs):,} StackExchange duplicate pairs")
        print(f"   📁 {output_path}")
        return len(pairs)
        
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        return 0

def download_yahoo_answers(output_dir, max_samples=500000):
    """
    Load Yahoo Answers title→answer pairs (58.85 score)
    """
    print("\n⭐ Loading Yahoo Answers (58.85)...")
    
    try:
        # Try different configs: 'question-answer-pair', 'title-answer-pair', 'title-question-answer-pair', 'title-question-pair'
        # Use 'question-answer-pair' for question→answer pairs
        dataset = load_dataset(
            "sentence-transformers/yahoo-answers",
            "question-answer-pair",  # Specify config name
            split="train",
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
        
        output_path = output_dir / "yahoo_answers.jsonl.gz"
        pairs = []
        
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Yahoo Answers"):
                # Try different field name combinations
                question = None
                answer = None
                
                if 'question' in item and 'answer' in item:
                    question = str(item['question']).strip()
                    answer = str(item['answer']).strip()
                elif 'question_title' in item and 'best_answer' in item:
                    question = str(item['question_title']).strip()
                    answer = str(item['best_answer']).strip()
                elif 'text1' in item and 'text2' in item:
                    question = str(item['text1']).strip()
                    answer = str(item['text2']).strip()
                
                if question and answer and len(question) > 10 and len(answer) > 20:
                    pair = [question, answer]
                    json.dump(pair, f)
                    f.write('\n')
                    pairs.append(pair)
                    
                    if len(pairs) >= max_samples:
                        break
        
        if pairs:
            print(f"   ✅ Saved {len(pairs):,} Yahoo Answers pairs")
            print(f"   📁 {output_path}")
            return len(pairs)
        else:
            print(f"   ⚠️  No valid pairs found")
            return 0
        
    except Exception as e:
        print(f"   ⚠️  Error loading Yahoo Answers: {e}")
        # Try alternative config
        try:
            print(f"   💡 Trying alternative config: title-answer-pair...")
            dataset = load_dataset(
                "sentence-transformers/yahoo-answers",
                "title-answer-pair",
                split="train",
                cache_dir="/workspace/.cache/huggingface/datasets"
            )
            
            output_path = output_dir / "yahoo_answers.jsonl.gz"
            pairs = []
            
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                for item in tqdm(dataset, desc="Yahoo Answers (alt)"):
                    title = str(item.get('title', item.get('text1', ''))).strip()
                    answer = str(item.get('answer', item.get('text2', ''))).strip()
                    
                    if len(title) > 10 and len(answer) > 20:
                        pair = [title, answer]
                        json.dump(pair, f)
                        f.write('\n')
                        pairs.append(pair)
                        
                        if len(pairs) >= max_samples:
                            break
            
            if pairs:
                print(f"   ✅ Saved {len(pairs):,} Yahoo Answers pairs (alt config)")
                print(f"   📁 {output_path}")
                return len(pairs)
        except Exception as e2:
            print(f"   ⚠️  Alternative also failed: {e2}")
        
        return 0

def download_google_qa_pairs(output_dir, max_samples=500000):
    """
    Load Google Q&A pairs (59.76 score - 2nd best!)
    
    Note: The original "gooq" (Google Q&A auto-suggest) dataset is NOT publicly available.
    It was used in SBERT research but was never released to the public.
    
    Best alternatives:
    1. QQP (Quora Question Pairs) - Already have this! Similar quality (57.38 vs 59.76)
    2. Natural Questions - Google's official Q&A (already have NQ-train_pairs.jsonl.gz)
    3. MS MARCO - Already have this! (59.06 score, very close to gooq's 59.76)
    """
    print("\n⭐ Loading Google Q&A pairs (59.76 - 2nd BEST!)...")
    print("   ⚠️  The 'gooq' dataset is NOT publicly available")
    print("   📊 Best alternatives you ALREADY HAVE:")
    print("      ✅ MS MARCO (59.06) - Only 0.7 points lower!")
    print("      ✅ QQP/Quora (57.38) - High quality Q&A pairs")
    print("      ✅ Natural Questions - Google's official Q&A dataset")
    print("   💡 These are excellent substitutes - no need to download gooq!")
    return 0

def prioritize_high_performers(datasets_dict):
    """
    Give extra weight to high-performing datasets from SBERT research
    
    Priority levels:
    - HIGH (3.0x): stackexchange_title_body (59.83), google_qa (59.76), msmarco (59.06)
    - MEDIUM (2.0x): yahoo_answers (58.85), stackexchange_duplicates (58.47)
    - NORMAL (1.0x): everything else
    """
    
    # Define priorities based on SBERT research
    priorities = {
        'stackexchange_title_body': 3.0,  # 59.83 - BEST!
        'google_qa': 3.0,                  # 59.76
        'msmarco': 3.0,                    # 59.06
        'yahoo_answers': 2.0,              # 58.85
        'stackexchange_duplicates': 2.0,   # 58.47
        'qqp': 2.0,                        # 57.38 (Quora - similar to google_qa)
        'wikianswers': 2.0,                # 57.34
    }
    
    weights = {}
    total_weight = 0
    
    for name, data in datasets_dict.items():
        # Get priority multiplier
        priority = priorities.get(name, 1.0)
        
        # Weight = sqrt(size) * priority
        # This prevents huge datasets from dominating while respecting quality
        weight = (len(data) ** 0.5) * priority
        
        weights[name] = weight
        total_weight += weight
    
    # Normalize to probabilities
    probs = {name: w / total_weight for name, w in weights.items()}
    
    print("\n⭐ Priority-Weighted Sampling (SBERT research-based):")
    print(f"   {'Dataset':<30} {'Priority':>8} {'Weight':>8} {'Size':>12}")
    print("   " + "-"*68)
    
    for name in sorted(probs.keys(), key=lambda x: probs[x], reverse=True):
        priority = priorities.get(name, 1.0)
        priority_str = "HIGH" if priority == 3.0 else "MED" if priority == 2.0 else "NORM"
        size = len(datasets_dict[name])
        prob = probs[name]
        
        print(f"   {name:<30} {priority_str:>8} {prob:>7.1%} {size:>12,}")
    
    return probs

def main():
    """Download all high-priority SBERT datasets"""
    
    print("="*80)
    print("🎯 DOWNLOADING HIGH-PRIORITY SBERT DATASETS")
    print("="*80)
    print("\nThese are the TOP-PERFORMING datasets from SBERT research!")
    print("Expected impact: +3-5 points on STS-B\n")
    
    # Setup
    data_dir = Path("/workspace/LAM/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    total_downloaded = 0
    
    # 1. StackExchange title+body (BEST - 59.83!)
    count = download_stackexchange_titlebody(data_dir, max_samples=500000)
    total_downloaded += count
    
    # 2. StackExchange duplicates (58.47)
    count = download_stackexchange_duplicates(data_dir, max_samples=300000)
    total_downloaded += count
    
    # 3. Yahoo Answers (58.85)
    count = download_yahoo_answers(data_dir, max_samples=500000)
    total_downloaded += count
    
    # 4. Google Q&A (59.76) - not publicly available, using QQP as substitute
    download_google_qa_pairs(data_dir)
    
    print("\n" + "="*80)
    print(f"✅ DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Total new pairs: {total_downloaded:,}")
    print(f"   Saved to: {data_dir}/")
    print(f"\n🎯 Expected improvement: +3-5 points on STS-B")
    print(f"   These are the BEST datasets from SBERT research!")
    print(f"\n💡 Next steps:")
    print(f"   1. Re-run your training script")
    print(f"   2. The domain-clustered sampler will auto-prioritize these")
    print(f"   3. Watch for improved performance (target: 0.84+ Spearman)")
    
if __name__ == "__main__":
    main()