"""
Download all-MiniLM training datasets
Saves to /workspace/LAM/data/ in .jsonl.gz format
"""

import gzip
import json
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

output_dir = Path("/workspace/LAM/data")
output_dir.mkdir(exist_ok=True)

def save_pairs(filename, pairs):
    """Save pairs to .jsonl.gz format"""
    filepath = output_dir / filename
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        for pair in tqdm(pairs, desc=f"Saving {filename}"):
            json.dump(pair, f)
            f.write('\n')
    print(f"✅ Saved {len(pairs):,} pairs to {filepath}")

# 1. AllNLI - Load from HuggingFace
print("\n1️⃣  AllNLI - CRITICAL for hard negatives")
allnli_path = output_dir / "AllNLI.jsonl.gz"
if allnli_path.exists():
    print(f"   ✅ AllNLI.jsonl.gz already exists, skipping download")
else:
    try:
        print("   Loading from HuggingFace: sentence-transformers/all-nli")
        allnli = load_dataset("sentence-transformers/all-nli", split="train")
        pairs = []
        for item in tqdm(allnli, desc="Processing AllNLI"):
            # AllNLI has 'sentence1' and 'sentence2' fields
            if 'sentence1' in item and 'sentence2' in item:
                pairs.append([item['sentence1'], item['sentence2']])
        save_pairs("AllNLI.jsonl.gz", pairs)
    except Exception as e:
        print(f"   ⚠️  Failed to load AllNLI: {e}")
        print("   Manual download: https://huggingface.co/datasets/sentence-transformers/all-nli")

# 2. altlex - Skip (dataset not available on HuggingFace)
print("\n2️⃣  altlex...")
altlex_path = output_dir / "altlex.jsonl.gz"
if altlex_path.exists():
    print(f"   ✅ altlex.jsonl.gz already exists, skipping download")
else:
    print("   ⚠️  altlex dataset not available on HuggingFace")
    print("   This dataset is optional and not critical for training")
    print("   Training will continue without altlex")
    print("   If needed, download manually from: https://github.com/chridey/altlex")

# 3. SimpleWiki
print("\n3️⃣  Downloading SimpleWiki...")
try:
    simplewiki = load_dataset("sentence-transformers/simple-wiki", split="train")
    pairs = [[item['sentence1'], item['sentence2']] for item in simplewiki]
    save_pairs("SimpleWiki.jsonl.gz", pairs)
except:
    print("   ⚠️  Failed - may need to scrape Wikipedia")

# 4. SQuAD pairs
print("\n4️⃣  Downloading SQuAD...")
try:
    squad = load_dataset("squad", split="train")
    pairs = []
    for item in squad:
        question = item['question']
        for answer in item['answers']['text']:
            pairs.append([question, answer])
    save_pairs("squad_pairs.jsonl.gz", pairs)
except:
    print("   ⚠️  Failed")

# 5. MS MARCO triplets (with hard negatives) - FIXED VERSION
print("\n5️⃣  Downloading MS MARCO...")
msmarco_path = output_dir / "msmarco-triplets.jsonl.gz"
msmarco_pairs_path = output_dir / "msmarco-pairs.jsonl.gz"
if msmarco_path.exists() or msmarco_pairs_path.exists():
    print(f"   ✅ MS MARCO file already exists, skipping download")
else:
    # Try direct MS MARCO v1.1 first (more reliable)
    try:
        print("   Loading from: ms_marco v1.1 (direct approach - 500K pairs)...")
        msmarco = load_dataset("ms_marco", "v1.1", split="train[:500000]", cache_dir="/workspace/.cache/huggingface/datasets")
        pairs = []
        for item in tqdm(msmarco, desc="Processing MS MARCO"):
            try:
                query = item.get('query', '')
                if not query:
                    continue
                passages = item.get('passages', {})
                if isinstance(passages, str):
                    import json
                    passages = json.loads(passages)
                if isinstance(passages, dict) and 'is_selected' in passages:
                    selected = [i for i, sel in enumerate(passages.get('is_selected', [])) if sel == 1]
                    for idx in selected[:1]:  # Take first selected passage
                        if idx < len(passages.get('passage_text', [])):
                            passage = passages['passage_text'][idx]
                            if passage and len(passage) > 10:
                                pairs.append([query, passage])
                                break
            except Exception as e:
                continue
        
        if pairs:
            save_pairs("msmarco-pairs.jsonl.gz", pairs)
            print(f"   ✅ MS MARCO: {len(pairs):,} pairs saved")
        else:
            raise Exception("No pairs extracted")
    except Exception as e:
        print(f"   ⚠️  Direct approach failed: {e}")
        # Try alternative: sentence-transformers/msmarco-hard-negatives with streaming
        try:
            print("   Trying alternative: sentence-transformers/msmarco-hard-negatives (streaming)...")
            msmarco = load_dataset("sentence-transformers/msmarco-hard-negatives", split="train", streaming=True)
            triplets = []
            count = 0
            max_samples = 500000  # Limit to 500K samples
            
            for item in tqdm(msmarco, desc="Processing MS MARCO (streaming)"):
                if count >= max_samples:
                    break
                try:
                    if 'query' in item and 'pos' in item:
                        query = str(item['query']) if item['query'] else ""
                        pos = item['pos']
                        
                        # Handle list or single value for pos
                        if isinstance(pos, list) and len(pos) > 0:
                            pos_text = str(pos[0]) if pos[0] else ""
                        elif pos:
                            pos_text = str(pos)
                        else:
                            pos_text = ""
                        
                        if query and pos_text and len(query) > 5 and len(pos_text) > 10:
                            triplets.append([query, pos_text])
                            count += 1
                except Exception:
                    continue
            
            if triplets:
                save_pairs("msmarco-triplets.jsonl.gz", triplets)
                print(f"   ✅ MS MARCO (streaming): {len(triplets):,} pairs saved")
            else:
                print("   ⚠️  No valid pairs found")
        except Exception as e2:
            print(f"   ⚠️  Streaming also failed: {e2}")
            print("   Note: MS MARCO may need manual download or different approach")
            print("   Training can continue with other datasets")

# 6. Reddit (title-body) - Get 3M records total
print("\n6️⃣  Downloading Reddit title-body...")
reddit_path = output_dir / "reddit_title-body.jsonl.gz"
target_total = 3000000  # 3M total records

# Check existing file
existing_count = 0
if reddit_path.exists():
    try:
        with gzip.open(reddit_path, 'rt') as f:
            existing_count = sum(1 for _ in f)
        print(f"   📊 Existing file has {existing_count:,} pairs")
    except:
        existing_count = 0

if existing_count >= target_total:
    print(f"   ✅ reddit_title-body.jsonl.gz already has {existing_count:,} pairs (target: {target_total:,}), skipping download")
else:
    needed = target_total - existing_count
    print(f"   📥 Need {needed:,} more pairs to reach {target_total:,} total")
    
    try:
        print(f"   Loading {needed:,} additional records from: sentence-transformers/reddit-title-body")
        # Use non-streaming with slice for efficient appending
        start_idx = existing_count
        end_idx = target_total
        print(f"   📥 Fetching records {start_idx:,} to {end_idx:,}...")
        reddit = load_dataset("sentence-transformers/reddit-title-body", split=f"train[{start_idx}:{end_idx}]", cache_dir="/workspace/.cache/huggingface/datasets")
        new_pairs = []
        
        for item in tqdm(reddit, desc="Processing Reddit"):
            try:
                if 'title' in item and 'body' in item:
                    title = str(item['title']) if item['title'] else ""
                    body = str(item['body']) if item['body'] else ""
                    if title and body and len(title) > 5 and len(body) > 10:
                        new_pairs.append([title, body])
            except Exception:
                continue
        
        if new_pairs:
            # Append to existing file or create new
            if existing_count > 0:
                # Append mode
                print(f"   📝 Appending {len(new_pairs):,} pairs to existing file...")
                with gzip.open(reddit_path, 'at', encoding='utf-8') as f:
                    for pair in tqdm(new_pairs, desc="Appending"):
                        json.dump(pair, f)
                        f.write('\n')
                print(f"   ✅ Reddit: Appended {len(new_pairs):,} pairs (total now: {existing_count + len(new_pairs):,})")
            else:
                # New file
                save_pairs("reddit_title-body.jsonl.gz", new_pairs)
                print(f"   ✅ Reddit: {len(new_pairs):,} pairs saved")
        else:
            print("   ⚠️  No valid pairs found")
    except Exception as e:
        print(f"   ⚠️  Failed to load Reddit: {e}")
        # Try streaming as fallback
        try:
            print("   Trying streaming approach as fallback...")
            reddit = load_dataset("sentence-transformers/reddit-title-body", split="train", streaming=True)
            new_pairs = []
            count = 0
            skipped = 0
            
            for item in tqdm(reddit, desc="Processing Reddit (streaming)"):
                if count >= needed:
                    break
                try:
                    # Skip existing records
                    if skipped < existing_count:
                        skipped += 1
                        continue
                    
                    if 'title' in item and 'body' in item:
                        title = str(item['title']) if item['title'] else ""
                        body = str(item['body']) if item['body'] else ""
                        if title and body and len(title) > 5 and len(body) > 10:
                            new_pairs.append([title, body])
                            count += 1
                except Exception:
                    continue
            
            if new_pairs:
                if existing_count > 0:
                    with gzip.open(reddit_path, 'at', encoding='utf-8') as f:
                        for pair in tqdm(new_pairs, desc="Appending"):
                            json.dump(pair, f)
                            f.write('\n')
                    print(f"   ✅ Reddit (streaming): Appended {len(new_pairs):,} pairs (total: {existing_count + len(new_pairs):,})")
                else:
                    save_pairs("reddit_title-body.jsonl.gz", new_pairs)
                    print(f"   ✅ Reddit (streaming): {len(new_pairs):,} pairs saved")
        except Exception as e2:
            print(f"   ⚠️  Streaming also failed: {e2}")
            print("   Training can continue with existing Reddit data")

# 7. Reddit comments (parent-reply)
print("\n7️⃣  Downloading Reddit comments...")
print("   Use existing reddit_title_text.jsonl.gz if you have it")

# 8. Quora Question Pairs
print("\n8️⃣  Downloading Quora duplicates...")
try:
    qqp = load_dataset("glue", "qqp", split="train")
    pairs = [[item['question1'], item['question2']] 
             for item in qqp if item['label'] == 1]
    save_pairs("quora_duplicate_questions.jsonl.gz", pairs)
except:
    print("   ⚠️  Failed")

# 9. StackExchange duplicates
print("\n9️⃣  Downloading StackExchange duplicates...")
try:
    stack = load_dataset("flax-sentence-embeddings/stackexchange_duplicates", split="train[:1000000]")
    pairs = [[item['title_1'], item['title_2']] for item in stack]
    save_pairs("stackexchange_duplicate_questions.jsonl.gz", pairs)
except:
    print("   ⚠️  Failed")

# 10. TriviaQA
print("\n🔟 Downloading TriviaQA...")
try:
    trivia = load_dataset("trivia_qa", "rc.nocontext", split="train[:500000]")
    pairs = []
    for item in trivia:
        question = item['question']
        for answer in item['answer']['aliases']:
            pairs.append([question, answer])
    save_pairs("TriviaQA.jsonl.gz", pairs)
except:
    print("   ⚠️  Failed")

# 11. ParaNMT (HUGE - 50M pairs, use subset)
print("\n1️⃣1️⃣  Downloading ParaNMT (this will take time)...")
try:
    paranmt = load_dataset("embedding-data/sentence-compression", split="train[:10000000]")
    pairs = [[item['set'][0], item['set'][1]] for item in paranmt if len(item['set']) >= 2]
    save_pairs("ParaNMT.jsonl.gz", pairs)
except:
    print("   ⚠️  Failed - manually download from: https://www.cs.cmu.edu/~jwieting/")

print("\n" + "="*80)
print("✅ DOWNLOAD COMPLETE")
print("="*80)
print(f"All datasets saved to: {output_dir}")
print("\nNext step: Run training with:")
print("python train_many_data_files_v2.py --data_folder /workspace/LAM/data data_config_allmini.json output/")