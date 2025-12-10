"""
Download STS 2012-2017 datasets and save to /workspace/LAM/data/
"""
from datasets import load_dataset
from pathlib import Path
import csv
import json
import gzip

data_dir = Path("/workspace/LAM/data")
data_dir.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING STS 2012-2017 DATASETS")
print("="*80)

# STS datasets available on HuggingFace
sts_datasets = {
    "sts12": "mteb/sts12-sts",
    "sts13": "mteb/sts13-sts", 
    "sts14": "mteb/sts14-sts",
    "sts15": "mteb/sts15-sts",
    "sts16": "mteb/sts16-sts",
    "sts17": "mteb/sts17-crosslingual-sts",  # Note: crosslingual, may need filtering
}

all_sts_pairs = []

for year, dataset_name in sts_datasets.items():
    print(f"\n📥 Downloading STS {year}...")
    try:
        # Try train split first, then test split (STS 2013-2017 only have test)
        dataset = None
        try:
            dataset = load_dataset(dataset_name, split="train", cache_dir="/workspace/.cache/huggingface/datasets")
        except:
            try:
                dataset = load_dataset(dataset_name, split="test", cache_dir="/workspace/.cache/huggingface/datasets")
                print(f"   ℹ️  Using test split (no train available)")
            except:
                pass
        
        if dataset is None:
            print(f"   ⚠️  No data available for {year}")
            continue
        
        count = 0
        for item in dataset:
            try:
                # Different datasets have different field names
                s1 = item.get('sentence1', item.get('sentence_a', item.get('text1', item.get('sentence', ''))))
                s2 = item.get('sentence2', item.get('sentence_b', item.get('text2', '')))
                score = item.get('score', item.get('label', item.get('similarity_score', item.get('score', 0.0))))
                
                # Normalize score to 0-5 range if needed
                if isinstance(score, (int, float)):
                    if score > 5:
                        score = score / 2.0  # If 0-10 scale, convert to 0-5
                    if score < 0:
                        score = 0.0
                    if score > 5:
                        score = 5.0
                else:
                    continue
                
                if s1 and s2 and len(s1) > 5 and len(s2) > 5:
                    all_sts_pairs.append([s1, s2, score])
                    count += 1
            except Exception as e:
                continue
        
        print(f"   ✅ STS {year}: {count:,} pairs")
        
    except Exception as e:
        print(f"   ⚠️  Failed to load {year}: {e}")

# Also try loading from sentence-transformers/stsb which might have more STS data
print(f"\n📥 Trying sentence-transformers/stsb...")
try:
    stsb = load_dataset("sentence-transformers/stsb", split="train", cache_dir="/workspace/.cache/huggingface/datasets")
    for item in stsb:
        s1 = item.get('sentence1', '')
        s2 = item.get('sentence2', '')
        score = item.get('score', 0.0)
        if s1 and s2 and len(s1) > 5 and len(s2) > 5:
            all_sts_pairs.append([s1, s2, score])
    print(f"   ✅ Added {len(stsb):,} pairs from sentence-transformers/stsb")
except Exception as e:
    print(f"   ⚠️  Could not load: {e}")

# Save all STS pairs to a single file
if all_sts_pairs:
    output_file = data_dir / "sts_2012_2017_pairs.jsonl.gz"
    print(f"\n💾 Saving {len(all_sts_pairs):,} STS pairs to {output_file}...")
    
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        for pair in all_sts_pairs:
            json.dump(pair, f)
            f.write('\n')
    
    print(f"✅ Saved {len(all_sts_pairs):,} STS pairs to {output_file}")
else:
    print(f"\n⚠️  No STS pairs collected")

print("\n" + "="*80)
print("✅ DOWNLOAD COMPLETE")
print("="*80)

