"""
⚡ PRE-COMPUTE TEACHER EMBEDDINGS (One-Time Setup)

This script pre-computes teacher embeddings for all unique sentences
in your training datasets and saves them to disk.

Benefits:
- 5-10x training speedup (no forward_teacher() calls during training)
- Frees GPU memory (can remove teacher model from training script)
- One-time cost: ~10-20GB disk space

Usage:
    python precompute_teacher_embeddings.py
"""

import torch
import json
import gzip
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import hashlib
import pickle

def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling for sentence embeddings"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def load_all_unique_sentences():
    """Load all unique sentences from all training datasets"""
    print("📚 Loading all unique sentences from datasets...")
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    unique_sentences = set()
    
    # Load from all dataset files
    dataset_files = [
        "AllNLI.jsonl.gz",
        "QQP.jsonl.gz",
        "MS_MARCO.jsonl.gz",
        "Reddit.jsonl.gz",
        "WikiAnswers_2M.jsonl.gz",
        "WikiAnswers_1M.jsonl.gz",
        "WikiAnswers_100k.jsonl.gz",
        "StackExchange_title_body.jsonl.gz",
        "Yahoo_Answers.jsonl.gz",
        "TriviaQA.jsonl.gz",
        "SICK.jsonl.gz",
        "STS-B.jsonl.gz",
    ]
    
    for filename in dataset_files:
        filepath = data_dir / filename
        if not filepath.exists():
            continue
        
        print(f"   Loading {filename}...")
        count = 0
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if isinstance(data, list) and len(data) >= 2:
                        # Extract sentence pairs
                        s1, s2 = str(data[0]).strip(), str(data[1]).strip()
                        if len(s1) > 5 and len(s2) > 5:
                            unique_sentences.add(s1)
                            unique_sentences.add(s2)
                            count += 1
                except:
                    continue
        
        print(f"      Added {count} pairs from {filename}")
    
    print(f"\n✅ Total unique sentences: {len(unique_sentences):,}")
    return list(unique_sentences)

def precompute_embeddings(teacher_model_name, batch_size=64, output_path=None):
    """Pre-compute teacher embeddings for all unique sentences"""
    
    # Load teacher model
    print(f"\n🤖 Loading teacher model: {teacher_model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = AutoModel.from_pretrained(teacher_model_name)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model.to(device)
    teacher_model.eval()
    
    # Load all unique sentences
    sentences = load_all_unique_sentences()
    
    # Output path
    if output_path is None:
        output_path = Path(__file__).parent / "data" / "teacher_embeddings.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pre-compute embeddings
    print(f"\n⚡ Pre-computing embeddings (batch_size={batch_size})...")
    embeddings_dict = {}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
            batch_sentences = sentences[i:i+batch_size]
            
            # Tokenize
            tokens = teacher_tokenizer(
                batch_sentences,
                padding=True,
                max_length=256,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Encode
            outputs = teacher_model(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask']
            )
            
            # Mean pooling and normalize
            embeddings = mean_pooling(outputs.last_hidden_state, tokens['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Store (move to CPU to save GPU memory)
            for j, sentence in enumerate(batch_sentences):
                embeddings_dict[sentence] = embeddings[j].cpu()
    
    # Save to file
    print(f"\n💾 Saving embeddings to {output_path}...")
    torch.save(embeddings_dict, output_path)
    
    # Calculate size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved {len(embeddings_dict):,} embeddings ({size_mb:.1f} MB)")
    print(f"   File: {output_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    precompute_embeddings(
        teacher_model_name=args.teacher_model,
        batch_size=args.batch_size,
        output_path=args.output
    )
    
    print("\n🎉 Pre-computation complete!")
    print("   Next: Update train_6layer_deltanet_3.py to use pre-computed embeddings")

